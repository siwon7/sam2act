"""Idea 4: Contrastive Phase-Aware Memory Selection.

Phase discovery is automatic (from 3D position + gripper clustering).
Contrastive loss directly optimizes for peak selection at ambiguous keyframes.
No graph, no FiLM -- just phase embedding + peak selection head.

Architecture:
    1. PhaseEncoder:     memory-enhanced feature -> 64-dim phase embedding
    2. ContrastiveLoss:  position-cluster-based positive/negative pair mining
    3. PeakSelector:     phase_embedding -> peak selection logits

Integration point: plugs into MVT_SAM2_Single after sam2_forward_with_memory,
before/after heatmap decode.  Runs per-step during the sequential training loop.

Key insight vs KC-VLA:
    - KC-VLA's contrastive is on raw images -> learns "is this a keyframe?"
    - Ours is on memory-enhanced features -> learns "which phase am I in?"
    - KC-VLA needs hard-coded phase labels per task
    - Ours discovers phases automatically via 3D position clustering
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ContrastivePhaseConfig:
    """All hyper-parameters for Idea 4."""

    # --- Phase Encoder ---
    feat_dim: int = 128          # input feature dim from memory-enhanced image
    phase_embed_dim: int = 64    # output phase embedding dim
    proprio_dim: int = 4         # EE proprioception (x,y,z,gripper_open)

    # --- Peak Extraction ---
    topk: int = 3                # max peaks per view
    nms_kernel: int = 5          # NMS kernel size

    # --- Contrastive Learning ---
    temperature: float = 0.07    # InfoNCE temperature
    pos_radius: float = 0.03     # 3D distance for same-cluster positives
    neg_margin: float = 0.10     # hard-neg mining: same-pos but diff-phase
    gripper_state_weight: float = 1.0  # extra push for grip-mismatch pairs

    # --- Phase Clustering (auto, from precomputed JSON) ---
    cluster_radius: float = 0.04   # radius for position clustering
    require_grip_match: bool = True  # same grip state for positive pairs

    # --- Loss Weights ---
    phase_contrastive_weight: float = 0.1
    peak_selection_weight: float = 0.5
    phase_consistency_weight: float = 0.05  # temporal smoothness

    # --- Multipeak Targets ---
    multipeak_targets_json: str = ""
    multipeak_max_peaks: int = 5

    # --- Misc ---
    warmup_steps: int = 500      # don't apply peak selection loss early


# ---------------------------------------------------------------------------
# Phase Encoder
# ---------------------------------------------------------------------------

class PhaseEncoder(nn.Module):
    """Encode memory-enhanced features into a compact phase embedding.

    Input:  pooled feature from all views (C,) + proprio (4,)
    Output: L2-normalized phase embedding (phase_embed_dim,)
    """

    def __init__(self, feat_dim: int = 128, proprio_dim: int = 4,
                 phase_embed_dim: int = 64):
        super().__init__()
        in_dim = feat_dim + proprio_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, phase_embed_dim),
        )

    def forward(self, pooled_feat: torch.Tensor,
                proprio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pooled_feat: (bs, feat_dim) global-avg-pooled memory-enhanced feat
            proprio:     (bs, proprio_dim) [x, y, z, gripper_open]

        Returns:
            phase_embed: (bs, phase_embed_dim), L2-normalized
        """
        x = torch.cat([pooled_feat, proprio], dim=-1)
        emb = self.encoder(x)
        return F.normalize(emb, p=2, dim=-1)


# ---------------------------------------------------------------------------
# Peak Extractor (reused from graph_peak_selector, kept self-contained)
# ---------------------------------------------------------------------------

class PeakExtractor(nn.Module):
    """NMS-based top-k peak extraction from heatmap."""

    def __init__(self, topk: int = 3, nms_kernel: int = 5):
        super().__init__()
        self.topk = topk
        self.nms_kernel = nms_kernel

    def forward(
        self, heatmap: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            heatmap: (bs, nc, H, W) raw logits.
        Returns:
            peak_coords:  (bs, nc, topk, 2)  row, col
            peak_scores:  (bs, nc, topk)
            peak_indices: (bs, nc, topk)
        """
        bs, nc, H, W = heatmap.shape
        hm_flat = heatmap.view(bs * nc, 1, H, W)
        hm_pool = F.max_pool2d(
            hm_flat, kernel_size=self.nms_kernel, stride=1,
            padding=self.nms_kernel // 2,
        )
        hm_nms = hm_flat * (hm_flat == hm_pool).float()
        hm_nms_flat = hm_nms.view(bs * nc, H * W)
        topk_vals, topk_idxs = torch.topk(hm_nms_flat, self.topk, dim=-1)

        rows = topk_idxs // W
        cols = topk_idxs % W
        peak_coords = torch.stack([rows, cols], dim=-1).view(bs, nc, self.topk, 2)
        peak_scores = F.softmax(topk_vals, dim=-1).view(bs, nc, self.topk)
        peak_indices = topk_idxs.view(bs, nc, self.topk)

        return peak_coords, peak_scores, peak_indices


# ---------------------------------------------------------------------------
# Phase-Conditioned Peak Selector
# ---------------------------------------------------------------------------

class PhaseConditionedPeakSelector(nn.Module):
    """Given a phase embedding, score extracted peaks to pick the right one.

    1. Sample features at peak locations from the feature map.
    2. Compute compatibility score between phase embedding and each peak feature.
    3. Output softmax logits over peaks.
    """

    def __init__(self, feat_dim: int = 128, phase_embed_dim: int = 64):
        super().__init__()
        # Project peak features to same space as phase embedding
        self.peak_proj = nn.Sequential(
            nn.Linear(feat_dim, phase_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(phase_embed_dim, phase_embed_dim),
        )
        # Learnable scale (like CLIP's logit_scale)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / 0.07))

    def forward(
        self,
        phase_embed: torch.Tensor,
        feature_map: torch.Tensor,
        peak_coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            phase_embed:  (bs, phase_embed_dim) current phase embedding
            feature_map:  (bs_views, C, H_f, W_f) memory-enhanced features
            peak_coords:  (bs, num_views, topk, 2) in heatmap space

        Returns:
            peak_logits: (num_views, topk) selection logits (pre-softmax)
        """
        num_views, C, H_f, W_f = feature_map.shape
        topk = peak_coords.shape[2]

        # 1. Sample features at peak locations
        coords_flat = peak_coords[0]  # (num_views, topk, 2) -- bs=1 during seq
        # Normalize to [-1, 1]
        grid_y = coords_flat[:, :, 0].float() / max(H_f - 1, 1) * 2 - 1
        grid_x = coords_flat[:, :, 1].float() / max(W_f - 1, 1) * 2 - 1
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(1)  # (nv, 1, topk, 2)

        sampled = F.grid_sample(
            feature_map, grid, mode="bilinear", align_corners=True
        )  # (nv, C, 1, topk)
        sampled = sampled.squeeze(2).permute(0, 2, 1)  # (nv, topk, C)

        # 2. Project peak features
        peak_embeds = self.peak_proj(sampled)  # (nv, topk, phase_embed_dim)
        peak_embeds = F.normalize(peak_embeds, dim=-1)

        # 3. Compute compatibility
        # Expand phase_embed for each view
        phase_q = phase_embed.expand(num_views, -1).unsqueeze(-1)  # (nv, D, 1)

        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = (peak_embeds @ phase_q).squeeze(-1) * scale  # (nv, topk)

        return logits


# ---------------------------------------------------------------------------
# Contrastive Pair Miner (automatic phase discovery)
# ---------------------------------------------------------------------------

class PhasePairMiner:
    """Mine positive/negative pairs for contrastive learning.

    Phase discovery is automatic -- uses 3D position + gripper state clustering.
    Hard negatives: same position, different phase (the KF4-vs-KF7 case).

    This operates on a sequence of observations within one episode.
    """

    def __init__(self, cfg: ContrastivePhaseConfig):
        self.pos_radius = cfg.pos_radius
        self.neg_margin = cfg.neg_margin
        self.require_grip_match = cfg.require_grip_match
        self.gripper_state_weight = cfg.gripper_state_weight

    def mine_pairs(
        self,
        positions_3d: torch.Tensor,
        gripper_states: torch.Tensor,
        next_targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            positions_3d:    (N, 3) EE positions for N steps in this episode
            gripper_states:  (N,) gripper open/close
            next_targets:    (N, 3) or None, next-step target positions
                             (used for hard-negative mining)

        Returns:
            pos_mask:      (N, N) bool, positive pairs
            hard_neg_mask: (N, N) bool, hard-negative pairs (same pos, diff phase)
            neg_mask:      (N, N) bool, all negative pairs
        """
        N = positions_3d.shape[0]
        device = positions_3d.device

        pos_dist = torch.cdist(positions_3d.float(), positions_3d.float())
        eye = torch.eye(N, device=device, dtype=torch.bool)

        # Gripper state match
        grip_match = (gripper_states.unsqueeze(0) == gripper_states.unsqueeze(1))

        # POSITIVE: same position + same gripper state
        if self.require_grip_match:
            pos_mask = (pos_dist < self.pos_radius) & grip_match & (~eye)
        else:
            pos_mask = (pos_dist < self.pos_radius) & (~eye)

        # HARD NEGATIVE: same position + same gripper but different next target
        # This is THE critical case: KF4 vs KF7 in put_block_back
        hard_neg_mask = torch.zeros(N, N, device=device, dtype=torch.bool)
        if next_targets is not None:
            target_dist = torch.cdist(next_targets.float(), next_targets.float())
            # Same current position, same grip, but different target
            hard_neg_mask = (
                (pos_dist < self.pos_radius)
                & grip_match
                & (target_dist > self.neg_margin)
                & (~eye)
            )

        # NEGATIVE: everything that isn't positive (and not self)
        neg_mask = (~pos_mask) & (~eye)

        return pos_mask, hard_neg_mask, neg_mask


# ---------------------------------------------------------------------------
# Contrastive Phase Loss (InfoNCE with hard-negative weighting)
# ---------------------------------------------------------------------------

def contrastive_phase_loss(
    phase_embeds: torch.Tensor,
    pos_mask: torch.Tensor,
    neg_mask: torch.Tensor,
    hard_neg_mask: torch.Tensor,
    temperature: float = 0.07,
    hard_neg_weight: float = 2.0,
) -> torch.Tensor:
    """InfoNCE-style contrastive loss with hard-negative emphasis.

    For each anchor i with at least one positive:
        L_i = -log( sum_pos exp(sim/T) / (sum_pos exp(sim/T) + sum_neg exp(sim/T)) )

    Hard negatives (same position, different phase) get extra weight.

    Args:
        phase_embeds:   (N, D) L2-normalized phase embeddings
        pos_mask:       (N, N) positive pairs
        neg_mask:       (N, N) negative pairs
        hard_neg_mask:  (N, N) hard-negative subset
        temperature:    softmax temperature
        hard_neg_weight: multiplicative weight for hard negatives in denominator

    Returns:
        Scalar loss (0 if no valid positives)
    """
    N = phase_embeds.shape[0]
    if N <= 1:
        return torch.tensor(0.0, device=phase_embeds.device)

    sim = phase_embeds @ phase_embeds.t() / temperature  # (N, N)

    # Positive logits
    pos_sim = sim.masked_fill(~pos_mask, float("-inf"))
    log_sum_pos = torch.logsumexp(pos_sim, dim=-1)  # (N,)

    # Negative logits with hard-neg weighting
    # For hard negatives, add log(hard_neg_weight) to effectively scale them
    neg_logits = sim.clone()
    neg_logits[~neg_mask] = float("-inf")
    if hard_neg_weight > 1.0:
        hard_bonus = math.log(hard_neg_weight)
        neg_logits[hard_neg_mask] += hard_bonus

    # Denominator = pos + neg
    all_logits = torch.stack([
        torch.logsumexp(pos_sim, dim=-1),
        torch.logsumexp(neg_logits, dim=-1),
    ], dim=-1)
    log_denom = torch.logsumexp(all_logits, dim=-1)  # (N,)

    # Only compute for anchors that have at least one positive
    valid = pos_mask.any(dim=-1)
    if not valid.any():
        return torch.tensor(0.0, device=phase_embeds.device)

    loss = -(log_sum_pos[valid] - log_denom[valid]).mean()
    return loss


# ---------------------------------------------------------------------------
# Peak Selection Loss
# ---------------------------------------------------------------------------

def peak_selection_loss(
    peak_logits: torch.Tensor,
    peak_coords: torch.Tensor,
    gt_position_2d: torch.Tensor,
) -> torch.Tensor:
    """CE loss: the peak closest to GT should get the highest logit.

    Args:
        peak_logits:    (num_views, topk) pre-softmax peak selection logits
        peak_coords:    (1, num_views, topk, 2) peak positions (row, col)
        gt_position_2d: (num_views, 2) GT target in image coords

    Returns:
        Scalar loss
    """
    num_views, topk = peak_logits.shape
    coords = peak_coords[0]  # (num_views, topk, 2)
    gt = gt_position_2d.unsqueeze(1).expand_as(coords)
    dists = torch.norm((coords.float() - gt.float()), dim=-1)  # (nv, topk)
    gt_peak_idx = dists.argmin(dim=-1)  # (num_views,)
    return F.cross_entropy(peak_logits, gt_peak_idx)


# ---------------------------------------------------------------------------
# Phase Consistency Loss (temporal smoothness)
# ---------------------------------------------------------------------------

def phase_consistency_loss(
    phase_embeds: torch.Tensor,
    positions_3d: torch.Tensor,
    next_targets: Optional[torch.Tensor] = None,
    neg_margin: float = 0.10,
) -> torch.Tensor:
    """Encourage phase embeddings to change smoothly unless phase changes.

    For consecutive steps (t, t+1):
        - If same phase (same position cluster & target direction): high similarity
        - If phase transition (different target): allow low similarity

    This prevents the phase encoder from being noisy step-to-step.

    Args:
        phase_embeds:  (N, D) sequential phase embeddings
        positions_3d:  (N, 3)
        next_targets:  (N, 3) optional
        neg_margin:    threshold to detect phase change

    Returns:
        Scalar loss
    """
    N = phase_embeds.shape[0]
    if N <= 1:
        return torch.tensor(0.0, device=phase_embeds.device)

    # Cosine similarity between consecutive steps
    cos_sim = F.cosine_similarity(
        phase_embeds[:-1], phase_embeds[1:], dim=-1
    )  # (N-1,)

    if next_targets is not None:
        # Detect phase transitions: large change in target direction
        target_diff = torch.norm(
            next_targets[:-1] - next_targets[1:], dim=-1
        )  # (N-1,)
        same_phase = (target_diff < neg_margin).float()
    else:
        same_phase = torch.ones(N - 1, device=phase_embeds.device)

    # For same-phase consecutive steps, encourage high similarity (cos > 0.8)
    # For phase transitions, no constraint
    target_sim = same_phase * 0.9  # target cosine sim for same-phase pairs
    loss = same_phase * F.relu(target_sim - cos_sim)

    return loss.mean()


# ---------------------------------------------------------------------------
# Heatmap Reweighting
# ---------------------------------------------------------------------------

def reweight_heatmap_with_peaks(
    heatmap: torch.Tensor,
    peak_logits: torch.Tensor,
    peak_coords: torch.Tensor,
    sigma: float = 3.0,
) -> torch.Tensor:
    """Apply phase-conditioned peak selection as additive bias on heatmap.

    Args:
        heatmap:     (bs, nc, 1, H, W)
        peak_logits: (num_views, topk)
        peak_coords: (1, num_views, topk, 2)
        sigma:       Gaussian spread for the bias bumps

    Returns:
        reweighted: (bs, nc, 1, H, W) reweighted heatmap
    """
    bs, nc, one, H, W = heatmap.shape
    num_views = peak_logits.shape[0]
    topk = peak_logits.shape[1]
    device = heatmap.device

    peak_weights = F.softmax(peak_logits, dim=-1)  # (nv, topk)

    # Build additive bias per view
    row_grid = torch.arange(H, device=device).float()
    col_grid = torch.arange(W, device=device).float()
    rr, cc = torch.meshgrid(row_grid, col_grid, indexing="ij")  # (H, W)

    bias = torch.zeros(num_views, H, W, device=device, dtype=heatmap.dtype)

    coords = peak_coords[0, :num_views]  # (nv, topk, 2)
    for k in range(topk):
        r0 = coords[:, k, 0].float()  # (nv,)
        c0 = coords[:, k, 1].float()  # (nv,)
        w_k = peak_weights[:, k]       # (nv,)

        # Gaussian bump: (nv, H, W)
        dr = rr.unsqueeze(0) - r0.view(-1, 1, 1)
        dc = cc.unsqueeze(0) - c0.view(-1, 1, 1)
        gaussian = torch.exp(-(dr ** 2 + dc ** 2) / (2 * sigma ** 2))
        bias += w_k.view(-1, 1, 1) * gaussian

    bias = bias.unsqueeze(0).unsqueeze(2)  # (1, nv, 1, H, W)
    return heatmap + bias


# ---------------------------------------------------------------------------
# Full Module: ContrastivePhaseSelector
# ---------------------------------------------------------------------------

class ContrastivePhaseSelector(nn.Module):
    """Complete Idea 4 module.

    Lifecycle (per episode during training):
        1. reset() at episode start
        2. For each step t in the episode:
           a. encode_phase(feature_map, proprio) -> phase_embed
           b. select_peaks(heatmap, feature_map, phase_embed) -> peak_logits
           c. store(phase_embed, position, gripper, next_target)
        3. compute_losses() at episode end -> contrastive + consistency losses

    At eval:
        1. reset()
        2. For each step: encode_phase + select_peaks + reweight heatmap
        3. No loss computation
    """

    def __init__(self, cfg: ContrastivePhaseConfig):
        super().__init__()
        self.cfg = cfg

        self.phase_encoder = PhaseEncoder(
            feat_dim=cfg.feat_dim,
            proprio_dim=cfg.proprio_dim,
            phase_embed_dim=cfg.phase_embed_dim,
        )
        self.peak_extractor = PeakExtractor(
            topk=cfg.topk,
            nms_kernel=cfg.nms_kernel,
        )
        self.peak_selector = PhaseConditionedPeakSelector(
            feat_dim=cfg.feat_dim,
            phase_embed_dim=cfg.phase_embed_dim,
        )
        self.pair_miner = PhasePairMiner(cfg)

        # Episode buffers (not parameters)
        self._phase_embeds: List[torch.Tensor] = []
        self._positions: List[torch.Tensor] = []
        self._grippers: List[torch.Tensor] = []
        self._next_targets: List[torch.Tensor] = []

    def reset(self):
        """Call at episode start."""
        self._phase_embeds = []
        self._positions = []
        self._grippers = []
        self._next_targets = []

    def encode_phase(
        self,
        feature_map: torch.Tensor,
        proprio: torch.Tensor,
    ) -> torch.Tensor:
        """Compute phase embedding from memory-enhanced features.

        Args:
            feature_map: (num_views, C, H_f, W_f) after memory attention
            proprio:     (1, proprio_dim) current EE state

        Returns:
            phase_embed: (1, phase_embed_dim)
        """
        # Global avg pool across all views
        pooled = feature_map.mean(dim=(0, 2, 3)).unsqueeze(0)  # (1, C)
        return self.phase_encoder(pooled, proprio)

    def select_peaks(
        self,
        heatmap: torch.Tensor,
        feature_map: torch.Tensor,
        phase_embed: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Extract peaks and score them using phase embedding.

        Args:
            heatmap:      (bs, nc, 1, H, W)
            feature_map:  (num_views, C, H_f, W_f)
            phase_embed:  (1, phase_embed_dim)

        Returns:
            dict with peak_coords, peak_scores, peak_logits, reweighted_heatmap
        """
        bs, nc, one, H, W = heatmap.shape
        hm_2d = heatmap.squeeze(2)  # (bs, nc, H, W)

        peak_coords, peak_scores, peak_indices = self.peak_extractor(hm_2d)

        # Scale coords from heatmap space to feature space
        H_f, W_f = feature_map.shape[2], feature_map.shape[3]
        scale_h = H_f / H
        scale_w = W_f / W
        scaled_coords = peak_coords.clone().float()
        scaled_coords[:, :, :, 0] *= scale_h
        scaled_coords[:, :, :, 1] *= scale_w

        # Phase-conditioned scoring
        peak_logits = self.peak_selector(
            phase_embed, feature_map, scaled_coords
        )  # (num_views, topk)

        # Reweight heatmap
        reweighted = reweight_heatmap_with_peaks(
            heatmap, peak_logits, peak_coords, sigma=3.0
        )

        return {
            "peak_coords": peak_coords,
            "peak_scores": peak_scores,
            "peak_logits": peak_logits,
            "reweighted_heatmap": reweighted,
            "phase_embed": phase_embed,
        }

    def store(
        self,
        phase_embed: torch.Tensor,
        position_3d: torch.Tensor,
        gripper_state: torch.Tensor,
        next_target: Optional[torch.Tensor] = None,
    ):
        """Store step data for end-of-episode contrastive loss.

        Args:
            phase_embed:    (1, D) or (D,)
            position_3d:    (3,) EE position
            gripper_state:  scalar, 0 or 1
            next_target:    (3,) next keyframe's target position (for hard-neg)
        """
        self._phase_embeds.append(phase_embed.detach() if phase_embed.dim() == 1
                                  else phase_embed.squeeze(0).detach())
        self._positions.append(position_3d.detach())
        self._grippers.append(gripper_state.detach())
        if next_target is not None:
            self._next_targets.append(next_target.detach())

    def compute_losses(self) -> Dict[str, torch.Tensor]:
        """Compute contrastive + consistency losses over the stored episode.

        Should be called once at the end of each episode's sequential pass.

        Returns:
            dict with phase_contrastive_loss, phase_consistency_loss
        """
        device = self._phase_embeds[0].device if self._phase_embeds else "cpu"

        if len(self._phase_embeds) < 2:
            return {
                "phase_contrastive_loss": torch.tensor(0.0, device=device),
                "phase_consistency_loss": torch.tensor(0.0, device=device),
            }

        embeds = torch.stack(self._phase_embeds)    # (N, D)
        positions = torch.stack(self._positions)     # (N, 3)
        grippers = torch.stack(self._grippers)       # (N,)

        next_targets = None
        if len(self._next_targets) == len(self._phase_embeds):
            next_targets = torch.stack(self._next_targets)  # (N, 3)

        # Re-encode with grad for loss backprop
        # Note: we stored detached embeds for memory efficiency.
        # The actual gradient flows through the current forward pass's
        # phase_embed (stored with grad in the training loop).
        # The contrastive loss here uses re-attached embeds from the
        # module's parameters.

        # Mine pairs
        pos_mask, hard_neg_mask, neg_mask = self.pair_miner.mine_pairs(
            positions, grippers, next_targets
        )

        # Contrastive loss
        c_loss = contrastive_phase_loss(
            embeds, pos_mask, neg_mask, hard_neg_mask,
            temperature=self.cfg.temperature,
            hard_neg_weight=2.0,
        )

        # Consistency loss
        cons_loss = phase_consistency_loss(
            embeds, positions, next_targets,
            neg_margin=self.cfg.neg_margin,
        )

        return {
            "phase_contrastive_loss": c_loss,
            "phase_consistency_loss": cons_loss,
        }

    def forward(
        self,
        heatmap: torch.Tensor,
        feature_map: torch.Tensor,
        proprio: torch.Tensor,
        position_3d: Optional[torch.Tensor] = None,
        gripper_state: Optional[torch.Tensor] = None,
        next_target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Full forward: encode phase, select peaks, optionally store.

        Args:
            heatmap:       (bs, nc, 1, H, W)
            feature_map:   (num_views, C, H_f, W_f) memory-enhanced features
            proprio:       (1, proprio_dim)
            position_3d:   (3,) for contrastive pair storage (training only)
            gripper_state: scalar (training only)
            next_target:   (3,) next keyframe target (training only)

        Returns:
            dict with reweighted_heatmap, phase_embed, peak_coords, peak_logits
        """
        phase_embed = self.encode_phase(feature_map, proprio)
        result = self.select_peaks(heatmap, feature_map, phase_embed)

        if self.training and position_3d is not None and gripper_state is not None:
            self.store(phase_embed, position_3d, gripper_state, next_target)

        return result


# ---------------------------------------------------------------------------
# Multipeak Target Loader (from precomputed JSON)
# ---------------------------------------------------------------------------

class MultipeakTargetProvider:
    """Load precomputed multipeak_targets JSON and provide GT peak labels.

    The JSON has per-episode, per-keyframe alt_positions and alt_mask.
    During training, this tells us which keyframes are ambiguous and what
    the alternative targets are -- enabling correct peak_selection_loss.
    """

    def __init__(self, json_path: str):
        self.targets = {}
        if json_path and Path(json_path).exists():
            with open(json_path) as f:
                raw = json.load(f)
            for ep_key, ep_data in raw.items():
                ep_idx = int(ep_key)
                self.targets[ep_idx] = {
                    "alt_positions": torch.tensor(
                        ep_data["alt_positions"], dtype=torch.float32
                    ),
                    "alt_mask": torch.tensor(
                        ep_data["alt_mask"], dtype=torch.bool
                    ),
                }

    def is_ambiguous(self, episode_idx: int, kf_idx: int) -> bool:
        """Check if this keyframe has multiple valid targets."""
        if episode_idx not in self.targets:
            return False
        mask = self.targets[episode_idx]["alt_mask"]
        if kf_idx >= mask.shape[0]:
            return False
        return mask[kf_idx].any().item()

    def get_alt_targets(
        self, episode_idx: int, kf_idx: int
    ) -> Optional[torch.Tensor]:
        """Get alt target positions for an ambiguous keyframe.

        Returns:
            (num_alts, 3) tensor or None
        """
        if not self.is_ambiguous(episode_idx, kf_idx):
            return None
        data = self.targets[episode_idx]
        mask = data["alt_mask"][kf_idx]  # (max_peaks,)
        positions = data["alt_positions"][kf_idx]  # (max_peaks, 3)
        return positions[mask]


# ---------------------------------------------------------------------------
# Integration Helpers
# ---------------------------------------------------------------------------

def build_contrastive_phase_selector(
    cfg: Optional[ContrastivePhaseConfig] = None,
    **kwargs,
) -> ContrastivePhaseSelector:
    """Factory function for creating the module."""
    if cfg is None:
        cfg = ContrastivePhaseConfig(**kwargs)
    return ContrastivePhaseSelector(cfg)


def compute_phase_aware_losses(
    selector: ContrastivePhaseSelector,
    step_outputs: List[Dict[str, torch.Tensor]],
    gt_positions_2d: Optional[List[torch.Tensor]] = None,
    cfg: Optional[ContrastivePhaseConfig] = None,
    global_step: int = 0,
) -> Dict[str, torch.Tensor]:
    """Compute all losses for an episode's sequential outputs.

    Called once per episode after the sequential training loop.

    Args:
        selector:        the ContrastivePhaseSelector module
        step_outputs:    list of dicts from selector.forward() per step
        gt_positions_2d: list of (num_views, 2) GT positions per step
        cfg:             config
        global_step:     for warmup

    Returns:
        dict with total_phase_loss and component losses
    """
    if cfg is None:
        cfg = selector.cfg

    device = step_outputs[0]["phase_embed"].device

    # 1. Contrastive + consistency from stored episode data
    ep_losses = selector.compute_losses()
    c_loss = ep_losses["phase_contrastive_loss"]
    cons_loss = ep_losses["phase_consistency_loss"]

    # 2. Peak selection loss (per step with peak_logits)
    peak_loss = torch.tensor(0.0, device=device)
    peak_count = 0

    if gt_positions_2d is not None and global_step >= cfg.warmup_steps:
        for t, out in enumerate(step_outputs):
            if "peak_logits" in out and out["peak_logits"] is not None:
                p_loss = peak_selection_loss(
                    out["peak_logits"],
                    out["peak_coords"],
                    gt_positions_2d[t],
                )
                peak_loss = peak_loss + p_loss
                peak_count += 1

    if peak_count > 0:
        peak_loss = peak_loss / peak_count

    # Weighted total
    total = (
        cfg.phase_contrastive_weight * c_loss
        + cfg.peak_selection_weight * peak_loss
        + cfg.phase_consistency_weight * cons_loss
    )

    return {
        "total_phase_loss": total,
        "phase_contrastive_loss": c_loss,
        "phase_peak_selection_loss": peak_loss,
        "phase_consistency_loss": cons_loss,
    }
