"""
Graph-Guided Peak Selection Network (GPSN)
============================================
Stage2 module for SAM2Act that explicitly selects among Stage1 multi-peak
candidates using graph-based transition scoring with language conditioning.

Key idea:
  - Each peak from Stage1 multi-peak heatmap = a candidate node in a graph
  - TransitionScorer(prev_node, curr_node, peak_embeds, lang_emb) computes
    selection weights over peaks
  - AmbiguityGate bypasses graph overhead when Stage1 output is unambiguous

Integration target:
  - mvt_sam2_single.py (MVT class)
  - sam2act_agent.py (loss computation)
  - config.py (hyperparameters)

Author: GPSN Design (Idea 1)
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Peak Extraction (non-learned, differentiable-friendly)
# ---------------------------------------------------------------------------

class PeakExtractor(nn.Module):
    """Extract top-k peaks from heatmap using soft-NMS style selection.

    Works on the Stage1 coarse heatmap before softmax, producing candidate
    peak positions and their raw logit scores.
    """

    def __init__(self, topk: int = 3, nms_kernel: int = 5, score_threshold: float = -float("inf")):
        super().__init__()
        self.topk = topk
        self.nms_kernel = nms_kernel
        self.score_threshold = score_threshold

    @torch.no_grad()
    def forward(
        self,
        heatmap_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            heatmap_logits: [bs, num_img, 1, H, W] — raw logits from Stage1

        Returns:
            peak_positions: [bs, k, 3] — (view_idx, row, col) per peak, float
            peak_scores:    [bs, k]    — logit value at each peak
            peak_mask:      [bs, k]    — True for valid peaks (above threshold)
        """
        bs, num_img, _, H, W = heatmap_logits.shape
        device = heatmap_logits.device

        # Flatten views into spatial: [bs, 1, num_img*H, W]
        hm = heatmap_logits.view(bs, 1, num_img * H, W)

        # Local-max detection via max-pooling
        pad = self.nms_kernel // 2
        hm_max = F.max_pool2d(hm, kernel_size=self.nms_kernel, stride=1, padding=pad)
        is_peak = (hm == hm_max)  # [bs, 1, num_img*H, W]

        # Suppress non-peaks
        hm_peaks = hm * is_peak.float() + (~is_peak).float() * (-1e9)
        hm_flat = hm_peaks.view(bs, -1)  # [bs, num_img*H*W]

        # Top-k selection
        k = min(self.topk, hm_flat.shape[1])
        scores, indices = torch.topk(hm_flat, k, dim=-1)  # [bs, k]

        # Convert flat index back to (view_idx, row, col)
        view_idx = indices // (H * W)
        remainder = indices % (H * W)
        row = remainder // W
        col = remainder % W

        peak_positions = torch.stack(
            [view_idx.float(), row.float(), col.float()], dim=-1
        )  # [bs, k, 3]

        peak_mask = scores > self.score_threshold  # [bs, k]

        return peak_positions, scores, peak_mask


# ---------------------------------------------------------------------------
# 2. Peak Node Embedding
# ---------------------------------------------------------------------------

class PeakNodeEmbedding(nn.Module):
    """Extract node embeddings at each peak position from encoder feature maps.

    Uses bilinear sampling on the intermediate feature map (post-transformer,
    pre-decoder) at each peak's spatial location.
    """

    def __init__(self, feat_dim: int = 128, node_dim: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, node_dim),
            nn.ReLU(inplace=True),
            nn.Linear(node_dim, node_dim),
        )
        self.feat_dim = feat_dim
        self.node_dim = node_dim

    def forward(
        self,
        feature_map: torch.Tensor,
        peak_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            feature_map:    [bs, C, num_img, H, W] — intermediate encoder feature
                            (after self-attention, before decoder).
                            This is the `x` tensor in mvt_sam2_single.py right after
                            rearrange back to [B, C, num_img, np, np].
            peak_positions: [bs, k, 3] — (view_idx, row, col), float

        Returns:
            node_embeds: [bs, k, node_dim] — L2-normalized
        """
        bs, C, num_img, H, W = feature_map.shape
        k = peak_positions.shape[1]

        # Gather features via grid_sample per peak
        # Normalize coordinates to [-1, 1] for grid_sample
        view_idx = peak_positions[:, :, 0].long().clamp(0, num_img - 1)  # [bs, k]
        row_norm = (peak_positions[:, :, 1] / (H - 1)) * 2 - 1         # [bs, k]
        col_norm = (peak_positions[:, :, 2] / (W - 1)) * 2 - 1         # [bs, k]

        peak_feats = []
        for b in range(bs):
            for ki in range(k):
                v = view_idx[b, ki].item()
                # Single point grid_sample: [1, C, 1, 1]
                grid = torch.tensor(
                    [[[[col_norm[b, ki].item(), row_norm[b, ki].item()]]]],
                    device=feature_map.device, dtype=feature_map.dtype,
                )  # [1, 1, 1, 2]
                feat_slice = feature_map[b : b + 1, :, v, :, :]  # [1, C, H, W]
                sampled = F.grid_sample(
                    feat_slice, grid, mode="bilinear", align_corners=True
                )  # [1, C, 1, 1]
                peak_feats.append(sampled.view(C))

        peak_feats = torch.stack(peak_feats, dim=0).view(bs, k, C)  # [bs, k, C]
        node_embeds = self.proj(peak_feats)  # [bs, k, node_dim]
        node_embeds = F.normalize(node_embeds, p=2, dim=-1)

        return node_embeds


class PeakNodeEmbeddingFast(nn.Module):
    """Vectorized version of PeakNodeEmbedding — avoids Python loops.

    Uses advanced indexing instead of grid_sample for speed during training.
    Falls back to nearest-neighbor sampling (sufficient for 14x14 or 16x16 patches).
    """

    def __init__(self, feat_dim: int = 128, node_dim: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, node_dim),
            nn.ReLU(inplace=True),
            nn.Linear(node_dim, node_dim),
        )
        self.feat_dim = feat_dim
        self.node_dim = node_dim

    def forward(
        self,
        feature_map: torch.Tensor,
        peak_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            feature_map:    [bs, C, num_img, H, W]
            peak_positions: [bs, k, 3] — (view_idx, row, col), float

        Returns:
            node_embeds: [bs, k, node_dim] — L2-normalized
        """
        bs, C, num_img, H, W = feature_map.shape
        k = peak_positions.shape[1]

        view_idx = peak_positions[:, :, 0].long().clamp(0, num_img - 1)
        row_idx = peak_positions[:, :, 1].long().clamp(0, H - 1)
        col_idx = peak_positions[:, :, 2].long().clamp(0, W - 1)

        # Advanced indexing: [bs, k, C]
        batch_idx = torch.arange(bs, device=feature_map.device).unsqueeze(1).expand(-1, k)
        peak_feats = feature_map.permute(0, 2, 3, 4, 1)[
            batch_idx, view_idx, row_idx, col_idx
        ]  # [bs, k, C]

        node_embeds = self.proj(peak_feats)
        node_embeds = F.normalize(node_embeds, p=2, dim=-1)

        return node_embeds


# ---------------------------------------------------------------------------
# 3. Current-frame Node Embedding Head
# ---------------------------------------------------------------------------

class NodeEmbedHead(nn.Module):
    """Produce a single node embedding for the current observation from the
    memory-enhanced feature map (post SAM2 memory attention).

    This is stored in the memory bank for use as prev_node in the next step.
    """

    def __init__(self, feat_dim: int = 128, node_dim: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, node_dim),
            nn.ReLU(inplace=True),
            nn.Linear(node_dim, node_dim),
        )

    def forward(self, memory_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            memory_feat: [bs, num_img, C, H, W] — post memory-attention features
                         (image_embeds from sam2_forward_with_memory)

        Returns:
            node_embed: [bs, node_dim] — L2-normalized
        """
        # Global average pool across views and spatial dims
        pooled = memory_feat.mean(dim=(1, 3, 4))  # [bs, C]
        node_embed = self.proj(pooled)
        return F.normalize(node_embed, p=2, dim=-1)


# ---------------------------------------------------------------------------
# 4. Language-Conditioned Transition Scorer
# ---------------------------------------------------------------------------

class LanguageConditionedTransitionScorer(nn.Module):
    """Score each candidate peak using prev_node, curr_node, and language.

    Uses FiLM (Feature-wise Linear Modulation) from language embedding to
    modulate node representations before computing transition context.
    This is inspired by KC-VLA's FiLM query generation but uses continuous
    language embeddings instead of discrete task IDs.

    Transition context:
        lang_emb -> FiLM(gamma, beta)
        prev_mod = gamma * prev_node + beta
        curr_mod = gamma * curr_node + beta
        context  = MLP([prev_mod; curr_mod])
        score_i  = dot(context, peak_embed_i)
    """

    def __init__(
        self,
        node_dim: int = 64,
        lang_dim: int = 128,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        hidden_dim = hidden_dim or node_dim * 2

        # Language -> FiLM parameters
        self.lang_film = nn.Sequential(
            nn.Linear(lang_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, node_dim * 2),  # gamma + beta
        )

        # Context projection: [prev_mod; curr_mod] -> context prototype
        self.context_proj = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.ReLU(inplace=True),
        )
        self.score_proj = nn.Linear(node_dim, node_dim)

        # Temperature for softmax (learnable)
        self.log_temperature = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        prev_node: torch.Tensor,
        curr_node: torch.Tensor,
        peak_embeds: torch.Tensor,
        lang_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            prev_node:    [bs, node_dim]  — previous step's node embedding
                          (zero-initialized if first step)
            curr_node:    [bs, node_dim]  — current step's node embedding
            peak_embeds:  [bs, k, node_dim] — candidate peak node embeddings
            lang_emb:     [bs, lang_dim]  — language instruction embedding
                          (use pooled CLIP embedding or mean of lang tokens)

        Returns:
            scores: [bs, k] — unnormalized logits for peak selection
        """
        # FiLM modulation
        gamma_beta = self.lang_film(lang_emb)  # [bs, node_dim*2]
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # each [bs, node_dim]

        prev_mod = gamma * prev_node + beta  # [bs, node_dim]
        curr_mod = gamma * curr_node + beta  # [bs, node_dim]

        # Context prototype
        context = self.context_proj(
            torch.cat([prev_mod, curr_mod], dim=-1)
        )  # [bs, node_dim]
        next_proto = self.score_proj(context)  # [bs, node_dim]

        # Dot-product scoring with temperature
        temperature = self.log_temperature.exp().clamp(min=0.01, max=10.0)
        scores = torch.bmm(
            peak_embeds, next_proto.unsqueeze(-1)
        ).squeeze(-1) / temperature  # [bs, k]

        return scores


# ---------------------------------------------------------------------------
# 5. Ambiguity Gate
# ---------------------------------------------------------------------------

class AmbiguityGate(nn.Module):
    """Determine whether Stage1 output is ambiguous (multi-peak).

    If ambiguous: activate graph-guided selection.
    If not: bypass graph, use Stage1 heatmap directly (zero overhead).

    Two criteria combined:
      1. Entropy of softmax heatmap > threshold (distribution is spread)
      2. Ratio of 2nd-peak to 1st-peak score > ratio_threshold
    """

    def __init__(
        self,
        entropy_threshold: float = 0.3,
        ratio_threshold: float = 0.5,
        mode: str = "entropy",  # "entropy", "ratio", "both"
    ):
        super().__init__()
        self.entropy_threshold = entropy_threshold
        self.ratio_threshold = ratio_threshold
        self.mode = mode

    @torch.no_grad()
    def forward(
        self,
        heatmap_logits: torch.Tensor,
        peak_scores: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            heatmap_logits: [bs, num_img, 1, H, W]
            peak_scores:    [bs, k] — optional, from PeakExtractor

        Returns:
            is_ambiguous: [bs] — boolean tensor
        """
        bs = heatmap_logits.shape[0]
        device = heatmap_logits.device

        if self.mode in ("entropy", "both"):
            # Per-view entropy
            hm = heatmap_logits.view(bs, heatmap_logits.shape[1], -1)  # [bs, V, H*W]
            hm_prob = F.softmax(hm, dim=-1)
            entropy = -(hm_prob * (hm_prob + 1e-8).log()).sum(dim=-1)  # [bs, V]
            # Normalize by max possible entropy: log(H*W)
            max_ent = math.log(hm.shape[-1])
            norm_entropy = entropy / max_ent  # [bs, V]
            entropy_flag = norm_entropy.max(dim=-1).values > self.entropy_threshold

        if self.mode in ("ratio", "both"):
            if peak_scores is not None and peak_scores.shape[1] >= 2:
                sorted_scores, _ = peak_scores.sort(dim=-1, descending=True)
                # Ratio of 2nd-best to best (in probability space)
                top2_probs = F.softmax(sorted_scores[:, :2], dim=-1)
                ratio_flag = top2_probs[:, 1] > self.ratio_threshold
            else:
                ratio_flag = torch.zeros(bs, dtype=torch.bool, device=device)

        if self.mode == "entropy":
            return entropy_flag
        elif self.mode == "ratio":
            return ratio_flag
        else:  # both — ambiguous if either criterion met
            return entropy_flag | ratio_flag


# ---------------------------------------------------------------------------
# 6. Heatmap Reweighting
# ---------------------------------------------------------------------------

def reweight_heatmap(
    heatmap_logits: torch.Tensor,
    peak_positions: torch.Tensor,
    peak_weights: torch.Tensor,
    sigma: float = 3.0,
) -> torch.Tensor:
    """Reweight Stage1 heatmap using graph-selected peak weights.

    Creates a soft selection mask: Gaussian blobs centered at each peak,
    weighted by graph-predicted peak_weights, multiplied with original heatmap.

    Args:
        heatmap_logits: [bs, num_img, 1, H, W]
        peak_positions: [bs, k, 3] — (view_idx, row, col)
        peak_weights:   [bs, k]    — softmax weights from TransitionScorer
        sigma:          Gaussian blob standard deviation in pixels

    Returns:
        refined_logits: [bs, num_img, 1, H, W]
    """
    bs, num_img, _, H, W = heatmap_logits.shape
    k = peak_positions.shape[1]
    device = heatmap_logits.device

    # Build weight map from peak Gaussians
    row_grid = torch.arange(H, device=device, dtype=torch.float32)
    col_grid = torch.arange(W, device=device, dtype=torch.float32)
    rr, cc = torch.meshgrid(row_grid, col_grid, indexing="ij")  # [H, W]

    weight_map = torch.zeros(bs, num_img, 1, H, W, device=device, dtype=heatmap_logits.dtype)

    for ki in range(k):
        view_idx = peak_positions[:, ki, 0].long()  # [bs]
        row_c = peak_positions[:, ki, 1]             # [bs]
        col_c = peak_positions[:, ki, 2]             # [bs]
        w = peak_weights[:, ki]                      # [bs]

        # Gaussian blob: [bs, H, W]
        gauss = torch.exp(
            -((rr.unsqueeze(0) - row_c.view(-1, 1, 1)) ** 2
              + (cc.unsqueeze(0) - col_c.view(-1, 1, 1)) ** 2)
            / (2 * sigma ** 2)
        )  # [bs, H, W]

        # Scatter into correct view
        for b in range(bs):
            v = view_idx[b].item()
            weight_map[b, v, 0] += w[b] * gauss[b]

    # Ensure non-zero (keep original where no peaks)
    weight_map = weight_map + 1e-6

    # Multiply original logits with weight map (soft gating)
    refined = heatmap_logits * weight_map

    return refined


def reweight_heatmap_fast(
    heatmap_logits: torch.Tensor,
    peak_positions: torch.Tensor,
    peak_weights: torch.Tensor,
    sigma: float = 3.0,
) -> torch.Tensor:
    """Vectorized heatmap reweighting — avoids Python loops over batch.

    Uses additive bias in logit space instead of multiplicative gating,
    which is more numerically stable for cross-entropy loss.

    refined_logits = heatmap_logits + sum_k( w_k * gaussian_k ) * scale

    Args:
        heatmap_logits: [bs, num_img, 1, H, W]
        peak_positions: [bs, k, 3]
        peak_weights:   [bs, k]
        sigma:          float

    Returns:
        refined_logits: [bs, num_img, 1, H, W]
    """
    bs, num_img, _, H, W = heatmap_logits.shape
    k = peak_positions.shape[1]
    device = heatmap_logits.device

    # Spatial grids
    row_grid = torch.arange(H, device=device, dtype=torch.float32)
    col_grid = torch.arange(W, device=device, dtype=torch.float32)
    rr, cc = torch.meshgrid(row_grid, col_grid, indexing="ij")
    rr = rr.view(1, 1, H, W)  # [1, 1, H, W]
    cc = cc.view(1, 1, H, W)

    # Peak coords: [bs, k] each
    view_idx = peak_positions[:, :, 0].long()
    row_c = peak_positions[:, :, 1].view(bs, k, 1, 1)
    col_c = peak_positions[:, :, 2].view(bs, k, 1, 1)

    # Gaussians: [bs, k, H, W]
    gauss = torch.exp(
        -((rr - row_c) ** 2 + (cc - col_c) ** 2) / (2 * sigma ** 2)
    )

    # Weight by peak_weights: [bs, k, H, W]
    weighted_gauss = gauss * peak_weights.view(bs, k, 1, 1)

    # Scatter-add into view dimension: [bs, num_img, H, W]
    bias_map = torch.zeros(bs, num_img, H, W, device=device, dtype=heatmap_logits.dtype)
    for ki in range(k):
        # One-hot view scatter
        v_idx = view_idx[:, ki]  # [bs]
        for b in range(bs):
            bias_map[b, v_idx[b]] += weighted_gauss[b, ki]

    # Scale factor: make bias meaningful relative to logit range
    logit_range = heatmap_logits.view(bs, -1).max(dim=-1).values - \
                  heatmap_logits.view(bs, -1).min(dim=-1).values  # [bs]
    scale = logit_range.view(bs, 1, 1, 1) * 0.5  # moderate scaling

    refined = heatmap_logits + bias_map.unsqueeze(2) * scale.unsqueeze(2)

    return refined


# ---------------------------------------------------------------------------
# 7. Graph Peak Selector (top-level module)
# ---------------------------------------------------------------------------

class GraphPeakSelector(nn.Module):
    """Top-level GPSN module integrating all components.

    Orchestrates: peak extraction -> node embedding -> ambiguity gating ->
    transition scoring -> heatmap reweighting.

    Designed to be instantiated inside MVT and called during Stage2 forward.
    """

    def __init__(
        self,
        feat_dim: int = 128,
        node_dim: int = 64,
        lang_dim: int = 128,
        topk: int = 3,
        nms_kernel: int = 5,
        entropy_threshold: float = 0.3,
        ratio_threshold: float = 0.5,
        ambiguity_mode: str = "entropy",
        reweight_sigma: float = 3.0,
    ):
        super().__init__()

        self.peak_extractor = PeakExtractor(
            topk=topk, nms_kernel=nms_kernel,
        )
        self.peak_node_embed = PeakNodeEmbeddingFast(
            feat_dim=feat_dim, node_dim=node_dim,
        )
        self.node_embed_head = NodeEmbedHead(
            feat_dim=feat_dim, node_dim=node_dim,
        )
        self.transition_scorer = LanguageConditionedTransitionScorer(
            node_dim=node_dim, lang_dim=lang_dim,
        )
        self.ambiguity_gate = AmbiguityGate(
            entropy_threshold=entropy_threshold,
            ratio_threshold=ratio_threshold,
            mode=ambiguity_mode,
        )

        self.node_dim = node_dim
        self.reweight_sigma = reweight_sigma

    def forward(
        self,
        heatmap_logits: torch.Tensor,
        encoder_feat: torch.Tensor,
        memory_feat: torch.Tensor,
        lang_emb: torch.Tensor,
        prev_node: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full GPSN forward pass for a single observation step.

        Args:
            heatmap_logits: [bs, num_img, 1, H, W] — Stage1 coarse heatmap (logits)
            encoder_feat:   [bs, C, num_img, H_enc, W_enc] — intermediate encoder feature
                            (post-transformer `x` in MVT, shape [B, 128, num_img, np, np])
            memory_feat:    [bs, num_img, C, H_enc, W_enc] — memory-enhanced features
                            (output of sam2_forward_with_memory)
            lang_emb:       [bs, lang_dim] — pooled language embedding
            prev_node:      [bs, node_dim] — previous step's node embed (None for first step)

        Returns:
            dict with keys:
                "trans_refined":       [bs, num_img, 1, H, W] — refined heatmap
                "curr_node_embed":     [bs, node_dim] — to store in memory bank
                "peak_positions":      [bs, k, 3]
                "peak_scores":         [bs, k]
                "peak_weights":        [bs, k] or None — softmax weights from graph
                "transition_scores":   [bs, k] or None — raw logits
                "is_ambiguous":        [bs] — boolean
        """
        bs = heatmap_logits.shape[0]
        device = heatmap_logits.device

        # 1. Extract peaks from coarse heatmap
        peak_positions, peak_scores, peak_mask = self.peak_extractor(heatmap_logits)

        # 2. Compute current node embedding from memory-enhanced features
        curr_node = self.node_embed_head(memory_feat)  # [bs, node_dim]

        # 3. Check ambiguity
        is_ambiguous = self.ambiguity_gate(heatmap_logits, peak_scores)  # [bs]

        # 4. If any sample in batch is ambiguous, compute graph scores
        transition_scores = None
        peak_weights = None
        trans_refined = heatmap_logits

        if is_ambiguous.any() or self.training:
            # During training, always compute for gradient flow
            # During eval, only compute if ambiguous

            # Get peak node embeddings from encoder features
            # Need to map peak positions from heatmap space to encoder feature space
            _, _, _, H_hm, W_hm = heatmap_logits.shape
            _, _, _, H_enc, W_enc = encoder_feat.shape

            # Scale peak positions from heatmap resolution to encoder resolution
            scale_h = H_enc / H_hm
            scale_w = W_enc / W_hm
            peak_pos_enc = peak_positions.clone()
            peak_pos_enc[:, :, 1] = peak_positions[:, :, 1] * scale_h
            peak_pos_enc[:, :, 2] = peak_positions[:, :, 2] * scale_w

            peak_embeds = self.peak_node_embed(encoder_feat, peak_pos_enc)  # [bs, k, node_dim]

            # Use zero vector if no prev_node (first step)
            if prev_node is None:
                prev_node = torch.zeros(bs, self.node_dim, device=device, dtype=curr_node.dtype)

            # Transition scoring
            transition_scores = self.transition_scorer(
                prev_node, curr_node, peak_embeds, lang_emb,
            )  # [bs, k]

            # Mask invalid peaks
            transition_scores = transition_scores.masked_fill(~peak_mask, -1e9)

            # Softmax -> peak weights
            peak_weights = F.softmax(transition_scores, dim=-1)  # [bs, k]

            # Reweight heatmap
            if self.training:
                # During training: always apply (gradient signal needed)
                trans_refined = reweight_heatmap_fast(
                    heatmap_logits, peak_positions, peak_weights,
                    sigma=self.reweight_sigma,
                )
            else:
                # During eval: only for ambiguous samples
                trans_refined = heatmap_logits.clone()
                if is_ambiguous.any():
                    amb_idx = is_ambiguous.nonzero(as_tuple=True)[0]
                    refined_amb = reweight_heatmap_fast(
                        heatmap_logits[amb_idx],
                        peak_positions[amb_idx],
                        peak_weights[amb_idx],
                        sigma=self.reweight_sigma,
                    )
                    trans_refined[amb_idx] = refined_amb

        return {
            "trans_refined": trans_refined,
            "curr_node_embed": curr_node,
            "peak_positions": peak_positions,
            "peak_scores": peak_scores,
            "peak_weights": peak_weights,
            "transition_scores": transition_scores,
            "is_ambiguous": is_ambiguous,
        }


# ---------------------------------------------------------------------------
# 8. Loss Functions
# ---------------------------------------------------------------------------

def peak_selection_loss(
    transition_scores: torch.Tensor,
    peak_positions: torch.Tensor,
    gt_position_2d: torch.Tensor,
    gt_view_idx: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy loss for selecting the correct peak.

    The GT peak is the one closest (in 2D pixel distance) to the ground-truth
    waypoint position.

    Args:
        transition_scores: [bs, k] — logits from TransitionScorer
        peak_positions:    [bs, k, 3] — (view_idx, row, col)
        gt_position_2d:    [bs, 2] — (row, col) of GT waypoint in heatmap
        gt_view_idx:       [bs] — which view the GT waypoint is in

    Returns:
        loss: scalar
    """
    bs, k = transition_scores.shape

    # Distance from each peak to GT
    peak_views = peak_positions[:, :, 0]  # [bs, k]
    peak_rows = peak_positions[:, :, 1]
    peak_cols = peak_positions[:, :, 2]

    gt_row = gt_position_2d[:, 0].unsqueeze(1)  # [bs, 1]
    gt_col = gt_position_2d[:, 1].unsqueeze(1)
    gt_v = gt_view_idx.unsqueeze(1).float()

    # Large penalty for wrong view
    view_penalty = (peak_views != gt_v).float() * 1e4

    dist = torch.sqrt(
        (peak_rows - gt_row) ** 2 + (peak_cols - gt_col) ** 2 + 1e-6
    ) + view_penalty  # [bs, k]

    # GT label: index of closest peak
    gt_peak_idx = dist.argmin(dim=-1)  # [bs]

    loss = F.cross_entropy(transition_scores, gt_peak_idx)
    return loss


def node_contrastive_loss(
    node_embeds: torch.Tensor,
    positions_3d: torch.Tensor,
    temperature: float = 0.1,
    pos_radius: float = 0.03,
) -> torch.Tensor:
    """InfoNCE-style contrastive loss on node embeddings.

    Nodes at similar 3D positions (within pos_radius) should have similar
    embeddings; nodes at different positions should have different embeddings.

    This is computed across all nodes in a training sequence.

    Args:
        node_embeds:  [N, node_dim] — all node embeddings in sequence
        positions_3d: [N, 3] — 3D gripper positions
        temperature:  float — InfoNCE temperature
        pos_radius:   float — distance threshold for positive pairs (meters)

    Returns:
        loss: scalar
    """
    N = node_embeds.shape[0]
    if N < 2:
        return torch.tensor(0.0, device=node_embeds.device, requires_grad=True)

    # Pairwise 3D distance
    pos_diff = positions_3d.unsqueeze(0) - positions_3d.unsqueeze(1)  # [N, N, 3]
    pos_dist = pos_diff.norm(dim=-1)  # [N, N]

    # Positive pairs: close in 3D (same location, different timestep)
    pos_mask = (pos_dist < pos_radius).float()
    pos_mask.fill_diagonal_(0)  # exclude self

    if pos_mask.sum() == 0:
        # No positive pairs — skip
        return torch.tensor(0.0, device=node_embeds.device, requires_grad=True)

    # Cosine similarity
    sim = torch.mm(node_embeds, node_embeds.t()) / temperature  # [N, N]

    # For each anchor with at least one positive, compute InfoNCE
    loss = torch.tensor(0.0, device=node_embeds.device)
    count = 0
    for i in range(N):
        pos_indices = pos_mask[i].nonzero(as_tuple=True)[0]
        if len(pos_indices) == 0:
            continue
        # Numerator: log-sum-exp over positives
        pos_sim = sim[i, pos_indices]
        # Denominator: all except self
        all_sim = torch.cat([sim[i, :i], sim[i, i + 1:]])
        log_denom = torch.logsumexp(all_sim, dim=0)

        # Average over positives
        loss_i = -(pos_sim - log_denom).mean()
        loss = loss + loss_i
        count += 1

    return loss / max(count, 1)


def transition_prediction_loss(
    node_embeds: torch.Tensor,
    transition_scorer: LanguageConditionedTransitionScorer,
    lang_emb: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Self-supervised loss: given (node_t, node_{t+1}), predict node_{t+2}.

    Uses the TransitionScorer to predict which node comes next in the sequence,
    treating consecutive node embeddings as the ground truth transition.

    Args:
        node_embeds:       [N, node_dim] — sequential node embeddings
        transition_scorer: the TransitionScorer module
        lang_emb:          [1, lang_dim] or [N, lang_dim] — language embedding
                           (use first one if all same instruction)

    Returns:
        loss: scalar
    """
    N = node_embeds.shape[0]
    if N < 3:
        return torch.tensor(0.0, device=node_embeds.device, requires_grad=True)

    device = node_embeds.device
    node_dim = node_embeds.shape[1]

    # For each triplet (t, t+1, t+2): predict t+2 from (t, t+1)
    # Candidates: all nodes in the sequence
    losses = []
    for t in range(N - 2):
        prev_node = node_embeds[t].unsqueeze(0)    # [1, D]
        curr_node = node_embeds[t + 1].unsqueeze(0)  # [1, D]

        # Candidates: all nodes except t and t+1
        candidate_embeds = node_embeds.unsqueeze(0)  # [1, N, D]

        # Language embedding
        if lang_emb is not None:
            if lang_emb.shape[0] > 1:
                l = lang_emb[t + 1:t + 2]
            else:
                l = lang_emb
        else:
            # Use zeros as fallback
            l = torch.zeros(1, transition_scorer.lang_film[0].in_features, device=device)

        scores = transition_scorer(prev_node, curr_node, candidate_embeds, l)  # [1, N]

        # GT: index t+2
        gt_idx = torch.tensor([t + 2], device=device)
        loss_t = F.cross_entropy(scores, gt_idx)
        losses.append(loss_t)

    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# 9. Utility: Language Embedding Pooling
# ---------------------------------------------------------------------------

def pool_language_embedding(
    lang_emb: torch.Tensor,
    mode: str = "mean",
) -> torch.Tensor:
    """Pool sequence-level language embedding to a single vector.

    Args:
        lang_emb: [bs, seq_len, dim] — from CLIP/T5 language encoder
        mode:     "mean", "first", or "max"

    Returns:
        pooled: [bs, dim]
    """
    if lang_emb.dim() == 2:
        return lang_emb
    if mode == "mean":
        return lang_emb.mean(dim=1)
    elif mode == "first":
        return lang_emb[:, 0]
    elif mode == "max":
        return lang_emb.max(dim=1).values
    else:
        raise ValueError(f"Unknown pooling mode: {mode}")


# ===========================================================================
# INTEGRATION INSTRUCTIONS
# ===========================================================================
"""
=============================================================================
INTEGRATION GUIDE: How to integrate GPSN into SAM2Act Stage2 pipeline
=============================================================================

Files to modify:
  1. sam2act/mvt/config.py
  2. sam2act/mvt/mvt_sam2_single.py
  3. sam2act/models/sam2act_agent.py
  4. sam2act/mvt/configs/sam2act_plus.yaml

Detailed instructions below.

-----------------------------------------------------------------------------
FILE 1: sam2act/mvt/config.py  (already has placeholder params — verify)
-----------------------------------------------------------------------------
The existing config.py already has the needed parameters (lines 82-90):

    _C.use_graph_peak_select = False
    _C.graph_node_embed_dim = 64
    _C.graph_peak_topk = 3
    _C.graph_peak_select_loss_weight = 0.5
    _C.graph_transition_loss_weight = 0.1
    _C.graph_contrastive_loss_weight = 0.05
    _C.graph_contrastive_temperature = 0.1
    _C.graph_contrastive_pos_radius = 0.03

Add these additional params after line 90:

    _C.graph_ambiguity_mode = "entropy"        # "entropy", "ratio", "both"
    _C.graph_entropy_threshold = 0.3
    _C.graph_ratio_threshold = 0.5
    _C.graph_reweight_sigma = 3.0
    _C.graph_lang_pool_mode = "mean"           # "mean", "first", "max"

-----------------------------------------------------------------------------
FILE 2: sam2act/mvt/mvt_sam2_single.py
-----------------------------------------------------------------------------

### Step 2a: Import GPSN (after existing imports, ~line 15)

    from sam2act.mvt.graph_peak_selector import (
        GraphPeakSelector,
        pool_language_embedding,
    )

NOTE: Copy idea1_gpsn.py to sam2act/mvt/graph_peak_selector.py
      (or adjust import path accordingly).

### Step 2b: Instantiate GPSN in __init__ (after line ~250, after self.use_graph_peak_select = ...)

    Add these lines:

        self.graph_peak_selector = None
        if self.use_graph_peak_select:
            self.graph_peak_selector = GraphPeakSelector(
                feat_dim=self.input_dim_before_seq,   # 128
                node_dim=graph_node_embed_dim,         # 64
                lang_dim=self.lang_emb_dim,            # 1024 (CLIP) or 128 (projected)
                topk=graph_peak_topk,
                nms_kernel=5,
                entropy_threshold=0.3,
                ratio_threshold=0.5,
                ambiguity_mode="entropy",
                reweight_sigma=3.0,
            )
            # Language pooling projector (CLIP 1024-dim -> node scorer lang_dim)
            self.graph_lang_proj = nn.Linear(self.lang_emb_dim, 128)

### Step 2c: Integrate into Stage2 training loop (lines ~1019-1074)

    In the training loop inside `forward()`, where `use_memory=True` and
    `no_feat=True` (the coarse branch), after each observation's heatmap
    `trans` is computed and before `trans_.append(trans)`:

    Find this block (~line 1055):
        trans_.append(trans)

    Replace with:
        # --- GPSN integration ---
        if self.use_graph_peak_select and self.graph_peak_selector is not None:
            # Get encoder feature [1, C, num_img, np, np]
            enc_feat = x_i.view(1, num_img, self.input_dim_before_seq,
                                num_pat_img, num_pat_img)
            enc_feat = enc_feat.permute(0, 2, 1, 3, 4)  # [1, C, V, H, W]

            # Memory-enhanced feature
            mem_feat = x_i.view(1, num_img, -1, num_pat_img, num_pat_img)

            # Pool language embedding
            lang_pooled = pool_language_embedding(
                lang_emb[idx:idx+1], mode="mean"
            )
            lang_proj = self.graph_lang_proj(lang_pooled)

            # Get prev_node from graph outputs list
            prev_n = None
            if hasattr(self, '_graph_outputs') and len(self._graph_outputs) > 0:
                prev_n = self._graph_outputs[-1]["curr_node_embed"].detach()

            # GPSN forward
            graph_out = self.graph_peak_selector(
                heatmap_logits=trans,         # [1, num_img, 1, H, W]
                encoder_feat=enc_feat,
                memory_feat=mem_feat,
                lang_emb=lang_proj,
                prev_node=prev_n,
            )

            trans = graph_out["trans_refined"]

            if not hasattr(self, '_graph_outputs'):
                self._graph_outputs = []
            self._graph_outputs.append(graph_out)
        # --- end GPSN ---
        trans_.append(trans)

### Step 2d: Reset graph outputs at start of each sequence

    After `self.reset_memory_bank()` (~line 1020), add:
        if self.use_graph_peak_select:
            self._graph_outputs = []

### Step 2e: Pass graph outputs in return dict

    Before the final `return out` (~line 1250), add:
        if self.use_graph_peak_select and hasattr(self, '_graph_outputs'):
            out["graph_outputs"] = self._graph_outputs

### Step 2f: Integrate into eval path (~lines 1078-1125)

    Similar to training but single-step. After trans is computed in eval:
        if self.use_graph_peak_select and self.graph_peak_selector is not None:
            enc_feat = x.view(bs, num_img, self.input_dim_before_seq,
                              num_pat_img, num_pat_img).permute(0, 2, 1, 3, 4)
            mem_feat = x.view(bs, num_img, -1, num_pat_img, num_pat_img)
            lang_pooled = pool_language_embedding(lang_emb, mode="mean")
            lang_proj = self.graph_lang_proj(lang_pooled)

            prev_n = getattr(self, '_eval_prev_node', None)

            graph_out = self.graph_peak_selector(
                heatmap_logits=trans,
                encoder_feat=enc_feat,
                memory_feat=mem_feat,
                lang_emb=lang_proj,
                prev_node=prev_n,
            )

            trans = graph_out["trans_refined"]
            self._eval_prev_node = graph_out["curr_node_embed"].detach()

-----------------------------------------------------------------------------
FILE 3: sam2act/models/sam2act_agent.py
-----------------------------------------------------------------------------

The existing code already has partial GPSN loss integration (lines 928-987).
It needs these updates:

### Step 3a: Add peak_selection_loss computation (lines 968-976 are incomplete)

    Replace the incomplete peak_selection_loss block with:

        p_loss = torch.tensor(0.0, device=self._device)
        p_count = 0
        for step_i, g_out in enumerate(graph_outputs):
            if g_out["transition_scores"] is not None:
                from sam2act.mvt.graph_peak_selector import peak_selection_loss
                # Get GT position for this step
                # wpt_img contains the 2D projection of GT waypoint
                wpt = replay_sample["wpt_img"]  # [bs, 2] or similar
                gt_view = replay_sample.get("gt_view_idx", torch.zeros(1, device=self._device).long())

                step_gt_2d = wpt[step_i:step_i+1]  # [1, 2]
                step_gt_view = gt_view[step_i:step_i+1] if gt_view.numel() > 1 else gt_view

                p_loss_i = peak_selection_loss(
                    g_out["transition_scores"],
                    g_out["peak_positions"],
                    step_gt_2d,
                    step_gt_view,
                )
                p_loss = p_loss + p_loss_i
                p_count += 1

        if p_count > 0:
            p_loss = p_loss / p_count

### Step 3b: Add peak_selection_loss to total (replace lines 978-982)

        graph_aux = (
            self.graph_contrastive_loss_weight * c_loss
            + self.graph_transition_loss_weight * t_loss
            + self.graph_peak_select_loss_weight * p_loss
        )
        total_loss = total_loss + graph_aux

### Step 3c: Add to loss log (replace lines 984-987)

        graph_loss_log = {
            "graph_contrastive_loss": c_loss.item(),
            "graph_transition_loss": t_loss.item(),
            "graph_peak_select_loss": p_loss.item(),
        }

### Step 3d: Log graph losses to wandb

    In train_plus.py, add to the wandb.log dict (after line ~124):
        if 'graph_contrastive_loss' in loss_log:
            wandb.log({
                'graph_contrastive_loss': loss_log['graph_contrastive_loss'][iteration],
                'graph_transition_loss': loss_log['graph_transition_loss'][iteration],
                'graph_peak_select_loss': loss_log['graph_peak_select_loss'][iteration],
            }, step=log_iter)

-----------------------------------------------------------------------------
FILE 4: sam2act/mvt/configs/sam2act_plus.yaml
-----------------------------------------------------------------------------

    Add to the end:

    use_graph_peak_select: true
    graph_node_embed_dim: 64
    graph_peak_topk: 3
    graph_peak_select_loss_weight: 0.5
    graph_transition_loss_weight: 0.1
    graph_contrastive_loss_weight: 0.05
    graph_contrastive_temperature: 0.1
    graph_contrastive_pos_radius: 0.03
    graph_ambiguity_mode: entropy
    graph_entropy_threshold: 0.3
    graph_ratio_threshold: 0.5
    graph_reweight_sigma: 3.0

=============================================================================
PARAMETER COUNT SUMMARY
=============================================================================

Module                                   Parameters
------                                   ----------
PeakNodeEmbeddingFast.proj               128*64 + 64 + 64*64 + 64 = 12,416
NodeEmbedHead.proj                       128*64 + 64 + 64*64 + 64 = 12,416
LanguageConditionedTransitionScorer      128*128+128 + 128*128+128 + 128*64+64 + 64*64+64 + 64*64+64 + 1 ≈ 49,089
graph_lang_proj                          1024*128 + 128 = 131,200 (if CLIP)
                                         or 128*128 + 128 = 16,512 (if pre-projected)
------
Total (CLIP):                            ~205K
Total (pre-projected):                   ~90K

SAM2Act total: ~100M → GPSN overhead: 0.09-0.21%
"""
