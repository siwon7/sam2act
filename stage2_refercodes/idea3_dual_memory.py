"""
Idea 3: Dual-Level Memory with Episodic Graph
=============================================

Adds a lightweight episodic graph memory alongside SAM2's existing spatial memory.

Level 1 - Spatial Memory (existing SAM2):
    pixel-level (feature, pos_enc) per past keyframe stored in memory_bank_multiview.
    Cross-attention produces memory-enhanced features for precise localization.

Level 2 - Episodic Graph Memory (new, ~12K params):
    Each keyframe -> compact 64-dim node embedding.
    Sequential edges capture phase transitions (what happened before/after).
    Graph query reweights multi-peak heatmap to select the correct peak.

Key insight vs MemoryVLA:
    MemoryVLA uses a VLM to produce cognitive memory tokens (expensive, ~7B params).
    We use learned graph embeddings with explicit transition edges (~12K params).
    The episodic graph models state transitions structurally, while MemoryVLA's
    memory bank is an unstructured bag of tokens.

Integration point:
    sam2_forward_with_memory() produces pix_feat_with_mem  [existing]
    -> NodeEmbedHead extracts curr_node from memory-enhanced features [new]
    -> TransitionScorer(prev_node, curr_node, peak_embeds, lang) -> peak scores [new]
    -> Reweight multi-peak heatmap with peak scores [new]

All supervision is auto-derived from existing data (no extra annotation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper: bilinear sampling of features at sub-pixel peak locations
# ---------------------------------------------------------------------------

def bilinear_sample_at_peaks(
    feature_map: torch.Tensor,
    peak_positions: torch.Tensor,
) -> torch.Tensor:
    """Sample feature vectors at peak positions using grid_sample.

    Args:
        feature_map: [B, C, H, W]
        peak_positions: [B, K, 2]  in (row, col) pixel coords, float

    Returns:
        [B, K, C]
    """
    B, C, H, W = feature_map.shape
    K = peak_positions.shape[1]

    # Convert (row, col) pixel coords -> grid_sample coords in [-1, 1]
    # grid_sample expects (x=col, y=row)
    grid_y = 2.0 * peak_positions[:, :, 0] / max(H - 1, 1) - 1.0  # [B, K]
    grid_x = 2.0 * peak_positions[:, :, 1] / max(W - 1, 1) - 1.0  # [B, K]
    grid = torch.stack([grid_x, grid_y], dim=-1)  # [B, K, 2]
    grid = grid.unsqueeze(1)  # [B, 1, K, 2]

    sampled = F.grid_sample(
        feature_map, grid, mode="bilinear", align_corners=True
    )  # [B, C, 1, K]
    sampled = sampled.squeeze(2).permute(0, 2, 1)  # [B, K, C]
    return sampled


# ---------------------------------------------------------------------------
# Helper: NMS-based top-k peak extraction from a heatmap
# ---------------------------------------------------------------------------

def extract_topk_peaks(
    heatmap: torch.Tensor,
    k: int = 3,
    nms_radius: int = 5,
    score_threshold: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract top-k peaks from a single-channel heatmap with NMS.

    Args:
        heatmap: [B, 1, H, W]  (logits or probabilities, any range)
        k: max number of peaks to return
        nms_radius: suppression radius in pixels
        score_threshold: minimum relative score (fraction of max) for a peak

    Returns:
        positions: [B, K, 2]   (row, col) in pixel coordinates
        scores:    [B, K]      peak scores (softmax-normalised)
        valid:     [B, K]      bool mask (True = real peak, False = padding)
    """
    B, _, H, W = heatmap.shape
    hm = heatmap.squeeze(1)  # [B, H, W]

    # Local max via max-pool
    pad = nms_radius
    hm_padded = F.pad(hm, [pad] * 4, mode="constant", value=-float("inf"))
    hm_max = F.max_pool2d(
        hm_padded.unsqueeze(1),
        kernel_size=2 * nms_radius + 1,
        stride=1,
        padding=0,
    ).squeeze(1)  # [B, H, W]

    is_peak = (hm >= hm_max) & (hm >= hm.flatten(1).max(dim=1).values.view(B, 1, 1) * score_threshold)

    # Flatten and get top-k
    flat_hm = hm.flatten(1)  # [B, H*W]
    flat_peak = is_peak.flatten(1).float()  # [B, H*W]
    flat_scores = flat_hm * flat_peak + (~is_peak.flatten(1)).float() * (-1e9)

    topk_scores, topk_idx = flat_scores.topk(k, dim=1)  # [B, K]
    rows = topk_idx // W  # [B, K]
    cols = topk_idx % W   # [B, K]

    positions = torch.stack([rows.float(), cols.float()], dim=-1)  # [B, K, 2]

    # Normalise scores
    topk_probs = F.softmax(topk_scores, dim=-1)

    valid = topk_scores > -1e8

    return positions, topk_probs, valid


# ---------------------------------------------------------------------------
# 1. NodeEmbedHead: extract compact node embedding from memory-enhanced feat
# ---------------------------------------------------------------------------

class NodeEmbedHead(nn.Module):
    """Compresses the memory-enhanced spatial feature into a single 64-dim node.

    This is applied to the *full* feature map (after SAM2 memory attention).
    Global-average-pool + 2-layer MLP produces a compact episodic embedding.

    Params: feat_dim=128, node_dim=64 -> 128*64 + 64 + 64*64 + 64 = ~12.5K
    """

    def __init__(self, feat_dim: int = 128, node_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, node_dim),
            nn.ReLU(inplace=True),
            nn.Linear(node_dim, node_dim),
        )

    def forward(self, memory_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            memory_feat: [B, C, H, W]  (pix_feat_with_mem from SAM2)
        Returns:
            node_embed: [B, node_dim]  L2-normalised
        """
        gap = memory_feat.flatten(2).mean(dim=-1)  # [B, C]
        node = self.mlp(gap)
        return F.normalize(node, dim=-1)


# ---------------------------------------------------------------------------
# 2. PeakNodeEmbedding: extract per-peak node embeddings from encoder feat
# ---------------------------------------------------------------------------

class PeakNodeEmbedding(nn.Module):
    """For each candidate peak in the multi-peak heatmap, extract a node embed
    from Stage1 encoder features via bilinear sampling + projection.

    Params: feat_dim=128, node_dim=64 -> ~12.5K
    """

    def __init__(self, feat_dim: int = 128, node_dim: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, node_dim),
            nn.ReLU(inplace=True),
            nn.Linear(node_dim, node_dim),
        )

    def forward(
        self, feature_map: torch.Tensor, peak_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            feature_map: [B, C, H, W]  (Stage1 encoder intermediate feature)
            peak_positions: [B, K, 2]  (row, col) from extract_topk_peaks
        Returns:
            peak_embeds: [B, K, node_dim]  L2-normalised
        """
        peak_feats = bilinear_sample_at_peaks(feature_map, peak_positions)  # [B, K, C]
        return F.normalize(self.proj(peak_feats), dim=-1)


# ---------------------------------------------------------------------------
# 3. LanguageConditionedTransitionScorer
# ---------------------------------------------------------------------------

class LanguageConditionedTransitionScorer(nn.Module):
    """Scores each candidate peak given episodic context (prev_node, curr_node)
    and language instruction.

    Uses FiLM (Feature-wise Linear Modulation) inspired by KC-VLA:
        language -> (gamma, beta)
        modulated_node = gamma * node + beta

    Then: context = MLP([prev_mod, curr_mod])  -> prototype
          score_i = dot(prototype, peak_embed_i)

    Params: node_dim=64, lang_dim=128 -> ~20K
    """

    def __init__(self, node_dim: int = 64, lang_dim: int = 128):
        super().__init__()
        # Language -> FiLM parameters
        self.lang_film = nn.Sequential(
            nn.Linear(lang_dim, node_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(node_dim * 2, node_dim * 2),  # outputs gamma, beta each node_dim
        )
        # Context: concat modulated (prev, curr) -> prototype
        self.context_proj = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.ReLU(inplace=True),
        )
        self.score_proj = nn.Linear(node_dim, node_dim)

    def forward(
        self,
        prev_node: torch.Tensor,
        curr_node: torch.Tensor,
        peak_embeds: torch.Tensor,
        lang_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            prev_node:   [B, node_dim]   (previous keyframe's node embed)
            curr_node:   [B, node_dim]   (current keyframe's node embed)
            peak_embeds: [B, K, node_dim] (candidate peak embeddings)
            lang_emb:    [B, lang_dim]   (language instruction embedding)

        Returns:
            scores: [B, K]  (unnormalised logits for peak selection)
        """
        # FiLM modulation
        film_params = self.lang_film(lang_emb)  # [B, 2*node_dim]
        gamma, beta = film_params.chunk(2, dim=-1)  # each [B, node_dim]

        prev_mod = gamma * prev_node + beta  # [B, node_dim]
        curr_mod = gamma * curr_node + beta  # [B, node_dim]

        # Build transition context prototype
        context = self.context_proj(torch.cat([prev_mod, curr_mod], dim=-1))  # [B, node_dim]
        proto = self.score_proj(context)  # [B, node_dim]

        # Dot-product scoring against each peak
        scores = torch.bmm(
            peak_embeds, proto.unsqueeze(-1)
        ).squeeze(-1)  # [B, K]

        return scores


# ---------------------------------------------------------------------------
# 4. AmbiguityGate: decide whether to activate graph peak selection
# ---------------------------------------------------------------------------

class AmbiguityGate(nn.Module):
    """Determines whether the heatmap is ambiguous (multi-peak) by checking
    the entropy of the spatial softmax distribution.

    Non-ambiguous keyframes (75-80%) skip graph scoring entirely.
    No learnable parameters.
    """

    def __init__(self, entropy_threshold: float = 0.3):
        super().__init__()
        self.entropy_threshold = entropy_threshold

    @torch.no_grad()
    def forward(self, heatmap_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            heatmap_logits: [B, num_views, 1, H, W]
        Returns:
            is_ambiguous: [B]  bool
        """
        B, V, _, H, W = heatmap_logits.shape
        hm = heatmap_logits.view(B, V, -1)
        probs = F.softmax(hm, dim=-1)  # [B, V, H*W]

        # Per-view entropy, take max across views
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1)  # [B, V]
        max_entropy = entropy.max(dim=-1).values  # [B]

        return max_entropy > self.entropy_threshold


# ---------------------------------------------------------------------------
# 5. Heatmap reweighting with peak scores
# ---------------------------------------------------------------------------

def reweight_heatmap(
    heatmap: torch.Tensor,
    peak_positions: torch.Tensor,
    peak_weights: torch.Tensor,
    peak_valid: torch.Tensor,
    sigma: float = 5.0,
) -> torch.Tensor:
    """Reweight a multi-peak heatmap using per-peak attention weights.

    Constructs Gaussian blobs centered at each peak, weighted by peak_weights,
    and uses them as a multiplicative mask on the original heatmap.

    Args:
        heatmap:        [B, 1, H, W]  original multi-peak heatmap logits
        peak_positions: [B, K, 2]     (row, col) of each peak
        peak_weights:   [B, K]        softmax weights from TransitionScorer
        peak_valid:     [B, K]        bool mask for valid peaks
        sigma:          Gaussian spread in pixels

    Returns:
        reweighted: [B, 1, H, W]
    """
    B, _, H, W = heatmap.shape
    K = peak_positions.shape[1]
    device = heatmap.device

    # Build coordinate grids
    rows = torch.arange(H, device=device, dtype=torch.float32)
    cols = torch.arange(W, device=device, dtype=torch.float32)
    grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")  # [H, W]

    # Gaussian weight map per peak
    pr = peak_positions[:, :, 0].view(B, K, 1, 1)  # [B, K, 1, 1]
    pc = peak_positions[:, :, 1].view(B, K, 1, 1)
    pw = peak_weights.view(B, K, 1, 1)
    pv = peak_valid.float().view(B, K, 1, 1)

    dist_sq = (grid_r - pr) ** 2 + (grid_c - pc) ** 2  # [B, K, H, W]
    gaussians = torch.exp(-dist_sq / (2 * sigma ** 2))  # [B, K, H, W]
    gaussians = gaussians * pw * pv  # weighted, zero-out invalid

    weight_map = gaussians.sum(dim=1, keepdim=True)  # [B, 1, H, W]

    # Normalise weight map to [0, 1] then apply
    weight_map = weight_map / (weight_map.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values + 1e-8)

    reweighted = heatmap * (0.5 + 0.5 * weight_map)  # soft multiplicative reweight
    return reweighted


# ---------------------------------------------------------------------------
# 6. EpisodicGraphMemory: container for node embeddings across an episode
# ---------------------------------------------------------------------------

@dataclass
class EpisodicNodeEntry:
    """Single node in the episodic graph."""
    obs_idx: int
    node_embed: torch.Tensor  # [node_dim]
    keyframe_3d: Optional[torch.Tensor] = None  # [3], GT or predicted 3D waypoint


class EpisodicGraphMemory:
    """Manages the episodic graph for one episode.

    Nodes are added sequentially (one per keyframe).  The graph is an
    ordered list -- edges are implicit sequential transitions (i -> i+1).

    This replaces the need for an explicit adjacency matrix: transition
    scoring uses (prev_node, curr_node) pairs directly.
    """

    def __init__(self, max_nodes: int = 24):
        self.max_nodes = max_nodes
        self.nodes: List[EpisodicNodeEntry] = []

    def reset(self) -> None:
        self.nodes.clear()

    def add_node(
        self,
        obs_idx: int,
        node_embed: torch.Tensor,
        keyframe_3d: Optional[torch.Tensor] = None,
    ) -> None:
        entry = EpisodicNodeEntry(
            obs_idx=obs_idx,
            node_embed=node_embed.detach(),
            keyframe_3d=keyframe_3d.detach() if keyframe_3d is not None else None,
        )
        self.nodes.append(entry)
        # Evict oldest if over capacity
        if len(self.nodes) > self.max_nodes:
            self.nodes.pop(0)

    @property
    def prev_node(self) -> Optional[torch.Tensor]:
        """Return the most recent node embedding, or None if empty."""
        if len(self.nodes) == 0:
            return None
        return self.nodes[-1].node_embed

    def get_all_embeds(self) -> Optional[torch.Tensor]:
        """Stack all node embeddings: [N, node_dim]."""
        if len(self.nodes) == 0:
            return None
        return torch.stack([n.node_embed for n in self.nodes], dim=0)

    def __len__(self) -> int:
        return len(self.nodes)


# ---------------------------------------------------------------------------
# 7. DualLevelMemoryModule: top-level orchestrator
# ---------------------------------------------------------------------------

class DualLevelMemoryModule(nn.Module):
    """Orchestrates the episodic graph memory on top of SAM2's spatial memory.

    Usage in forward pass (see integration notes below):
        1. SAM2 memory attention produces memory_feat  [existing pipeline]
        2. Stage1 produces multi-peak heatmap           [existing pipeline]
        3. This module:
           a. Extracts curr_node from memory_feat       (NodeEmbedHead)
           b. Extracts peak_embeds from encoder feat    (PeakNodeEmbedding)
           c. Checks ambiguity                          (AmbiguityGate)
           d. If ambiguous: scores peaks via transition  (TransitionScorer)
           e. Reweights heatmap                          (reweight_heatmap)
           f. Stores curr_node in episodic graph         (EpisodicGraphMemory)

    Total new parameters: ~12.4K + 12.4K + 45.4K + 64 = ~70K  (<0.02% of SAM2Act)
    """

    def __init__(
        self,
        feat_dim: int = 128,
        node_dim: int = 64,
        lang_dim: int = 128,
        max_peaks: int = 3,
        nms_radius: int = 5,
        entropy_threshold: float = 0.3,
        reweight_sigma: float = 5.0,
        max_graph_nodes: int = 24,
        peak_score_threshold: float = 0.05,
    ):
        super().__init__()
        self.max_peaks = max_peaks
        self.nms_radius = nms_radius
        self.reweight_sigma = reweight_sigma
        self.peak_score_threshold = peak_score_threshold

        # Sub-modules
        self.node_embed_head = NodeEmbedHead(feat_dim, node_dim)
        self.peak_node_embed = PeakNodeEmbedding(feat_dim, node_dim)
        self.transition_scorer = LanguageConditionedTransitionScorer(node_dim, lang_dim)
        self.ambiguity_gate = AmbiguityGate(entropy_threshold)

        # Learnable "no-previous-node" token (first keyframe in episode)
        self.no_prev_token = nn.Parameter(torch.randn(node_dim) * 0.02)

        # Episodic graph (not a nn.Module, just state -- one per episode)
        self._graph = EpisodicGraphMemory(max_nodes=max_graph_nodes)

    def reset_episodic_memory(self) -> None:
        """Call at the start of each episode."""
        self._graph.reset()

    @property
    def graph(self) -> EpisodicGraphMemory:
        return self._graph

    def forward(
        self,
        memory_feat: torch.Tensor,
        encoder_feat: torch.Tensor,
        heatmap_logits: torch.Tensor,
        lang_emb: torch.Tensor,
        force_graph: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            memory_feat:    [B, C, H, W]  SAM2 memory-enhanced features
            encoder_feat:   [B, C, H, W]  Stage1 encoder intermediate features
            heatmap_logits: [B, num_views, 1, H, W]  Stage1 multi-peak heatmap
            lang_emb:       [B, lang_dim]  language instruction embedding
            force_graph:    if True, skip ambiguity gate (for ablation)

        Returns dict with:
            "heatmap_refined":  [B, num_views, 1, H, W]
            "curr_node":        [B, node_dim]
            "peak_scores":      [B, K] or None  (for loss computation)
            "peak_positions":   [B, K, 2] or None
            "peak_valid":       [B, K] or None
            "is_ambiguous":     [B] bool
        """
        B = memory_feat.shape[0]
        device = memory_feat.device

        # (a) Current node embedding from memory-enhanced feature
        curr_node = self.node_embed_head(memory_feat)  # [B, node_dim]

        # (b) Check ambiguity across all views
        is_ambiguous = self.ambiguity_gate(heatmap_logits)  # [B]
        if force_graph:
            is_ambiguous = torch.ones_like(is_ambiguous)

        result = {
            "curr_node": curr_node,
            "is_ambiguous": is_ambiguous,
            "heatmap_refined": heatmap_logits,  # default: pass through
            "peak_scores": None,
            "peak_positions": None,
            "peak_valid": None,
        }

        any_ambiguous = is_ambiguous.any().item()

        if any_ambiguous:
            # Process per-view: pick the view with highest entropy for peak extraction
            # In practice, apply to the best view or all views; here we do all views
            B, V, _, H, W = heatmap_logits.shape
            refined_views = []

            # Aggregate peak info across views for loss
            all_peak_scores = []
            all_peak_positions = []
            all_peak_valid = []

            for v in range(V):
                hm_v = heatmap_logits[:, v]  # [B, 1, H, W]

                # Extract peaks
                positions, scores, valid = extract_topk_peaks(
                    hm_v, k=self.max_peaks, nms_radius=self.nms_radius,
                    score_threshold=self.peak_score_threshold,
                )  # [B, K, 2], [B, K], [B, K]

                # Peak node embeddings from encoder features
                peak_embeds = self.peak_node_embed(encoder_feat, positions)  # [B, K, D]

                # Previous node (from episodic graph or learned token)
                prev = self._graph.prev_node
                if prev is None:
                    prev_node = self.no_prev_token.unsqueeze(0).expand(B, -1)
                else:
                    prev_node = prev.unsqueeze(0).expand(B, -1)

                # Transition scoring
                peak_logits = self.transition_scorer(
                    prev_node, curr_node, peak_embeds, lang_emb
                )  # [B, K]

                # Mask invalid peaks
                peak_logits = peak_logits.masked_fill(~valid, -1e9)
                peak_weights = F.softmax(peak_logits, dim=-1)  # [B, K]

                # Reweight heatmap for ambiguous samples only
                hm_refined = hm_v.clone()
                if is_ambiguous.any():
                    amb_mask = is_ambiguous  # [B]
                    hm_reweighted = reweight_heatmap(
                        hm_v, positions, peak_weights, valid, sigma=self.reweight_sigma
                    )
                    # Blend: ambiguous samples get reweighted, others keep original
                    hm_refined = torch.where(
                        amb_mask.view(B, 1, 1, 1), hm_reweighted, hm_v
                    )

                refined_views.append(hm_refined.unsqueeze(1))
                all_peak_scores.append(peak_logits)
                all_peak_positions.append(positions)
                all_peak_valid.append(valid)

            result["heatmap_refined"] = torch.cat(refined_views, dim=1)  # [B, V, 1, H, W]
            # Return per-view peak info (stack along a new dim for loss)
            result["peak_scores"] = torch.stack(all_peak_scores, dim=1)      # [B, V, K]
            result["peak_positions"] = torch.stack(all_peak_positions, dim=1) # [B, V, K, 2]
            result["peak_valid"] = torch.stack(all_peak_valid, dim=1)         # [B, V, K]

        return result

    def update_episodic_memory(
        self,
        obs_idx: int,
        curr_node: torch.Tensor,
        keyframe_3d: Optional[torch.Tensor] = None,
    ) -> None:
        """Store current node embedding in the episodic graph.

        Call after forward() and loss computation for this keyframe.

        Args:
            obs_idx:      integer index of the current observation
            curr_node:    [node_dim] (take [0] if batched)
            keyframe_3d:  [3] optional 3D waypoint for contrastive loss
        """
        node = curr_node[0] if curr_node.ndim == 2 else curr_node
        kf = keyframe_3d[0] if (keyframe_3d is not None and keyframe_3d.ndim == 2) else keyframe_3d
        self._graph.add_node(obs_idx, node, kf)


# ---------------------------------------------------------------------------
# 8. Loss functions
# ---------------------------------------------------------------------------

def peak_selection_loss(
    peak_scores: torch.Tensor,
    peak_positions: torch.Tensor,
    peak_valid: torch.Tensor,
    gt_position_2d: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy loss for selecting the correct peak.

    The "correct" peak is the one closest to the ground-truth 2D position.

    Args:
        peak_scores:    [B, K] logits from TransitionScorer
        peak_positions: [B, K, 2] (row, col) of each peak
        peak_valid:     [B, K] bool
        gt_position_2d: [B, 2] (row, col) ground-truth target on heatmap

    Returns:
        scalar loss
    """
    # Distance from each peak to GT
    gt_exp = gt_position_2d.unsqueeze(1)  # [B, 1, 2]
    dist = ((peak_positions - gt_exp) ** 2).sum(dim=-1)  # [B, K]

    # Mask invalid peaks with large distance
    dist = dist.masked_fill(~peak_valid, 1e9)

    # Target: index of nearest peak
    target = dist.argmin(dim=-1)  # [B]

    # Mask invalid logits
    logits = peak_scores.masked_fill(~peak_valid, -1e9)

    loss = F.cross_entropy(logits, target)
    return loss


def node_contrastive_loss(
    node_embeds: torch.Tensor,
    positions_3d: torch.Tensor,
    margin: float = 0.5,
    pos_threshold: float = 0.05,
) -> torch.Tensor:
    """Contrastive loss: nodes at similar 3D positions should have similar embeddings.

    Uses a simple pair-wise margin loss within the episode's nodes.

    Args:
        node_embeds:  [N, node_dim]  all node embeddings in the episode
        positions_3d: [N, 3]         3D waypoint positions
        margin:       margin for negative pairs
        pos_threshold: 3D distance threshold for positive pairs (metres)

    Returns:
        scalar loss
    """
    N = node_embeds.shape[0]
    if N < 2:
        return torch.tensor(0.0, device=node_embeds.device)

    # Pairwise cosine similarity
    sim = torch.mm(node_embeds, node_embeds.t())  # [N, N]

    # Pairwise 3D distance
    diff = positions_3d.unsqueeze(0) - positions_3d.unsqueeze(1)  # [N, N, 3]
    dist_3d = diff.norm(dim=-1)  # [N, N]

    # Positive pairs: same position (revisit)
    pos_mask = (dist_3d < pos_threshold) & ~torch.eye(N, dtype=torch.bool, device=node_embeds.device)
    # Negative pairs: different position
    neg_mask = dist_3d >= pos_threshold

    loss = torch.tensor(0.0, device=node_embeds.device)
    count = 0

    if pos_mask.any():
        # Positive: maximise similarity (minimise 1 - sim)
        pos_loss = (1.0 - sim[pos_mask]).mean()
        loss = loss + pos_loss
        count += 1

    if neg_mask.any():
        # Negative: push similarity below margin
        neg_loss = F.relu(sim[neg_mask] - margin).mean()
        loss = loss + neg_loss
        count += 1

    return loss / max(count, 1)


def transition_prediction_loss(
    prev_node: torch.Tensor,
    curr_node: torch.Tensor,
    next_node: torch.Tensor,
    lang_emb: torch.Tensor,
    scorer: LanguageConditionedTransitionScorer,
) -> torch.Tensor:
    """Self-supervised transition loss: given (prev, curr), predict next.

    Constructs a set of candidates from the episode's nodes plus negatives,
    and trains the scorer to rank the true next_node highest.

    For simplicity, uses a direct cosine target: the prototype from
    (prev, curr) should be close to next_node's embedding.

    Args:
        prev_node: [B, node_dim]
        curr_node: [B, node_dim]
        next_node: [B, node_dim]  ground-truth next node embedding
        lang_emb:  [B, lang_dim]
        scorer:    the LanguageConditionedTransitionScorer module

    Returns:
        scalar loss
    """
    # Use scorer internals to get the prototype
    film_params = scorer.lang_film(lang_emb)
    gamma, beta = film_params.chunk(2, dim=-1)
    prev_mod = gamma * prev_node + beta
    curr_mod = gamma * curr_node + beta
    context = scorer.context_proj(torch.cat([prev_mod, curr_mod], dim=-1))
    proto = scorer.score_proj(context)
    proto = F.normalize(proto, dim=-1)

    # Target: proto should be close to next_node
    next_norm = F.normalize(next_node, dim=-1)
    loss = 1.0 - (proto * next_norm).sum(dim=-1).mean()

    return loss


# ---------------------------------------------------------------------------
# 9. Composite loss helper
# ---------------------------------------------------------------------------

class DualMemoryLoss(nn.Module):
    """Combines all loss terms for the episodic graph memory.

    Args:
        lambda_peak:        weight for peak selection CE loss
        lambda_contrastive: weight for node contrastive loss
        lambda_transition:  weight for transition prediction loss
    """

    def __init__(
        self,
        lambda_peak: float = 1.0,
        lambda_contrastive: float = 0.1,
        lambda_transition: float = 0.5,
    ):
        super().__init__()
        self.lambda_peak = lambda_peak
        self.lambda_contrastive = lambda_contrastive
        self.lambda_transition = lambda_transition

    def forward(
        self,
        peak_scores: Optional[torch.Tensor],
        peak_positions: Optional[torch.Tensor],
        peak_valid: Optional[torch.Tensor],
        gt_position_2d: Optional[torch.Tensor],
        episode_node_embeds: Optional[torch.Tensor],
        episode_positions_3d: Optional[torch.Tensor],
        prev_node: Optional[torch.Tensor],
        curr_node: Optional[torch.Tensor],
        next_node: Optional[torch.Tensor],
        lang_emb: Optional[torch.Tensor],
        scorer: Optional[LanguageConditionedTransitionScorer] = None,
    ) -> Dict[str, torch.Tensor]:
        """Returns dict of individual + total loss."""
        device = peak_scores.device if peak_scores is not None else torch.device("cuda")
        losses = {}
        total = torch.tensor(0.0, device=device)

        # Peak selection loss (per-view, average)
        if peak_scores is not None and gt_position_2d is not None:
            if peak_scores.ndim == 3:  # [B, V, K]
                V = peak_scores.shape[1]
                l_peak = torch.tensor(0.0, device=device)
                for v in range(V):
                    l_peak = l_peak + peak_selection_loss(
                        peak_scores[:, v], peak_positions[:, v],
                        peak_valid[:, v], gt_position_2d,
                    )
                l_peak = l_peak / V
            else:
                l_peak = peak_selection_loss(
                    peak_scores, peak_positions, peak_valid, gt_position_2d
                )
            losses["L_peak_select"] = l_peak
            total = total + self.lambda_peak * l_peak

        # Node contrastive loss
        if episode_node_embeds is not None and episode_positions_3d is not None:
            l_con = node_contrastive_loss(episode_node_embeds, episode_positions_3d)
            losses["L_node_contrastive"] = l_con
            total = total + self.lambda_contrastive * l_con

        # Transition prediction loss
        if (
            prev_node is not None
            and curr_node is not None
            and next_node is not None
            and lang_emb is not None
            and scorer is not None
        ):
            l_trans = transition_prediction_loss(
                prev_node, curr_node, next_node, lang_emb, scorer
            )
            losses["L_transition"] = l_trans
            total = total + self.lambda_transition * l_trans

        losses["L_total_graph"] = total
        return losses


# ---------------------------------------------------------------------------
# 10. Parameter count verification
# ---------------------------------------------------------------------------

def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def print_module_params():
    """Verify parameter counts match design spec (~45K total)."""
    m = DualLevelMemoryModule(feat_dim=128, node_dim=64, lang_dim=128)
    parts = {
        "NodeEmbedHead": m.node_embed_head,
        "PeakNodeEmbedding": m.peak_node_embed,
        "TransitionScorer": m.transition_scorer,
        "AmbiguityGate": m.ambiguity_gate,
        "no_prev_token": None,
    }
    total = 0
    for name, mod in parts.items():
        if mod is not None:
            n = count_parameters(mod)
        else:
            n = m.no_prev_token.numel()
        total += n
        print(f"  {name}: {n:,} params")
    print(f"  TOTAL: {total:,} params")
    return total


if __name__ == "__main__":
    print("=== Idea 3: Dual-Level Memory with Episodic Graph ===")
    print()
    print("Parameter counts:")
    print_module_params()

    print()
    print("Smoke test...")
    B, C, H, W, V = 2, 128, 64, 64, 3
    node_dim, lang_dim = 64, 128

    module = DualLevelMemoryModule(
        feat_dim=C, node_dim=node_dim, lang_dim=lang_dim,
        max_peaks=3, entropy_threshold=0.0,  # force ambiguity for test
    )

    memory_feat = torch.randn(B, C, H, W)
    encoder_feat = torch.randn(B, C, H, W)
    heatmap = torch.randn(B, V, 1, H, W)
    lang = torch.randn(B, lang_dim)

    module.reset_episodic_memory()
    out = module(memory_feat, encoder_feat, heatmap, lang, force_graph=True)

    print(f"  heatmap_refined shape: {out['heatmap_refined'].shape}")
    print(f"  curr_node shape:       {out['curr_node'].shape}")
    print(f"  peak_scores shape:     {out['peak_scores'].shape}")
    print(f"  is_ambiguous:          {out['is_ambiguous']}")

    # Update episodic memory
    module.update_episodic_memory(0, out["curr_node"])
    print(f"  episodic graph size:   {len(module.graph)}")

    # Second step
    out2 = module(memory_feat, encoder_feat, heatmap, lang, force_graph=True)
    module.update_episodic_memory(1, out2["curr_node"])
    print(f"  episodic graph size:   {len(module.graph)}")

    # Loss
    loss_fn = DualMemoryLoss()
    gt_pos = torch.tensor([[32.0, 32.0], [16.0, 48.0]])
    losses = loss_fn(
        peak_scores=out2["peak_scores"],
        peak_positions=out2["peak_positions"],
        peak_valid=out2["peak_valid"],
        gt_position_2d=gt_pos,
        episode_node_embeds=module.graph.get_all_embeds(),
        episode_positions_3d=torch.randn(2, 3),
        prev_node=out["curr_node"],
        curr_node=out2["curr_node"],
        next_node=torch.randn(B, node_dim),  # placeholder
        lang_emb=lang,
        scorer=module.transition_scorer,
    )
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")

    print()
    print("Smoke test passed.")
