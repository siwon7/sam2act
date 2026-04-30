"""Graph-based peak selector for Stage2 memory-guided action selection.

Extracts peaks from Stage1 heatmap, computes node embeddings per peak,
and uses a learned transition scorer to reweight peaks based on the
previous step's node embedding (graph edge).
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PeakExtractor(nn.Module):
    """Extract top-k peaks from a heatmap using non-maximum suppression."""

    def __init__(self, topk: int = 3, nms_kernel: int = 5):
        super().__init__()
        self.topk = topk
        self.nms_kernel = nms_kernel

    def forward(
        self, heatmap: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            heatmap: (bs, nc, H, W) raw heatmap logits (per-view).

        Returns:
            peak_coords: (bs, nc, topk, 2) peak positions (row, col) per view.
            peak_scores: (bs, nc, topk) softmaxed peak scores.
            peak_indices: (bs, nc, topk) flat indices for grid_sample.
        """
        bs, nc, H, W = heatmap.shape
        hm_flat = heatmap.view(bs * nc, 1, H, W)

        # Simple NMS via max-pool comparison
        hm_pool = F.max_pool2d(
            hm_flat, kernel_size=self.nms_kernel, stride=1,
            padding=self.nms_kernel // 2,
        )
        hm_nms = hm_flat * (hm_flat == hm_pool).float()

        hm_nms_flat = hm_nms.view(bs * nc, H * W)
        topk_vals, topk_idxs = torch.topk(hm_nms_flat, self.topk, dim=-1)

        rows = topk_idxs // W
        cols = topk_idxs % W
        peak_coords = torch.stack([rows, cols], dim=-1)  # (bs*nc, topk, 2)

        # Softmax over topk scores for relative weighting
        peak_scores = F.softmax(topk_vals, dim=-1)  # (bs*nc, topk)

        peak_coords = peak_coords.view(bs, nc, self.topk, 2)
        peak_scores = peak_scores.view(bs, nc, self.topk)
        peak_indices = topk_idxs.view(bs, nc, self.topk)

        return peak_coords, peak_scores, peak_indices


class NodeEmbeddingHead(nn.Module):
    """Extract node embedding from feature map at peak locations."""

    def __init__(self, feat_dim: int = 128, node_embed_dim: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, node_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(node_embed_dim, node_embed_dim),
        )

    def forward(
        self, feature_map: torch.Tensor, peak_coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            feature_map: (bs, C, H_feat, W_feat) features from memory-enhanced image.
            peak_coords: (bs, topk, 2) peak positions (row, col) in heatmap space.

        Returns:
            node_embeds: (bs, topk, node_embed_dim) L2-normalized embeddings.
        """
        bs, C, H_f, W_f = feature_map.shape
        topk = peak_coords.shape[1]

        # Normalize coords to [-1, 1] for grid_sample
        # peak_coords are in heatmap space; we assume feature_map is at
        # a potentially different resolution, so normalize to relative coords
        grid_y = peak_coords[:, :, 0].float() / max(H_f - 1, 1) * 2 - 1
        grid_x = peak_coords[:, :, 1].float() / max(W_f - 1, 1) * 2 - 1
        grid = torch.stack([grid_x, grid_y], dim=-1)  # (bs, topk, 2)
        grid = grid.unsqueeze(1)  # (bs, 1, topk, 2)

        # Sample features at peak locations
        sampled = F.grid_sample(
            feature_map, grid, mode="bilinear", align_corners=True
        )  # (bs, C, 1, topk)
        sampled = sampled.squeeze(2).permute(0, 2, 1)  # (bs, topk, C)

        embeds = self.proj(sampled)  # (bs, topk, node_embed_dim)
        return F.normalize(embeds, dim=-1)


class TransitionScorer(nn.Module):
    """Score candidate peaks based on transition context (prev + curr node)."""

    def __init__(self, node_embed_dim: int = 64):
        super().__init__()
        self.context_proj = nn.Sequential(
            nn.Linear(node_embed_dim * 2, node_embed_dim),
            nn.ReLU(inplace=True),
        )
        self.score_proj = nn.Linear(node_embed_dim, node_embed_dim)

    def forward(
        self,
        prev_node_embed: torch.Tensor,
        curr_node_embed: torch.Tensor,
        candidate_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            prev_node_embed: (bs, node_embed_dim) previous step's selected node.
            curr_node_embed: (bs, node_embed_dim) current step's pooled node.
            candidate_embeds: (bs, topk, node_embed_dim) peak node embeddings.

        Returns:
            scores: (bs, topk) transition compatibility scores (pre-softmax).
        """
        context = self.context_proj(
            torch.cat([prev_node_embed, curr_node_embed], dim=-1)
        )  # (bs, node_embed_dim)
        next_proto = self.score_proj(context)  # (bs, node_embed_dim)

        scores = torch.bmm(
            candidate_embeds,
            next_proto.unsqueeze(-1),
        ).squeeze(-1)  # (bs, topk)

        return scores


class GraphPeakSelector(nn.Module):
    """Complete graph-based peak selection module.

    Combines peak extraction, node embedding, and transition scoring
    to reweight a multi-peak heatmap based on learned graph structure.
    """

    def __init__(
        self,
        feat_dim: int = 128,
        node_embed_dim: int = 64,
        topk: int = 3,
        nms_kernel: int = 5,
    ):
        super().__init__()
        self.topk = topk
        self.peak_extractor = PeakExtractor(topk=topk, nms_kernel=nms_kernel)
        self.node_embed_head = NodeEmbeddingHead(
            feat_dim=feat_dim, node_embed_dim=node_embed_dim
        )
        self.transition_scorer = TransitionScorer(node_embed_dim=node_embed_dim)

        # Pooling head: aggregate multi-view features into single node embed
        self.pool_proj = nn.Sequential(
            nn.Linear(feat_dim, node_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(node_embed_dim, node_embed_dim),
        )

    def get_pooled_node_embed(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Pool entire feature map into a single node embedding.

        Args:
            feature_map: (num_views, C, H, W) memory-enhanced features.

        Returns:
            node_embed: (1, node_embed_dim) L2-normalized.
        """
        pooled = feature_map.mean(dim=(0, 2, 3))  # (C,)
        embed = self.pool_proj(pooled.unsqueeze(0))  # (1, node_embed_dim)
        return F.normalize(embed, dim=-1)

    def forward(
        self,
        heatmap: torch.Tensor,
        feature_map: torch.Tensor,
        prev_node_embed: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            heatmap: (bs, nc, 1, H, W) Stage1 heatmap logits.
            feature_map: (bs_views, C, H_f, W_f) memory-enhanced image features.
                         In SAM2Act, this is (num_views, 128, 16, 16) after
                         memory attention.
            prev_node_embed: (1, node_embed_dim) or None if first step.

        Returns:
            dict with:
                reweighted_heatmap: (bs, nc, 1, H, W)
                curr_node_embed: (1, node_embed_dim) for storing in memory
                peak_coords: (bs, nc, topk, 2)
                transition_scores: (bs_views, topk) or None
                peak_node_embeds: (bs_views, topk, node_embed_dim)
        """
        bs, nc, one, H, W = heatmap.shape
        hm_2d = heatmap.squeeze(2)  # (bs, nc, H, W)

        # 1. Extract peaks
        peak_coords, peak_scores, peak_indices = self.peak_extractor(hm_2d)
        # peak_coords: (bs, nc, topk, 2)

        # 2. Node embeddings at peak locations (process all views)
        num_views = feature_map.shape[0]
        H_f, W_f = feature_map.shape[2], feature_map.shape[3]

        # Scale peak coords from heatmap space to feature space
        scale_h = H_f / H
        scale_w = W_f / W
        scaled_coords = peak_coords.view(-1, self.topk, 2).clone().float()
        scaled_coords[:, :, 0] *= scale_h
        scaled_coords[:, :, 1] *= scale_w

        peak_node_embeds = self.node_embed_head(
            feature_map, scaled_coords[:num_views]
        )  # (num_views, topk, node_embed_dim)

        # 3. Current node embedding (pooled from all views)
        curr_node_embed = self.get_pooled_node_embed(feature_map)  # (1, D)

        # 4. Transition scoring (if prev_node available)
        transition_scores = None
        reweighted = heatmap.clone()

        if prev_node_embed is not None:
            # Score per-view peaks using transition context
            prev_expanded = prev_node_embed.expand(num_views, -1)
            curr_expanded = curr_node_embed.expand(num_views, -1)
            transition_scores = self.transition_scorer(
                prev_expanded, curr_expanded, peak_node_embeds
            )  # (num_views, topk)

            # Reweight heatmap: boost selected peaks, suppress others
            peak_weights = F.softmax(transition_scores, dim=-1)  # (num_views, topk)

            # Build additive bias in heatmap space
            bias = torch.zeros(num_views, H, W, device=heatmap.device, dtype=heatmap.dtype)
            sigma = 3.0
            for k in range(self.topk):
                coords_k = peak_coords[0, :num_views, k, :]  # (num_views, 2)
                weight_k = peak_weights[:, k]  # (num_views,)
                # Create Gaussian bump at peak location
                row_grid = torch.arange(H, device=heatmap.device).float()
                col_grid = torch.arange(W, device=heatmap.device).float()
                rr, cc = torch.meshgrid(row_grid, col_grid, indexing="ij")
                for v in range(num_views):
                    r0, c0 = coords_k[v, 0].float(), coords_k[v, 1].float()
                    gaussian = torch.exp(-((rr - r0) ** 2 + (cc - c0) ** 2) / (2 * sigma ** 2))
                    bias[v] += weight_k[v] * gaussian

            # Apply as additive logit bias
            bias = bias.unsqueeze(0).unsqueeze(2)  # (1, num_views, 1, H, W)
            reweighted = heatmap + bias

        return {
            "reweighted_heatmap": reweighted,
            "curr_node_embed": curr_node_embed,
            "peak_coords": peak_coords,
            "transition_scores": transition_scores,
            "peak_node_embeds": peak_node_embeds,
        }


# ─── Auxiliary Loss Functions ───


def peak_selection_loss(
    transition_scores: torch.Tensor,
    peak_coords: torch.Tensor,
    gt_position_2d: torch.Tensor,
) -> torch.Tensor:
    """CE loss: the peak closest to GT should have the highest score.

    Args:
        transition_scores: (num_views, topk) pre-softmax scores.
        peak_coords: (1, num_views, topk, 2) peak positions.
        gt_position_2d: (num_views, 2) GT target in image coords.

    Returns:
        Scalar loss.
    """
    num_views, topk = transition_scores.shape
    coords = peak_coords[0]  # (num_views, topk, 2)
    gt = gt_position_2d.unsqueeze(1).expand_as(coords[:, :, :2])  # (nv, topk, 2)
    dists = torch.norm((coords.float() - gt.float()), dim=-1)  # (nv, topk)
    gt_peak_idx = dists.argmin(dim=-1)  # (num_views,)
    return F.cross_entropy(transition_scores, gt_peak_idx)


def node_contrastive_loss(
    node_embeds: torch.Tensor,
    positions_3d: torch.Tensor,
    temperature: float = 0.1,
    pos_radius: float = 0.03,
) -> torch.Tensor:
    """InfoNCE contrastive: same 3D position → similar embeddings.

    Args:
        node_embeds: (num_steps, embed_dim) pooled node embeds per step.
        positions_3d: (num_steps, 3) gripper positions.
        temperature: softmax temperature.
        pos_radius: distance threshold for positive pairs.

    Returns:
        Scalar loss (0 if no valid positive pairs).
    """
    n = node_embeds.shape[0]
    if n <= 1:
        return torch.tensor(0.0, device=node_embeds.device)

    embeds = F.normalize(node_embeds.float(), dim=-1)
    sim = embeds @ embeds.t() / temperature  # (n, n)
    dist = torch.cdist(positions_3d.float(), positions_3d.float(), p=2)

    eye = torch.eye(n, device=sim.device, dtype=torch.bool)
    pos_mask = (dist < pos_radius) & (~eye)

    if not pos_mask.any():
        return torch.tensor(0.0, device=node_embeds.device)

    neg_mask = (~eye)
    sim_neg = sim.masked_fill(~neg_mask, float("-inf"))
    log_denom = torch.logsumexp(sim_neg, dim=-1)  # (n,)

    pos_sim = torch.where(pos_mask, sim, torch.full_like(sim, float("-inf")))
    log_num = torch.logsumexp(pos_sim, dim=-1)  # (n,)

    valid = pos_mask.any(dim=-1)
    if not valid.any():
        return torch.tensor(0.0, device=node_embeds.device)

    loss = -(log_num[valid] - log_denom[valid]).mean()
    return loss


def transition_prediction_loss(
    node_embeds: torch.Tensor,
    transition_scorer: TransitionScorer,
) -> torch.Tensor:
    """For consecutive steps, the transition scorer should predict the next embed.

    Uses cosine similarity between predicted next and actual next node embed.

    Args:
        node_embeds: (num_steps, embed_dim) sequential node embeddings.
        transition_scorer: the TransitionScorer module.

    Returns:
        Scalar loss.
    """
    n = node_embeds.shape[0]
    if n <= 2:
        return torch.tensor(0.0, device=node_embeds.device)

    losses = []
    for t in range(1, n - 1):
        prev = node_embeds[t - 1].unsqueeze(0)  # (1, D)
        curr = node_embeds[t].unsqueeze(0)       # (1, D)
        actual_next = node_embeds[t + 1]         # (D,)

        context = transition_scorer.context_proj(
            torch.cat([prev, curr], dim=-1)
        )
        predicted_next = F.normalize(
            transition_scorer.score_proj(context), dim=-1
        ).squeeze(0)  # (D,)

        cos_sim = F.cosine_similarity(
            predicted_next.unsqueeze(0),
            actual_next.detach().unsqueeze(0),
            dim=-1,
        )
        losses.append(1.0 - cos_sim)

    return torch.stack(losses).mean()
