"""Lightweight top-K peak selector for SAM2Act V9 experiments.

The selector does not replace the Stage1 heatmap objective. It learns a small
ranking head over 3D peak proposals extracted from the Stage1 heatmap, so we
can test whether keeping top-K candidates can recover top1 crop failures.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


class TopK3DPeakSelector(nn.Module):
    """Score Stage1 3D peak candidates with proprio and language context."""

    def __init__(
        self,
        topk: int = 3,
        proprio_dim: int = 4,
        lang_dim: int = 512,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.topk = int(topk)
        self.proprio_dim = int(proprio_dim)
        self.lang_dim = int(lang_dim)
        self.lang_proj = nn.Sequential(
            nn.Linear(self.lang_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        in_dim = 3 + 1 + self.topk + self.proprio_dim + hidden_dim
        self.scorer = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        candidate_xyz: torch.Tensor,
        candidate_scores: torch.Tensor,
        proprio: torch.Tensor | None,
        lang_emb: torch.Tensor | None,
        candidate_view_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits over K candidates.

        Args:
            candidate_xyz: ``[B, K, 3]`` local cube coordinates.
            candidate_scores: ``[B, K]`` heatmap mass sampled at candidates.
            proprio: ``[B, proprio_dim]`` low-dimensional state.
            lang_emb: ``[B, T, lang_dim]`` CLIP token embeddings.
        """
        bs, k, _ = candidate_xyz.shape
        if k != self.topk:
            raise ValueError(f"expected topk={self.topk}, got {k}")
        candidate_scores = candidate_scores.to(dtype=candidate_xyz.dtype)

        if proprio is None:
            proprio = candidate_xyz.new_zeros(bs, self.proprio_dim)
        else:
            proprio = proprio.to(dtype=candidate_xyz.dtype)
            if proprio.shape[-1] != self.proprio_dim:
                proprio = proprio[..., : self.proprio_dim]

        if lang_emb is None:
            lang_ctx = candidate_xyz.new_zeros(bs, self.lang_dim)
        else:
            lang_ctx = lang_emb.to(dtype=candidate_xyz.dtype).mean(dim=1)
            if lang_ctx.shape[-1] != self.lang_dim:
                lang_ctx = lang_ctx[..., : self.lang_dim]
        lang_ctx = self.lang_proj(lang_ctx)

        rank_eye = torch.eye(
            self.topk, device=candidate_xyz.device, dtype=candidate_xyz.dtype
        ).unsqueeze(0).expand(bs, -1, -1)
        context = torch.cat([proprio, lang_ctx], dim=-1).unsqueeze(1).expand(-1, k, -1)
        feat = torch.cat(
            [
                candidate_xyz.detach(),
                candidate_scores.unsqueeze(-1),
                rank_eye,
                context,
            ],
            dim=-1,
        )
        return self.scorer(feat).squeeze(-1)


class Stage2CandidateGraphSelector(nn.Module):
    """Pairwise graph selector over Stage1 candidates for V10 experiments.

    Each Stage1 proposal is a node. The selector keeps the V9 inputs
    (candidate xyz, fused heatmap score, proprio, language) and adds pairwise
    relative geometry plus per-view support when available. It deliberately
    stays small so it can be enabled as an auxiliary head without changing the
    SAM2Act backbone.
    """

    def __init__(
        self,
        topk: int = 3,
        proprio_dim: int = 4,
        lang_dim: int = 512,
        hidden_dim: int = 64,
        max_views: int = 5,
    ):
        super().__init__()
        self.topk = int(topk)
        self.proprio_dim = int(proprio_dim)
        self.lang_dim = int(lang_dim)
        self.max_views = int(max_views)

        self.lang_proj = nn.Sequential(
            nn.Linear(self.lang_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.node_proj = nn.Sequential(
            nn.Linear(3 + 1 + self.topk + self.max_views + self.proprio_dim + hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.edge_proj = nn.Sequential(
            nn.Linear((hidden_dim * 2) + 4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def _context(
        self,
        candidate_xyz: torch.Tensor,
        proprio: torch.Tensor | None,
        lang_emb: torch.Tensor | None,
    ) -> torch.Tensor:
        bs = candidate_xyz.shape[0]
        if proprio is None:
            proprio = candidate_xyz.new_zeros(bs, self.proprio_dim)
        else:
            proprio = proprio.to(dtype=candidate_xyz.dtype)
            if proprio.shape[-1] != self.proprio_dim:
                proprio = proprio[..., : self.proprio_dim]

        if lang_emb is None:
            lang_ctx = candidate_xyz.new_zeros(bs, self.lang_dim)
        else:
            lang_ctx = lang_emb.to(dtype=candidate_xyz.dtype).mean(dim=1)
            if lang_ctx.shape[-1] != self.lang_dim:
                lang_ctx = lang_ctx[..., : self.lang_dim]
        return torch.cat([proprio, self.lang_proj(lang_ctx)], dim=-1)

    def forward(
        self,
        candidate_xyz: torch.Tensor,
        candidate_scores: torch.Tensor,
        proprio: torch.Tensor | None,
        lang_emb: torch.Tensor | None,
        candidate_view_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bs, k, _ = candidate_xyz.shape
        if k != self.topk:
            raise ValueError(f"expected topk={self.topk}, got {k}")

        candidate_scores = candidate_scores.to(dtype=candidate_xyz.dtype)
        if candidate_view_scores is None:
            view_scores = candidate_xyz.new_zeros(bs, k, self.max_views)
        else:
            view_scores = candidate_view_scores.to(dtype=candidate_xyz.dtype)
            if view_scores.shape[-1] < self.max_views:
                pad = self.max_views - view_scores.shape[-1]
                view_scores = F.pad(view_scores, (0, pad))
            view_scores = view_scores[..., : self.max_views]

        rank_eye = torch.eye(
            self.topk, device=candidate_xyz.device, dtype=candidate_xyz.dtype
        ).unsqueeze(0).expand(bs, -1, -1)
        context = self._context(candidate_xyz, proprio, lang_emb).unsqueeze(1).expand(-1, k, -1)
        node_in = torch.cat(
            [
                candidate_xyz.detach(),
                candidate_scores.unsqueeze(-1),
                rank_eye,
                view_scores,
                context,
            ],
            dim=-1,
        )
        node = self.node_proj(node_in)

        src = node.unsqueeze(2).expand(-1, -1, k, -1)
        dst = node.unsqueeze(1).expand(-1, k, -1, -1)
        rel = candidate_xyz.unsqueeze(2) - candidate_xyz.unsqueeze(1)
        dist = torch.norm(rel, dim=-1, keepdim=True)
        edge_in = torch.cat([src, dst, rel.detach(), dist.detach()], dim=-1)
        msg = self.edge_proj(edge_in)

        eye = torch.eye(k, device=candidate_xyz.device, dtype=torch.bool).view(1, k, k, 1)
        msg = msg.masked_fill(eye, 0)
        denom = max(k - 1, 1)
        pooled = msg.sum(dim=2) / float(denom)
        return self.scorer(torch.cat([node, pooled], dim=-1)).squeeze(-1)


def nearest_candidate_targets(
    candidate_xyz: torch.Tensor,
    gt_xyz: torch.Tensor,
    positive_radius: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return nearest candidate index, distance, and coverage mask."""
    dists = torch.norm(candidate_xyz - gt_xyz.unsqueeze(1), dim=-1)
    min_dist, target_idx = dists.min(dim=1)
    covered = min_dist <= float(positive_radius)
    return target_idx, min_dist, covered


def peak_selection_loss(
    logits: torch.Tensor,
    target_idx: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross entropy over candidate logits, optionally masking invalid rows."""
    if valid is not None:
        valid = valid.bool()
        if not valid.any():
            return logits.sum() * 0.0
        logits = logits[valid]
        target_idx = target_idx[valid]
    return F.cross_entropy(logits.float(), target_idx.long())
