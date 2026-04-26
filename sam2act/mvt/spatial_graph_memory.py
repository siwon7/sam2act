from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SpatialMemoryNode:
    feat: torch.Tensor
    pos: torch.Tensor
    count: int = 1


def _as_pos_3(pos: torch.Tensor) -> torch.Tensor:
    if not isinstance(pos, torch.Tensor):
        pos = torch.as_tensor(pos)
    if pos.shape == (3,):
        return pos
    if pos.ndim == 2 and pos.shape[-2:] == (1, 3):
        return pos[0]
    raise ValueError(f"Expected pos shape (3,) or (1,3); got {tuple(pos.shape)}")


class SpatialEMAMemoryBank:
    def __init__(
        self,
        eps: float,
        alpha: float,
        *,
        max_nodes: Optional[int] = None,
    ) -> None:
        if eps <= 0:
            raise ValueError(f"`eps` must be > 0; got {eps}")
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"`alpha` must be in (0, 1]; got {alpha}")
        if max_nodes is not None and max_nodes <= 0:
            raise ValueError(f"`max_nodes` must be >= 1; got {max_nodes}")

        self.eps = float(eps)
        self.alpha = float(alpha)
        self.max_nodes = max_nodes
        self.nodes: List[SpatialMemoryNode] = []

    def __len__(self) -> int:
        return len(self.nodes)

    def reset(self) -> None:
        self.nodes.clear()

    @property
    def device(self) -> Optional[torch.device]:
        if not self.nodes:
            return None
        return self.nodes[0].feat.device

    def to(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "SpatialEMAMemoryBank":
        if device is None and dtype is None:
            return self
        for node in self.nodes:
            node.feat = node.feat.to(device=device, dtype=dtype)
            node.pos = node.pos.to(device=device, dtype=torch.float32)
        return self

    def stacked_feats(self) -> torch.Tensor:
        if not self.nodes:
            raise RuntimeError("Memory bank is empty.")
        return torch.stack([n.feat for n in self.nodes], dim=0)

    def stacked_pos(self) -> torch.Tensor:
        if not self.nodes:
            raise RuntimeError("Memory bank is empty.")
        return torch.stack([n.pos for n in self.nodes], dim=0)

    def update(
        self,
        new_kf_feat: torch.Tensor,
        new_kf_pos: torch.Tensor,
        *,
        eps: Optional[float] = None,
        alpha: Optional[float] = None,
    ) -> int:
        eps = self.eps if eps is None else float(eps)
        alpha = self.alpha if alpha is None else float(alpha)
        if eps <= 0:
            raise ValueError(f"`eps` must be > 0; got {eps}")
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"`alpha` must be in (0, 1]; got {alpha}")

        new_kf_pos = _as_pos_3(new_kf_pos).to(device=new_kf_feat.device, dtype=torch.float32)

        if not self.nodes:
            self.nodes.append(SpatialMemoryNode(feat=new_kf_feat, pos=new_kf_pos, count=1))
            return 0

        mem_pos = torch.stack([n.pos.to(device=new_kf_feat.device) for n in self.nodes], dim=0).float()
        dists = torch.linalg.norm(mem_pos - new_kf_pos.float(), dim=-1)  # (N,)
        min_idx = int(torch.argmin(dists).item())
        d_min = float(dists[min_idx].item())

        if d_min < eps:
            node = self.nodes[min_idx]
            node.feat = alpha * new_kf_feat + (1.0 - alpha) * node.feat.to(device=new_kf_feat.device)
            node.pos = alpha * new_kf_pos + (1.0 - alpha) * node.pos.to(device=new_kf_feat.device)
            node.count += 1
            return min_idx

        self.nodes.append(SpatialMemoryNode(feat=new_kf_feat, pos=new_kf_pos, count=1))

        if self.max_nodes is not None and len(self.nodes) > self.max_nodes:
            self.nodes.pop(0)
            return len(self.nodes) - 1

        return len(self.nodes) - 1


def batched_spatial_ema_update(
    memory_banks: Sequence[SpatialEMAMemoryBank],
    new_kf_feat: torch.Tensor,
    new_kf_pos: torch.Tensor,
    *,
    eps: Optional[float] = None,
    alpha: Optional[float] = None,
) -> List[int]:
    if new_kf_feat.ndim < 1:
        raise ValueError(f"`new_kf_feat` must have a batch dim; got shape {tuple(new_kf_feat.shape)}")
    if new_kf_pos.shape[-1] != 3:
        raise ValueError(f"`new_kf_pos` must have last dim 3; got shape {tuple(new_kf_pos.shape)}")
    batch_size = new_kf_feat.shape[0]
    if len(memory_banks) != batch_size:
        raise ValueError(f"len(memory_banks) must match batch size {batch_size}; got {len(memory_banks)}")

    indices: List[int] = []
    for b in range(batch_size):
        indices.append(
            memory_banks[b].update(
                new_kf_feat[b],
                new_kf_pos[b],
                eps=eps,
                alpha=alpha,
            )
        )
    return indices


def spatial_adjacency_mask(
    pos_q: torch.Tensor,
    pos_k: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if eps <= 0:
        raise ValueError(f"`eps` must be > 0; got {eps}")
    if pos_q.shape[-1] != 3 or pos_k.shape[-1] != 3:
        raise ValueError(
            f"pos tensors must have last dim 3; got pos_q={tuple(pos_q.shape)}, pos_k={tuple(pos_k.shape)}"
        )
    if pos_q.ndim != 3 or pos_k.ndim != 3:
        raise ValueError(
            f"pos tensors must be [B, L, 3]; got pos_q={tuple(pos_q.shape)}, pos_k={tuple(pos_k.shape)}"
        )
    if pos_q.shape[0] != pos_k.shape[0]:
        raise ValueError(
            f"batch size mismatch; got pos_q={tuple(pos_q.shape)}, pos_k={tuple(pos_k.shape)}"
        )

    dists = torch.cdist(pos_q.float(), pos_k.float())  # [B, Lq, Lk]
    return dists < eps


def spatial_masked_scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    pos_q: torch.Tensor,
    pos_k: torch.Tensor,
    eps: float,
    dropout_p: float = 0.0,
    training: Optional[bool] = None,
    return_weights: bool = False,
) -> torch.Tensor:
    """
    q: [B, Lq, Dk]
    k: [B, Lk, Dk]
    v: [B, Lk, Dv]
    pos_q: [B, Lq, 3]
    pos_k: [B, Lk, 3]
    """
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError(f"q/k/v must be rank-3; got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError(f"batch mismatch; got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}")
    if k.shape[1] != v.shape[1]:
        raise ValueError(f"key/value length mismatch; got k={tuple(k.shape)}, v={tuple(v.shape)}")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(f"q/k dim mismatch; got q={tuple(q.shape)}, k={tuple(k.shape)}")
    if dropout_p < 0.0 or dropout_p >= 1.0:
        raise ValueError(f"`dropout_p` must be in [0, 1); got {dropout_p}")

    mask = spatial_adjacency_mask(pos_q=pos_q, pos_k=pos_k, eps=eps)  # [B, Lq, Lk] bool

    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # [B, Lq, Lk]

    scores_fp32 = scores.float()
    scores_fp32 = scores_fp32.masked_fill(~mask, torch.finfo(scores_fp32.dtype).min)

    attn = torch.softmax(scores_fp32, dim=-1).to(dtype=scores.dtype)

    attn = attn * mask.to(dtype=attn.dtype)
    denom = attn.sum(dim=-1, keepdim=True)
    attn = torch.where(denom > 0, attn / denom, torch.zeros_like(attn))

    if dropout_p > 0.0:
        if training is None:
            training = False
        attn = F.dropout(attn, p=dropout_p, training=training)

    out = torch.matmul(attn, v)  # [B, Lq, Dv]
    if return_weights:
        return out, attn
    return out


def spatial_graph_contrastive_loss(
    h: torch.Tensor,
    pos: torch.Tensor,
    *,
    eps: float,
    tau: float,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    InfoNCE-style loss where positives are defined by 3D proximity.

    h: [B, Seq, D]
    pos: [B, Seq, 3]
    """
    if eps <= 0:
        raise ValueError(f"`eps` must be > 0; got {eps}")
    if tau <= 0:
        raise ValueError(f"`tau` must be > 0; got {tau}")
    if h.ndim != 3 or pos.ndim != 3:
        raise ValueError(f"h/pos must be rank-3; got h={tuple(h.shape)}, pos={tuple(pos.shape)}")
    if pos.shape[-1] != 3:
        raise ValueError(f"`pos` must have last dim 3; got {tuple(pos.shape)}")
    if h.shape[0] != pos.shape[0] or h.shape[1] != pos.shape[1]:
        raise ValueError(f"batch/seq mismatch; got h={tuple(h.shape)}, pos={tuple(pos.shape)}")
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"Invalid reduction: {reduction!r}")

    batch_size, seq_len, feat_dim = h.shape
    num_states = batch_size * seq_len
    if num_states <= 1:
        if reduction == "none":
            return h.new_zeros((batch_size, seq_len))
        return h.new_tensor(0.0)

    h_flat = h.reshape(num_states, feat_dim).float()
    pos_flat = pos.reshape(num_states, 3).float()

    h_norm = F.normalize(h_flat, dim=-1)
    sim = torch.matmul(h_norm, h_norm.transpose(0, 1))  # [N, N]

    dists = torch.cdist(pos_flat, pos_flat)  # [N, N]
    eye = torch.eye(num_states, device=h.device, dtype=torch.bool)
    pos_mask = (dists < eps) & ~eye

    if not bool(pos_mask.any().item()):
        if reduction == "none":
            return h.new_zeros((batch_size, seq_len))
        return h.new_tensor(0.0)

    logits = (sim / tau).to(device=h.device)
    logits = logits.masked_fill(eye, torch.finfo(logits.dtype).min)
    log_prob = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

    pos_log_prob = log_prob.masked_fill(~pos_mask, 0.0)
    pos_count = pos_mask.sum(dim=-1)  # [N]

    loss_per_anchor = -(pos_log_prob.sum(dim=-1) / pos_count.clamp(min=1))
    valid = pos_count > 0

    if reduction == "none":
        loss = loss_per_anchor * valid.to(dtype=loss_per_anchor.dtype)
        return loss.reshape(batch_size, seq_len)
    if reduction == "sum":
        return loss_per_anchor[valid].sum()
    return loss_per_anchor[valid].mean()


class SpatialGraphContrastiveLoss(nn.Module):
    def __init__(
        self,
        *,
        eps: float,
        tau: float,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.tau = float(tau)
        self.reduction = reduction

    def forward(self, h: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return spatial_graph_contrastive_loss(
            h,
            pos,
            eps=self.eps,
            tau=self.tau,
            reduction=self.reduction,
        )
