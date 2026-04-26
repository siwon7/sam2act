import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def pairwise_l2_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.reshape(a.shape[0], -1)
    b = b.reshape(b.shape[0], -1)
    return torch.cdist(a, b, p=2)


class SpatialGraphMemoryBank:
    def __init__(
        self,
        distance_threshold: float,
        ema_alpha: float,
        grip_threshold: float = 1.0,
        feature_similarity_threshold: float = -1.0,
    ):
        self.distance_threshold = float(distance_threshold)
        self.ema_alpha = float(ema_alpha)
        self.grip_threshold = float(grip_threshold)
        self.feature_similarity_threshold = float(feature_similarity_threshold)

    @staticmethod
    def pooled_feature(memory_feat: torch.Tensor) -> torch.Tensor:
        pooled = memory_feat.mean(dim=0).reshape(-1).float()
        return F.normalize(pooled, dim=0)

    def update(
        self,
        memory_bank: List[Dict[str, torch.Tensor]],
        new_feat: torch.Tensor,
        new_pos_enc: torch.Tensor,
        new_kf_pos: torch.Tensor,
        new_grip_state: Optional[torch.Tensor],
        obs_idx: int,
    ) -> Tuple[List[Dict[str, torch.Tensor]], bool, int]:
        new_pos = new_kf_pos.reshape(-1, 3)[0]
        new_grip = (
            new_grip_state.reshape(-1)[0].float().to(new_pos.device)
            if new_grip_state is not None
            else None
        )
        new_feat_mean = self.pooled_feature(new_feat).to(new_pos.device)
        if len(memory_bank) == 0:
            memory_bank.append(
                {
                    "memory": new_feat,
                    "memory_pos": new_pos_enc,
                    "pos3d": new_pos,
                    "grip": new_grip if new_grip is not None else torch.tensor(0.0, device=new_pos.device),
                    "feat_mean": new_feat_mean,
                    "last_obs_idx": int(obs_idx),
                    "count": 1,
                }
            )
            return memory_bank, False, 0

        bank_pos = torch.stack([node["pos3d"].to(new_pos.device) for node in memory_bank], dim=0)
        dists = pairwise_l2_distance(new_pos.unsqueeze(0), bank_pos).squeeze(0)

        eligible = dists < self.distance_threshold
        if new_grip is not None:
            bank_grip = torch.stack(
                [node.get("grip", torch.tensor(0.0, device=new_pos.device)).to(new_pos.device) for node in memory_bank],
                dim=0,
            ).reshape(-1)
            grip_diffs = torch.abs(bank_grip - new_grip)
            eligible = eligible & (grip_diffs < self.grip_threshold)
        if self.feature_similarity_threshold > -1.0:
            bank_feat = torch.stack(
                [node.get("feat_mean", self.pooled_feature(node["memory"]).to(new_pos.device)) for node in memory_bank],
                dim=0,
            )
            sims = F.cosine_similarity(
                new_feat_mean.unsqueeze(0), bank_feat.to(new_pos.device), dim=-1
            )
            eligible = eligible & (sims > self.feature_similarity_threshold)

        if bool(eligible.any().item()):
            masked_dists = dists.masked_fill(~eligible, float("inf"))
            min_dist, min_idx = torch.min(masked_dists, dim=0)
            min_idx_int = int(min_idx.item())
            node = memory_bank[min_idx_int]
            alpha = self.ema_alpha
            node["memory"] = alpha * new_feat + (1.0 - alpha) * node["memory"]
            node["memory_pos"] = alpha * new_pos_enc + (1.0 - alpha) * node["memory_pos"]
            node["pos3d"] = alpha * new_pos + (1.0 - alpha) * node["pos3d"].to(new_pos.device)
            if new_grip is not None:
                node["grip"] = alpha * new_grip + (1.0 - alpha) * node["grip"].to(new_pos.device)
            node["feat_mean"] = alpha * new_feat_mean + (1.0 - alpha) * node["feat_mean"].to(new_pos.device)
            node["last_obs_idx"] = int(obs_idx)
            node["count"] = int(node.get("count", 1)) + 1
            return memory_bank, True, min_idx_int

        memory_bank.append(
            {
                "memory": new_feat,
                "memory_pos": new_pos_enc,
                "pos3d": new_pos,
                "grip": new_grip if new_grip is not None else torch.tensor(0.0, device=new_pos.device),
                "feat_mean": new_feat_mean,
                "last_obs_idx": int(obs_idx),
                "count": 1,
            }
        )
        return memory_bank, False, len(memory_bank) - 1

    @staticmethod
    def recent_nodes(
        memory_bank: Sequence[Dict[str, torch.Tensor]], max_nodes: int
    ) -> List[Dict[str, torch.Tensor]]:
        nodes = sorted(memory_bank, key=lambda x: int(x["last_obs_idx"]), reverse=True)
        return nodes[:max_nodes]


def build_spatial_attention_bias(
    query_pos: Optional[torch.Tensor],
    memory_nodes: Sequence[Dict[str, torch.Tensor]],
    distance_threshold: float,
    mask_value: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    if query_pos is None or len(memory_nodes) == 0:
        return None

    q = query_pos.reshape(-1, 3).to(device=device, dtype=torch.float32)
    if q.shape[0] != 1:
        q = q[:1]

    node_pos = torch.stack([node["pos3d"].reshape(-1, 3)[0] for node in memory_nodes], dim=0)
    node_pos = node_pos.to(device=device, dtype=torch.float32)
    dists = pairwise_l2_distance(q, node_pos).squeeze(0)
    keep_nodes = dists < float(distance_threshold)
    if not bool(keep_nodes.any().item()):
        keep_nodes[torch.argmin(dists)] = True

    token_bias = []
    for keep, node in zip(keep_nodes.tolist(), memory_nodes):
        num_tokens = int(node["memory"].shape[0])
        value = 0.0 if keep else float(mask_value)
        token_bias.append(torch.full((num_tokens,), value, device=device, dtype=dtype))

    if not token_bias:
        return None

    bias = torch.cat(token_bias, dim=0).view(1, 1, 1, -1)
    return bias


def build_summary_attention_bias(
    summary_nodes: Sequence[Dict[str, torch.Tensor]],
    prev_summary_idx: Optional[int],
    transition_counts: Optional[Dict[Tuple[int, int], int]],
    current_grip_state: Optional[torch.Tensor],
    query_pos: Optional[torch.Tensor],
    distance_threshold: float,
    transition_bias_scale: float,
    grip_bias_scale: float,
    spatial_bias_scale: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    if len(summary_nodes) == 0:
        return None

    current_grip = (
        current_grip_state.reshape(-1)[0].float().to(device)
        if current_grip_state is not None
        else None
    )
    current_pos = (
        query_pos.reshape(-1, 3)[0].float().to(device)
        if query_pos is not None
        else None
    )

    max_transition = 1.0
    if transition_counts:
        max_transition = float(max(transition_counts.values()))

    token_bias = []
    for node_idx, node in enumerate(summary_nodes):
        bias_value = 0.0
        if prev_summary_idx is not None and transition_counts:
            count = transition_counts.get((int(prev_summary_idx), int(node_idx)), 0)
            if count > 0:
                bias_value += float(transition_bias_scale) * math.log1p(float(count)) / math.log1p(max_transition)

        if current_grip is not None:
            node_grip = node.get("grip", torch.tensor(0.0, device=device)).reshape(-1)[0].float().to(device)
            grip_sim = 1.0 - float(torch.abs(current_grip - node_grip).item())
            bias_value += float(grip_bias_scale) * max(grip_sim, 0.0)

        if current_pos is not None and spatial_bias_scale > 0.0:
            node_pos = node["pos3d"].reshape(-1, 3)[0].float().to(device)
            dist = float(torch.norm(current_pos - node_pos, p=2).item())
            spatial_sim = math.exp(-dist / max(float(distance_threshold), 1e-6))
            bias_value += float(spatial_bias_scale) * spatial_sim

        num_tokens = int(node["memory"].shape[0])
        token_bias.append(
            torch.full((num_tokens,), float(bias_value), device=device, dtype=dtype)
        )

    if not token_bias:
        return None
    return torch.cat(token_bias, dim=0).view(1, 1, 1, -1)


def spatial_graph_contrastive_loss(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    temperature: float,
    distance_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    hidden_states: [B, S, D]
    positions: [B, S, 3]
    """
    device = hidden_states.device
    dtype = hidden_states.dtype
    bsz, seq_len, dim = hidden_states.shape
    if bsz == 0 or seq_len <= 1:
        return (
            torch.zeros((), device=device, dtype=dtype),
            torch.zeros((), device=device, dtype=dtype),
        )

    h = F.normalize(hidden_states.reshape(bsz * seq_len, dim).float(), dim=-1)
    pos = positions.reshape(bsz * seq_len, 3).float()
    batch_ids = (
        torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, seq_len).reshape(-1)
    )

    sim = h @ h.t()
    sim = sim / max(float(temperature), 1e-6)
    dist = torch.cdist(pos, pos, p=2)

    same_batch = batch_ids[:, None].eq(batch_ids[None, :])
    eye = torch.eye(bsz * seq_len, device=device, dtype=torch.bool)
    positive_mask = (dist < float(distance_threshold)) & same_batch & (~eye)
    valid_mask = same_batch & (~eye)

    if positive_mask.sum() == 0:
        return (
            torch.zeros((), device=device, dtype=dtype),
            torch.zeros((), device=device, dtype=dtype),
        )

    sim = sim.masked_fill(~valid_mask, float("-inf"))
    log_denom = torch.logsumexp(sim, dim=-1)
    pos_scores = torch.where(positive_mask, sim, torch.full_like(sim, float("-inf")))
    log_num = torch.logsumexp(pos_scores, dim=-1)
    valid_rows = positive_mask.any(dim=-1)
    loss = -(log_num[valid_rows] - log_denom[valid_rows]).mean()

    topk = min(3, sim.shape[-1] - 1)
    if topk <= 0:
        acc = torch.zeros((), device=device, dtype=dtype)
    else:
        topk_idx = torch.topk(sim.masked_fill(~valid_mask, float("-inf")), k=topk, dim=-1).indices
        hit = positive_mask.gather(1, topk_idx).any(dim=-1)
        if valid_rows.any():
            acc = hit[valid_rows].float().mean().to(dtype)
        else:
            acc = torch.zeros((), device=device, dtype=dtype)

    return loss.to(dtype), acc
