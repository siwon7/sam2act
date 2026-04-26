from typing import Optional, Sequence, Tuple

import torch


def _gaussian_similarity(dist: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return torch.zeros_like(dist)
    return torch.exp(-(dist**2) / (2 * sigma * sigma))


def build_memory_edge_bias(
    memory_entries: Sequence[dict],
    query_pos: Optional[torch.Tensor],
    query_grip: Optional[torch.Tensor],
    prev_ref_obs_idx: Optional[int],
    temporal_scale: float,
    revisit_scale: float,
    transition_scale: float,
    grip_scale: float,
    revisit_sigma: float,
    transition_sigma: float,
) -> Tuple[Optional[torch.Tensor], list[tuple[int, int]], list[int]]:
    if len(memory_entries) == 0:
        return None, [], []

    token_ranges = []
    obs_indices = []
    token_bias_chunks = []
    token_offset = 0

    query_pos = query_pos.detach() if query_pos is not None else None
    query_grip = query_grip.detach() if query_grip is not None else None
    max_rank = max(len(memory_entries) - 1, 1)

    for rank, entry in enumerate(memory_entries):
        memory = entry["memory"]
        token_count = int(memory.shape[0])
        token_ranges.append((token_offset, token_offset + token_count))
        token_offset += token_count
        obs_indices.append(int(entry["obs_idx"]))

        step_bias = memory.new_zeros((1,), dtype=memory.dtype, device=memory.device)

        if temporal_scale != 0.0:
            recency_bonus = 1.0 - (rank / max_rank)
            step_bias = step_bias + temporal_scale * recency_bonus

        if revisit_scale != 0.0 and query_pos is not None and entry.get("query_pos") is not None:
            dist = torch.linalg.vector_norm(query_pos - entry["query_pos"].to(query_pos.device), dim=-1)
            step_bias = step_bias + revisit_scale * _gaussian_similarity(dist, revisit_sigma).to(step_bias.dtype)

        if transition_scale != 0.0 and prev_ref_obs_idx is not None:
            obs_delta = memory.new_tensor([abs(int(entry["obs_idx"]) - int(prev_ref_obs_idx))], dtype=memory.dtype)
            step_bias = step_bias + transition_scale * torch.exp(-obs_delta / max(transition_sigma, 1e-6))

        if grip_scale != 0.0 and query_grip is not None and entry.get("grip") is not None:
            grip_delta = torch.abs(query_grip - entry["grip"].to(query_grip.device)).mean(dim=-1)
            grip_bonus = torch.clamp(1.0 - grip_delta, min=0.0)
            step_bias = step_bias + grip_scale * grip_bonus.to(step_bias.dtype)

        token_bias_chunks.append(step_bias.view(1, 1).expand(1, token_count))

    token_bias = torch.cat(token_bias_chunks, dim=1)
    return token_bias, token_ranges, obs_indices


def summarize_attn_over_memory_steps(
    attn_weights: Optional[torch.Tensor],
    token_ranges: Sequence[tuple[int, int]],
) -> Optional[torch.Tensor]:
    if attn_weights is None or len(token_ranges) == 0:
        return None

    step_scores = []
    for start, end in token_ranges:
        step_scores.append(attn_weights[..., start:end].sum(dim=-1))
    stacked = torch.stack(step_scores, dim=-1)
    return stacked.mean(dim=(1, 2))
