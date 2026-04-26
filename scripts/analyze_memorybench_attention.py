#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import torch


def to_torch_obs(obs_history: dict[str, list[np.ndarray]], device: str) -> dict[str, torch.Tensor]:
    out = {}
    for k, vals in obs_history.items():
        out[k] = torch.tensor(np.array([vals]), device=device)
    return out


def import_repo_modules(repo_code_root: str):
    if repo_code_root not in sys.path:
        sys.path.insert(0, repo_code_root)

    from rlbench.backend.utils import task_file_to_task_class
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.gripper_action_modes import Discrete

    from sam2act.eval import load_agent
    from sam2act.mvt.sam2_train.modeling.memory_attention import MemoryAttention
    from sam2act.mvt.sam2_train.modeling.sam.transformer import RoPEAttention, apply_rotary_enc
    from sam2act.utils.custom_rlbench_env import CustomMultiTaskRLBenchEnv2
    from sam2act.utils.peract_utils import CAMERAS, IMAGE_SIZE
    from sam2act.utils.rlbench_planning import EndEffectorPoseViaPlanning2
    from sam2act.libs.peract.helpers import utils

    return {
        "task_file_to_task_class": task_file_to_task_class,
        "MoveArmThenGripper": MoveArmThenGripper,
        "Discrete": Discrete,
        "load_agent": load_agent,
        "MemoryAttention": MemoryAttention,
        "RoPEAttention": RoPEAttention,
        "apply_rotary_enc": apply_rotary_enc,
        "CustomMultiTaskRLBenchEnv2": CustomMultiTaskRLBenchEnv2,
        "CAMERAS": CAMERAS,
        "IMAGE_SIZE": IMAGE_SIZE,
        "EndEffectorPoseViaPlanning2": EndEffectorPoseViaPlanning2,
        "utils": utils,
    }


class AttentionRecorder:
    def __init__(self) -> None:
        self.pending_contexts: list[dict] = []
        self.records: list[dict] = []

    def enqueue(self, contexts: list[dict]) -> None:
        self.pending_contexts.extend(contexts)

    def next_context(self) -> dict | None:
        if not self.pending_contexts:
            return None
        return self.pending_contexts.pop(0)

    def add_record(self, record: dict) -> None:
        self.records.append(record)


def compute_attention_probs(
    cross_attn_image,
    q: torch.Tensor,
    memory: torch.Tensor,
    num_k_exclude_rope: int,
    apply_rotary_enc_fn,
):
    q_proj = cross_attn_image.q_proj(q)
    k_proj = cross_attn_image.k_proj(memory)

    q_heads = cross_attn_image._separate_heads(q_proj, cross_attn_image.num_heads)
    k_heads = cross_attn_image._separate_heads(k_proj, cross_attn_image.num_heads)

    if hasattr(cross_attn_image, "rope_k_repeat"):
        w = h = int(math.sqrt(q_heads.shape[-2]))
        cross_attn_image.freqs_cis = cross_attn_image.freqs_cis.to(q_heads.device)
        if cross_attn_image.freqs_cis.shape[0] != q_heads.shape[-2]:
            cross_attn_image.freqs_cis = cross_attn_image.compute_cis(end_x=w, end_y=h).to(q_heads.device)
        num_k_rope = k_heads.size(-2) - num_k_exclude_rope
        q_heads, k_heads[:, :, :num_k_rope] = apply_rotary_enc_fn(
            q_heads,
            k_heads[:, :, :num_k_rope],
            freqs_cis=cross_attn_image.freqs_cis,
            repeat_freqs_k=cross_attn_image.rope_k_repeat,
        )

    scale = 1.0 / math.sqrt(q_heads.shape[-1])
    logits = torch.matmul(q_heads, k_heads.transpose(-2, -1)) * scale
    return torch.softmax(logits, dim=-1)


def patch_attention(
    agent,
    recorder: AttentionRecorder,
    modules: dict,
):
    MemoryAttention = modules["MemoryAttention"]
    RoPEAttention = modules["RoPEAttention"]
    apply_rotary_enc_fn = modules["apply_rotary_enc"]

    mvt = agent._network.module.mvt1 if hasattr(agent._network, "module") else agent._network.mvt1

    original_memory_forward = MemoryAttention.forward
    original_sam2_forward_with_memory = mvt.sam2_forward_with_memory

    def wrapped_sam2_forward_with_memory(net, idx, num_views, feat_sizes):
        contexts = []
        for view_idx in range(num_views):
            memory_bank_list = mvt.memory_bank_multiview[view_idx]
            if len(memory_bank_list) == 0:
                continue
            num_mem = min(len(memory_bank_list), net.num_maskmem)
            first_prev = memory_bank_list.get(mvt.curr_obs_idx - 1)
            if first_prev is None:
                continue
            chunk_tokens = int(first_prev[0].shape[0])
            contexts.append(
                {
                    "step": int(mvt.curr_obs_idx),
                    "view_idx": int(view_idx),
                    "num_mem": int(num_mem),
                    "chunk_tokens": chunk_tokens,
                    "chunk_t_pos": list(range(1, num_mem + 1)),
                    "past_steps": [int(mvt.curr_obs_idx - t_pos) for t_pos in range(1, num_mem + 1)],
                }
            )
        recorder.enqueue(contexts)
        return original_sam2_forward_with_memory(net, idx, num_views, feat_sizes)

    def wrapped_memory_forward(self, curr, memory, curr_pos=None, memory_pos=None, num_obj_ptr_tokens=0):
        context = recorder.next_context()

        if isinstance(curr, list):
            curr, curr_pos = curr[0], curr_pos[0]

        output = curr
        if self.pos_enc_at_input and curr_pos is not None:
            output = output + 0.1 * curr_pos

        if self.batch_first:
            output = output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)
            memory = memory.transpose(0, 1)
            memory_pos = memory_pos.transpose(0, 1)

        for layer_idx, layer in enumerate(self.layers):
            tgt = layer._forward_sa(output, curr_pos)

            kwds = {}
            if isinstance(layer.cross_attn_image, RoPEAttention):
                kwds = {"num_k_exclude_rope": num_obj_ptr_tokens}

            tgt2 = layer.norm2(tgt)
            q = tgt2 + curr_pos if layer.pos_enc_at_cross_attn_queries else tgt2
            k = memory + memory_pos if layer.pos_enc_at_cross_attn_keys else memory

            probs = compute_attention_probs(
                layer.cross_attn_image,
                q=q,
                memory=k,
                num_k_exclude_rope=kwds.get("num_k_exclude_rope", 0),
                apply_rotary_enc_fn=apply_rotary_enc_fn,
            )
            probs_mean = probs.mean(dim=(0, 1, 2)).detach().cpu().numpy()

            if context is not None:
                chunk_scores = []
                chunk_tokens = context["chunk_tokens"]
                for chunk_idx, t_pos in enumerate(context["chunk_t_pos"]):
                    start = chunk_idx * chunk_tokens
                    end = start + chunk_tokens
                    score = float(probs_mean[start:end].sum())
                    chunk_scores.append(
                        {
                            "t_pos": int(t_pos),
                            "past_step": int(context["past_steps"][chunk_idx]),
                            "score": score,
                        }
                    )
                recorder.add_record(
                    {
                        "step": int(context["step"]),
                        "view_idx": int(context["view_idx"]),
                        "layer_idx": int(layer_idx),
                        "num_mem": int(context["num_mem"]),
                        "chunk_scores": chunk_scores,
                    }
                )

            attn_out = layer.cross_attn_image(q=q, k=k, v=memory, **kwds)
            tgt = tgt + layer.dropout2(attn_out)

            tgt2 = layer.norm3(tgt)
            tgt2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(tgt2))))
            output = tgt + layer.dropout3(tgt2)

        normed_output = self.norm(output)
        if self.batch_first:
            normed_output = normed_output.transpose(0, 1)
        return normed_output

    MemoryAttention.forward = wrapped_memory_forward
    mvt.sam2_forward_with_memory = wrapped_sam2_forward_with_memory

    return original_memory_forward, original_sam2_forward_with_memory, mvt


def restore_attention(original_memory_forward, original_sam2_forward_with_memory, modules, mvt) -> None:
    modules["MemoryAttention"].forward = original_memory_forward
    mvt.sam2_forward_with_memory = original_sam2_forward_with_memory


def summarize_records(records: list[dict]) -> list[dict]:
    by_step: dict[int, dict[int, list[float]]] = {}
    for rec in records:
        step = rec["step"]
        for chunk in rec["chunk_scores"]:
            past_step = chunk["past_step"]
            by_step.setdefault(step, {}).setdefault(past_step, []).append(chunk["score"])

    summary = []
    for step in sorted(by_step):
        past = by_step[step]
        items = [
            {
                "past_step": int(past_step),
                "score_mean": float(np.mean(scores)),
                "score_max": float(np.max(scores)),
                "num_entries": len(scores),
            }
            for past_step, scores in sorted(past.items())
        ]
        items.sort(key=lambda x: x["score_mean"], reverse=True)
        summary.append({"step": int(step), "top_past_steps": items})
    return summary


def build_heatmap(summary: list[dict], output_html: Path, task: str, episode: int) -> None:
    if not summary:
        return

    steps = [x["step"] for x in summary]
    max_step = max(steps)
    z = np.full((max_step + 1, max_step + 1), np.nan, dtype=np.float32)
    text = [["" for _ in range(max_step + 1)] for _ in range(max_step + 1)]

    for item in summary:
        step = item["step"]
        for past in item["top_past_steps"]:
            past_step = past["past_step"]
            if past_step < 0 or past_step > max_step:
                continue
            z[step, past_step] = past["score_mean"]
            text[step][past_step] = f"step={step}<br>past={past_step}<br>score={past['score_mean']:.4f}"

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=list(range(max_step + 1)),
            y=list(range(max_step + 1)),
            colorscale="Viridis",
            text=text,
            hoverinfo="text",
        )
    )
    fig.update_layout(
        title=f"Memory attention summary | task={task} episode={episode}",
        xaxis_title="past step",
        yaxis_title="current step",
    )
    fig.write_html(output_html)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--episode-length", type=int, default=25)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--repo-code-root", default="/home/cv25/siwon/sam2act/sam2act")
    parser.add_argument("--data-root", default="./data_memory/test")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-html", required=True)
    args = parser.parse_args()

    modules = import_repo_modules(args.repo_code_root)
    load_agent = modules["load_agent"]
    task_file_to_task_class = modules["task_file_to_task_class"]
    MoveArmThenGripper = modules["MoveArmThenGripper"]
    Discrete = modules["Discrete"]
    CustomMultiTaskRLBenchEnv2 = modules["CustomMultiTaskRLBenchEnv2"]
    CAMERAS = modules["CAMERAS"]
    IMAGE_SIZE = modules["IMAGE_SIZE"]
    EndEffectorPoseViaPlanning2 = modules["EndEffectorPoseViaPlanning2"]
    utils = modules["utils"]

    recorder = AttentionRecorder()
    agent = load_agent(model_path=args.model_path, eval_log_dir="/tmp/sam2act_attn_debug", device=args.device)
    agent.load_clip()
    agent.reset()

    orig_memory_forward, orig_sam2_forward_with_memory, mvt = patch_attention(agent, recorder, modules)

    obs_config = utils.create_obs_config(
        CAMERAS, [IMAGE_SIZE, IMAGE_SIZE], method_name="", use_mask_from_replay=False
    )
    task_class = task_file_to_task_class(args.task)
    env = CustomMultiTaskRLBenchEnv2(
        task_classes=[task_class],
        observation_config=obs_config,
        action_mode=MoveArmThenGripper(EndEffectorPoseViaPlanning2(), Discrete()),
        episode_length=args.episode_length,
        dataset_root=args.data_root,
        headless=False,
        swap_task_every=1,
        include_lang_goal_in_obs=True,
    )
    env.eval = True
    env.launch()

    try:
        obs = env.reset_to_demo(args.episode)
        obs_history = {k: [np.array(v)] for k, v in obs.items()}
        terminal = False
        step = 0
        while not terminal and step < args.episode_length:
            act_result = agent.act(-1, to_torch_obs(obs_history, f"cuda:{args.device}"), deterministic=True)
            transition = env.step(act_result)
            terminal = bool(transition.terminal)
            obs = transition.observation
            for k, v in obs.items():
                obs_history.setdefault(k, []).append(np.array(v))
                obs_history[k].pop(0)
            step += 1
    finally:
        env.shutdown()
        restore_attention(orig_memory_forward, orig_sam2_forward_with_memory, modules, mvt)

    summary = summarize_records(recorder.records)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": args.task,
        "episode": args.episode,
        "model_path": args.model_path,
        "records": recorder.records,
        "summary": summary,
    }
    output_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2))
    build_heatmap(summary, Path(args.output_html), args.task, args.episode)
    print(json.dumps({"json": str(output_json), "html": str(args.output_html), "records": len(recorder.records)}))


if __name__ == "__main__":
    main()
