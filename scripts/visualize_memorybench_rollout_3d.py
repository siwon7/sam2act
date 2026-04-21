#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
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


def fuse_point_cloud(obs: dict[str, np.ndarray], sample_points: int, cameras: list[str]) -> np.ndarray:
    pcs = []
    for cam in cameras:
        key = f"{cam}_point_cloud"
        if key not in obs:
            continue
        pc = np.asarray(obs[key], dtype=np.float32)
        pc = np.moveaxis(pc, 0, -1).reshape(-1, 3)
        finite = np.isfinite(pc).all(axis=1)
        pcs.append(pc[finite])
    if not pcs:
        return np.zeros((0, 3), dtype=np.float32)
    fused = np.concatenate(pcs, axis=0)
    if len(fused) > sample_points:
        idx = np.linspace(0, len(fused) - 1, sample_points, dtype=np.int64)
        fused = fused[idx]
    return fused


def get_gt_action(env, episode: int, step: int) -> np.ndarray | None:
    try:
        actions = env.get_ground_truth_action(episode)
    except Exception:
        return None
    if not actions:
        return None
    idx = min(step, len(actions) - 1)
    return np.asarray(actions[idx], dtype=np.float32)


def import_repo_modules(repo_code_root: str):
    if repo_code_root not in sys.path:
        sys.path.insert(0, repo_code_root)

    from rlbench.backend.utils import task_file_to_task_class
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.gripper_action_modes import Discrete
    from yarr.agents.agent import ActResult

    from sam2act.eval import load_agent
    from sam2act.libs.peract.helpers import utils
    from sam2act.utils.custom_rlbench_env import CustomMultiTaskRLBenchEnv2
    from sam2act.utils.peract_utils import CAMERAS, IMAGE_SIZE
    from sam2act.utils.rlbench_planning import EndEffectorPoseViaPlanning2

    return {
        "task_file_to_task_class": task_file_to_task_class,
        "MoveArmThenGripper": MoveArmThenGripper,
        "Discrete": Discrete,
        "ActResult": ActResult,
        "load_agent": load_agent,
        "utils": utils,
        "CustomMultiTaskRLBenchEnv2": CustomMultiTaskRLBenchEnv2,
        "CAMERAS": CAMERAS,
        "IMAGE_SIZE": IMAGE_SIZE,
        "EndEffectorPoseViaPlanning2": EndEffectorPoseViaPlanning2,
    }


def build_figure(frames_payload: list[dict], task: str, episode: int) -> go.Figure:
    frames = []
    slider_steps = []

    for i, payload in enumerate(frames_payload):
        pc = payload["pc"]
        pred = payload["pred_wpt"]
        gt = payload["gt_wpt"]

        traces = [
            go.Scatter3d(
                x=pc[:, 0] if len(pc) else [],
                y=pc[:, 1] if len(pc) else [],
                z=pc[:, 2] if len(pc) else [],
                mode="markers",
                marker=dict(size=1.5, color="rgba(90,90,90,0.35)"),
                name="fused_pc",
            ),
            go.Scatter3d(
                x=[pred[0]],
                y=[pred[1]],
                z=[pred[2]],
                mode="markers",
                marker=dict(size=7, color="#e74c3c"),
                name="pred_wpt",
            ),
        ]

        if gt is not None:
            traces.append(
                go.Scatter3d(
                    x=[gt[0]],
                    y=[gt[1]],
                    z=[gt[2]],
                    mode="markers",
                    marker=dict(size=7, color="#2ecc71"),
                    name="gt_wpt",
                )
            )

        frames.append(go.Frame(data=traces, name=str(i)))
        slider_steps.append(
            {
                "args": [[str(i)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                "label": str(i),
                "method": "animate",
            }
        )

    fig = go.Figure(frames=frames)
    if frames:
        fig.add_traces(frames[0].data)
    fig.update_layout(
        title=f"MemoryBench 3D Rollout | task={task} episode={episode}",
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode="data",
        ),
        showlegend=True,
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {"frame": {"duration": 600, "redraw": True}, "fromcurrent": True},
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[{"steps": slider_steps, "currentvalue": {"prefix": "step="}}],
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--repo-code-root", default=None)
    parser.add_argument("--data-root", default="./data_memory/test")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--episode-length", type=int, default=25)
    parser.add_argument("--sample-points", type=int, default=4000)
    parser.add_argument("--output-html", required=True)
    parser.add_argument("--deterministic", action="store_true", default=True)
    args = parser.parse_args()

    repo_code_root = args.repo_code_root
    if repo_code_root is None:
        repo_code_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sam2act"))
    modules = import_repo_modules(repo_code_root)

    task_file_to_task_class = modules["task_file_to_task_class"]
    MoveArmThenGripper = modules["MoveArmThenGripper"]
    Discrete = modules["Discrete"]
    ActResult = modules["ActResult"]
    load_agent = modules["load_agent"]
    utils = modules["utils"]
    CustomMultiTaskRLBenchEnv2 = modules["CustomMultiTaskRLBenchEnv2"]
    CAMERAS = modules["CAMERAS"]
    IMAGE_SIZE = modules["IMAGE_SIZE"]
    EndEffectorPoseViaPlanning2 = modules["EndEffectorPoseViaPlanning2"]

    device = f"cuda:{args.device}"
    agent = load_agent(model_path=args.model_path, eval_log_dir="/tmp/sam2act_mb3d", device=args.device)
    agent.load_clip()

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

    frames_payload = []
    trace_json = []
    try:
        obs = env.reset_to_demo(args.episode)
        agent.reset()
        obs_history = {k: [np.array(v)] for k, v in obs.items()}
        terminal = False
        step = 0
        while not terminal and step < args.episode_length:
            act_result = agent.act(-1, to_torch_obs(obs_history, device), deterministic=args.deterministic)
            pred = np.asarray(act_result.action, dtype=np.float32)
            gt = get_gt_action(env, args.episode, step)
            pc = fuse_point_cloud(obs, args.sample_points, CAMERAS)
            frames_payload.append(
                {
                    "step": step,
                    "pc": pc,
                    "pred_wpt": pred[:3].tolist(),
                    "gt_wpt": gt[:3].tolist() if gt is not None else None,
                }
            )
            trace_json.append(
                {
                    "step": step,
                    "pred_action": pred.tolist(),
                    "gt_action": gt.tolist() if gt is not None else None,
                    "lang_goal": str(np.asarray(obs["lang_goal"]).reshape(-1)[0]) if "lang_goal" in obs else "",
                }
            )
            transition = env.step(ActResult(action=pred))
            terminal = bool(transition.terminal)
            obs = transition.observation
            for k in obs_history.keys():
                obs_history[k].append(np.array(obs[k]))
                obs_history[k].pop(0)
            step += 1
    finally:
        env.shutdown()

    output_html = Path(args.output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure(frames_payload, args.task, args.episode)
    fig.write_html(str(output_html), include_plotlyjs="cdn")
    trace_path = output_html.with_suffix(".json")
    trace_path.write_text(json.dumps(trace_json, ensure_ascii=True, indent=2))
    print(json.dumps({"html": str(output_html), "trace": str(trace_path), "steps": len(frames_payload)}))


if __name__ == "__main__":
    main()
