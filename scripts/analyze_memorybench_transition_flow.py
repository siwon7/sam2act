#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


def to_torch_obs(obs_history: dict[str, list[np.ndarray]], device: str) -> dict[str, torch.Tensor]:
    return {k: torch.tensor(np.array([vals]), device=device) for k, vals in obs_history.items()}


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


PUT_BLOCK_BACK_LABELS = {
    0: "move_to_block",
    1: "lift_block",
    2: "move_to_center",
    3: "place_center",
    4: "rise_for_button",
    5: "move_over_button",
    6: "press_button",
    7: "rise_over_center",
    8: "grasp_block",
    9: "lift_block_again",
    10: "move_to_initial_slot",
    11: "place_at_initial_slot",
}


def nearest_keyframe(pos: np.ndarray, gt_actions: np.ndarray) -> tuple[int, float]:
    gt_xyz = gt_actions[:, :3]
    dists = np.linalg.norm(gt_xyz - pos[None, :], axis=1)
    idx = int(np.argmin(dists))
    return idx, float(dists[idx])


def extract_actual_gripper_pose(obs: dict) -> np.ndarray:
    if "gripper_pose" in obs:
        return np.asarray(obs["gripper_pose"], dtype=np.float32)
    low_dim = np.asarray(obs["low_dim_state"], dtype=np.float32).reshape(-1)
    # RLBench low_dim_state layout in this setup:
    # joint_velocities(7), joint_positions(7), joint_forces(7), gripper_open(1),
    # gripper_pose(7), ...
    if low_dim.shape[0] >= 29:
        return low_dim[22:29].astype(np.float32)
    raise KeyError("Could not recover gripper pose from observation")


def compress_labels(labels: list[int]) -> list[dict]:
    if not labels:
        return []
    runs = []
    curr = labels[0]
    count = 1
    start = 0
    for i, lab in enumerate(labels[1:], start=1):
        if lab == curr:
            count += 1
            continue
        runs.append({"label": int(curr), "count": int(count), "start_step": int(start), "end_step": int(i - 1)})
        curr = lab
        count = 1
        start = i
    runs.append({"label": int(curr), "count": int(count), "start_step": int(start), "end_step": int(len(labels) - 1)})
    return runs


def find_backtracks(labels: list[int]) -> list[dict]:
    issues = []
    best = -1
    for step, lab in enumerate(labels):
        if lab > best:
            best = lab
            continue
        if lab < best:
            issues.append({"step": int(step), "label": int(lab), "best_seen": int(best)})
    return issues


def render_markdown(records: list[dict], runs: list[dict], backtracks: list[dict], task: str) -> str:
    lines = [
        "# MemoryBench transition flow",
        "",
        f"- task: `{task}`",
        "",
        "## Runs",
        "",
        "| label | name | repeat | step range |",
        "|---|---|---:|---|",
    ]
    for run in runs:
        name = PUT_BLOCK_BACK_LABELS.get(run["label"], "")
        lines.append(
            f"| {run['label']} | `{name}` | {run['count']} | {run['start_step']}..{run['end_step']} |"
        )

    lines.extend(["", "## Step Table", "", "| step | pred_label | pred_dist | actual_label | actual_dist | note |", "|---|---:|---:|---:|---:|---|"])
    for rec in records:
        note = []
        if rec["actual_label"] != rec["pred_label"]:
            note.append("pred/actual mismatch")
        if rec["step"] > 0 and rec["actual_label"] == records[rec["step"] - 1]["actual_label"]:
            note.append("repeat")
        lines.append(
            f"| {rec['step']} | {rec['pred_label']} | {rec['pred_dist']:.4f} | {rec['actual_label']} | {rec['actual_dist']:.4f} | {'; '.join(note)} |"
        )

    lines.extend(["", "## Backtracks", ""])
    if not backtracks:
        lines.append("- none")
    else:
        for item in backtracks:
            lines.append(
                f"- step {item['step']}: returned to label {item['label']} after already reaching {item['best_seen']}"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--repo-code-root", required=True)
    parser.add_argument("--data-root", default="./data_memory/test")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--episode-length", type=int, default=25)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--deterministic", action="store_true", default=True)
    args = parser.parse_args()

    mods = import_repo_modules(args.repo_code_root)
    task_file_to_task_class = mods["task_file_to_task_class"]
    MoveArmThenGripper = mods["MoveArmThenGripper"]
    Discrete = mods["Discrete"]
    ActResult = mods["ActResult"]
    load_agent = mods["load_agent"]
    utils = mods["utils"]
    CustomMultiTaskRLBenchEnv2 = mods["CustomMultiTaskRLBenchEnv2"]
    CAMERAS = mods["CAMERAS"]
    IMAGE_SIZE = mods["IMAGE_SIZE"]
    EndEffectorPoseViaPlanning2 = mods["EndEffectorPoseViaPlanning2"]

    device = f"cuda:{args.device}"
    agent = load_agent(model_path=args.model_path, eval_log_dir="/tmp/sam2act_transition_flow", device=args.device)
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

    records: list[dict] = []
    try:
        obs = env.reset_to_demo(args.episode)
        agent.reset()
        gt_actions = np.asarray(env.get_ground_truth_action(args.episode), dtype=np.float32)
        obs_history = {k: [np.array(v)] for k, v in obs.items()}
        terminal = False
        step = 0
        while not terminal and step < args.episode_length:
            act_result = agent.act(-1, to_torch_obs(obs_history, device), deterministic=args.deterministic)
            pred = np.asarray(act_result.action, dtype=np.float32)
            transition = env.step(ActResult(action=pred))
            terminal = bool(transition.terminal)
            obs = transition.observation

            actual_pose = extract_actual_gripper_pose(obs)
            pred_label, pred_dist = nearest_keyframe(pred[:3], gt_actions)
            actual_label, actual_dist = nearest_keyframe(actual_pose[:3], gt_actions)
            records.append(
                {
                    "step": int(step),
                    "pred_action": pred.tolist(),
                    "actual_gripper_pose": actual_pose.tolist(),
                    "pred_label": int(pred_label),
                    "pred_dist": float(pred_dist),
                    "actual_label": int(actual_label),
                    "actual_dist": float(actual_dist),
                }
            )

            for k in obs_history.keys():
                obs_history[k].append(np.array(obs[k]))
                obs_history[k].pop(0)
            step += 1
    finally:
        env.shutdown()

    actual_labels = [r["actual_label"] for r in records]
    runs = compress_labels(actual_labels)
    backtracks = find_backtracks(actual_labels)

    payload = {
        "task": args.task,
        "episode": int(args.episode),
        "records": records,
        "runs": runs,
        "backtracks": backtracks,
        "gt_labels": PUT_BLOCK_BACK_LABELS if args.task == "put_block_back" else {},
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2))
    out_md.write_text(render_markdown(records, runs, backtracks, args.task))
    print(json.dumps({"json": str(out_json), "md": str(out_md), "steps": len(records)}))


if __name__ == "__main__":
    main()
