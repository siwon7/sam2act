#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import cm

REPO_CODE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sam2act"))
if REPO_CODE_ROOT not in sys.path:
    sys.path.insert(0, REPO_CODE_ROOT)

from sam2act.libs.peract.helpers.demo_loading_utils import keypoint_discovery


def natural_episode_key(path: Path) -> int:
    return int(path.name.replace("episode", ""))


def load_demo(episode_dir: Path):
    with (episode_dir / "low_dim_obs.pkl").open("rb") as f:
        return pickle.load(f)


def extract_episode_keyframes(episode_dir: Path) -> dict:
    demo = load_demo(episode_dir)
    keyframes = keypoint_discovery(demo)
    positions = []
    for order, keyframe in enumerate(keyframes):
        obs = demo[keyframe]
        pose = obs.gripper_pose.astype("float32").tolist()
        positions.append(
            {
                "order": order,
                "keyframe": int(keyframe),
                "pose": pose,
                "gripper_open": float(obs.gripper_open),
            }
        )
    return {
        "episode": natural_episode_key(episode_dir),
        "num_frames": len(demo),
        "num_keyframes": len(keyframes),
        "keyframes": positions,
    }


def draw_episode(ax_xy, ax_xz, payload: dict) -> None:
    kfs = payload["keyframes"]
    xs = [kf["pose"][0] for kf in kfs]
    ys = [kf["pose"][1] for kf in kfs]
    zs = [kf["pose"][2] for kf in kfs]
    n = max(len(kfs), 1)
    colors = cm.turbo([i / max(n - 1, 1) for i in range(n)])

    ax_xy.plot(xs, ys, color="#888888", linewidth=1.5, alpha=0.8)
    ax_xz.plot(xs, zs, color="#888888", linewidth=1.5, alpha=0.8)

    for i, (x, y, z, color) in enumerate(zip(xs, ys, zs, colors)):
        size = 42 if i not in (0, len(xs) - 1) else 70
        ax_xy.scatter([x], [y], s=size, color=color, edgecolors="black", linewidths=0.4, zorder=3)
        ax_xz.scatter([x], [z], s=size, color=color, edgecolors="black", linewidths=0.4, zorder=3)
        ax_xy.text(x, y, str(i), fontsize=7, ha="center", va="center", color="black")
        ax_xz.text(x, z, str(i), fontsize=7, ha="center", va="center", color="black")

    if xs:
        ax_xy.scatter([xs[0]], [ys[0]], s=120, facecolors="none", edgecolors="#2ecc71", linewidths=2, zorder=4)
        ax_xy.scatter([xs[-1]], [ys[-1]], s=120, facecolors="none", edgecolors="#e74c3c", linewidths=2, zorder=4)
        ax_xz.scatter([xs[0]], [zs[0]], s=120, facecolors="none", edgecolors="#2ecc71", linewidths=2, zorder=4)
        ax_xz.scatter([xs[-1]], [zs[-1]], s=120, facecolors="none", edgecolors="#e74c3c", linewidths=2, zorder=4)

    ax_xy.set_title(f"ep{payload['episode']} xy", fontsize=10)
    ax_xz.set_title(f"ep{payload['episode']} xz", fontsize=10)
    ax_xy.set_xlabel("x")
    ax_xy.set_ylabel("y")
    ax_xz.set_xlabel("x")
    ax_xz.set_ylabel("z")
    ax_xy.grid(alpha=0.25)
    ax_xz.grid(alpha=0.25)
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xz.set_aspect("equal", adjustable="box")


def build_figure(episodes_payload: list[dict], task: str):
    rows = len(episodes_payload)
    fig, axes = plt.subplots(rows, 2, figsize=(10, max(3 * rows, 8)), constrained_layout=True)
    if rows == 1:
        axes = [axes]

    for row_axes, payload in zip(axes, episodes_payload):
        draw_episode(row_axes[0], row_axes[1], payload)

    fig.suptitle(
        f"MemoryBench keyframes | {task} | green=start red=end",
        fontsize=14,
        y=1.01,
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="put_block_back")
    parser.add_argument("--split", default="test")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--data-root",
        default="/home/cv25/siwon/sam2act/sam2act/data_memory",
    )
    parser.add_argument(
        "--output-png",
        default="/home/cv25/siwon/sam2act/logs/memorybench_keyframes_gallery/put_block_back_keyframes_10eps_simple.png",
    )
    args = parser.parse_args()

    episodes_root = (
        Path(args.data_root)
        / args.split
        / args.task
        / "all_variations"
        / "episodes"
    )
    episode_dirs = sorted(episodes_root.glob("episode*"), key=natural_episode_key)[: args.episodes]
    payload = [extract_episode_keyframes(ep_dir) for ep_dir in episode_dirs]

    output_png = Path(args.output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure(payload, args.task)
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    json_path = output_png.with_suffix(".json")
    json_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2))
    print(json.dumps({"png": str(output_png), "json": str(json_path), "episodes": len(payload)}))


if __name__ == "__main__":
    main()
