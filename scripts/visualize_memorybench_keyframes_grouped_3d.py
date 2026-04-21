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
import numpy as np
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


def cluster_keyframes(keyframes: list[dict], tol: float) -> list[dict]:
    clusters: list[dict] = []
    for kf in keyframes:
        pos = np.asarray(kf["pose"][:3], dtype=np.float32)
        matched = None
        for cluster in clusters:
            center = np.asarray(cluster["center"], dtype=np.float32)
            if np.linalg.norm(pos - center) <= tol:
                matched = cluster
                break
        if matched is None:
            clusters.append(
                {
                    "center": pos.tolist(),
                    "orders": [kf["order"]],
                    "keyframes": [kf["keyframe"]],
                }
            )
        else:
            matched["orders"].append(kf["order"])
            matched["keyframes"].append(kf["keyframe"])
            pts = np.asarray([matched["center"], pos.tolist()], dtype=np.float32)
            matched["center"] = pts.mean(axis=0).tolist()
    return clusters


def format_orders(orders: list[int]) -> str:
    return "(" + ",".join(str(x) for x in orders) + ")"


def draw_episode(ax, payload: dict, tol: float) -> None:
    keyframes = payload["keyframes"]
    xs = [kf["pose"][0] for kf in keyframes]
    ys = [kf["pose"][1] for kf in keyframes]
    zs = [kf["pose"][2] for kf in keyframes]
    n = max(len(keyframes), 1)
    colors = cm.turbo([i / max(n - 1, 1) for i in range(n)])

    ax.plot(xs, ys, zs, color="#888888", linewidth=1.4, alpha=0.9)
    for (x, y, z), color in zip(zip(xs, ys, zs), colors):
        ax.scatter([x], [y], [z], s=28, color=color, edgecolors="black", linewidths=0.3, alpha=0.9)

    clusters = cluster_keyframes(keyframes, tol)
    for idx, cluster in enumerate(clusters):
        cx, cy, cz = cluster["center"]
        label = format_orders(cluster["orders"])
        ax.text(cx, cy, cz + 0.01 + (idx % 2) * 0.005, label, fontsize=7, ha="center", va="bottom")

    if keyframes:
        sx, sy, sz = keyframes[0]["pose"][:3]
        ex, ey, ez = keyframes[-1]["pose"][:3]
        ax.scatter([sx], [sy], [sz], s=100, facecolors="none", edgecolors="#2ecc71", linewidths=2.0)
        ax.scatter([ex], [ey], [ez], s=100, facecolors="none", edgecolors="#e74c3c", linewidths=2.0)

    ax.set_title(f"ep{payload['episode']} | kf={payload['num_keyframes']}", fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def set_axes_equal(ax) -> None:
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()], dtype=float)
    spans = limits[:, 1] - limits[:, 0]
    centers = np.mean(limits, axis=1)
    radius = max(spans) / 2
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def build_figure(episodes_payload: list[dict], task: str, tol: float):
    cols = 5
    rows = math.ceil(len(episodes_payload) / cols)
    fig = plt.figure(figsize=(18, max(3.6 * rows, 7)))
    for idx, payload in enumerate(episodes_payload, start=1):
        ax = fig.add_subplot(rows, cols, idx, projection="3d")
        draw_episode(ax, payload, tol)
        set_axes_equal(ax)
        ax.view_init(elev=24, azim=-58)
    fig.suptitle(
        f"MemoryBench grouped keyframes | {task} | green=start red=end | labels=(order group)",
        fontsize=14,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="put_block_back")
    parser.add_argument("--split", default="test")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--cluster-tol", type=float, default=0.015)
    parser.add_argument(
        "--data-root",
        default="/home/cv25/siwon/sam2act/sam2act/data_memory",
    )
    parser.add_argument(
        "--output-png",
        default="/home/cv25/siwon/sam2act/logs/memorybench_keyframes_gallery/put_block_back_keyframes_10eps_grouped3d.png",
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
    fig = build_figure(payload, args.task, args.cluster_tol)
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    json_path = output_png.with_suffix(".json")
    json_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2))
    print(json.dumps({"png": str(output_png), "json": str(json_path), "episodes": len(payload)}))


if __name__ == "__main__":
    main()
