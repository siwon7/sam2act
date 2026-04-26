#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    for keyframe in keyframes:
        obs = demo[keyframe]
        pose = obs.gripper_pose.astype("float32").tolist()
        positions.append(
            {
                "keyframe": int(keyframe),
                "pose": pose,
                "gripper_open": float(obs.gripper_open),
                "ignore_collisions": bool(obs.ignore_collisions),
            }
        )
    return {
        "episode": natural_episode_key(episode_dir),
        "num_frames": len(demo),
        "num_keyframes": len(keyframes),
        "keyframes": positions,
    }


def build_figure(episodes_payload: list[dict], task: str) -> go.Figure:
    cols = 5
    rows = math.ceil(len(episodes_payload) / cols)
    specs = [[{"type": "scene"} for _ in range(cols)] for _ in range(rows)]
    titles = [
        f"ep{item['episode']} | keyframes={item['num_keyframes']}"
        for item in episodes_payload
    ]
    titles.extend([""] * (rows * cols - len(titles)))
    fig = make_subplots(rows=rows, cols=cols, specs=specs, subplot_titles=titles)

    colorscale = "Turbo"
    for idx, item in enumerate(episodes_payload):
        row = idx // cols + 1
        col = idx % cols + 1
        xs = [kf["pose"][0] for kf in item["keyframes"]]
        ys = [kf["pose"][1] for kf in item["keyframes"]]
        zs = [kf["pose"][2] for kf in item["keyframes"]]
        order = list(range(len(item["keyframes"])))
        labels = [f"kf={kf['keyframe']}" for kf in item["keyframes"]]

        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines+markers+text",
                text=labels,
                textposition="top center",
                line=dict(color="rgba(80,80,80,0.45)", width=4),
                marker=dict(
                    size=5,
                    color=order,
                    colorscale=colorscale,
                    colorbar=dict(title="order") if idx == 0 else None,
                ),
                name=f"ep{item['episode']}",
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        scene_name = "scene" if idx == 0 else f"scene{idx + 1}"
        fig.layout[scene_name].update(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode="data",
        )

    fig.update_layout(
        title=f"MemoryBench keyframes | task={task}",
        height=max(320 * rows, 700),
        width=1800,
        margin=dict(l=10, r=10, t=60, b=10),
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
        "--output-html",
        default="/home/cv25/siwon/sam2act/logs/memorybench_put_block_back_keyframes_10eps.html",
    )
    parser.add_argument("--output-png", default=None)
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

    output_html = Path(args.output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure(payload, args.task)
    fig.write_html(str(output_html), include_plotlyjs="cdn")
    if args.output_png:
        output_png = Path(args.output_png)
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(output_png), width=1800, height=max(320 * math.ceil(len(payload) / 5), 700), scale=2)
    json_path = output_html.with_suffix(".json")
    json_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2))
    print(
        json.dumps(
            {
                "html": str(output_html),
                "json": str(json_path),
                "png": args.output_png,
                "episodes": len(payload),
            }
        )
    )


if __name__ == "__main__":
    main()
