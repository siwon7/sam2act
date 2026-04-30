#!/usr/bin/env python3
"""Evaluate Stage1 heatmap quality on ambiguous keyframes.

Loads a trained Stage1 model (use_memory=False), runs inference on
GT keyframe observations, and checks whether the predicted heatmap
peaks align with the correct target or get confused.

Usage:
    conda run -n sam2act5090 python scripts/eval_stage1_heatmap.py \
        --model-path /path/to/runs/sam2act_memorybench_put_block_back/model_last.pth \
        --task put_block_back \
        --data-root /hdd3/siwon_data/sam2act/data_memory/test \
        --repo-code-root /home/cv11/project/siwon/sam2act_dirty/sam2act \
        --episodes 10 \
        --output-dir results/stage1_eval/
"""

import argparse
import json
import math
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def find_peaks_nms(heatmap_2d, topk=5, nms_kernel=5):
    """NMS peak extraction from 2D heatmap."""
    h, w = heatmap_2d.shape
    hm = torch.tensor(heatmap_2d, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    hm_pool = F.max_pool2d(hm, kernel_size=nms_kernel, stride=1, padding=nms_kernel // 2)
    hm_nms = hm * (hm == hm_pool).float()
    hm_flat = hm_nms.view(-1)
    vals, idxs = torch.topk(hm_flat, min(topk, len(hm_flat)))
    peaks = []
    for v, idx in zip(vals.numpy(), idxs.numpy()):
        r, c = divmod(int(idx), w)
        peaks.append({"row": r, "col": c, "score": float(v)})
    return peaks


def heatmap_entropy(hm_2d):
    hm = hm_2d.flatten()
    hm = hm[hm > 1e-10]
    return float(-np.sum(hm * np.log(hm)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--repo-code-root", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output-dir", default="results/stage1_eval/")
    args = parser.parse_args()

    # Import from the correct repo
    repo_root = os.path.abspath(args.repo_code_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    import clip
    from sam2act.libs.peract.helpers.demo_loading_utils import keypoint_discovery
    from sam2act.libs.peract.helpers.utils import extract_obs
    from sam2act.utils.peract_utils import CAMERAS

    # Also need multipeak_utils from our repo
    multipeak_root = os.path.join(os.path.dirname(__file__), "..", "sam2act")
    if multipeak_root not in sys.path:
        sys.path.insert(0, multipeak_root)
    from mvt.multipeak_utils import track_object_positions, collect_alt_targets

    # Load demo data
    from sam2act.libs.peract_colab.peract_colab.rlbench.utils import get_stored_demo

    episode_base = os.path.join(args.data_root, args.task, "all_variations", "episodes")

    # Load agent
    from sam2act.eval import load_agent
    agent = load_agent(
        model_path=args.model_path,
        eval_log_dir="/tmp/stage1_eval",
        device=args.device,
    )
    agent.load_clip()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for ep_idx in range(args.episodes):
        print(f"\n--- Episode {ep_idx} ---")

        try:
            demo = get_stored_demo(str(episode_base), ep_idx)
        except Exception as e:
            print(f"  Skip ep{ep_idx}: {e}")
            continue

        keyframes = keypoint_discovery(demo)
        n = len(keyframes)

        # Compute ambiguity info
        poses = np.array([demo[keyframes[k]].gripper_pose for k in range(n)], dtype=np.float32)
        opens = np.array([demo[keyframes[k]].gripper_open for k in range(n)], dtype=np.float32)
        obj_pos = track_object_positions(poses, opens)
        alt_pos, alt_mask = collect_alt_targets(poses, opens, n, max_peaks=3)

        # Load language
        desc_path = os.path.join(episode_base, f"episode{ep_idx}", "variation_descriptions.pkl")
        with open(desc_path, "rb") as f:
            descs = pickle.load(f)
        desc = descs[0]

        # Reset agent for new episode
        agent.reset()

        # Run through all keyframes
        for k in range(n):
            obs = demo[keyframes[k]]
            obs_dict = extract_obs(obs, CAMERAS, t=k, prev_action=None, episode_length=n)

            # Language
            lang_tokens = clip.tokenize([desc]).numpy()
            obs_dict["lang_goal_tokens"] = lang_tokens[0]

            obs_torch = {}
            for key, val in obs_dict.items():
                obs_torch[key] = torch.tensor(np.array([[val]]), device=f"cuda:{args.device}")

            with torch.no_grad():
                out = agent.act(k, obs_torch, deterministic=True)

            # Extract predicted waypoint from ActResult
            action = out.action
            if hasattr(action, 'cpu'):
                action = action.cpu().numpy()
            pred_wpt = np.array(action).flatten()[:3]
            gt_target = poses[k + 1, :3] if k + 1 < n else pred_wpt

            dist_to_gt = float(np.linalg.norm(pred_wpt - gt_target))

            is_intra_ambig = bool(alt_mask[k].any())
            ee = poses[k, :3]

            result = {
                "episode": ep_idx,
                "keyframe_idx": k,
                "ee_pos": ee.tolist(),
                "gripper_open": float(opens[k]),
                "obj_pos": obj_pos[k].tolist(),
                "gt_target": gt_target.tolist(),
                "pred_wpt": pred_wpt.tolist(),
                "dist_to_gt": dist_to_gt,
                "is_intra_ambiguous": is_intra_ambig,
            }

            if is_intra_ambig:
                alt_targets = alt_pos[k][alt_mask[k]]
                alt_dists = [float(np.linalg.norm(pred_wpt - at)) for at in alt_targets]
                result["alt_targets"] = alt_targets.tolist()
                result["dist_to_alts"] = alt_dists
                result["closest_is_gt"] = dist_to_gt <= min(alt_dists) if alt_dists else True

                marker = "OK" if dist_to_gt < 0.05 else "MISS"
                closest_alt_d = min(alt_dists) if alt_dists else 999
                print(f"  KF{k} [AMBIG] gt_d={dist_to_gt:.3f} alt_d={closest_alt_d:.3f} → {marker}"
                      f"  pred=({pred_wpt[0]:.3f},{pred_wpt[1]:.3f},{pred_wpt[2]:.3f})"
                      f"  gt=({gt_target[0]:.3f},{gt_target[1]:.3f},{gt_target[2]:.3f})")
            else:
                marker = "OK" if dist_to_gt < 0.05 else "MISS"
                print(f"  KF{k}         gt_d={dist_to_gt:.3f} → {marker}")

            all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print(f"  Summary: {args.task}")
    print(f"{'='*60}")

    ambig_results = [r for r in all_results if r["is_intra_ambiguous"]]
    non_ambig = [r for r in all_results if not r["is_intra_ambiguous"]]

    if non_ambig:
        non_ambig_dists = [r["dist_to_gt"] for r in non_ambig]
        non_ambig_ok = sum(1 for d in non_ambig_dists if d < 0.05)
        print(f"\n  Non-ambiguous keyframes: {non_ambig_ok}/{len(non_ambig)} within 5cm")
        print(f"    mean dist: {np.mean(non_ambig_dists):.4f}m")
        print(f"    max dist:  {np.max(non_ambig_dists):.4f}m")

    if ambig_results:
        ambig_gt_dists = [r["dist_to_gt"] for r in ambig_results]
        ambig_ok = sum(1 for d in ambig_gt_dists if d < 0.05)
        ambig_closest_gt = sum(1 for r in ambig_results if r.get("closest_is_gt", True))
        print(f"\n  Ambiguous keyframes: {ambig_ok}/{len(ambig_results)} within 5cm of GT")
        print(f"    mean dist to GT: {np.mean(ambig_gt_dists):.4f}m")
        print(f"    predicted closer to GT than alt: {ambig_closest_gt}/{len(ambig_results)}")

        # Check if predictions land on GT or on alt target
        chose_gt = 0
        chose_alt = 0
        chose_neither = 0
        for r in ambig_results:
            gt_d = r["dist_to_gt"]
            alt_dists = r.get("dist_to_alts", [])
            min_alt_d = min(alt_dists) if alt_dists else 999
            if gt_d < 0.05:
                chose_gt += 1
            elif min_alt_d < 0.05:
                chose_alt += 1
            else:
                chose_neither += 1
        print(f"    chose GT target: {chose_gt}")
        print(f"    chose alt target: {chose_alt}")
        print(f"    chose neither (average?): {chose_neither}")

    # Save
    out_path = output_dir / f"{args.task}_stage1_eval.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved to: {out_path}")


if __name__ == "__main__":
    main()
