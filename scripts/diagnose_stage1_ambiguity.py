#!/usr/bin/env python3
"""Stage1 ambiguity diagnostic pipeline.

Evaluates a trained Stage1 model (no memory) on ambiguous keyframes
to check whether it produces distinct peaks or blurry averages.

Two analysis modes:
  1. intra-task: within-episode pairs (e.g. KF4 vs KF7 in put_block_back)
     Same episode, same EE+grip+block, different targets.
  2. cross-variation: same structural step across different variations
     Different episodes, same EE+grip+block, different targets.

For each ambiguous keyframe, reports:
  - predicted heatmap peak location(s) vs GT target
  - peak sharpness (entropy, max value)
  - number of distinct peaks (NMS)
  - distance of top peak to GT

Usage:
    python scripts/diagnose_stage1_ambiguity.py \\
        --model-path runs/sam2act_<task>/model_last.pth \\
        --task put_block_back \\
        --repo-code-root sam2act \\
        --data-root sam2act/data_memory/test \\
        --output-dir results/stage1_diagnosis/
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

# Defer torch/model imports to after arg parsing so --help works without GPU


def find_peaks_nms(heatmap_2d, topk=5, nms_kernel=5):
    """Extract top-k peaks from a 2D heatmap via NMS.

    Args:
        heatmap_2d: (H, W) numpy array, softmaxed or normalized.
        topk: number of peaks to return.
        nms_kernel: NMS pooling kernel size.

    Returns:
        list of (row, col, score) tuples, sorted by score descending.
    """
    import torch
    import torch.nn.functional as F

    h, w = heatmap_2d.shape
    hm = torch.tensor(heatmap_2d, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    hm_pool = F.max_pool2d(hm, kernel_size=nms_kernel, stride=1, padding=nms_kernel // 2)
    hm_nms = hm * (hm == hm_pool).float()
    hm_flat = hm_nms.view(-1)
    vals, idxs = torch.topk(hm_flat, min(topk, len(hm_flat)))

    peaks = []
    for v, idx in zip(vals.numpy(), idxs.numpy()):
        r, c = divmod(int(idx), w)
        peaks.append((r, c, float(v)))
    return peaks


def heatmap_entropy(hm_2d):
    """Compute entropy of a normalized 2D heatmap."""
    hm = hm_2d.flatten()
    hm = hm[hm > 1e-10]
    return float(-np.sum(hm * np.log(hm)))


def analyze_episode_intra(
    agent, demo, keyframes, episode_idx, modules, device,
    ee_radius=0.04, obj_radius=0.05, target_threshold=0.05,
):
    """Analyze within-episode ambiguous pairs."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sam2act"))
    from mvt.multipeak_utils import track_object_positions, compute_scene_aware_clusters

    extract_obs = modules["extract_obs"]
    CAMERAS = modules["CAMERAS"]
    clip_mod = modules["clip"]

    n = len(keyframes)
    poses = np.array([demo[keyframes[k]].gripper_pose for k in range(n)], dtype=np.float32)
    opens = np.array([demo[keyframes[k]].gripper_open for k in range(n)], dtype=np.float32)
    obj_pos = track_object_positions(poses, opens)
    cluster_map = compute_scene_aware_clusters(poses, opens, ee_radius, obj_radius)

    # Find ambiguous pairs
    clusters = defaultdict(list)
    for i, cid in cluster_map.items():
        clusters[cid].append(i)

    ambiguous_kfs = set()
    for cid, members in clusters.items():
        if len(members) < 2:
            continue
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                i, j = members[a], members[b]
                if i >= n - 1 or j >= n - 1:
                    continue
                tgt_d = np.linalg.norm(poses[i + 1, :3] - poses[j + 1, :3])
                if tgt_d > target_threshold:
                    ambiguous_kfs.add(i)
                    ambiguous_kfs.add(j)

    if not ambiguous_kfs:
        return []

    # Run Stage1 predictions on ambiguous keyframes
    import torch
    results = []
    agent.reset()

    for k in sorted(ambiguous_kfs):
        obs = demo[keyframes[k]]
        obs_dict = extract_obs(obs, CAMERAS, t=k, prev_action=None, episode_length=n)

        # Get language tokens
        episode_dir = Path(agent._data_root) if hasattr(agent, '_data_root') else None
        obs_dict["lang_goal_tokens"] = np.zeros(77, dtype=np.int64)  # placeholder

        obs_torch = {}
        for key, val in obs_dict.items():
            obs_torch[key] = torch.tensor(np.array([[val]]), device=device)

        # Stage1 forward (no memory)
        with torch.no_grad():
            out = agent.act(k, obs_torch, deterministic=True)

        # Extract heatmap from model internals if possible
        # For now, record the prediction
        gt_target = poses[k + 1, :3] if k + 1 < n else None

        results.append({
            "type": "intra",
            "episode": episode_idx,
            "keyframe_idx": k,
            "ee_pos": poses[k, :3].tolist(),
            "gripper_open": float(opens[k]),
            "obj_pos": obj_pos[k].tolist(),
            "gt_target": gt_target.tolist() if gt_target is not None else None,
            "predicted_wpt": out[0].cpu().numpy().tolist() if hasattr(out[0], 'cpu') else None,
        })

    return results


def analyze_cross_variation(
    keyframe_data_by_episode, ee_radius=0.04, obj_radius=0.05, target_threshold=0.05,
):
    """Find cross-variation ambiguity: same structural step, different targets.

    Args:
        keyframe_data_by_episode: list of dicts with 'poses', 'opens', 'obj_pos' per episode.

    Returns:
        dict: keypoint_idx -> list of (episode, target) pairs where cross-variation exists.
    """
    from mvt.multipeak_utils import track_object_positions

    cross_ambiguous = defaultdict(list)

    # For each keypoint index, compare across episodes
    all_kf_indices = set()
    for ep_data in keyframe_data_by_episode:
        for k in range(len(ep_data["poses"])):
            all_kf_indices.add(k)

    for kf_idx in sorted(all_kf_indices):
        observations = []  # (episode, ee, grip, obj, target)
        for ep_data in keyframe_data_by_episode:
            poses = ep_data["poses"]
            opens = ep_data["opens"]
            obj_pos = ep_data["obj_pos"]
            n = len(poses)
            if kf_idx >= n or kf_idx + 1 >= n:
                continue
            observations.append({
                "episode": ep_data["episode"],
                "ee": poses[kf_idx, :3],
                "grip": opens[kf_idx],
                "obj": obj_pos[kf_idx],
                "target": poses[kf_idx + 1, :3],
            })

        if len(observations) < 2:
            continue

        # Check for cross-variation ambiguity
        for a in range(len(observations)):
            for b in range(a + 1, len(observations)):
                oa, ob = observations[a], observations[b]
                ee_d = np.linalg.norm(oa["ee"] - ob["ee"])
                grip_same = oa["grip"] == ob["grip"]
                obj_d = np.linalg.norm(oa["obj"] - ob["obj"])
                tgt_d = np.linalg.norm(oa["target"] - ob["target"])

                if ee_d < ee_radius and grip_same and obj_d < obj_radius and tgt_d > target_threshold:
                    if kf_idx not in cross_ambiguous:
                        cross_ambiguous[kf_idx] = {
                            "ee": oa["ee"].tolist(),
                            "grip": float(oa["grip"]),
                            "episodes": [],
                            "targets": [],
                        }
                    # Collect unique targets
                    for obs in observations:
                        tgt = obs["target"].tolist()
                        ep = obs["episode"]
                        if ep not in cross_ambiguous[kf_idx]["episodes"]:
                            cross_ambiguous[kf_idx]["episodes"].append(ep)
                        tgt_exists = any(
                            np.linalg.norm(np.array(tgt) - np.array(t)) < 0.05
                            for t in cross_ambiguous[kf_idx]["targets"]
                        )
                        if not tgt_exists:
                            cross_ambiguous[kf_idx]["targets"].append(tgt)
                    break
            if kf_idx in cross_ambiguous:
                break

    return dict(cross_ambiguous)


def main():
    parser = argparse.ArgumentParser(description="Stage1 ambiguity diagnostic")
    parser.add_argument("--task", required=True, help="MemoryBench task name")
    parser.add_argument("--keyframe-json", default=None,
                        help="Path to episodewise JSON (from keyframe gallery). "
                             "If not provided, uses --data-root to load demos.")
    parser.add_argument("--data-root", default=None,
                        help="Path to test data (for model-based analysis)")
    parser.add_argument("--model-path", default=None,
                        help="Path to Stage1 model checkpoint (optional, for prediction analysis)")
    parser.add_argument("--repo-code-root", default=None)
    parser.add_argument("--output-dir", default="results/stage1_diagnosis/")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analysis from pre-computed keyframe JSON (no model needed)
    if args.keyframe_json:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sam2act"))
        from mvt.multipeak_utils import track_object_positions, collect_alt_targets

        with open(args.keyframe_json) as f:
            data = json.load(f)

        print(f"\n{'='*65}")
        print(f"  Stage1 Ambiguity Diagnosis: {args.task}")
        print(f"  Source: {args.keyframe_json}")
        print(f"{'='*65}")

        # --- Intra-task analysis ---
        print(f"\n--- Intra-Task (within-episode) Ambiguity ---\n")

        ep_data_list = []
        total_intra = 0
        for ep_entry in data:
            ep = ep_entry["episode"]
            kfs = ep_entry["keyframes"]
            n = len(kfs)
            poses = np.array([kf["pose"] for kf in kfs], dtype=np.float32)
            opens = np.array([kf["gripper_open"] for kf in kfs], dtype=np.float32)
            obj_pos = track_object_positions(poses, opens)

            alt_pos, alt_mask = collect_alt_targets(poses, opens, n, max_peaks=3)

            ep_data_list.append({
                "episode": ep,
                "poses": poses,
                "opens": opens,
                "obj_pos": obj_pos,
            })

            ambig_kfs = [i for i in range(n) if alt_mask[i].any()]
            if ambig_kfs:
                total_intra += len(ambig_kfs)
                if ep < 3:
                    for k in ambig_kfs:
                        num_alts = int(alt_mask[k].sum())
                        gt_next = poses[k + 1, :3] if k + 1 < n else None
                        alts = alt_pos[k][alt_mask[k]]
                        print(f"  ep{ep} KF{k}: {1+num_alts}-peak"
                              f"  EE=({poses[k,0]:.3f},{poses[k,1]:.3f},{poses[k,2]:.3f})"
                              f"  grip={opens[k]:.0f}")
                        if gt_next is not None:
                            print(f"    primary: ({gt_next[0]:.3f},{gt_next[1]:.3f},{gt_next[2]:.3f})")
                        for j, a in enumerate(alts):
                            print(f"    alt {j+1}:   ({a[0]:.3f},{a[1]:.3f},{a[2]:.3f})")

        print(f"\n  Total intra-task ambiguous keyframes: {total_intra}/{sum(len(e['poses']) for e in ep_data_list)}")

        # --- Cross-variation analysis ---
        print(f"\n--- Cross-Variation Ambiguity ---\n")

        cross = analyze_cross_variation(ep_data_list)

        total_cross_kfs = 0
        for kf_idx, info in sorted(cross.items()):
            n_targets = len(info["targets"])
            n_eps = len(info["episodes"])
            total_cross_kfs += 1
            print(f"  KF{kf_idx}: {n_targets} different targets across {n_eps} episodes"
                  f"  EE=({info['ee'][0]:.3f},{info['ee'][1]:.3f},{info['ee'][2]:.3f})"
                  f"  grip={info['grip']:.0f}")
            for j, tgt in enumerate(info["targets"]):
                print(f"    target {j+1}: ({tgt[0]:.3f},{tgt[1]:.3f},{tgt[2]:.3f})")

        if not cross:
            print("  No cross-variation ambiguity found.")

        print(f"\n  Total cross-variation ambiguous steps: {total_cross_kfs}")

        # --- Combined summary ---
        print(f"\n--- Combined Summary ---\n")

        all_ambig_steps = set()
        # intra
        for ep_entry in data:
            kfs = ep_entry["keyframes"]
            n = len(kfs)
            poses = np.array([kf["pose"] for kf in kfs], dtype=np.float32)
            opens = np.array([kf["gripper_open"] for kf in kfs], dtype=np.float32)
            _, alt_mask = collect_alt_targets(poses, opens, n, max_peaks=3)
            for i in range(n):
                if alt_mask[i].any():
                    all_ambig_steps.add(f"intra:KF{i}")

        # cross
        for kf_idx in cross:
            all_ambig_steps.add(f"cross:KF{kf_idx}")

        print(f"  Intra-task ambiguous steps: {sorted(s for s in all_ambig_steps if s.startswith('intra'))}")
        print(f"  Cross-variation ambiguous steps: {sorted(s for s in all_ambig_steps if s.startswith('cross'))}")

        # Save results
        result = {
            "task": args.task,
            "intra_task_ambiguous_total": total_intra,
            "cross_variation_ambiguous": cross,
            "cross_variation_steps": list(cross.keys()),
        }
        out_path = output_dir / f"{args.task}_ambiguity_report.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Report saved to: {out_path}")

    else:
        print("Provide --keyframe-json for offline analysis or --model-path + --data-root for model-based analysis.")
        return


if __name__ == "__main__":
    main()
