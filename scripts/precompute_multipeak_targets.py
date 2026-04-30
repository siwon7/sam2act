#!/usr/bin/env python3
"""Precompute multi-peak alt targets for MemoryBench tasks.

Scans all training episodes, identifies intra-episode and cross-variation
ambiguity, and saves per-episode alt_targets as a single JSON file.

Three modes:
  - intra:  within-episode only (same EE+grip+block, different next target)
  - cross:  cross-variation only (same structural step across episodes,
            same EE+grip+block, different next target)
  - both:   union of intra and cross

Output: {task}_multipeak_{mode}.json containing per-episode alt targets
that dataset.py loads at replay fill time.

Usage:
    conda run -n sam2act5090 python scripts/precompute_multipeak_targets.py \
        --data-root /hdd3/siwon_data/sam2act/data_memory/train \
        --task put_block_back \
        --mode both \
        --output-dir sam2act/data_memory/multipeak_targets/
"""

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def get_repo_root():
    return os.path.join(os.path.dirname(__file__), "..")


def load_episode_keyframes(data_root, task, episode_idx):
    """Load keyframe poses from a demo episode."""
    repo = get_repo_root()
    sys.path.insert(0, os.path.join(repo, "sam2act"))

    episode_base = os.path.join(data_root, task, "all_variations", "episodes")
    demo_path = os.path.join(episode_base, f"episode{episode_idx}", "low_dim_obs.pkl")

    if not os.path.exists(demo_path):
        return None

    with open(demo_path, "rb") as f:
        demo = pickle.load(f)

    # Keypoint discovery (simplified — same heuristic as SAM2Act)
    from mvt.multipeak_utils import track_object_positions

    # Import keypoint_discovery
    sys.path.insert(0, os.path.join(repo, "sam2act", "libs", "peract", "helpers"))
    try:
        from demo_loading_utils import keypoint_discovery
    except ImportError:
        # Fallback: try from the dirty repo
        for candidate in [
            "/home/cv11/project/siwon/sam2act_dirty/sam2act",
            "/home/cv11/project/siwon/sam2act_tgm/SAM2Act/sam2act",
        ]:
            p = os.path.join(candidate, "libs", "peract", "helpers")
            if os.path.exists(p):
                sys.path.insert(0, p)
                from demo_loading_utils import keypoint_discovery
                break

    keyframes = keypoint_discovery(demo)
    n = len(keyframes)

    poses = np.array([demo[keyframes[k]].gripper_pose for k in range(n)], dtype=np.float32)
    opens = np.array([demo[keyframes[k]].gripper_open for k in range(n)], dtype=np.float32)
    obj_pos = track_object_positions(poses, opens)

    return {
        "episode": episode_idx,
        "num_keyframes": n,
        "poses": poses,
        "opens": opens,
        "obj_pos": obj_pos,
    }


def compute_intra_targets(ep_data, max_peaks=5, dedup_radius=0.05,
                          ee_radius=0.04, obj_radius=0.05, target_threshold=0.05):
    """Compute within-episode alt targets."""
    from mvt.multipeak_utils import collect_alt_targets
    poses = ep_data["poses"]
    opens = ep_data["opens"]
    n = ep_data["num_keyframes"]
    return collect_alt_targets(
        poses, opens, n, max_peaks=max_peaks,
        dedup_radius=dedup_radius, ee_radius=ee_radius,
        obj_radius=obj_radius, target_diverge_threshold=target_threshold,
    )


def compute_cross_targets(all_ep_data, max_peaks=5, dedup_radius=0.05,
                          ee_radius=0.04, obj_radius=0.05, target_threshold=0.05,
                          exclude_random_spawn_z=0.914, exclude_random_spawn_tol=0.01):
    """Compute cross-variation alt targets.

    For each keypoint_idx, compare across all episodes.
    If same EE+grip+obj but different target → add alts from other episodes.

    Targets near exclude_random_spawn_z (e.g. button z=0.914) are excluded
    from cross-variation because they are visually identifiable objects
    that the model can locate from RGB.

    Returns: dict[episode_idx] -> (alt_positions, alt_mask) arrays
    """
    # Collect per-step data across episodes
    step_data = defaultdict(list)  # kf_idx -> list of {episode, ee, grip, obj, target}
    for ep in all_ep_data:
        poses = ep["poses"]
        opens = ep["opens"]
        obj_pos = ep["obj_pos"]
        n = ep["num_keyframes"]
        for k in range(n):
            if k + 1 >= n:
                continue
            step_data[k].append({
                "episode": ep["episode"],
                "ee": poses[k, :3],
                "grip": float(opens[k]),
                "obj": obj_pos[k],
                "target": poses[k + 1, :3],
            })

    # For each step, find cross-variation ambiguity
    cross_alts = {}  # (episode, kf_idx) -> list of alt target xyz

    for kf_idx, observations in step_data.items():
        if len(observations) < 2:
            continue

        # Group observations that look the same (same EE+grip+obj)
        groups = []
        assigned = [False] * len(observations)
        for i in range(len(observations)):
            if assigned[i]:
                continue
            group = [i]
            assigned[i] = True
            for j in range(i + 1, len(observations)):
                if assigned[j]:
                    continue
                oi, oj = observations[i], observations[j]
                ee_d = np.linalg.norm(oi["ee"] - oj["ee"])
                grip_same = oi["grip"] == oj["grip"]
                obj_d = np.linalg.norm(oi["obj"] - oj["obj"])
                if ee_d < ee_radius and grip_same and obj_d < obj_radius:
                    group.append(j)
                    assigned[j] = True
            groups.append(group)

        for group in groups:
            if len(group) < 2:
                continue

            # Collect unique targets in this group
            group_obs = [observations[i] for i in group]
            targets = [o["target"] for o in group_obs]

            unique_targets = [targets[0]]
            for t in targets[1:]:
                if not any(np.linalg.norm(t - u) < dedup_radius for u in unique_targets):
                    unique_targets.append(t)

            if len(unique_targets) < 2:
                continue

            # Assign alt targets to each member
            for o in group_obs:
                ep_idx = o["episode"]
                own_target = o["target"]
                alts = [
                    t for t in unique_targets
                    if np.linalg.norm(t - own_target) >= dedup_radius
                ]
                # Filter out random-spawn objects (e.g. button at z≈0.914)
                # These are visually identifiable, so cross-variation from
                # their random placement is resolvable by visual cues
                if exclude_random_spawn_z is not None:
                    alts = [
                        t for t in alts
                        if abs(t[2] - exclude_random_spawn_z) > exclude_random_spawn_tol
                    ]
                if alts:
                    key = (ep_idx, kf_idx)
                    if key not in cross_alts:
                        cross_alts[key] = []
                    for a in alts:
                        if not any(np.linalg.norm(a - existing) < dedup_radius
                                   for existing in cross_alts[key]):
                            cross_alts[key].append(a)

    # Convert to per-episode arrays
    result = {}
    for ep in all_ep_data:
        ep_idx = ep["episode"]
        n = ep["num_keyframes"]
        alt_pos = np.zeros((n, max_peaks, 3), dtype=np.float32)
        alt_mask = np.zeros((n, max_peaks), dtype=bool)

        for k in range(n):
            key = (ep_idx, k)
            if key in cross_alts:
                for i, t in enumerate(cross_alts[key][:max_peaks]):
                    alt_pos[k, i] = t
                    alt_mask[k, i] = True

        result[ep_idx] = (alt_pos, alt_mask)

    return result


def merge_targets(intra_pos, intra_mask, cross_pos, cross_mask, max_peaks=5, dedup_radius=0.05):
    """Merge intra and cross alt targets, deduplicating."""
    n = intra_pos.shape[0]
    merged_pos = np.zeros((n, max_peaks, 3), dtype=np.float32)
    merged_mask = np.zeros((n, max_peaks), dtype=bool)

    for k in range(n):
        targets = []
        # Collect intra targets
        for i in range(intra_mask.shape[1]):
            if intra_mask[k, i]:
                targets.append(intra_pos[k, i])
        # Collect cross targets, dedup against existing
        for i in range(cross_mask.shape[1]):
            if cross_mask[k, i]:
                t = cross_pos[k, i]
                if not any(np.linalg.norm(t - existing) < dedup_radius for existing in targets):
                    targets.append(t)

        for i, t in enumerate(targets[:max_peaks]):
            merged_pos[k, i] = t
            merged_mask[k, i] = True

    return merged_pos, merged_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--mode", choices=["intra", "cross", "both"], default="both")
    parser.add_argument("--max-episodes", type=int, default=100)
    parser.add_argument("--max-peaks", type=int, default=5)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    # Need CoppeliaSim env for RLBench imports
    repo = get_repo_root()
    sys.path.insert(0, os.path.join(repo, "sam2act"))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all episodes
    print(f"Loading episodes for {args.task}...")
    all_ep_data = []
    for ep_idx in range(args.max_episodes):
        ep = load_episode_keyframes(args.data_root, args.task, ep_idx)
        if ep is None:
            break
        all_ep_data.append(ep)
        if (ep_idx + 1) % 10 == 0:
            print(f"  loaded {ep_idx + 1} episodes")

    print(f"  total: {len(all_ep_data)} episodes")

    # Compute targets per mode
    results = {}
    total_intra = 0
    total_cross = 0
    total_both = 0

    if args.mode in ("intra", "both"):
        print("Computing intra-episode targets...")
    if args.mode in ("cross", "both"):
        print("Computing cross-variation targets...")
        cross_results = compute_cross_targets(all_ep_data, max_peaks=args.max_peaks)

    for ep in all_ep_data:
        ep_idx = ep["episode"]
        n = ep["num_keyframes"]

        intra_pos = np.zeros((n, args.max_peaks, 3), dtype=np.float32)
        intra_mask = np.zeros((n, args.max_peaks), dtype=bool)
        cross_pos = np.zeros((n, args.max_peaks, 3), dtype=np.float32)
        cross_mask = np.zeros((n, args.max_peaks), dtype=bool)

        if args.mode in ("intra", "both"):
            intra_pos, intra_mask = compute_intra_targets(ep, max_peaks=args.max_peaks)
            total_intra += int(intra_mask.any(axis=1).sum())

        if args.mode in ("cross", "both"):
            if ep_idx in cross_results:
                cross_pos, cross_mask = cross_results[ep_idx]
                total_cross += int(cross_mask.any(axis=1).sum())

        if args.mode == "intra":
            final_pos, final_mask = intra_pos, intra_mask
        elif args.mode == "cross":
            final_pos, final_mask = cross_pos, cross_mask
        else:  # both
            final_pos, final_mask = merge_targets(
                intra_pos, intra_mask, cross_pos, cross_mask, max_peaks=args.max_peaks
            )
            total_both += int(final_mask.any(axis=1).sum())

        results[ep_idx] = {
            "episode": ep_idx,
            "num_keyframes": n,
            "alt_positions": final_pos.tolist(),
            "alt_mask": final_mask.tolist(),
        }

    # Save
    out_path = output_dir / f"{args.task}_multipeak_{args.mode}.json"
    with open(out_path, "w") as f:
        json.dump(results, f)

    total_kf = sum(ep["num_keyframes"] for ep in all_ep_data)
    print(f"\nResults:")
    if args.mode in ("intra", "both"):
        print(f"  intra-episode ambiguous keyframes: {total_intra}/{total_kf}")
    if args.mode in ("cross", "both"):
        print(f"  cross-variation ambiguous keyframes: {total_cross}/{total_kf}")
    if args.mode == "both":
        print(f"  combined (deduped): {total_both}/{total_kf}")
    print(f"  saved to: {out_path}")


if __name__ == "__main__":
    main()
