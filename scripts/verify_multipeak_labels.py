#!/usr/bin/env python3
"""Verify multi-peak label assignment on MemoryBench keyframe data.

Reads the episodewise JSON files (from keyframe gallery) and runs the
scene-aware clustering to show which keyframes get multi-peak labels
and why.

Usage:
    python scripts/verify_multipeak_labels.py \
        --data-dir /home/cv11/project/siwon/memorybench_keyframes_gallery_20260427
"""

import argparse
import json
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sam2act"))

from mvt.multipeak_utils import (
    track_object_positions,
    compute_scene_aware_clusters,
    collect_alt_targets,
)


def verify_task(filepath: str, task_name: str):
    with open(filepath) as f:
        data = json.load(f)

    print(f"\n{'='*65}")
    print(f"  {task_name} ({len(data)} episodes)")
    print(f"{'='*65}")

    total_kf = 0
    total_multipeak = 0

    for ep_data in data:
        ep = ep_data["episode"]
        kfs = ep_data["keyframes"]
        n = len(kfs)

        poses = np.array([kf["pose"] for kf in kfs], dtype=np.float32)
        opens = np.array([kf["gripper_open"] for kf in kfs], dtype=np.float32)

        alt_pos, alt_mask = collect_alt_targets(poses, opens, n, max_peaks=3)

        multipeak_kfs = []
        for i in range(n):
            if alt_mask[i].any():
                num_alts = int(alt_mask[i].sum())
                multipeak_kfs.append((i, num_alts))

        total_kf += n
        total_multipeak += len(multipeak_kfs)

        if multipeak_kfs and ep < 3:
            obj_pos = track_object_positions(poses, opens)
            print(f"\n  ep{ep}: {len(multipeak_kfs)} multi-peak keyframes")
            for kf_idx, num_alts in multipeak_kfs:
                ee = poses[kf_idx, :3]
                grip = opens[kf_idx]
                obj = obj_pos[kf_idx]
                own_next = poses[kf_idx + 1, :3] if kf_idx + 1 < n else None
                alt_targets = alt_pos[kf_idx][alt_mask[kf_idx]]

                print(f"    KF{kf_idx}: EE=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f})"
                      f" grip={grip:.0f}"
                      f" obj=({obj[0]:.3f},{obj[1]:.3f},{obj[2]:.3f})")
                if own_next is not None:
                    print(f"      primary target: ({own_next[0]:.3f},{own_next[1]:.3f},{own_next[2]:.3f})")
                for j, at in enumerate(alt_targets):
                    print(f"      alt target {j+1}: ({at[0]:.3f},{at[1]:.3f},{at[2]:.3f})")

    pct = total_multipeak / max(total_kf, 1) * 100
    print(f"\n  Summary: {total_multipeak}/{total_kf} keyframes get multi-peak ({pct:.1f}%)")
    return total_kf, total_multipeak


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="/home/cv11/project/siwon/memorybench_keyframes_gallery_20260427",
    )
    args = parser.parse_args()

    tasks = {
        "put_block_back": "episodewise_put_block_back/put_block_back_episodewise.json",
        "rearrange_block": "episodewise_rearrange_block/rearrange_block_episodewise.json",
        "reopen_drawer": "episodewise_reopen_drawer/reopen_drawer_episodewise.json",
    }

    grand_kf = 0
    grand_mp = 0

    for task_name, relpath in tasks.items():
        fp = os.path.join(args.data_dir, relpath)
        if not os.path.exists(fp):
            print(f"  [skip] {fp} not found")
            continue
        kf, mp = verify_task(fp, task_name)
        grand_kf += kf
        grand_mp += mp

    print(f"\n{'='*65}")
    print(f"  TOTAL: {grand_mp}/{grand_kf} keyframes get multi-peak"
          f" ({grand_mp/max(grand_kf,1)*100:.1f}%)")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
