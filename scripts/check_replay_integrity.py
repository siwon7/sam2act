#!/usr/bin/env python3

import argparse
import glob
import os
import pickle

import numpy as np


def is_corrupt_like(sample):
    action = np.asarray(sample["action"], dtype=np.float64)
    terminal = int(sample["terminal"])
    sample_frame = int(sample["sample_frame"])
    initial_frame = int(sample["initial_frame"])
    keypoint_frame = int(sample["keypoint_frame"])
    next_keypoint_frame = int(sample["next_keypoint_frame"])

    # add_final() writes placeholder transitions with terminal == -1.
    # They are part of the replay layout but are intentionally excluded from
    # sampling by UniformReplayBuffer_temporal.is_valid_transition().
    if terminal == -1:
        return False, False, True

    zero_action = (
        float(np.linalg.norm(action[:3])) < 1e-6
        and float(np.linalg.norm(action[3:])) < 1e-5
    )
    weird_meta = (
        sample_frame < 0
        or sample_frame > 1000
        or abs(initial_frame) > 100000
        or abs(keypoint_frame) > 100000
        or abs(next_keypoint_frame) > 100000
    )
    return zero_action, weird_meta, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replay-root",
        default="/home/cv25/siwon/sam2act/sam2act/replay_temporal/replay_train",
    )
    parser.add_argument("--limit-per-task", type=int, default=3000)
    parser.add_argument("--fail-on-corrupt", action="store_true", default=False)
    args = parser.parse_args()

    has_corrupt = False
    for task in sorted(os.listdir(args.replay_root)):
        task_dir = os.path.join(args.replay_root, task)
        files = glob.glob(os.path.join(task_dir, "*.replay"))[: args.limit_per_task]
        if not files:
            continue

        zero_count = 0
        weird_count = 0
        placeholder_count = 0
        for path in files:
            with open(path, "rb") as f:
                sample = pickle.load(f)
            zero_action, weird_meta, is_placeholder = is_corrupt_like(sample)
            placeholder_count += int(is_placeholder)
            zero_count += int(zero_action)
            weird_count += int(weird_meta)

        ratio = weird_count / len(files)
        has_corrupt = has_corrupt or weird_count > 0
        print(
            f"{task}: total={len(files)} corrupt_like={weird_count} "
            f"ratio={ratio:.4f} zero_action={zero_count} "
            f"placeholder_final={placeholder_count}"
        )

    if args.fail_on_corrupt and has_corrupt:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
