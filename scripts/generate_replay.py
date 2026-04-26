#!/usr/bin/env python3

import argparse

import sam2act.config as exp_cfg_mod
import sam2act.mvt.config as mvt_cfg_mod
from sam2act.utils.get_dataset import get_dataset_temporal
from sam2act.utils.rvt_utils import RLBENCH_TASKS
from sam2act.utils.peract_utils import DATA_FOLDER


def get_tasks(exp_cfg):
    parsed_tasks = exp_cfg.tasks.split(",")
    return RLBENCH_TASKS if parsed_tasks[0] == "all" else parsed_tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--exp_cfg_path", default="configs/sam2act.yaml")
    parser.add_argument("--mvt_cfg_path", default="mvt/configs/sam2act.yaml")
    parser.add_argument("--exp_cfg_opts", default="")
    parser.add_argument("--mvt_cfg_opts", default="")
    parser.add_argument("--refresh_replay", action="store_true", default=False)
    parser.add_argument("--num_workers", type=int, default=None)
    args = parser.parse_args()

    exp_cfg = exp_cfg_mod.get_cfg_defaults()
    if args.exp_cfg_path:
        exp_cfg.merge_from_file(args.exp_cfg_path)
    if args.exp_cfg_opts:
        exp_cfg.merge_from_list(args.exp_cfg_opts.split(" "))
    exp_cfg.freeze()

    mvt_cfg = mvt_cfg_mod.get_cfg_defaults()
    if args.mvt_cfg_path:
        mvt_cfg.merge_from_file(args.mvt_cfg_path)
    if args.mvt_cfg_opts:
        mvt_cfg.merge_from_list(args.mvt_cfg_opts.split(" "))
    mvt_cfg.freeze()

    tasks = get_tasks(exp_cfg)
    num_workers = exp_cfg.num_workers if args.num_workers is None else args.num_workers

    get_dataset_temporal(
        tasks=tasks,
        BATCH_SIZE_TRAIN=exp_cfg.bs,
        BATCH_SIZE_TEST=None,
        TRAIN_REPLAY_STORAGE_DIR="replay_temporal/replay_train",
        TEST_REPLAY_STORAGE_DIR=None,
        DATA_FOLDER=DATA_FOLDER,
        NUM_TRAIN=exp_cfg.demo,
        NUM_VAL=None,
        refresh_replay=args.refresh_replay,
        device=args.device,
        num_workers=num_workers,
        only_train=True,
        num_maskmem=mvt_cfg.num_maskmem,
        rank=0,
        sample_distribution_mode=exp_cfg.sample_distribution_mode,
    )


if __name__ == "__main__":
    main()
