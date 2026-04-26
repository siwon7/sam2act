#!/usr/bin/env python3

import argparse
import json
import numpy as np
import os
import sys
import torch

REPO_CODE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sam2act"))
if REPO_CODE_ROOT not in sys.path:
    sys.path.insert(0, REPO_CODE_ROOT)

from rlbench.backend.utils import task_file_to_task_class
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.action_mode import MoveArmThenGripper

from sam2act.libs.peract.helpers import utils
from sam2act.utils.custom_rlbench_env import CustomMultiTaskRLBenchEnv2
from sam2act.utils.rlbench_planning import EndEffectorPoseViaPlanning2
from sam2act.utils.peract_utils import CAMERAS, IMAGE_SIZE
from sam2act.eval import load_agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-root", default="./data/test")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--tasks", nargs="+", required=True)
    args = parser.parse_args()

    agent = load_agent(
        model_path=args.model_path,
        eval_log_dir="/tmp/sam2act_dbg",
        device=args.device,
    )
    agent.load_clip()

    obs_config = utils.create_obs_config(
        CAMERAS, [IMAGE_SIZE, IMAGE_SIZE], method_name="", use_mask_from_replay=False
    )

    for task_name in args.tasks:
        task_class = task_file_to_task_class(task_name)
        action_mode = MoveArmThenGripper(EndEffectorPoseViaPlanning2(), Discrete())
        env = CustomMultiTaskRLBenchEnv2(
            task_classes=[task_class],
            observation_config=obs_config,
            action_mode=action_mode,
            episode_length=25,
            dataset_root=args.data_root,
            headless=False,
            swap_task_every=1,
            include_lang_goal_in_obs=True,
        )
        env.eval = True
        env.launch()
        obs = env.reset_to_demo(0)
        prepped = {
            k: torch.tensor(np.array([[v]]), device=f"cuda:{args.device}")
            for k, v in obs.items()
        }
        act = agent.act(-1, prepped, deterministic=True)
        print(
            json.dumps(
                {"task": task_name, "pred": np.asarray(act.action).tolist()},
                ensure_ascii=True,
            )
        )
        env.shutdown()


if __name__ == "__main__":
    main()
