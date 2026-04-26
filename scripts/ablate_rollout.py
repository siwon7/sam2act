#!/usr/bin/env python3

import argparse
import faulthandler
import json
import os
import sys
import time

import numpy as np
import torch

REPO_CODE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sam2act"))
if REPO_CODE_ROOT not in sys.path:
    sys.path.insert(0, REPO_CODE_ROOT)

from rlbench.backend.utils import task_file_to_task_class
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.action_modes.action_mode import MoveArmThenGripper
from yarr.agents.agent import ActResult

from sam2act.eval import load_agent
from sam2act.libs.peract.helpers import utils
from sam2act.utils.custom_rlbench_env import CustomMultiTaskRLBenchEnv2
from sam2act.utils.peract_utils import CAMERAS, IMAGE_SIZE
from sam2act.utils.rlbench_planning import EndEffectorPoseViaPlanning2


def _to_torch_obs(obs, device):
    return {
        k: torch.tensor(np.array([v]), device=device)
        for k, v in obs.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-root", default="./data/test")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--task", required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--episode-length", type=int, default=25)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--dump-traceback-after", type=int, default=0)
    parser.add_argument("--force-collision", choices=["pred", "zero", "gt"], default="pred")
    parser.add_argument("--rotation-source", choices=["pred", "gt"], default="pred")
    parser.add_argument("--translation-source", choices=["pred", "gt"], default="pred")
    parser.add_argument("--grip-source", choices=["pred", "gt"], default="pred")
    args = parser.parse_args()

    device = f"cuda:{args.device}"
    if args.dump_traceback_after > 0:
        faulthandler.enable()
        faulthandler.dump_traceback_later(args.dump_traceback_after, repeat=True)

    agent = load_agent(
        model_path=args.model_path,
        eval_log_dir="/tmp/sam2act_ablate",
        device=args.device,
    )
    agent.load_clip()

    obs_config = utils.create_obs_config(
        CAMERAS, [IMAGE_SIZE, IMAGE_SIZE], method_name="", use_mask_from_replay=False
    )
    task_class = task_file_to_task_class(args.task)
    env = CustomMultiTaskRLBenchEnv2(
        task_classes=[task_class],
        observation_config=obs_config,
        action_mode=MoveArmThenGripper(EndEffectorPoseViaPlanning2(), Discrete()),
        episode_length=args.episode_length,
        dataset_root=args.data_root,
        headless=False,
        swap_task_every=1,
        include_lang_goal_in_obs=True,
    )
    env.eval = True
    env.launch()

    try:
        obs = env.reset_to_demo(args.episode)
        agent.reset()
        gt_actions = env.get_ground_truth_action(args.episode)
        obs_history = {
            k: [np.array(v)] for k, v in obs.items()
        }
        step = 0
        total_reward = 0.0
        terminal = False
        trace = []

        max_steps = args.max_steps if args.max_steps is not None else args.episode_length
        while not terminal and step < args.episode_length and step < max_steps:
            print(f"[step {step}] act", flush=True)
            act_result = agent.act(-1, _to_torch_obs(obs_history, device), deterministic=True)
            pred = np.asarray(act_result.action, dtype=np.float32).copy()
            gt = np.asarray(gt_actions[min(step, len(gt_actions) - 1)], dtype=np.float32).copy()
            action = pred.copy()

            if args.translation_source == "gt":
                action[:3] = gt[:3]
            if args.rotation_source == "gt":
                action[3:7] = gt[3:7]
            if args.grip_source == "gt":
                action[7] = gt[7]

            if args.force_collision == "zero":
                action[8] = 0.0
            elif args.force_collision == "gt":
                action[8] = gt[8]

            print(
                f"[step {step}] pred={pred.tolist()} gt={gt.tolist()} used={action.tolist()}",
                flush=True,
            )
            step_start = time.time()
            transition = env.step(ActResult(action=action))
            step_dur = time.time() - step_start
            terminal = bool(transition.terminal)
            total_reward += float(transition.reward)
            obs = transition.observation
            for k in obs_history.keys():
                obs_history[k].append(np.array(obs[k]))
                obs_history[k].pop(0)
            last_exc = getattr(env, "_last_exception", None)
            print(
                f"[step {step}] reward={float(transition.reward):.1f} terminal={terminal} "
                f"duration={step_dur:.2f}s exception={type(last_exc).__name__ if last_exc else 'None'}",
                flush=True,
            )
            trace.append(
                {
                    "step": step,
                    "pred": pred.tolist(),
                    "gt": gt.tolist(),
                    "used": action.tolist(),
                    "reward": float(transition.reward),
                    "terminal": terminal,
                    "duration_sec": step_dur,
                    "exception": str(last_exc) if last_exc else None,
                }
            )
            step += 1

        print(
            json.dumps(
                {
                    "task": args.task,
                    "episode": args.episode,
                    "steps": step,
                    "total_reward": total_reward,
                    "success": total_reward >= 100.0,
                    "force_collision": args.force_collision,
                    "rotation_source": args.rotation_source,
                    "translation_source": args.translation_source,
                    "grip_source": args.grip_source,
                    "trace": trace,
                },
                ensure_ascii=True,
            )
        )
    finally:
        env.shutdown()


if __name__ == "__main__":
    main()
