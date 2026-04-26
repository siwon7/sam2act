# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import numpy as np
from clip import tokenize

from sam2act.libs.peract.helpers.custom_rlbench_env import CustomMultiTaskRLBenchEnv
from sam2act.libs.peract.helpers.demo_loading_utils import keypoint_discovery
from sam2act.libs.peract.helpers.utils import extract_obs as replay_extract_obs
from sam2act.utils.peract_utils import CAMERAS


class CustomMultiTaskRLBenchEnv2(CustomMultiTaskRLBenchEnv):
    def __init__(self, *args, **kwargs):
        super(CustomMultiTaskRLBenchEnv2, self).__init__(*args, **kwargs)
        self._current_demo = None

    def reset(self) -> dict:
        super().reset()
        self._record_current_episode = (
            self.eval
            and self._record_every_n > 0
            and self._episode_index % self._record_every_n == 0
        )
        return self._previous_obs_dict

    def extract_obs(self, obs, t=None, prev_action=None):
        step_idx = self._i if t is None else t
        extracted_obs = replay_extract_obs(
            obs,
            CAMERAS,
            t=step_idx,
            prev_action=prev_action,
            episode_length=self._episode_length,
        )
        if self._include_lang_goal_in_obs:
            extracted_obs["lang_goal_tokens"] = tokenize([self._lang_goal])[0].numpy()
        return extracted_obs

    def reset_to_demo(self, i, variation_number=-1):
        if self._episodes_this_task == self._swap_task_every:
            self._set_new_task()
            self._episodes_this_task = 0
        self._episodes_this_task += 1

        self._i = 0
        self._task.set_variation(-1)
        d = self._task.get_demos(
            1, live_demos=False, random_selection=False, from_episode_number=i
        )[0]
        self._current_demo = d

        self._task.set_variation(d.variation_number)
        desc, obs = self._task.reset_to_demo(d)
        self._lang_goal = desc[0]

        self._previous_obs_dict = self.extract_obs(obs)
        self._record_current_episode = (
            self.eval
            and self._record_every_n > 0
            and self._episode_index % self._record_every_n == 0
        )
        self._episode_index += 1
        self._recorded_images.clear()

        return self._previous_obs_dict

    def get_ground_truth_action(self, i):
        # Replay keypoint actions to match the policy evaluation semantics.
        d = self._current_demo
        if d is None:
            raise RuntimeError(
                "Ground-truth action requested before reset_to_demo cached a demo."
            )

        actions = []
        for keypoint in keypoint_discovery(d):
            obs = d[keypoint]
            actions.append(
                np.concatenate(
                    [
                        np.asarray(obs.gripper_pose, dtype=np.float32),
                        np.asarray([obs.gripper_open], dtype=np.float32),
                        np.asarray([obs.ignore_collisions], dtype=np.float32),
                    ]
                )
            )
        return actions
