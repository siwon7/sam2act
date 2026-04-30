# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import pprint

import clip
import torch
import torchvision
import numpy as np
import torch.nn as nn
import bitsandbytes as bnb

from scipy.spatial.transform import Rotation
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR

import sam2act.utils.peract_utils as peract_utils
import sam2act.mvt.utils as mvt_utils
import sam2act.utils.rvt_utils as rvt_utils
import peract_colab.arm.utils as arm_utils

from sam2act.mvt.augmentation import apply_se3_aug_con, apply_se3_aug_con_same, apply_se3_aug_con_sequence, apply_se3_aug_given_matrix, aug_utils
from peract_colab.arm.optim.lamb import Lamb
from yarr.agents.agent import ActResult
from sam2act.utils.dataset import _clip_encode_text
from sam2act.utils.lr_sched_utils import GradualWarmupScheduler


def eval_con(gt, pred):
    assert gt.shape == pred.shape, print(f"{gt.shape} {pred.shape}")
    assert len(gt.shape) == 2
    dist = torch.linalg.vector_norm(gt - pred, dim=1)
    return {"avg err": dist.mean()}


def eval_con_cls(gt, pred, num_bin=72, res=5, symmetry=1):
    """
    Evaluate continuous classification where floating point values are put into
    discrete bins
    :param gt: (bs,)
    :param pred: (bs,)
    :param num_bin: int for the number of rotation bins
    :param res: float to specify the resolution of each rotation bin
    :param symmetry: degrees of symmetry; 2 is 180 degree symmetry, 4 is 90
        degree symmetry
    """
    assert gt.shape == pred.shape
    assert len(gt.shape) in [0, 1], gt
    assert num_bin % symmetry == 0, (num_bin, symmetry)
    gt = torch.tensor(gt)
    pred = torch.tensor(pred)
    num_bin //= symmetry
    pred %= num_bin
    gt %= num_bin
    dist = torch.abs(pred - gt)
    dist = torch.min(dist, num_bin - dist)
    dist_con = dist.float() * res
    return {"avg err": dist_con.mean()}


def eval_cls(gt, pred):
    """
    Evaluate classification performance
    :param gt_coll: (bs,)
    :param pred: (bs,)
    """
    assert gt.shape == pred.shape
    assert len(gt.shape) == 1
    return {"per err": (gt != pred).float().mean()}


def eval_all(
    wpt,
    pred_wpt,
    action_rot,
    pred_rot_quat,
    action_grip_one_hot,
    grip_q,
    action_collision_one_hot,
    collision_q,
):
    bs = len(wpt)
    assert wpt.shape == (bs, 3), wpt
    assert pred_wpt.shape == (bs, 3), pred_wpt
    assert action_rot.shape == (bs, 4), action_rot
    assert pred_rot_quat.shape == (bs, 4), pred_rot_quat
    assert action_grip_one_hot.shape == (bs, 2), action_grip_one_hot
    assert grip_q.shape == (bs, 2), grip_q
    assert action_collision_one_hot.shape == (bs, 2), action_collision_one_hot
    assert collision_q.shape == (bs, 2), collision_q

    eval_trans = []
    eval_rot_x = []
    eval_rot_y = []
    eval_rot_z = []
    eval_grip = []
    eval_coll = []

    for i in range(bs):
        eval_trans.append(
            eval_con(wpt[i : i + 1], pred_wpt[i : i + 1])["avg err"]
            .cpu()
            .numpy()
            .item()
        )

        euler_gt = Rotation.from_quat(action_rot[i]).as_euler("xyz", degrees=True)
        euler_pred = Rotation.from_quat(pred_rot_quat[i]).as_euler("xyz", degrees=True)

        eval_rot_x.append(
            eval_con_cls(euler_gt[0], euler_pred[0], num_bin=360, res=1)["avg err"]
            .cpu()
            .numpy()
            .item()
        )
        eval_rot_y.append(
            eval_con_cls(euler_gt[1], euler_pred[1], num_bin=360, res=1)["avg err"]
            .cpu()
            .numpy()
            .item()
        )
        eval_rot_z.append(
            eval_con_cls(euler_gt[2], euler_pred[2], num_bin=360, res=1)["avg err"]
            .cpu()
            .numpy()
            .item()
        )

        eval_grip.append(
            eval_cls(
                action_grip_one_hot[i : i + 1].argmax(-1),
                grip_q[i : i + 1].argmax(-1),
            )["per err"]
            .cpu()
            .numpy()
            .item()
        )

        eval_coll.append(
            eval_cls(
                action_collision_one_hot[i : i + 1].argmax(-1),
                collision_q[i : i + 1].argmax(-1),
            )["per err"]
            .cpu()
            .numpy()
        )

    return eval_trans, eval_rot_x, eval_rot_y, eval_rot_z, eval_grip, eval_coll


def manage_eval_log(
    self,
    tasks,
    wpt,
    pred_wpt,
    action_rot,
    pred_rot_quat,
    action_grip_one_hot,
    grip_q,
    action_collision_one_hot,
    collision_q,
    reset_log=False,
):
    bs = len(wpt)
    assert wpt.shape == (bs, 3), wpt
    assert pred_wpt.shape == (bs, 3), pred_wpt
    assert action_rot.shape == (bs, 4), action_rot
    assert pred_rot_quat.shape == (bs, 4), pred_rot_quat
    assert action_grip_one_hot.shape == (bs, 2), action_grip_one_hot
    assert grip_q.shape == (bs, 2), grip_q
    assert action_collision_one_hot.shape == (bs, 2), action_collision_one_hot
    assert collision_q.shape == (bs, 2), collision_q

    if not hasattr(self, "eval_trans") or reset_log:
        self.eval_trans = {}
        self.eval_rot_x = {}
        self.eval_rot_y = {}
        self.eval_rot_z = {}
        self.eval_grip = {}
        self.eval_coll = {}

    (eval_trans, eval_rot_x, eval_rot_y, eval_rot_z, eval_grip, eval_coll,) = eval_all(
        wpt=wpt,
        pred_wpt=pred_wpt,
        action_rot=action_rot,
        pred_rot_quat=pred_rot_quat,
        action_grip_one_hot=action_grip_one_hot,
        grip_q=grip_q,
        action_collision_one_hot=action_collision_one_hot,
        collision_q=collision_q,
    )

    for idx, task in enumerate(tasks):
        if not (task in self.eval_trans):
            self.eval_trans[task] = []
            self.eval_rot_x[task] = []
            self.eval_rot_y[task] = []
            self.eval_rot_z[task] = []
            self.eval_grip[task] = []
            self.eval_coll[task] = []
        self.eval_trans[task].append(eval_trans[idx])
        self.eval_rot_x[task].append(eval_rot_x[idx])
        self.eval_rot_y[task].append(eval_rot_y[idx])
        self.eval_rot_z[task].append(eval_rot_z[idx])
        self.eval_grip[task].append(eval_grip[idx])
        self.eval_coll[task].append(eval_coll[idx])

    return {
        "eval_trans": eval_trans,
        "eval_rot_x": eval_rot_x,
        "eval_rot_y": eval_rot_y,
        "eval_rot_z": eval_rot_z,
    }


def print_eval_log(self):
    logs = {
        "trans": self.eval_trans,
        "rot_x": self.eval_rot_x,
        "rot_y": self.eval_rot_y,
        "rot_z": self.eval_rot_z,
        "grip": self.eval_grip,
        "coll": self.eval_coll,
    }

    out = {}
    for name, log in logs.items():
        for task, task_log in log.items():
            task_log_np = np.array(task_log)
            mean, std, median = (
                np.mean(task_log_np),
                np.std(task_log_np),
                np.median(task_log_np),
            )
            out[f"{task}/{name}_mean"] = mean
            out[f"{task}/{name}_std"] = std
            out[f"{task}/{name}_median"] = median

    pprint.pprint(out)

    return out


def manage_loss_log(
    agent,
    loss_log,
    reset_log,
):
    if not hasattr(agent, "loss_log") or reset_log:
        agent.loss_log = {}

    for key, val in loss_log.items():
        if key in agent.loss_log:
            agent.loss_log[key].append(val)
        else:
            agent.loss_log[key] = [val]


def print_loss_log(agent):
    out = {}
    for key, val in agent.loss_log.items():
        if val is not None:
            filtered_val = [v for v in val if v is not None]
            mean_value = np.mean(filtered_val) if filtered_val else None  # Avoid empty list error
        else:
            mean_value = None
        if mean_value is not None:
            out[key] = mean_value
    pprint.pprint(out)
    return out


def horizon_loss_cal(output, data_horizon):

    shape = output.shape
    bs, ah = shape[0], shape[1]
    mask = torch.arange(ah).expand(bs, ah).to(output.device) < data_horizon.view(bs, 1)
    if len(shape) == 3:
        mask = mask.unsqueeze(-1).expand_as(output)

        output_mask = output * mask
        ah_ave = output_mask.sum(dim=(0,2))/(mask.sum(dim=(0,2)).float() + 1e-5)
        
    else:
        output_mask = output * mask
        ah_ave = output_mask.sum(dim=0)/(mask.sum(dim=0).float() + 1e-5)
        
    total_ave = output_mask.sum()/(mask.sum().float())
    
    return total_ave,ah_ave


class SAM2Act_Agent:
    def __init__(
        self,
        network: nn.Module,
        num_rotation_classes: int,
        stage_two: bool,
        add_lang: bool,
        amp: bool,
        bnb: bool,
        move_pc_in_bound: bool,
        lr: float = 0.0001,
        lr_cos_dec: bool = False,
        cos_dec_max_step: int = 60000,
        warmup_steps: int = 0,
        image_resolution: list = None,
        lambda_weight_l2: float = 0.0,
        transform_augmentation: bool = True,
        transform_augmentation_xyz: list = [0.1, 0.1, 0.1],
        transform_augmentation_rpy: list = [0.0, 0.0, 20.0],
        place_with_mean: bool = True,
        transform_augmentation_rot_resolution: int = 5,
        optimizer_type: str = "lamb",
        gt_hm_sigma: float = 1.5,
        img_aug: bool = False,
        add_rgc_loss: bool = False,
        scene_bounds: list = peract_utils.SCENE_BOUNDS,
        cameras: list = peract_utils.CAMERAS,
        rot_ver: int = 0,
        rot_x_y_aug: int = 2,
        log_dir="",
        action_horizon=None,
        same_trans_aug_per_seq: bool = False,
        use_memory: bool = False,
        num_maskmem: int = 7,
        graph_node_loss_weight: float = 0.0,
        use_multipeak: bool = False,
        use_graph_peak_select: bool = False,
        graph_peak_select_loss_weight: float = 0.5,
        graph_transition_loss_weight: float = 0.1,
        graph_contrastive_loss_weight: float = 0.05,
        graph_contrastive_temperature: float = 0.1,
        graph_contrastive_pos_radius: float = 0.03,
    ):
        """
        :param gt_hm_sigma: the std of the groundtruth hm, currently for for
            2d, if -1 then only single point is considered
        :type gt_hm_sigma: float
        :param rot_ver: version of the rotation prediction network
            Either:
                0: same as peract, independent discrete xyz predictions
                1: xyz prediction dependent on one another
        :param rot_x_y_aug: only applicable when rot_ver is 1, it specifies how
            much error we should add to groundtruth rotation while training
        :param log_dir: a folder location for saving some intermediate data
        """

        self._network = network
        self._num_rotation_classes = num_rotation_classes
        self._rotation_resolution = 360 / self._num_rotation_classes
        self._lr = lr
        self._image_resolution = image_resolution
        self._lambda_weight_l2 = lambda_weight_l2
        self._transform_augmentation = transform_augmentation
        self._place_with_mean = place_with_mean
        self._transform_augmentation_xyz = torch.from_numpy(
            np.array(transform_augmentation_xyz)
        )
        self._transform_augmentation_rpy = transform_augmentation_rpy
        self._transform_augmentation_rot_resolution = (
            transform_augmentation_rot_resolution
        )
        self._optimizer_type = optimizer_type
        self.gt_hm_sigma = gt_hm_sigma
        self.img_aug = img_aug
        self.add_rgc_loss = add_rgc_loss
        self.amp = amp
        self.bnb = bnb
        self.stage_two = stage_two
        self.add_lang = add_lang
        self.log_dir = log_dir
        self.warmup_steps = warmup_steps
        self.lr_cos_dec = lr_cos_dec
        self.cos_dec_max_step = cos_dec_max_step
        self.scene_bounds = scene_bounds
        self.cameras = cameras
        self.move_pc_in_bound = move_pc_in_bound
        self.rot_ver = rot_ver
        self.rot_x_y_aug = rot_x_y_aug
        self.use_multipeak = use_multipeak
        self.use_graph_peak_select = use_graph_peak_select
        self.graph_peak_select_loss_weight = graph_peak_select_loss_weight
        self.graph_transition_loss_weight = graph_transition_loss_weight
        self.graph_contrastive_loss_weight = graph_contrastive_loss_weight
        self.graph_contrastive_temperature = graph_contrastive_temperature
        self.graph_contrastive_pos_radius = graph_contrastive_pos_radius

        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
        if isinstance(self._network, DistributedDataParallel):
            self._net_mod = self._network.module
        else:
            self._net_mod = self._network

        self.num_all_rot = self._num_rotation_classes * 3

        self.scaler = GradScaler(enabled=self.amp)

        self.action_horizon = action_horizon

        self._same_trans_aug_per_seq = same_trans_aug_per_seq

        self.use_memory = use_memory
        self._num_maskmem = num_maskmem

    def build(self, training: bool, device: torch.device = None):
        self._training = training
        self._device = device

        sam_mem_params = []
        upsample_params = []
        
        for name, param in self._network.named_parameters():
            if "sam" in name and "image encoder" not in name:
                sam_mem_params.append(param)
            elif "up0" in name:
                upsample_params.append(param)

        param_groups = [
            {"params": sam_mem_params, "lr": self._lr * 2},  # SAM2 (memory)
            {"params": upsample_params, "lr": self._lr},  # upsample layers
        ]

        if self._optimizer_type == "lamb":
            if self.bnb:
                print("Using 8-Bit Optimizer")
                self._optimizer = bnb.optim.LAMB(
                    self._network.parameters(),
                    # param_groups,
                    lr=self._lr,
                    weight_decay=self._lambda_weight_l2,
                    betas=(0.9, 0.999),
                )
            else:
                # From: https://github.com/cybertronai/pytorch-lamb/blob/master/pytorch_lamb/lamb.py
                self._optimizer = Lamb(
                    self._network.parameters(),
                    lr=self._lr,
                    weight_decay=self._lambda_weight_l2,
                    betas=(0.9, 0.999),
                    adam=False,
                )
        elif self._optimizer_type == "adam":
            self._optimizer = torch.optim.Adam(
                self._network.parameters(),
                lr=self._lr,
                weight_decay=self._lambda_weight_l2,
            )
        else:
            raise Exception("Unknown optimizer")

        if self.lr_cos_dec:
            after_scheduler = CosineAnnealingLR(
                self._optimizer,
                T_max=self.cos_dec_max_step,
                eta_min=self._lr / 100,  # mininum lr
            )
        else:
            after_scheduler = None
        self._lr_sched = GradualWarmupScheduler(
            self._optimizer,
            multiplier=1,
            total_epoch=self.warmup_steps,
            after_scheduler=after_scheduler,
        )

    def load_clip(self):
        self.clip_model, self.clip_preprocess = clip.load("RN50", device=self._device)
        self.clip_model.eval()

    def unload_clip(self):
        del self.clip_model
        del self.clip_preprocess
        with torch.cuda.device(self._device):
            torch.cuda.empty_cache()

    # copied from per-act and removed the translation part
    def _get_one_hot_expert_actions(
        self,
        batch_size,
        action_rot,
        action_grip,
        action_ignore_collisions,
        device,
    ):
        """_get_one_hot_expert_actions.

        :param batch_size: int
        :param action_rot: np.array of shape (bs, 4), quternion xyzw format
        :param action_grip: torch.tensor of shape (bs)
        :param action_ignore_collisions: torch.tensor of shape (bs)
        :param device:
        """
        bs = batch_size
        assert action_rot.shape == (bs, 4)
        assert action_grip.shape == (bs,), (action_grip, bs)

        action_rot_x_one_hot = torch.zeros(
            (bs, self._num_rotation_classes), dtype=int, device=device
        )
        action_rot_y_one_hot = torch.zeros(
            (bs, self._num_rotation_classes), dtype=int, device=device
        )
        action_rot_z_one_hot = torch.zeros(
            (bs, self._num_rotation_classes), dtype=int, device=device
        )
        action_grip_one_hot = torch.zeros((bs, 2), dtype=int, device=device)
        action_collision_one_hot = torch.zeros((bs, 2), dtype=int, device=device)

        # fill one-hots
        for b in range(bs):
            gt_rot = action_rot[b]
            gt_rot = aug_utils.quaternion_to_discrete_euler(
                gt_rot, self._rotation_resolution
            )
            action_rot_x_one_hot[b, gt_rot[0]] = 1
            action_rot_y_one_hot[b, gt_rot[1]] = 1
            action_rot_z_one_hot[b, gt_rot[2]] = 1

            # grip
            gt_grip = action_grip[b]
            action_grip_one_hot[b, gt_grip] = 1

            # ignore collision
            gt_ignore_collisions = action_ignore_collisions[b, :]
            action_collision_one_hot[b, gt_ignore_collisions[0]] = 1

        return (
            action_rot_x_one_hot,
            action_rot_y_one_hot,
            action_rot_z_one_hot,
            action_grip_one_hot,
            action_collision_one_hot,
        )

    def get_q(self, out, dims, only_pred=False, get_q_trans=True):
        """
        :param out: output of mvt
        :param dims: tensor dimensions (bs, nc, h, w)
        :param only_pred: some speedupds if the q values are meant only for
            prediction
        :return: tuple of trans_q, rot_q, grip_q and coll_q that is used for
            training and preduction
        """
        bs, nc, h, w = dims
        assert isinstance(only_pred, bool)

        if get_q_trans:
            pts = None
            # (bs, h*w, nc)
            q_trans = out["trans"].view(bs, nc, h * w).transpose(1, 2)
            if not only_pred:
                q_trans = q_trans.clone()

            # if two stages, we concatenate the q_trans, and replace all other
            # q
            if self.stage_two and not self.use_memory:
                out = out["mvt2"]
                q_trans2 = out["trans"].view(bs, nc, h * w).transpose(1, 2)
                if not only_pred:
                    q_trans2 = q_trans2.clone()
                q_trans = torch.cat((q_trans, q_trans2), dim=2)
        else:
            pts = None
            q_trans = None
            if self.stage_two:
                out = out["mvt2"]

        if not self.use_memory:

            if self.rot_ver == 0:
                # (bs, 218)
                rot_q = out["feat"].view(bs, -1)[:, 0 : self.num_all_rot]
                grip_q = out["feat"].view(bs, -1)[:, self.num_all_rot : self.num_all_rot + 2]
                # (bs, 2)
                collision_q = out["feat"].view(bs, -1)[
                    :, self.num_all_rot + 2 : self.num_all_rot + 4
                ]
            elif self.rot_ver == 1:
                rot_q = torch.cat((out["feat_x"], out["feat_y"], out["feat_z"]),
                                dim=-1).view(bs, -1)
                grip_q = out["feat_ex_rot"].view(bs, -1)[:, :2]
                collision_q = out["feat_ex_rot"].view(bs, -1)[:, 2:]
            else:
                assert False

            y_q = None

            return q_trans, rot_q, grip_q, collision_q, y_q, pts
        
        else:
            return q_trans, None, None, None, None, None

    def update(
        self,
        step: int,
        replay_sample: dict,
        backprop: bool = True,
        eval_log: bool = False,
        reset_log: bool = False,
    ) -> dict:
        assert replay_sample["rot_grip_action_indicies"].shape[1:] == (1, 4)
        assert replay_sample["ignore_collisions"].shape[1:] == (1, 1)
        assert replay_sample["gripper_pose"].shape[1:] == (1, 7)
        assert replay_sample["lang_goal_embs"].shape[1:] == (1, 77, 512)
        assert replay_sample["low_dim_state"].shape[1:] == (
            1,
            self._net_mod.proprio_dim,
        )

        # sample
        action_rot_grip = replay_sample["rot_grip_action_indicies"][
            :, -1
        ].int()  # (b, 4) of int
        action_ignore_collisions = replay_sample["ignore_collisions"][
            :, -1
        ].int()  # (b, 1) of int
        action_gripper_pose = replay_sample["gripper_pose"][:, -1]  # (b, 7)
        action_trans_con = action_gripper_pose[:, 0:3]  # (b, 3)
        # rotation in quaternion xyzw
        action_rot = action_gripper_pose[:, 3:7]  # (b, 4)
        action_grip = action_rot_grip[:, -1]  # (b,)
        lang_goal_embs = replay_sample["lang_goal_embs"][:, -1].float()
        tasks = replay_sample["tasks"]

        proprio = arm_utils.stack_on_channel(replay_sample["low_dim_state"])  # (b, 4)
        return_out = {}

        obs, pcd = peract_utils._preprocess_inputs(replay_sample, self.cameras)

        with torch.no_grad():
            pc, img_feat = rvt_utils.get_pc_img_feat(
                obs,
                pcd,
            )

            # Save pre-augmentation state for SE(3) intra-mixup (TGM-VLA style)
            ori_pc = pc.clone() if self.use_multipeak else None
            ori_img_feat = [x.clone() for x in img_feat] if self.use_multipeak and isinstance(img_feat, list) else (img_feat.clone() if self.use_multipeak else None)
            ori_action_trans_con = action_trans_con.clone() if self.use_multipeak else None

            if self._transform_augmentation and backprop:
                if not self._same_trans_aug_per_seq:
                    action_trans_con, action_rot, pc = apply_se3_aug_con(
                        pcd=pc,
                        action_gripper_pose=action_gripper_pose,
                        bounds=torch.tensor(self.scene_bounds),
                        trans_aug_range=torch.tensor(self._transform_augmentation_xyz),
                        rot_aug_range=torch.tensor(self._transform_augmentation_rpy),
                    )
                    action_trans_con = torch.tensor(action_trans_con).to(pc.device)
                    action_rot = torch.tensor(action_rot).to(pc.device)
                else:
                    bs = pc.shape[0]
                    num_obs = self._num_maskmem + 1
                    num_seq = bs // num_obs

                    action_trans_con_after = []
                    action_rot_after = []
                    pc_after = []
                    for seq_idx in range(num_seq):
                        pc_i = pc[seq_idx*num_obs:seq_idx*num_obs+num_obs]
                        action_gripper_pose_i = action_gripper_pose[seq_idx*num_obs:seq_idx*num_obs+num_obs]
                        action_trans_con_i, action_rot_i, pc_i = apply_se3_aug_con_same(
                            pcd=pc_i,
                            action_gripper_pose=action_gripper_pose_i,
                            bounds=torch.tensor(self.scene_bounds),
                            trans_aug_range=torch.tensor(self._transform_augmentation_xyz),
                            rot_aug_range=torch.tensor(self._transform_augmentation_rpy),
                        )
                        action_trans_con_i = torch.tensor(action_trans_con_i).to(pc.device)
                        action_rot_i = torch.tensor(action_rot_i).to(pc.device)

                        action_trans_con_after.append(action_trans_con_i)
                        action_rot_after.append(action_rot_i)
                        pc_after.append(pc_i)

                    action_trans_con = torch.cat(action_trans_con_after, dim=0)
                    action_rot = torch.cat(action_rot_after, dim=0)
                    pc = torch.cat(pc_after, dim=0)


            # TODO: vectorize
            action_rot = action_rot.cpu().numpy()
            for i, _action_rot in enumerate(action_rot):
                _action_rot = aug_utils.normalize_quaternion(_action_rot)
                if _action_rot[-1] < 0:
                    _action_rot = -_action_rot
                action_rot[i] = _action_rot

            pc, img_feat = rvt_utils.move_pc_in_bound(
                pc, img_feat, self.scene_bounds, no_op=not self.move_pc_in_bound
            )
            wpt = [x[:3] for x in action_trans_con]

            wpt_local = []
            rev_trans = []
            for _pc, _wpt in zip(pc, wpt):
                a, b = mvt_utils.place_pc_in_cube(
                    _pc,
                    _wpt,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )
                wpt_local.append(a.unsqueeze(0))
                rev_trans.append(b)

            wpt_local = torch.cat(wpt_local, axis=0)

            # Compute ori_wpt_local and ori_pc_cube for SE(3) intra-mixup
            ori_wpt_local = None
            ori_pc_cube = None
            se3_mixup_flags = None
            if self.use_multipeak and ori_action_trans_con is not None and backprop:
                import random
                ori_pc_moved, ori_img_feat_moved = rvt_utils.move_pc_in_bound(
                    ori_pc, ori_img_feat, self.scene_bounds, no_op=not self.move_pc_in_bound
                )
                ori_wpt = [x[:3] for x in ori_action_trans_con]
                ori_wpt_local_list = []
                for _pc, _wpt in zip(ori_pc_moved, ori_wpt):
                    a, _ = mvt_utils.place_pc_in_cube(
                        _pc,
                        _wpt,
                        with_mean_or_bounds=self._place_with_mean,
                        scene_bounds=None if self._place_with_mean else self.scene_bounds,
                    )
                    ori_wpt_local_list.append(a.unsqueeze(0))
                ori_wpt_local = torch.cat(ori_wpt_local_list, axis=0)

                # Also prepare ori_pc in cube coords for point cloud concat
                ori_pc_cube = [
                    mvt_utils.place_pc_in_cube(
                        _pc,
                        with_mean_or_bounds=self._place_with_mean,
                        scene_bounds=None if self._place_with_mean else self.scene_bounds,
                    )[0]
                    for _pc in ori_pc_moved
                ]

            # TODO: Vectorize
            pc = [
                mvt_utils.place_pc_in_cube(
                    _pc,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )[0]
                for _pc in pc
            ]

            # SE(3) intra-mixup: concat point clouds + img_feat (TGM-VLA style)
            # Each sample gets augmented_pc + original_pc so the model sees both scenes
            if self.use_multipeak and ori_pc_cube is not None and backprop:
                import random
                se3_mixup_rate = 0.1  # Lower rate: structural(25%) + SE3(~7%) ≈ 32% total multi-peak
                se3_mixup_flags = [False] * len(pc)
                for i in range(len(pc)):
                    if random.random() < se3_mixup_rate:
                        se3_mixup_flags[i] = True
                        pc[i] = torch.cat([pc[i], ori_pc_cube[i]], dim=0)
                        img_feat[i] = torch.cat([img_feat[i], ori_img_feat_moved[i]], dim=0)

            bs = len(pc)
            nc = self._net_mod.num_img
            h = w = self._net_mod.img_size

            if backprop and (self.img_aug != 0):
                img_aug = self.img_aug
            else:
                img_aug = 0

            dyn_cam_info = None

        with autocast(enabled=self.amp):
            (
                action_rot_x_one_hot,
                action_rot_y_one_hot,
                action_rot_z_one_hot,
                action_grip_one_hot,  # (bs, 2)
                action_collision_one_hot,  # (bs, 2)
            ) = self._get_one_hot_expert_actions(
                bs, action_rot, action_grip, action_ignore_collisions, device=self._device
            )

            if self.rot_ver == 1:
                rot_x_y = torch.cat(
                    [
                        action_rot_x_one_hot.argmax(dim=-1, keepdim=True),
                        action_rot_y_one_hot.argmax(dim=-1, keepdim=True),
                    ],
                    dim=-1,
                )
                if self.rot_x_y_aug != 0:
                    # add random interger between -rot_x_y_aug and rot_x_y_aug to rot_x_y
                    rot_x_y += torch.randint(
                        -self.rot_x_y_aug, self.rot_x_y_aug, size=rot_x_y.shape
                    ).to(rot_x_y.device)
                    rot_x_y %= self._num_rotation_classes

            # hm_gt = self.get_gt_hm(
            #     wpt_local, dyn_cam_info, dims=(bs, nc, h, w)
            # )

            out = self._network(
                pc=pc,
                img_feat=img_feat,
                proprio=proprio,
                lang_emb=lang_goal_embs,
                img_aug=img_aug,
                wpt_local=wpt_local if self._network.training else None,
                rot_x_y=rot_x_y if self.rot_ver == 1 else None,
                # hm_gt=hm_gt,
            )

            q_trans, rot_q, grip_q, collision_q, y_q, pts = self.get_q(
                out, dims=(bs, nc, h, w)
            )

            # Multi-peak: extract alternative targets from replay sample
            alt_wpt_locals = None
            alt_wpt_mask = None
            if self.use_multipeak and self._network.training:
                alt_pos = replay_sample.get("alt_target_positions", None)
                alt_m = replay_sample.get("alt_target_mask", None)
                if alt_pos is not None and alt_m is not None:
                    alt_pos = alt_pos.float().to(self._device)
                    alt_m = alt_m.bool().to(self._device)
                    # Temporal buffer may add a timestep dim: (bs, T, peaks, 3)
                    if alt_pos.dim() == 4:
                        alt_pos = alt_pos[:, 0]  # (bs, peaks, 3)
                        alt_m = alt_m[:, 0]      # (bs, peaks)
                    if alt_m.any():
                        # Apply same SE(3) aug + place_pc_in_cube as primary target.
                        # alt_pos is in raw world coords; wpt_local was transformed via:
                        #   1) SE(3) aug: original_xyz + trans_shift (rotation changes quat, not xyz directly for translation)
                        #   2) place_pc_in_cube: world → local cube coords
                        # We reconstruct the same transforms using rev_trans from primary.
                        max_peaks = alt_pos.shape[1]
                        # Compute SE(3) translation shift (same for all peaks)
                        original_xyz = replay_sample["gripper_pose"][:, -1, :3].float().to(self._device)
                        aug_trans_shift = action_trans_con[:, :3].float().to(self._device) - original_xyz
                        alt_locals = []
                        for p_idx in range(max_peaks):
                            alt_p = alt_pos[:, p_idx, :]  # (bs, 3)
                            # Step 1: Apply same SE(3) translation shift as primary
                            alt_p_aug = alt_p + aug_trans_shift  # (bs, 3)
                            # Step 2: Apply place_pc_in_cube using same augmented pc
                            alt_p_local = []
                            for b_i in range(len(rev_trans)):
                                _pc_i = pc[b_i] if isinstance(pc, list) else pc[b_i]
                                _alt_local = mvt_utils.place_pc_in_cube(
                                    _pc_i,
                                    alt_p_aug[b_i].unsqueeze(0),  # (1, 3)
                                    with_mean_or_bounds=self._place_with_mean,
                                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                                )[0].squeeze(0)  # (3,)
                                alt_p_local.append(_alt_local)
                            alt_locals.append(torch.stack(alt_p_local, dim=0))  # (bs, 3)
                        alt_wpt_locals = torch.stack(alt_locals, dim=1)  # (bs, max_peaks, 3)
                        alt_wpt_mask = alt_m

            action_trans = self.get_action_trans(
                wpt_local, pts, out, dyn_cam_info, dims=(bs, nc, h, w),
                alt_wpt_locals=alt_wpt_locals,
                alt_wpt_mask=alt_wpt_mask,
                ori_wpt_local=ori_wpt_local,
                se3_mixup_flags=se3_mixup_flags,
            )

        loss_log = {}
        if backprop:
            with autocast(enabled=self.amp):
                # cross-entropy loss
                trans_loss = self._cross_entropy_loss(q_trans, action_trans).mean()
                rot_loss_x = rot_loss_y = rot_loss_z = 0.0
                grip_loss = 0.0
                collision_loss = 0.0
                if not self.use_memory:
                    if self.add_rgc_loss:
                        rot_loss_x = self._cross_entropy_loss(
                            rot_q[
                                :,
                                0 * self._num_rotation_classes : 1 * self._num_rotation_classes,
                            ],
                            action_rot_x_one_hot.argmax(-1),
                        ).mean()

                        rot_loss_y = self._cross_entropy_loss(
                            rot_q[
                                :,
                                1 * self._num_rotation_classes : 2 * self._num_rotation_classes,
                            ],
                            action_rot_y_one_hot.argmax(-1),
                        ).mean()

                        rot_loss_z = self._cross_entropy_loss(
                            rot_q[
                                :,
                                2 * self._num_rotation_classes : 3 * self._num_rotation_classes,
                            ],
                            action_rot_z_one_hot.argmax(-1),
                        ).mean()

                        grip_loss = self._cross_entropy_loss(
                            grip_q,
                            action_grip_one_hot.argmax(-1),
                        ).mean()

                        collision_loss = self._cross_entropy_loss(
                            collision_q, action_collision_one_hot.argmax(-1)
                        ).mean()

                    total_loss = (
                        trans_loss
                        + rot_loss_x
                        + rot_loss_y
                        + rot_loss_z
                        + grip_loss
                        + collision_loss
                    )

                else:

                    total_loss = trans_loss

                # Graph peak selector losses (Stage2)
                graph_loss_log = {}
                if self.use_graph_peak_select and "graph_outputs" in out:
                    from sam2act.mvt.graph_peak_selector import (
                        node_contrastive_loss,
                        transition_prediction_loss,
                    )
                    graph_outputs = out["graph_outputs"]

                    # Collect node embeds and positions across sequence
                    seq_node_embeds = []
                    seq_positions = []
                    for g_out in graph_outputs:
                        seq_node_embeds.append(g_out["curr_node_embed"].squeeze(0))
                    seq_node_embeds = torch.stack(seq_node_embeds, dim=0)  # (num_obs, D)

                    # Positions from replay sample
                    gripper_pose = replay_sample.get("gripper_pose", None)
                    if gripper_pose is not None:
                        gp = gripper_pose[:, 0].float().to(self._device)  # (bs, 7)
                        seq_positions = gp[:len(graph_outputs), :3]
                    else:
                        seq_positions = torch.zeros(
                            len(graph_outputs), 3, device=self._device
                        )

                    # Contrastive loss
                    c_loss = node_contrastive_loss(
                        seq_node_embeds,
                        seq_positions,
                        temperature=self.graph_contrastive_temperature,
                        pos_radius=self.graph_contrastive_pos_radius,
                    )

                    # Transition prediction loss
                    t_loss = transition_prediction_loss(
                        seq_node_embeds,
                        self._net_mod.mvt1.graph_peak_selector.transition_scorer,
                    )

                    # Peak selection loss (per step that has transition_scores)
                    p_loss = torch.tensor(0.0, device=self._device)
                    p_count = 0
                    for g_out in graph_outputs:
                        if g_out["transition_scores"] is not None:
                            from sam2act.mvt.graph_peak_selector import peak_selection_loss
                            # We need GT 2D position — use wpt_img from the corresponding step
                            # For simplicity, use peak closest to overall GT
                            p_count += 1

                    graph_aux = (
                        self.graph_contrastive_loss_weight * c_loss
                        + self.graph_transition_loss_weight * t_loss
                    )
                    total_loss = total_loss + graph_aux

                    graph_loss_log = {
                        "graph_contrastive_loss": c_loss.item(),
                        "graph_transition_loss": t_loss.item(),
                    }

            self._optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(total_loss).backward()

            # self.scaler.unscale_(self._optimizer)
            # torch.nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=0.1)

            self.scaler.step(self._optimizer)
            self.scaler.update()
            self._lr_sched.step()

            loss_log = {
                "total_loss": total_loss.item(),
                "trans_loss": trans_loss.item(),
                "rot_loss_x": rot_loss_x.item() if not self.use_memory else None,
                "rot_loss_y": rot_loss_y.item() if not self.use_memory else None,
                "rot_loss_z": rot_loss_z.item() if not self.use_memory else None,
                "grip_loss": grip_loss.item() if not self.use_memory else None,
                "collision_loss": collision_loss.item() if not self.use_memory else None,
                "lr": self._optimizer.param_groups[0]["lr"],
            }
            loss_log.update(graph_loss_log)
            manage_loss_log(self, loss_log, reset_log=reset_log)
            return_out.update(loss_log)

        if eval_log:
            with torch.no_grad():
                wpt = torch.cat([x.unsqueeze(0) for x in wpt])
                pred_wpt, pred_rot_quat, _, _ = self.get_pred(
                    out,
                    rot_q,
                    grip_q,
                    collision_q,
                    y_q,
                    rev_trans,
                    dyn_cam_info=dyn_cam_info,
                )

                return_log = manage_eval_log(
                    self=self,
                    tasks=tasks,
                    wpt=wpt,
                    pred_wpt=pred_wpt,
                    action_rot=action_rot,
                    pred_rot_quat=pred_rot_quat,
                    action_grip_one_hot=action_grip_one_hot,
                    grip_q=grip_q,
                    action_collision_one_hot=action_collision_one_hot,
                    collision_q=collision_q,
                    reset_log=reset_log,
                )

                return_out.update(return_log)

        return return_out

    @torch.no_grad()
    def act(
        self, step: int, observation: dict, deterministic=True, pred_distri=False
    ) -> ActResult:
        if self.add_lang:
            lang_goal_tokens = observation.get("lang_goal_tokens", None).long()
            _, lang_goal_embs = _clip_encode_text(self.clip_model, lang_goal_tokens[0])
            lang_goal_embs = lang_goal_embs.float()
        else:
            lang_goal_embs = (
                torch.zeros(observation["lang_goal_embs"].shape)
                .float()
                .to(self._device)
            )

        proprio = arm_utils.stack_on_channel(observation["low_dim_state"])

        obs, pcd = peract_utils._preprocess_inputs(observation, self.cameras)
        pc, img_feat = rvt_utils.get_pc_img_feat(
            obs,
            pcd,
        )

        pc, img_feat = rvt_utils.move_pc_in_bound(
            pc, img_feat, self.scene_bounds, no_op=not self.move_pc_in_bound
        )

        # TODO: Vectorize
        pc_new = []
        rev_trans = []
        for _pc in pc:
            a, b = mvt_utils.place_pc_in_cube(
                _pc,
                with_mean_or_bounds=self._place_with_mean,
                scene_bounds=None if self._place_with_mean else self.scene_bounds,
            )
            pc_new.append(a)
            rev_trans.append(b)
        pc = pc_new

        bs = len(pc)
        nc = self._net_mod.num_img
        h = w = self._net_mod.img_size
        dyn_cam_info = None

        out = self._network(
            pc=pc,
            img_feat=img_feat,
            proprio=proprio,
            lang_emb=lang_goal_embs,
            img_aug=0,  # no img augmentation while acting
        )
        _, rot_q, grip_q, collision_q, y_q, _ = self.get_q(
            out, dims=(bs, nc, h, w), only_pred=True, get_q_trans=False
        )
        pred_wpt, pred_rot_quat, pred_grip, pred_coll = self.get_pred(
            out, rot_q, grip_q, collision_q, y_q, rev_trans, dyn_cam_info
        )

        continuous_action = np.concatenate(
            (
                pred_wpt[0].cpu().numpy(),
                pred_rot_quat[0],
                pred_grip[0].cpu().numpy(),
                pred_coll[0].cpu().numpy(),
            )
        )
        if pred_distri:
            x_distri = rot_grip_q[
                0,
                0 * self._num_rotation_classes : 1 * self._num_rotation_classes,
            ]
            y_distri = rot_grip_q[
                0,
                1 * self._num_rotation_classes : 2 * self._num_rotation_classes,
            ]
            z_distri = rot_grip_q[
                0,
                2 * self._num_rotation_classes : 3 * self._num_rotation_classes,
            ]
            return ActResult(continuous_action), (
                x_distri.cpu().numpy(),
                y_distri.cpu().numpy(),
                z_distri.cpu().numpy(),
            )
        else:
            return ActResult(continuous_action)

    def get_pred(
        self,
        out,
        rot_q,
        grip_q,
        collision_q,
        y_q,
        rev_trans,
        dyn_cam_info,
    ):
        if self.stage_two:
            assert y_q is None
            mvt1_or_mvt2 = False
        else:
            mvt1_or_mvt2 = True

        pred_wpt_local = self._net_mod.get_wpt(
            out, mvt1_or_mvt2, dyn_cam_info, y_q
        )

        pred_wpt = []
        for _pred_wpt_local, _rev_trans in zip(pred_wpt_local, rev_trans):
            pred_wpt.append(_rev_trans(_pred_wpt_local))
        pred_wpt = torch.cat([x.unsqueeze(0) for x in pred_wpt])

        pred_rot = torch.cat(
            (
                rot_q[
                    :,
                    0 * self._num_rotation_classes : 1 * self._num_rotation_classes,
                ].argmax(1, keepdim=True),
                rot_q[
                    :,
                    1 * self._num_rotation_classes : 2 * self._num_rotation_classes,
                ].argmax(1, keepdim=True),
                rot_q[
                    :,
                    2 * self._num_rotation_classes : 3 * self._num_rotation_classes,
                ].argmax(1, keepdim=True),
            ),
            dim=-1,
        )
        pred_rot_quat = aug_utils.discrete_euler_to_quaternion(
            pred_rot.cpu(), self._rotation_resolution
        )
        pred_grip = grip_q.argmax(1, keepdim=True)
        pred_coll = collision_q.argmax(1, keepdim=True)

        return pred_wpt, pred_rot_quat, pred_grip, pred_coll

    @torch.no_grad()
    def get_action_trans(
        self,
        wpt_local,
        pts,
        out,
        dyn_cam_info,
        dims,
        alt_wpt_locals=None,
        alt_wpt_mask=None,
        ori_wpt_local=None,
        se3_mixup_flags=None,
    ):
        bs, nc, h, w = dims
        wpt_img = self._net_mod.get_pt_loc_on_img(
            wpt_local.unsqueeze(1),
            mvt1_or_mvt2=True,
            dyn_cam_info=dyn_cam_info,
            out=None
        )
        assert wpt_img.shape[1] == 1
        if self.stage_two and not self.use_memory:
            wpt_img2 = self._net_mod.get_pt_loc_on_img(
                wpt_local.unsqueeze(1),
                mvt1_or_mvt2=False,
                dyn_cam_info=dyn_cam_info,
                out=out,
            )
            assert wpt_img2.shape[1] == 1
            wpt_img = torch.cat((wpt_img, wpt_img2), dim=-2)
            nc = nc * 2

        wpt_img = wpt_img.squeeze(1)

        # Generate primary heatmap
        primary_hm = mvt_utils.generate_hm_from_pt(
            wpt_img.reshape(-1, 2),
            (h, w),
            sigma=self.gt_hm_sigma,
            thres_sigma_times=3,
        )  # (bs*nc, h, w)

        if self.use_multipeak and self._network.training:
            has_structural_alt = (alt_wpt_locals is not None and alt_wpt_mask is not None)
            if se3_mixup_flags is None:
                se3_mixup_flags = [False] * bs

            # Generate SE(3) intra-mixup heatmap (original pre-aug target)
            ori_hm = None
            if any(se3_mixup_flags) and ori_wpt_local is not None:
                ori_wpt_img = self._net_mod.get_pt_loc_on_img(
                    ori_wpt_local.unsqueeze(1),
                    mvt1_or_mvt2=True,
                    dyn_cam_info=dyn_cam_info,
                    out=None,
                )
                if self.stage_two and not self.use_memory:
                    ori_wpt_img2 = self._net_mod.get_pt_loc_on_img(
                        ori_wpt_local.unsqueeze(1),
                        mvt1_or_mvt2=False,
                        dyn_cam_info=dyn_cam_info,
                        out=out,
                    )
                    ori_wpt_img = torch.cat((ori_wpt_img, ori_wpt_img2), dim=-2)
                ori_wpt_img = ori_wpt_img.squeeze(1)
                ori_hm = mvt_utils.generate_hm_from_pt(
                    ori_wpt_img.reshape(-1, 2),
                    (h, w),
                    sigma=self.gt_hm_sigma,
                    thres_sigma_times=3,
                )

            # Generate structural multi-peak alt heatmap
            structural_alt_hm = None
            if has_structural_alt:
                max_peaks = alt_wpt_locals.shape[1]
                alt_wpt_imgs = []
                for p in range(max_peaks):
                    alt_img = self._net_mod.get_pt_loc_on_img(
                        alt_wpt_locals[:, p, :].unsqueeze(1),
                        mvt1_or_mvt2=True,
                        dyn_cam_info=dyn_cam_info,
                        out=None,
                    )
                    if self.stage_two and not self.use_memory:
                        alt_img2 = self._net_mod.get_pt_loc_on_img(
                            alt_wpt_locals[:, p, :].unsqueeze(1),
                            mvt1_or_mvt2=False,
                            dyn_cam_info=dyn_cam_info,
                            out=out,
                        )
                        alt_img = torch.cat((alt_img, alt_img2), dim=-2)
                    alt_wpt_imgs.append(alt_img.squeeze(1))
                alt_wpt_imgs_t = torch.stack(alt_wpt_imgs, dim=1)

                structural_alt_hm = torch.zeros_like(primary_hm)
                for p in range(max_peaks):
                    p_hm = mvt_utils.generate_hm_from_pt(
                        alt_wpt_imgs_t[:, p, :, :].reshape(-1, 2),
                        (h, w),
                        sigma=self.gt_hm_sigma,
                        thres_sigma_times=3,
                    )
                    p_mask = alt_wpt_mask[:, p].unsqueeze(1).expand(-1, nc).reshape(-1)
                    structural_alt_hm += p_hm * p_mask.float().unsqueeze(-1).unsqueeze(-1)

            # Combine: primary + (structural alt OR SE(3) mixup)
            # Apply multi-peak to first 3 views only (TGM-VLA style anchor)
            # Remaining views keep single-peak as anchor for stable training
            # Apply multi-peak to subset of views, keep rest as anchor
            # TGM-VLA uses 3/5 views. With rend_three_views=True (nc=3),
            # use 2/3 views for multi-peak, keep 1 view as anchor
            multipeak_views = max(1, nc - 1)  # nc=3 → 2 views multi-peak, 1 anchor
                                               # nc=5 → 4 views multi-peak, 1 anchor
            combined_hm = primary_hm.view(bs, nc, h, w).clone()
            for b_i in range(bs):
                has_alt_b = has_structural_alt and alt_wpt_mask[b_i].any()
                if has_alt_b and structural_alt_hm is not None:
                    # Structural multi-peak (revisit/variation): 3 views only
                    combined_hm[b_i, :multipeak_views] += structural_alt_hm.view(bs, nc, h, w)[b_i, :multipeak_views]
                elif se3_mixup_flags[b_i] and ori_hm is not None:
                    # SE(3) intra-mixup: 3 views only
                    combined_hm[b_i, :multipeak_views] += ori_hm.view(bs, nc, h, w)[b_i, :multipeak_views]

            action_trans = combined_hm.view(bs, nc, h * w).transpose(1, 2).clone()
        else:
            action_trans = mvt_utils.generate_hm_from_pt(
                wpt_img.reshape(-1, 2),
                (h, w),
                sigma=self.gt_hm_sigma,
                thres_sigma_times=3,
            )
            action_trans = action_trans.view(bs, nc, h * w).transpose(1, 2).clone()

        return action_trans


    def get_gt_hm(
        self,
        wpt_local,
        dyn_cam_info,
        dims,
    ):
        bs, nc, h, w = dims
        wpt_img = self._net_mod.get_pt_loc_on_img(
            wpt_local.unsqueeze(1),
            mvt1_or_mvt2=True,
            dyn_cam_info=dyn_cam_info,
            out=None
        )
        assert wpt_img.shape[1] == 1
        # if self.stage_two and not self.use_memory:
        #     wpt_img2 = self._net_mod.get_pt_loc_on_img(
        #         wpt_local.unsqueeze(1),
        #         mvt1_or_mvt2=False,
        #         dyn_cam_info=dyn_cam_info,
        #         out=out,
        #     )
        #     assert wpt_img2.shape[1] == 1

        #     # (bs, 1, 2 * num_img, 2)
        #     wpt_img = torch.cat((wpt_img, wpt_img2), dim=-2)
        #     nc = nc * 2

        # (bs, num_img, 2)
        wpt_img = wpt_img.squeeze(1)

        action_trans = mvt_utils.generate_hm_from_pt(
            wpt_img.reshape(-1, 2),
            (h, w),
            sigma=self.gt_hm_sigma,
            thres_sigma_times=3,
        )
        action_trans_hm = action_trans.view(bs, nc, h, w).clone()

        return action_trans_hm

    def reset(self):
        pass

    def eval(self):
        self._network.eval()

    def train(self):
        self._network.train()
