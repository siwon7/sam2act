# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Utility function for Our Agent
"""
import pdb
import argparse
import sys
import signal
from datetime import datetime
from collections import Counter

import torch
# from torch.utils.tensorboard import SummaryWriter

from torch.nn.parallel import DistributedDataParallel as DDP

import sam2act.utils.peract_utils as peract_utils
from sam2act.models.peract_official import PreprocessAgent2


def _load_state_dict_with_report(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"WARNING: missing checkpoint keys ({len(missing_keys)}): {missing_keys[:20]}")
        print(
            "WARNING: missing checkpoint key prefixes: "
            f"{_summarize_key_prefixes(missing_keys)}"
        )
    if unexpected_keys:
        print(
            f"WARNING: unexpected checkpoint keys ({len(unexpected_keys)}): "
            f"{unexpected_keys[:20]}"
        )
        print(
            "WARNING: unexpected checkpoint key prefixes: "
            f"{_summarize_key_prefixes(unexpected_keys)}"
        )
    return missing_keys, unexpected_keys


def _adapt_state_dict_for_model(model, state_dict):
    adapted = dict(state_dict)
    target_state = model.state_dict()
    for key, value in list(adapted.items()):
        if key not in target_state:
            continue
        target = target_state[key]
        if not isinstance(value, torch.Tensor) or not isinstance(target, torch.Tensor):
            continue
        if value.shape == target.shape:
            continue

        # Allow longer temporal memory windows by copying the existing prefix and
        # leaving the newly added positions at their initialized values.
        if key.endswith("maskmem_tpos_enc"):
            if (
                value.ndim == target.ndim == 4
                and value.shape[1:] == target.shape[1:]
            ):
                patched = target.clone()
                copy_n = min(value.shape[0], target.shape[0])
                patched[:copy_n] = value[:copy_n]
                adapted[key] = patched
                print(
                    f"WARNING: adapted {key} from checkpoint shape {tuple(value.shape)} "
                    f"to model shape {tuple(target.shape)} by prefix copy ({copy_n})."
                )
                continue
    return adapted


def _summarize_key_prefixes(keys, depth=3, topk=8):
    if not keys:
        return "{}"
    counts = Counter(".".join(key.split(".")[:depth]) for key in keys)
    most_common = ", ".join(
        f"{prefix}: {count}" for prefix, count in counts.most_common(topk)
    )
    return "{" + most_common + "}"


def get_pc_img_feat(obs, pcd, bounds=None):
    """
    preprocess the data in the peract to our framework
    """
    # obs, pcd = peract_utils._preprocess_inputs(batch)
    bs = obs[0][0].shape[0]
    # concatenating the points from all the cameras
    # (bs, num_points, 3)
    pc = torch.cat([p.permute(0, 2, 3, 1).reshape(bs, -1, 3) for p in pcd], 1)
    _img_feat = [o[0] for o in obs]
    img_dim = _img_feat[0].shape[1]
    # (bs, num_points, 3)
    img_feat = torch.cat(
        [p.permute(0, 2, 3, 1).reshape(bs, -1, img_dim) for p in _img_feat], 1
    )

    img_feat = (img_feat + 1) / 2

    # x_min, y_min, z_min, x_max, y_max, z_max = bounds
    # inv_pnt = (
    #     (pc[:, :, 0] < x_min)
    #     | (pc[:, :, 0] > x_max)
    #     | (pc[:, :, 1] < y_min)
    #     | (pc[:, :, 1] > y_max)
    #     | (pc[:, :, 2] < z_min)
    #     | (pc[:, :, 2] > z_max)
    # )

    # # TODO: move from a list to a better batched version
    # pc = [pc[i, ~_inv_pnt] for i, _inv_pnt in enumerate(inv_pnt)]
    # img_feat = [img_feat[i, ~_inv_pnt] for i, _inv_pnt in enumerate(inv_pnt)]

    return pc, img_feat


def move_pc_in_bound(pc, img_feat, bounds, no_op=False):
    """
    :param no_op: no operation
    """
    if no_op:
        return pc, img_feat

    x_min, y_min, z_min, x_max, y_max, z_max = bounds
    inv_pnt = (
        (pc[:, :, 0] < x_min)
        | (pc[:, :, 0] > x_max)
        | (pc[:, :, 1] < y_min)
        | (pc[:, :, 1] > y_max)
        | (pc[:, :, 2] < z_min)
        | (pc[:, :, 2] > z_max)
        | torch.isnan(pc[:, :, 0])
        | torch.isnan(pc[:, :, 1])
        | torch.isnan(pc[:, :, 2])
    )

    # TODO: move from a list to a better batched version
    pc = [pc[i, ~_inv_pnt] for i, _inv_pnt in enumerate(inv_pnt)]
    img_feat = [img_feat[i, ~_inv_pnt] for i, _inv_pnt in enumerate(inv_pnt)]
    return pc, img_feat


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TensorboardManager:
    def __init__(self, path):
        self.writer = SummaryWriter(path)

    def update(self, split, step, vals):
        for k, v in vals.items():
            if "image" in k:
                for i, x in enumerate(v):
                    self.writer.add_image(f"{split}_{step}", x, i)
            elif "hist" in k:
                if isinstance(v, list):
                    self.writer.add_histogram(k, v, step)
                elif isinstance(v, dict):
                    hist_id = {}
                    for i, idx in enumerate(sorted(v.keys())):
                        self.writer.add_histogram(f"{split}_{k}_{step}", v[idx], i)
                        hist_id[i] = idx
                    self.writer.add_text(f"{split}_{k}_{step}_id", f"{hist_id}")
                else:
                    assert False
            else:
                self.writer.add_scalar("%s_%s" % (split, k), v, step)

    def close(self):
        self.writer.flush()
        self.writer.close()


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


def short_name(cfg_opts):
    SHORT_FORMS = {
        "peract": "PA",
        "sample_distribution_mode": "SDM",
        "optimizer_type": "OPT",
        "lr_cos_dec": "LCD",
        "num_workers": "NW",
        "True": "T",
        "False": "F",
        "pe_fix": "pf",
        "transform_augmentation_rpy": "tar",
        "lambda_weight_l2": "l2",
        "resume": "RES",
        "inp_pre_pro": "IPP",
        "inp_pre_con": "IPC",
        "cvx_up": "CU",
        "stage_two": "ST",
        "feat_ver": "FV",
        "lamb": "L",
        "img_size": "IS",
        "img_patch_size": "IPS",
        "rlbench": "RLB",
        "move_pc_in_bound": "MPIB",
        "rend": "R",
        "xops": "X",
        "warmup_steps": "WS",
        "epochs": "E",
        "amp": "A",
    }

    if "resume" in cfg_opts:
        cfg_opts = cfg_opts.split(" ")
        res_idx = cfg_opts.index("resume")
        cfg_opts.pop(res_idx + 1)
        cfg_opts = " ".join(cfg_opts)

    cfg_opts = cfg_opts.replace(" ", "_")
    cfg_opts = cfg_opts.replace("/", "_")
    cfg_opts = cfg_opts.replace("[", "")
    cfg_opts = cfg_opts.replace("]", "")
    cfg_opts = cfg_opts.replace("..", "")
    for a, b in SHORT_FORMS.items():
        cfg_opts = cfg_opts.replace(a, b)

    return cfg_opts


def get_num_feat(cfg):
    num_feat = cfg.num_rotation_classes * 3
    # 2 for grip, 2 for collision
    num_feat += 4
    return num_feat


def get_eval_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tasks", type=str, nargs="+", default=["insert_onto_square_peg"]
    )
    parser.add_argument("--model-folder", type=str, default=None)
    parser.add_argument("--eval-datafolder", type=str, default="./data/val/")
    parser.add_argument(
        "--start-episode",
        type=int,
        default=0,
        help="start to evaluate from which episode",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="how many episodes to be evaluated for each task",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=25,
        help="maximum control steps allowed for each episode",
    )
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--ground-truth", action="store_true", default=False)
    parser.add_argument("--exp_cfg_path", type=str, default=None)
    parser.add_argument("--mvt_cfg_path", type=str, default=None)
    parser.add_argument(
        "--mvt_cfg_opts",
        type=str,
        default="",
        help='Override mvt config via "<k1> <v1> <k2> <v2> ..."',
    )
    parser.add_argument("--peract_official", action="store_true")
    parser.add_argument(
        "--peract_model_dir",
        type=str,
        default="runs/peract_official/seed0/weights/600000",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--log-name", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--use-input-place-with-mean", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--skip", action="store_true")

    return parser


RLBENCH_TASKS = [
    "put_item_in_drawer",
    "reach_and_drag",
    "turn_tap",
    "slide_block_to_color_target",
    "open_drawer",
    "put_groceries_in_cupboard",
    "place_shape_in_shape_sorter",
    "put_money_in_safe",
    "push_buttons",
    "close_jar",
    "stack_blocks",
    "place_cups",
    "place_wine_at_rack_location",
    "light_bulb_in",
    "sweep_to_dustpan_of_size",
    "insert_onto_square_peg",
    "meat_off_grill",
    "stack_cups",
]


def load_agent(agent_path, agent=None, only_epoch=False):
    if isinstance(agent, PreprocessAgent2):
        assert not only_epoch
        agent._pose_agent.load_weights(agent_path)
        return 0

    checkpoint = torch.load(agent_path, map_location="cpu")
    epoch = checkpoint["epoch"]

    if not only_epoch:
        if hasattr(agent, "_q"):
            model = agent._q
        elif hasattr(agent, "_network"):
            model = agent._network
        optimizer = agent._optimizer
        lr_sched = agent._lr_sched

        if isinstance(model, DDP):
            model = model.module

        try:
            _load_state_dict_with_report(
                model, _adapt_state_dict_for_model(model, checkpoint["model_state"])
            )
        except RuntimeError:
            print(
                "WARNING: loading states in mvt1. "
                "Be cautious if you are using a two stage network."
            )
            _load_state_dict_with_report(
                model.mvt1,
                _adapt_state_dict_for_model(model.mvt1, checkpoint["model_state"]),
            )

        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        else:
            print(
                "WARNING: No optimizer_state in checkpoint" "KNOW WHAT YOU ARE DOING!!"
            )

        if "lr_sched_state" in checkpoint:
            lr_sched.load_state_dict(checkpoint["lr_sched_state"])
        else:
            print(
                "WARNING: No lr_sched_state in checkpoint" "KNOW WHAT YOU ARE DOING!!"
            )

    return epoch


def load_agent_only_model(agent_path, agent=None, only_epoch=False):
    if isinstance(agent, PreprocessAgent2):
        assert not only_epoch
        agent._pose_agent.load_weights(agent_path)
        return 0

    checkpoint = torch.load(agent_path, map_location="cpu")
    epoch = checkpoint["epoch"]

    if not only_epoch:
        if hasattr(agent, "_q"):
            model = agent._q
        elif hasattr(agent, "_network"):
            model = agent._network
        optimizer = agent._optimizer
        lr_sched = agent._lr_sched

        if isinstance(model, DDP):
            model = model.module

        try:
            _load_state_dict_with_report(
                model, _adapt_state_dict_for_model(model, checkpoint["model_state"])
            )
        except RuntimeError:
            print(
                "WARNING: loading states in mvt1. "
                "Be cautious if you are using a two stage network."
            )
            _load_state_dict_with_report(
                model.mvt1,
                _adapt_state_dict_for_model(model.mvt1, checkpoint["model_state"]),
            )

    return epoch


def load_agent_only_model_exclude(agent_path, agent=None, only_epoch=False, exclude_keys=None):
    if isinstance(agent, PreprocessAgent2):
        assert not only_epoch
        agent._pose_agent.load_weights(agent_path)
        return 0

    checkpoint = torch.load(agent_path, map_location="cpu")
    epoch = checkpoint["epoch"]

    if not only_epoch:
        if hasattr(agent, "_q"):
            model = agent._q
        elif hasattr(agent, "_network"):
            model = agent._network
        optimizer = agent._optimizer
        lr_sched = agent._lr_sched

        if isinstance(model, DDP):
            model = model.module


        sd = checkpoint["model_state"]
        if exclude_keys is not None:
            for exclude_key in exclude_keys:
                keys_to_delete = [key for key in sd if exclude_key in key]
                for key in keys_to_delete:
                    del sd[key]

        try:
            _load_state_dict_with_report(model, _adapt_state_dict_for_model(model, sd))
        except RuntimeError:
            print(
                "WARNING: loading states in mvt1. "
                "Be cautious if you are using a two stage network."
            )
            _load_state_dict_with_report(
                model.mvt1,
                _adapt_state_dict_for_model(model.mvt1, checkpoint["model_state"]),
            )

    return epoch
