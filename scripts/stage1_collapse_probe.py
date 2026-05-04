#!/usr/bin/env python3
"""Probe Stage1 heatmap collapse against GT and intra-episode alternatives.

This is stricter than a final-action audit: it captures the Stage1 waypoint
that is used to crop Stage2, then checks whether that Stage1 top-1 points to
the GT next target, an alternative target, a tie, or an off-target direction.
It also samples the Stage1 heatmap score at the projected GT/alt target pixels.
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import site
import sys
from datetime import datetime
from pathlib import Path


def _setup_paths() -> Path:
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")

    root = Path(
        os.environ.get(
            "SAM2ACT_ROOT",
            "/home/cv11/project/siwon/sam2act_stage2_oracle_ceiling/sam2act",
        )
    )
    try:
        user_site = site.getusersitepackages()
    except Exception:
        user_site = ""
    sys.path[:] = [
        p
        for p in sys.path
        if p and p != user_site and "/.local/lib/python" not in p
    ]

    pyrep = root / "libs/PyRep"
    pyrep_so = pyrep / "pyrep/backend/_sim_cffi.cpython-310-x86_64-linux-gnu.so"
    if not pyrep_so.exists():
        pyrep = Path(
            "/home/cv11/project/siwon/.sync_unpack_20260426/"
            "sam2act_dirty/sam2act/sam2act/libs/PyRep"
        )

    sys.path[:0] = [
        str(root.parent),
        str(root),
        str(pyrep),
        str(root / "libs/RLBench"),
        str(root / "libs/YARR"),
        str(root / "libs/peract"),
    ]

    coppelia = os.environ.get(
        "COPPELIASIM_ROOT",
        "/home/cv11/project/siwon/coppeliasim/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04",
    )
    os.environ["COPPELIASIM_ROOT"] = coppelia
    os.environ["LD_LIBRARY_PATH"] = f"{coppelia}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    os.chdir(root)
    return root


SAM2ACT_ROOT = _setup_paths()

import clip  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from eval import load_agent  # noqa: E402
from libs.peract.helpers.demo_loading_utils import keypoint_discovery  # noqa: E402
from libs.peract.helpers.utils import extract_obs  # noqa: E402
from libs.peract_colab.peract_colab.rlbench.utils import get_stored_demo  # noqa: E402
from mvt import utils as mvt_utils  # noqa: E402
from mvt.multipeak_utils import collect_alt_targets  # noqa: E402
from utils import peract_utils, rvt_utils  # noqa: E402
from utils.peract_utils import CAMERAS  # noqa: E402


def _ratio_status(ratio: float) -> str:
    if ratio <= 3:
        return "alive"
    if ratio <= 20:
        return "weak"
    if ratio <= 100:
        return "faint"
    return "collapsed"


def _median(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _fmt(values: list[float], digits: int = 3) -> str:
    if any(np.isposinf(v) for v in values):
        finite_vals = [v for v in values if np.isfinite(v)]
        if not finite_vals:
            return "inf-inf-inf"
        fmt = f"%.{digits}f-%.{digits}f-inf"
        return fmt % (min(finite_vals), _median(finite_vals))
    vals = [v for v in values if np.isfinite(v)]
    if not vals:
        return "-"
    fmt = f"%.{digits}f-%.{digits}f-%.{digits}f"
    return fmt % (min(vals), _median(vals), max(vals))


def _short(value: float) -> str:
    if not np.isfinite(value):
        return "-"
    if abs(value) >= 1000:
        return f"{value:.0f}"
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an < 1e-6 or bn < 1e-6:
        return float("nan")
    return float(np.dot(a, b) / (an * bn))


def _classify_direction(
    current: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    alts: np.ndarray,
    cos_margin: float,
    off_cos: float,
) -> tuple[str, float, float, float, float, str]:
    pred_vec = pred - current
    gt_vec = gt - current
    cos_gt = _cosine(pred_vec, gt_vec)

    alt_cos_values = [_cosine(pred_vec, alt - current) for alt in alts]
    finite_alt_cos = [x for x in alt_cos_values if np.isfinite(x)]
    best_alt_cos = max(finite_alt_cos) if finite_alt_cos else float("nan")
    best_alt_idx = int(np.nanargmax(np.asarray(alt_cos_values))) if finite_alt_cos else -1

    targets = [gt] + [alt for alt in alts]
    target_names = ["gt"] + [f"alt{i}" for i in range(len(alts))]
    dists = [float(np.linalg.norm(pred - target)) for target in targets]
    nearest_idx = int(np.argmin(dists))
    nearest_target = target_names[nearest_idx]

    if not np.isfinite(cos_gt):
        direction_pick = "off"
    elif not finite_alt_cos:
        direction_pick = "gt" if cos_gt >= off_cos else "off"
    elif cos_gt >= best_alt_cos + cos_margin and cos_gt >= off_cos:
        direction_pick = "gt"
    elif best_alt_cos >= cos_gt + cos_margin and best_alt_cos >= off_cos:
        direction_pick = f"alt{best_alt_idx}"
    elif max(cos_gt, best_alt_cos) >= off_cos:
        direction_pick = "tie"
    else:
        direction_pick = "off"

    return (
        direction_pick,
        cos_gt,
        best_alt_cos,
        float(np.linalg.norm(pred - gt)),
        dists[nearest_idx],
        nearest_target,
    )


def _stage1_hm(trans: torch.Tensor) -> torch.Tensor:
    if trans.dim() == 5:
        logits = trans[0, :, 0]
    elif trans.dim() == 4:
        logits = trans[0]
    elif trans.dim() == 3:
        # Fallback for (bs, hw, nc)-style q tensors.
        bs, hw, nc = trans.shape
        side = int(hw**0.5)
        if side * side != hw:
            raise RuntimeError(f"unexpected trans shape: {tuple(trans.shape)}")
        logits = trans[0].transpose(0, 1).reshape(nc, side, side)
    else:
        raise RuntimeError(f"unexpected trans shape: {tuple(trans.shape)}")
    nc, h, w = logits.shape
    return F.softmax(logits.reshape(nc, h * w), dim=1).reshape(nc, h, w).cpu()


def _stage1_topk_world(
    agent,
    hm: torch.Tensor,
    rev_trans,
    device: str,
    topk_3d: int,
    nms_dist: float,
) -> tuple[np.ndarray, list[float]]:
    """Return Stage1 top-K candidate crop centers in world coordinates."""
    if topk_3d <= 0:
        return np.zeros((0, 3), dtype=np.float32), []

    renderer = agent._net_mod.mvt1.renderer
    hm_device = hm.to(device).unsqueeze(0)
    with torch.no_grad():
        try:
            topk_local = renderer.get_max_3d_frm_hm_cube(
                hm_device,
                fix_cam=True,
                dyn_cam_info=None,
                topk=topk_3d,
                non_max_sup=True,
                non_max_sup_dist=nms_dist,
            )
        except TypeError:
            topk_local = renderer.get_max_3d_frm_hm_cube(
                hm_device,
                fix_cam=True,
                dyn_cam_info=None,
            )
            if topk_local.dim() == 2:
                topk_local = topk_local.unsqueeze(0)

        if topk_local.dim() == 2:
            topk_local = topk_local.unsqueeze(0)
        topk_local = topk_local[0].to(device)
        topk_world = rev_trans(topk_local).detach().cpu().numpy()

    score_rows = _sample_target_scores(agent, hm, topk_local)
    scores = [row["score_mean"] for row in score_rows]
    return topk_world, scores


def _view_peak_stats(hm: torch.Tensor, topk: int = 5) -> dict:
    ratios: list[float] = []
    p1s: list[float] = []
    p2s: list[float] = []
    peaks = []
    entropies: list[float] = []
    for view_idx in range(hm.shape[0]):
        view = hm[view_idx]
        pooled = F.max_pool2d(view.unsqueeze(0).unsqueeze(0), 5, 1, 2)
        nms = view.unsqueeze(0).unsqueeze(0) * (
            view.unsqueeze(0).unsqueeze(0) == pooled
        ).float()
        vals, idxs = torch.topk(nms.reshape(-1), min(topk, nms.numel()))
        p1 = float(vals[0])
        p2 = float(vals[1]) if len(vals) > 1 else 0.0
        ratio = p1 / max(p2, 1e-10)
        h, w = view.shape
        row_peaks = []
        for val, idx in zip(vals, idxs):
            r, c = divmod(int(idx), w)
            row_peaks.append(f"{view_idx}:{r},{c}:{float(val):.6g}")
        ratios.append(ratio)
        p1s.append(p1)
        p2s.append(p2)
        peaks.append(";".join(row_peaks))
        flat = view.reshape(-1)
        ent = -float(torch.sum(flat.clamp_min(1e-12) * torch.log(flat.clamp_min(1e-12))))
        entropies.append(ent)
    return {
        "ratios": ratios,
        "p1s": p1s,
        "p2s": p2s,
        "peaks": "|".join(peaks),
        "entropy": entropies,
    }


def _localize_points(agent, obs_torch: dict, points_world: np.ndarray, device: str):
    obs, pcd = peract_utils._preprocess_inputs(obs_torch, agent.cameras)
    pc, img_feat = rvt_utils.get_pc_img_feat(obs, pcd)
    pc, img_feat = rvt_utils.move_pc_in_bound(
        pc,
        img_feat,
        agent.scene_bounds,
        no_op=not agent.move_pc_in_bound,
    )
    points = torch.as_tensor(points_world, dtype=torch.float32, device=device)
    local, rev_trans = mvt_utils.place_pc_in_cube(
        pc[0],
        points,
        with_mean_or_bounds=agent._place_with_mean,
        scene_bounds=None if agent._place_with_mean else agent.scene_bounds,
    )
    return local, rev_trans


def _sample_target_scores(agent, hm: torch.Tensor, target_locals: torch.Tensor) -> list[dict]:
    if target_locals.numel() == 0:
        return []
    with torch.no_grad():
        pt_img = agent._net_mod.get_pt_loc_on_img(
            target_locals.unsqueeze(0),
            mvt1_or_mvt2=True,
            dyn_cam_info=None,
            out=None,
        )[0].detach().cpu()

    nc, h, w = hm.shape
    rows = []
    for target_idx in range(pt_img.shape[0]):
        scores = []
        score_over_p1 = []
        ranks = []
        rank_pct = []
        in_frame = 0
        coords = []
        for view_idx in range(nc):
            x_f = float(pt_img[target_idx, view_idx, 0])
            y_f = float(pt_img[target_idx, view_idx, 1])
            inside = 0.0 <= x_f < w and 0.0 <= y_f < h
            in_frame += int(inside)
            x = int(round(min(max(x_f, 0.0), w - 1.0)))
            y = int(round(min(max(y_f, 0.0), h - 1.0)))
            score = float(hm[view_idx, y, x])
            p1 = float(hm[view_idx].max())
            rank = int(torch.sum(hm[view_idx].reshape(-1) > score).item()) + 1
            scores.append(score)
            score_over_p1.append(score / max(p1, 1e-12))
            ranks.append(rank)
            rank_pct.append(rank / float(h * w))
            coords.append(f"{view_idx}:{x},{y}")
        rows.append(
            {
                "score_mean": float(np.mean(scores)),
                "score_max": float(np.max(scores)),
                "score_over_p1_mean": float(np.mean(score_over_p1)),
                "rank_median": float(np.median(ranks)),
                "rank_pct_median": float(np.median(rank_pct)),
                "in_frame_frac": in_frame / float(nc),
                "coords": "|".join(coords),
            }
        )
    return rows


def _score_pick(gt_score: float, alt_score: float, margin: float) -> str:
    if not np.isfinite(alt_score):
        return "gt_only"
    if gt_score >= alt_score * margin:
        return "gt"
    if alt_score >= gt_score * margin:
        return "alt"
    return "tie"


def _cause_hint(row: dict) -> str:
    if row["direction_pick"] == "no_next":
        return "terminal"
    if row["is_correct"]:
        if row["status"] == "collapsed":
            return "benign_gt_collapse"
        return "correct_noncollapsed"
    if row["direction_pick"] == "gt" and row["stage1_gt_err_m"] > 0.05:
        return "gt_direction_but_far_crop"
    if row["gt_score_over_p1_mean"] < 0.1:
        if row["label_mp"] and row["score_pick"] == "alt":
            return "heatmap_favors_alt_candidate"
        return "gt_score_suppressed"
    if row["label_mp"]:
        if row["score_pick"] == "alt":
            return "heatmap_favors_alt_candidate"
        if row["score_pick"] == "tie":
            return "gt_alt_scores_tied_but_top1_wrong"
        return "top1_wrong_despite_gt_score"
    return "single_step_direction_wrong"


def _new_agg(task: str, kf: int) -> dict:
    return {
        "task": task,
        "kf": kf,
        "episodes": 0,
        "valid": 0,
        "label_hits": 0,
        "top1_gt": 0,
        "top1_alt": 0,
        "top1_tie": 0,
        "top1_off": 0,
        "top1_correct": 0,
        "top1_wrong": 0,
        "collapsed_correct": 0,
        "collapsed_wrong": 0,
        "collapsed_total": 0,
        "score_pick_gt": 0,
        "score_pick_alt": 0,
        "score_pick_tie": 0,
        "score_pick_gt_only": 0,
        "ratios": [],
        "gt_err": [],
        "cos_gt": [],
        "gt_score_over_p1": [],
        "alt_over_gt": [],
        "gt_rank_pct": [],
        "entropy": [],
        "topk_gt_hit": 0,
        "topk_miss": 0,
        "top1_fail_topk_hit": 0,
        "topk_gt_err": [],
        "topk_best_rank": [],
        "cause_counts": {},
    }


def analyze(args) -> tuple[list[dict], list[dict]]:
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device}", flush=True)
    agent = load_agent(
        model_path=str(args.model),
        eval_log_dir="/tmp/sam2act_stage1_collapse_probe",
        device=args.device,
    )
    agent.load_clip()

    captured = {}
    orig_get_wpt = agent._net_mod.get_wpt

    def hooked_get_wpt(out, mvt1_or_mvt2, dyn_cam_info, y_q=None):
        wpt = orig_get_wpt(out, mvt1_or_mvt2, dyn_cam_info, y_q)
        if bool(mvt1_or_mvt2) and "stage1_trans" not in captured:
            captured["stage1_trans"] = out["trans"].detach().cpu()
            captured["stage1_wpt_local"] = wpt.detach().cpu()
        return wpt

    agent._net_mod.get_wpt = hooked_get_wpt

    detail_rows: list[dict] = []
    per_key: dict[tuple[str, int], dict] = {}

    for task in [x.strip() for x in args.tasks.split(",") if x.strip()]:
        ep_base = args.data_root / task / "all_variations/episodes"
        print(f"Task {task}: episodes 0..{args.episodes - 1}", flush=True)

        for ep_idx in range(args.episodes):
            demo = get_stored_demo(str(ep_base), ep_idx)
            keyframes = keypoint_discovery(demo)
            num_kf = len(keyframes)
            poses = np.asarray(
                [demo[keyframes[k]].gripper_pose for k in range(num_kf)],
                dtype=np.float32,
            )
            opens = np.asarray(
                [demo[keyframes[k]].gripper_open for k in range(num_kf)],
                dtype=np.float32,
            )
            alt_pos, alt_mask = collect_alt_targets(
                poses,
                opens,
                num_kf,
                max_peaks=args.max_peaks,
                spatial_only=True,
            )

            desc_path = ep_base / f"episode{ep_idx}" / "variation_descriptions.pkl"
            with desc_path.open("rb") as f:
                desc = pickle.load(f)[0]

            agent.reset()
            for k in range(num_kf):
                agg = per_key.setdefault((task, k), _new_agg(task, k))
                agg["episodes"] += 1
                label_mp = bool(alt_mask[k].any())
                agg["label_hits"] += int(label_mp)

                obs = demo[keyframes[k]]
                obs_dict = extract_obs(obs, CAMERAS, t=k, prev_action=None, episode_length=num_kf)
                obs_dict["lang_goal_tokens"] = clip.tokenize([desc]).numpy()[0]
                obs_torch = {
                    key: torch.tensor(np.asarray([[val]]), device=device)
                    for key, val in obs_dict.items()
                }

                valid_alts = alt_pos[k][alt_mask[k]]
                target_worlds = []
                target_names = []
                if k + 1 < num_kf:
                    target_worlds.append(poses[k + 1, :3])
                    target_names.append("gt")
                    for alt_idx, alt in enumerate(valid_alts):
                        target_worlds.append(alt)
                        target_names.append(f"alt{alt_idx}")

                if target_worlds:
                    target_locals, rev_trans = _localize_points(
                        agent,
                        obs_torch,
                        np.asarray(target_worlds, dtype=np.float32),
                        device,
                    )
                else:
                    target_locals = torch.empty((0, 3), device=device)
                    _, rev_trans = _localize_points(
                        agent,
                        obs_torch,
                        poses[k : k + 1, :3],
                        device,
                    )

                captured.clear()
                with torch.no_grad():
                    act_result = agent.act(k, obs_torch, deterministic=True)
                if "stage1_trans" not in captured or "stage1_wpt_local" not in captured:
                    raise RuntimeError("failed to capture stage1 trans/wpt")

                hm = _stage1_hm(captured["stage1_trans"])
                peak_stats = _view_peak_stats(hm, topk=args.topk)
                ratio_med = _median(peak_stats["ratios"])
                status = _ratio_status(ratio_med)
                entropy_med = _median(peak_stats["entropy"])

                stage1_local = captured["stage1_wpt_local"].to(device)
                stage1_world = rev_trans(stage1_local).detach().cpu().numpy()[0]
                topk_world, topk_scores = _stage1_topk_world(
                    agent,
                    hm,
                    rev_trans,
                    device,
                    args.topk_3d,
                    args.topk_nms_dist,
                )
                final_world = np.asarray(act_result.action[:3], dtype=np.float32)

                target_scores = _sample_target_scores(agent, hm, target_locals)
                score_by_name = dict(zip(target_names, target_scores))
                gt_score = score_by_name.get("gt", {}).get("score_mean", float("nan"))
                gt_score_over_p1 = score_by_name.get("gt", {}).get(
                    "score_over_p1_mean", float("nan")
                )
                gt_rank_pct = score_by_name.get("gt", {}).get("rank_pct_median", float("nan"))
                best_alt_score = float("nan")
                best_alt_name = ""
                if valid_alts.size > 0:
                    alt_scores = [
                        (name, score_by_name[name]["score_mean"])
                        for name in target_names
                        if name.startswith("alt") and name in score_by_name
                    ]
                    if alt_scores:
                        best_alt_name, best_alt_score = max(alt_scores, key=lambda x: x[1])
                score_pick = _score_pick(gt_score, best_alt_score, args.score_margin)
                if np.isfinite(best_alt_score):
                    alt_over_gt = (
                        float("inf") if gt_score < 1e-12 else best_alt_score / gt_score
                    )
                else:
                    alt_over_gt = float("nan")

                if k + 1 >= num_kf:
                    detail = {
                        "task": task,
                        "episode": ep_idx,
                        "kf": k,
                        "label_mp": int(label_mp),
                        "num_alt": int(alt_mask[k].sum()),
                        "status": status,
                        "direction_pick": "no_next",
                        "is_correct": "",
                        "nearest_target": "no_next",
                        "score_pick": "no_next",
                        "cause_hint": "terminal",
                        "ratio_view_min_med_max": _fmt(peak_stats["ratios"]),
                        "entropy_median": entropy_med,
                        "gt_score_mean": "",
                        "best_alt_score_mean": "",
                        "best_alt_name": "",
                        "gt_score_over_p1_mean": "",
                        "best_alt_over_gt_score": "",
                        "gt_rank_pct_median": "",
                        "stage1_gt_err_m": "",
                        "cos_gt": "",
                        "stage1_pred_xyz": "%.4f,%.4f,%.4f" % tuple(stage1_world),
                        "final_pred_xyz": "%.4f,%.4f,%.4f" % tuple(final_world),
                        "gt_xyz": "",
                        "topk_3d_gt_hit": "",
                        "topk_3d_min_gt_err_m": "",
                        "topk_3d_best_rank": "",
                        "topk_3d_xyz": ";".join(
                            "%.4f,%.4f,%.4f" % tuple(x) for x in topk_world
                        ),
                        "topk_3d_scores": ";".join(_short(x) for x in topk_scores),
                        "peak_coords": peak_stats["peaks"],
                        "target_coords": "",
                    }
                    detail_rows.append(detail)
                    continue

                agg["valid"] += 1
                current = poses[k, :3]
                gt = poses[k + 1, :3]
                (
                    direction_pick,
                    cos_gt,
                    _best_alt_cos,
                    gt_err,
                    nearest_err,
                    nearest_target,
                ) = _classify_direction(
                    current=current,
                    pred=stage1_world,
                    gt=gt,
                    alts=valid_alts,
                    cos_margin=args.cos_margin,
                    off_cos=args.off_cos,
                )
                is_correct = gt_err <= args.correct_dist
                if topk_world.shape[0] > 0:
                    topk_gt_dists = np.linalg.norm(topk_world - gt[None, :], axis=1)
                    topk_min_gt_err = float(np.min(topk_gt_dists))
                    topk_best_rank = int(np.argmin(topk_gt_dists)) + 1
                    topk_gt_hit = topk_min_gt_err <= args.correct_dist
                else:
                    topk_min_gt_err = float("nan")
                    topk_best_rank = -1
                    topk_gt_hit = False

                if direction_pick == "gt":
                    agg["top1_gt"] += 1
                elif direction_pick.startswith("alt"):
                    agg["top1_alt"] += 1
                elif direction_pick == "tie":
                    agg["top1_tie"] += 1
                else:
                    agg["top1_off"] += 1
                if is_correct:
                    agg["top1_correct"] += 1
                else:
                    agg["top1_wrong"] += 1
                if topk_gt_hit:
                    agg["topk_gt_hit"] += 1
                    if not is_correct:
                        agg["top1_fail_topk_hit"] += 1
                else:
                    agg["topk_miss"] += 1
                if status == "collapsed":
                    agg["collapsed_total"] += 1
                    if is_correct:
                        agg["collapsed_correct"] += 1
                    else:
                        agg["collapsed_wrong"] += 1

                score_key = {
                    "gt": "score_pick_gt",
                    "alt": "score_pick_alt",
                    "tie": "score_pick_tie",
                    "gt_only": "score_pick_gt_only",
                }.get(score_pick)
                if score_key:
                    agg[score_key] += 1

                agg["ratios"].append(ratio_med)
                agg["gt_err"].append(gt_err)
                if np.isfinite(cos_gt):
                    agg["cos_gt"].append(cos_gt)
                if np.isfinite(gt_score_over_p1):
                    agg["gt_score_over_p1"].append(gt_score_over_p1)
                if np.isfinite(alt_over_gt) or np.isposinf(alt_over_gt):
                    agg["alt_over_gt"].append(alt_over_gt)
                if np.isfinite(gt_rank_pct):
                    agg["gt_rank_pct"].append(gt_rank_pct)
                if np.isfinite(entropy_med):
                    agg["entropy"].append(entropy_med)
                if np.isfinite(topk_min_gt_err):
                    agg["topk_gt_err"].append(topk_min_gt_err)
                if topk_best_rank > 0:
                    agg["topk_best_rank"].append(float(topk_best_rank))

                detail = {
                    "task": task,
                    "episode": ep_idx,
                    "kf": k,
                    "label_mp": int(label_mp),
                    "num_alt": int(alt_mask[k].sum()),
                    "status": status,
                    "direction_pick": direction_pick,
                    "is_correct": int(is_correct),
                    "nearest_target": nearest_target,
                    "score_pick": score_pick,
                    "cause_hint": "",
                    "ratio_view_min_med_max": _fmt(peak_stats["ratios"]),
                    "entropy_median": entropy_med,
                    "gt_score_mean": gt_score,
                    "best_alt_score_mean": best_alt_score,
                    "best_alt_name": best_alt_name,
                    "gt_score_over_p1_mean": gt_score_over_p1,
                    "best_alt_over_gt_score": alt_over_gt,
                    "gt_rank_pct_median": gt_rank_pct,
                    "stage1_gt_err_m": gt_err,
                    "nearest_err_m": nearest_err,
                    "cos_gt": cos_gt,
                    "stage1_pred_xyz": "%.4f,%.4f,%.4f" % tuple(stage1_world),
                    "final_pred_xyz": "%.4f,%.4f,%.4f" % tuple(final_world),
                    "gt_xyz": "%.4f,%.4f,%.4f" % tuple(gt),
                    "topk_3d_gt_hit": int(topk_gt_hit),
                    "topk_3d_min_gt_err_m": topk_min_gt_err,
                    "topk_3d_best_rank": topk_best_rank,
                    "topk_3d_xyz": ";".join(
                        "%.4f,%.4f,%.4f" % tuple(x) for x in topk_world
                    ),
                    "topk_3d_scores": ";".join(_short(x) for x in topk_scores),
                    "peak_coords": peak_stats["peaks"],
                    "target_coords": ";".join(
                        f"{name}:{score.get('coords', '')}"
                        for name, score in score_by_name.items()
                    ),
                }
                detail["cause_hint"] = _cause_hint(detail)
                agg["cause_counts"][detail["cause_hint"]] = (
                    agg["cause_counts"].get(detail["cause_hint"], 0) + 1
                )
                detail_rows.append(detail)

    summary_rows = []
    for key in sorted(per_key):
        agg = per_key[key]
        ratio_med = _median(agg["ratios"])
        cause_summary = ",".join(
            f"{name}:{count}" for name, count in sorted(agg["cause_counts"].items())
        )
        summary_rows.append(
            {
                "task": agg["task"],
                "kf": agg["kf"],
                "episodes": agg["episodes"],
                "valid": agg["valid"],
                "label_hits": agg["label_hits"],
                "status": "terminal" if agg["valid"] == 0 else _ratio_status(ratio_med),
                "ratio_med_min_med_max": _fmt(agg["ratios"]),
                "top1_gt": agg["top1_gt"],
                "top1_alt": agg["top1_alt"],
                "top1_tie": agg["top1_tie"],
                "top1_off": agg["top1_off"],
                "top1_correct": agg["top1_correct"],
                "top1_wrong": agg["top1_wrong"],
                "collapsed_correct": agg["collapsed_correct"],
                "collapsed_wrong": agg["collapsed_wrong"],
                "collapsed_total": agg["collapsed_total"],
                "score_pick_gt": agg["score_pick_gt"],
                "score_pick_alt": agg["score_pick_alt"],
                "score_pick_tie": agg["score_pick_tie"],
                "score_pick_gt_only": agg["score_pick_gt_only"],
                "gt_score_over_p1_min_med_max": _fmt(agg["gt_score_over_p1"]),
                "best_alt_over_gt_min_med_max": _fmt(agg["alt_over_gt"]),
                "gt_rank_pct_min_med_max": _fmt(agg["gt_rank_pct"]),
                "stage1_gt_err_min_med_max_m": _fmt(agg["gt_err"]),
                "cos_gt_min_med_max": _fmt(agg["cos_gt"]),
                "entropy_min_med_max": _fmt(agg["entropy"]),
                "topk_gt_hit": agg["topk_gt_hit"],
                "topk_miss": agg["topk_miss"],
                "top1_fail_topk_hit": agg["top1_fail_topk_hit"],
                "topk_gt_err_min_med_max_m": _fmt(agg["topk_gt_err"]),
                "topk_best_rank_min_med_max": _fmt(agg["topk_best_rank"], digits=1),
                "cause_summary": cause_summary,
            }
        )
    return summary_rows, detail_rows


def _verdict(row: dict) -> str:
    valid = int(row["valid"])
    if valid <= 0:
        return "terminal"
    collapsed_total = int(row["collapsed_total"])
    collapsed_wrong = int(row["collapsed_wrong"])
    collapsed_correct = int(row["collapsed_correct"])
    correct = int(row["top1_correct"])
    wrong = int(row["top1_wrong"])
    status = row["status"]
    has_mp = int(row["label_hits"]) > 0
    if collapsed_total > 0 and collapsed_wrong > 0:
        return "wrong-collapse-mp" if has_mp else "wrong-collapse-single"
    if correct > 0 and wrong > 0:
        return "mixed"
    if correct == 0 and wrong > 0:
        return "wrong"
    if collapsed_total > 0 and collapsed_wrong == 0 and collapsed_correct > 0:
        return "ok-collapse"
    if status in {"alive", "weak", "faint"} and wrong > 0:
        return "alive-wrong"
    if correct == valid:
        return "ok"
    return "mixed"


def write_outputs(summary_rows: list[dict], detail_rows: list[dict], args) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem
    if not stem:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"{args.model.parent.name}_{args.model.stem}_{timestamp}"

    summary_csv = args.out_dir / f"stage1_collapse_probe_{stem}.csv"
    detail_csv = args.out_dir / f"stage1_collapse_probe_{stem}_per_episode.csv"
    md_path = args.out_dir / f"stage1_collapse_probe_{stem}.md"

    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    with detail_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)

    with md_path.open("w") as f:
        f.write(f"# Stage1 Collapse Probe: {args.model}\n\n")
        f.write(f"- episodes: 0..{args.episodes - 1}\n")
        f.write("- direction is computed from the captured Stage1 crop waypoint, not the final Stage2 action.\n")
        f.write(f"- `correct/wrong` and collapse verdicts use Stage1 crop position within {args.correct_dist:.3f}m of GT.\n")
        f.write("- `top1 gt/alt/tie/off` is still the direction classification from the current keyframe.\n")
        f.write("- `GT/p1` is the mean heatmap score at projected GT pixels divided by per-view peak score.\n")
        f.write("- `alt/GT` is the best alternative target score divided by GT score.\n")
        f.write("- status uses the median p1/p2 ratio across render views: alive<=3, weak<=20, faint<=100, collapsed>100.\n")
        f.write(f"- `topK` uses 3D NMS top-{args.topk_3d} with radius {args.topk_nms_dist:.3f}m.\n\n")

        for task in dict.fromkeys(row["task"] for row in summary_rows):
            f.write(f"## {task}\n\n")
            f.write(
                "| KF | MP | peak | top1 gt/alt/tie/off | correct/wrong | "
                "collapsed ok/wrong/total | score gt/alt/tie | GT/p1 | alt/GT | "
                "GT err m | topK hit/miss | top1 fail recovered | verdict | cause |\n"
            )
            f.write("|---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|\n")
            for row in summary_rows:
                if row["task"] != task:
                    continue
                verdict = _verdict(row)
                f.write(
                    "| {kf} | {label_hits}/{episodes} | {status} | "
                    "{top1_gt}/{top1_alt}/{top1_tie}/{top1_off} | "
                    "{top1_correct}/{top1_wrong} | "
                    "{collapsed_correct}/{collapsed_wrong}/{collapsed_total} | "
                    "{score_pick_gt}/{score_pick_alt}/{score_pick_tie} | "
                    "{gt_score} | {alt_gt} | {gt_err} | {topk_hit}/{topk_miss} | "
                    "{top1_fail_topk_hit} | `{verdict}` | {cause} |\n".format(
                        kf=row["kf"],
                        label_hits=row["label_hits"],
                        episodes=row["episodes"],
                        status=row["status"],
                        top1_gt=row["top1_gt"],
                        top1_alt=row["top1_alt"],
                        top1_tie=row["top1_tie"],
                        top1_off=row["top1_off"],
                        top1_correct=row["top1_correct"],
                        top1_wrong=row["top1_wrong"],
                        collapsed_correct=row["collapsed_correct"],
                        collapsed_wrong=row["collapsed_wrong"],
                        collapsed_total=row["collapsed_total"],
                        score_pick_gt=row["score_pick_gt"],
                        score_pick_alt=row["score_pick_alt"],
                        score_pick_tie=row["score_pick_tie"],
                        gt_score=row["gt_score_over_p1_min_med_max"],
                        alt_gt=row["best_alt_over_gt_min_med_max"],
                        gt_err=row["stage1_gt_err_min_med_max_m"],
                        topk_hit=row["topk_gt_hit"],
                        topk_miss=row["topk_miss"],
                        top1_fail_topk_hit=row["top1_fail_topk_hit"],
                        verdict=verdict,
                        cause=row["cause_summary"] or "-",
                    )
                )
            f.write("\n")

        bad = [
            row
            for row in summary_rows
            if _verdict(row) in {"wrong-collapse-mp", "wrong-collapse-single", "alive-wrong", "wrong", "mixed"}
        ]
        if bad:
            f.write("## Problem Focus\n\n")
            f.write(
                "| task | KF | MP | peak | top1 correct/wrong | collapsed ok/wrong/total | "
                "score gt/alt/tie | GT/p1 | alt/GT | topK hit/miss | top1 fail recovered | cause |\n"
            )
            f.write("|---|---:|:---:|:---:|:---:|:---:|:---:|---:|---:|---:|---:|---|\n")
            for row in bad:
                f.write(
                    "| {task} | {kf} | {label_hits}/{episodes} | {status} | "
                    "{top1_correct}/{top1_wrong} | "
                    "{collapsed_correct}/{collapsed_wrong}/{collapsed_total} | "
                    "{score_pick_gt}/{score_pick_alt}/{score_pick_tie} | "
                    "{gt_score} | {alt_gt} | {topk_hit}/{topk_miss} | "
                    "{top1_fail_topk_hit} | {cause} |\n".format(
                        task=row["task"],
                        kf=row["kf"],
                        label_hits=row["label_hits"],
                        episodes=row["episodes"],
                        status=row["status"],
                        top1_correct=row["top1_correct"],
                        top1_wrong=row["top1_wrong"],
                        collapsed_correct=row["collapsed_correct"],
                        collapsed_wrong=row["collapsed_wrong"],
                        collapsed_total=row["collapsed_total"],
                        score_pick_gt=row["score_pick_gt"],
                        score_pick_alt=row["score_pick_alt"],
                        score_pick_tie=row["score_pick_tie"],
                        gt_score=row["gt_score_over_p1_min_med_max"],
                        alt_gt=row["best_alt_over_gt_min_med_max"],
                        topk_hit=row["topk_gt_hit"],
                        topk_miss=row["topk_miss"],
                        top1_fail_topk_hit=row["top1_fail_topk_hit"],
                        cause=row["cause_summary"] or "-",
                    )
                )
            f.write("\n")

    print(f"Wrote {md_path}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {detail_csv}")
    print(md_path.read_text())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", type=int, default=7)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--tasks", default="put_block_back")
    parser.add_argument("--max-peaks", type=int, default=5)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--topk-3d", type=int, default=3)
    parser.add_argument("--topk-nms-dist", type=float, default=0.05)
    parser.add_argument("--cos-margin", type=float, default=0.05)
    parser.add_argument("--off-cos", type=float, default=0.5)
    parser.add_argument("--correct-dist", type=float, default=0.05)
    parser.add_argument("--score-margin", type=float, default=1.2)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/hdd4/siwon/datasets/sam2act/data_memory/test"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/home/cv11/project/siwon/sam2act_stage2_oracle_ceiling/logs"),
    )
    parser.add_argument("--output-stem", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_rows, detail_rows = analyze(args)
    write_outputs(summary_rows, detail_rows, args)


if __name__ == "__main__":
    main()
