"""Multi-peak heatmap utilities for Stage1.

Only truly ambiguous keyframes receive multi-peak labels.
"Truly ambiguous" = same EE position + same gripper state + same object
position, but different next targets. This is determined by tracking
object positions through gripper open/close transitions.

Non-ambiguous divergent pairs (e.g. same EE but different block position)
keep single-peak labels so the model learns to use visual cues.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Object position tracking
# ---------------------------------------------------------------------------

def track_object_positions(
    gripper_poses: np.ndarray,
    gripper_opens: np.ndarray,
) -> np.ndarray:
    """Track the primary manipulated object's position through an episode.

    Infers object location from gripper open/close transitions:
      - grip 0→1 (close): object grasped at current EE position
      - grip 1→0 (open):  object released at previous EE position
      - While holding (grip=1): object moves with EE
      - While not holding (grip=0): object stays at last release position

    Args:
        gripper_poses: (num_kf, 7) gripper poses (xyz + quat).
        gripper_opens: (num_kf,) gripper states (0=open, 1=closed/holding).

    Returns:
        object_positions: (num_kf, 3) inferred object xyz at each keyframe.
    """
    num_kf = len(gripper_poses)
    positions = gripper_poses[:, :3]
    obj_pos = positions[0].copy()  # assume object starts near first keyframe
    holding = False
    object_positions = np.zeros((num_kf, 3), dtype=np.float32)

    for i in range(num_kf):
        if i > 0:
            prev_g = float(gripper_opens[i - 1])
            curr_g = float(gripper_opens[i])
            if prev_g == 0.0 and curr_g == 1.0:
                # Grasped: object is now at EE
                holding = True
            elif prev_g == 1.0 and curr_g == 0.0:
                # Released: object placed at previous step's EE
                obj_pos = positions[i - 1].copy()
                holding = False

        if holding:
            object_positions[i] = positions[i]
        else:
            object_positions[i] = obj_pos

    return object_positions


# ---------------------------------------------------------------------------
# Scene-aware clustering: EE + gripper + object position
# ---------------------------------------------------------------------------

def compute_scene_aware_clusters(
    gripper_poses: np.ndarray,
    gripper_opens: np.ndarray,
    ee_radius: float = 0.04,
    obj_radius: float = 0.05,
    spatial_only: bool = False,
) -> Dict[int, int]:
    """Cluster keyframes by spatial proximity.

    When spatial_only=False (legacy):
      Two keyframes cluster only if ALL three match:
        - EE positions within ee_radius
        - Same gripper state
        - Object positions within obj_radius

    When spatial_only=True (v7+):
      Two keyframes cluster if EE positions are within ee_radius.
      Grip and object differences are left for the model to learn.
      This captures all intra-episode revisit candidates, letting
      the model's gradient naturally adjust peak ratios based on
      whether visual cues (grip, object position) are distinguishable.

    Args:
        gripper_poses: (num_kf, 7).
        gripper_opens: (num_kf,).
        ee_radius: max EE distance for same cluster.
        obj_radius: max object distance for same cluster (ignored if spatial_only).
        spatial_only: if True, cluster by EE position only.

    Returns:
        Dict mapping keyframe_index -> cluster_id.
    """
    ee_positions = gripper_poses[:, :3]
    num_kf = len(ee_positions)

    if not spatial_only:
        obj_positions = track_object_positions(gripper_poses, gripper_opens)

    cluster_ids = [-1] * num_kf
    next_cluster = 0

    for i in range(num_kf):
        if cluster_ids[i] >= 0:
            continue
        cluster_ids[i] = next_cluster
        for j in range(i + 1, num_kf):
            if cluster_ids[j] >= 0:
                continue
            ee_dist = np.linalg.norm(ee_positions[i] - ee_positions[j])
            if spatial_only:
                if ee_dist < ee_radius:
                    cluster_ids[j] = next_cluster
            else:
                grip_same = float(gripper_opens[i]) == float(gripper_opens[j])
                obj_dist = np.linalg.norm(obj_positions[i] - obj_positions[j])
                if ee_dist < ee_radius and grip_same and obj_dist < obj_radius:
                    cluster_ids[j] = next_cluster
        next_cluster += 1

    return {i: cid for i, cid in enumerate(cluster_ids)}


# ---------------------------------------------------------------------------
# Alternative target collection (only for truly ambiguous clusters)
# ---------------------------------------------------------------------------

def collect_alt_targets(
    gripper_poses: np.ndarray,
    gripper_opens: np.ndarray,
    num_keyframes: int,
    max_peaks: int = 3,
    dedup_radius: float = 0.05,
    ee_radius: float = 0.04,
    obj_radius: float = 0.05,
    target_diverge_threshold: float = 0.05,
    spatial_only: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect alternative next-targets for ambiguous keyframes.

    A keyframe gets alt targets when it belongs to a spatial cluster
    where members have divergent next-targets.

    When spatial_only=True (v7+), clusters are formed by EE position
    only, so KFs with different grip/object but same position are
    included. The model learns to distinguish them during training.

    Args:
        gripper_poses: (num_kf, 7).
        gripper_opens: (num_kf,).
        num_keyframes: total keyframes in episode.
        max_peaks: max alternative targets to store per keyframe.
        dedup_radius: dedup distance for next-targets.
        ee_radius: clustering threshold for EE position.
        obj_radius: clustering threshold for object position.
        target_diverge_threshold: min distance between next-targets
            to be considered divergent.
        spatial_only: if True, cluster by EE position only (v7+).

    Returns:
        alt_positions: (num_kf, max_peaks, 3) padded xyz.
        alt_mask: (num_kf, max_peaks) boolean validity mask.
    """
    from collections import defaultdict

    cluster_map = compute_scene_aware_clusters(
        gripper_poses, gripper_opens, ee_radius, obj_radius,
        spatial_only=spatial_only,
    )

    # Group cluster members
    cluster_to_kfs = defaultdict(list)
    for kf_idx, cid in cluster_map.items():
        cluster_to_kfs[cid].append(kf_idx)

    alt_positions = np.zeros((num_keyframes, max_peaks, 3), dtype=np.float32)
    alt_mask = np.zeros((num_keyframes, max_peaks), dtype=bool)

    for cid, members in cluster_to_kfs.items():
        if len(members) < 2:
            continue

        # Collect next-targets for all members
        member_next = {}
        for kf in members:
            if kf + 1 < num_keyframes:
                member_next[kf] = gripper_poses[kf + 1, :3]

        if len(member_next) < 2:
            continue

        # Check if any pair of next-targets actually diverges
        kf_list = list(member_next.keys())
        has_divergence = False
        for a in range(len(kf_list)):
            for b in range(a + 1, len(kf_list)):
                d = np.linalg.norm(
                    member_next[kf_list[a]] - member_next[kf_list[b]]
                )
                if d > target_diverge_threshold:
                    has_divergence = True
                    break
            if has_divergence:
                break

        if not has_divergence:
            continue

        # This cluster is truly ambiguous — assign alt targets
        # Collect all unique next-targets
        all_targets = list(member_next.values())
        unique_targets = [all_targets[0]]
        for t in all_targets[1:]:
            is_dup = any(
                np.linalg.norm(t - u) < dedup_radius for u in unique_targets
            )
            if not is_dup:
                unique_targets.append(t)

        # For each member, alt targets = unique targets minus own next
        for kf in members:
            if kf not in member_next:
                continue
            own_next = member_next[kf]
            alts = [
                t for t in unique_targets
                if np.linalg.norm(t - own_next) >= dedup_radius
            ]
            for i, t in enumerate(alts[:max_peaks]):
                alt_positions[kf, i] = t
                alt_mask[kf, i] = True

    return alt_positions, alt_mask


# ---------------------------------------------------------------------------
# Multi-peak heatmap generation (unchanged — works on whatever alt targets
# are provided, whether from scene-aware or any other source)
# ---------------------------------------------------------------------------

def generate_multipeak_hm(
    primary_pt: torch.Tensor,
    alt_pts: torch.Tensor,
    alt_mask: torch.Tensor,
    res: Tuple[int, int],
    sigma: float = 1.5,
    thres_sigma_times: int = 3,
) -> torch.Tensor:
    """Generate multi-peak heatmap by summing Gaussians.

    Args:
        primary_pt: (num_pt, 2) primary target in image coords.
        alt_pts: (num_pt, max_peaks, 2) alternative targets in image coords.
        alt_mask: (num_pt, max_peaks) boolean mask for valid alternatives.
        res: (H, W) resolution.
        sigma: Gaussian std.
        thres_sigma_times: truncation threshold.

    Returns:
        hm: (num_pt, H, W) normalized multi-peak heatmap.
    """
    from sam2act.mvt.utils import generate_hm_from_pt

    # Primary peak
    hm = generate_hm_from_pt(primary_pt, res, sigma, thres_sigma_times)

    num_pt = primary_pt.shape[0]
    max_peaks = alt_pts.shape[1]

    for peak_idx in range(max_peaks):
        mask = alt_mask[:, peak_idx]
        if not mask.any():
            continue
        alt_pt = alt_pts[:, peak_idx, :]
        alt_hm = generate_hm_from_pt(alt_pt, res, sigma, thres_sigma_times)
        hm = hm + alt_hm * mask.float().view(num_pt, 1, 1)

    # Re-normalize to probability distribution
    hm_sum = hm.sum(dim=(1, 2), keepdim=True).clamp(min=1e-6)
    hm = hm / hm_sum

    return hm
