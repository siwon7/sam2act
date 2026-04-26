from typing import Dict, List, Tuple

import numpy as np


ROLE_GRAPH_IGNORE = -1


PUT_BLOCK_BACK_ROLE_NAMES: List[str] = [
    "initial_slot_low",
    "initial_slot_high",
    "center_high",
    "center_low",
    "button_high",
    "button_low",
]


_PUT_BLOCK_BACK_ROLE_SEQUENCE: List[int] = [
    0,  # 0
    1,  # 1
    2,  # 2
    3,  # 3
    2,  # 4
    4,  # 5
    5,  # 6
    2,  # 7
    3,  # 8
    2,  # 9
    1,  # 10
    0,  # 11
]


_PUT_BLOCK_BACK_ROLE_REF_ROLES: Dict[int, List[int]] = {
    # Regrasp block from center: center roles remain the key reference.
    7: [2, 3],
    8: [2, 3],
    # Late return phase: initial-slot roles should dominate.
    9: [1, 2],
    10: [0, 1],
    11: [0, 1],
}


_PUT_BLOCK_BACK_ANCHOR_USE: Dict[int, int] = {
    9: 1,
    10: 1,
    11: 1,
}


def get_memorybench_role_graph_targets(
    task: str,
    keypoint_idx: int,
    num_keypoints: int,
) -> Tuple[int, int, np.ndarray, int]:
    """Return grouped role-graph targets for MemoryBench.

    Returns:
      role_label: grouped semantic role id
      role_ref_valid: whether role_ref_mask should be supervised
      role_ref_mask: multi-hot over grouped role ids
      anchor_use_label: whether persistent slot anchors should be emphasized
    """
    if keypoint_idx < 0:
        raise ValueError(f"keypoint_idx must be non-negative, got {keypoint_idx}")
    if num_keypoints <= 0:
        raise ValueError(f"num_keypoints must be positive, got {num_keypoints}")

    if task == "put_block_back" and num_keypoints == len(_PUT_BLOCK_BACK_ROLE_SEQUENCE):
        role_label = int(_PUT_BLOCK_BACK_ROLE_SEQUENCE[keypoint_idx])
        role_ref_mask = np.zeros(len(PUT_BLOCK_BACK_ROLE_NAMES), dtype=np.float32)
        role_ref_roles = _PUT_BLOCK_BACK_ROLE_REF_ROLES.get(keypoint_idx, [])
        for role_idx in role_ref_roles:
            role_ref_mask[role_idx] = 1.0
        role_ref_valid = int(len(role_ref_roles) > 0)
        anchor_use_label = int(_PUT_BLOCK_BACK_ANCHOR_USE.get(keypoint_idx, 0))
        return role_label, role_ref_valid, role_ref_mask, anchor_use_label

    # Fallback for tasks without explicit grouped-role annotation.
    role_label = ROLE_GRAPH_IGNORE
    role_ref_valid = 0
    role_ref_mask = np.zeros(len(PUT_BLOCK_BACK_ROLE_NAMES), dtype=np.float32)
    anchor_use_label = 0
    return role_label, role_ref_valid, role_ref_mask, anchor_use_label
