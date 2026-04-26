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

PUT_BLOCK_BACK_PERSISTENT_ANCHOR_ROLE_IDS = {0, 1}


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

_PUT_BLOCK_BACK_VISIT_MODE_SEQUENCE: List[int] = [
    0,  # 0: first visit to initial_slot_low
    0,  # 1: first visit to initial_slot_high
    0,  # 2: first visit to center_high
    0,  # 3: first visit to center_low
    1,  # 4: revisit center_high
    0,  # 5: first visit to button_high
    0,  # 6: first visit to button_low
    1,  # 7: revisit center_high
    1,  # 8: revisit center_low
    1,  # 9: revisit center_high
    1,  # 10: revisit initial_slot_high
    1,  # 11: revisit initial_slot_low
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
) -> Tuple[int, int, int, np.ndarray, int]:
    """Return grouped role-graph targets for MemoryBench.

    Returns:
      role_label: grouped semantic role id
      visit_mode_label: 0 for new target prototype, 1 for revisit target
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
        visit_mode_label = int(_PUT_BLOCK_BACK_VISIT_MODE_SEQUENCE[keypoint_idx])
        role_ref_mask = np.zeros(len(PUT_BLOCK_BACK_ROLE_NAMES), dtype=np.float32)
        role_ref_roles = _PUT_BLOCK_BACK_ROLE_REF_ROLES.get(keypoint_idx, [])
        for role_idx in role_ref_roles:
            role_ref_mask[role_idx] = 1.0
        role_ref_valid = int(len(role_ref_roles) > 0)
        anchor_use_label = int(_PUT_BLOCK_BACK_ANCHOR_USE.get(keypoint_idx, 0))
        return (
            role_label,
            visit_mode_label,
            role_ref_valid,
            role_ref_mask,
            anchor_use_label,
        )

    # Fallback for tasks without explicit grouped-role annotation.
    role_label = ROLE_GRAPH_IGNORE
    visit_mode_label = 0
    role_ref_valid = 0
    role_ref_mask = np.zeros(len(PUT_BLOCK_BACK_ROLE_NAMES), dtype=np.float32)
    anchor_use_label = 0
    return role_label, visit_mode_label, role_ref_valid, role_ref_mask, anchor_use_label


def should_store_persistent_anchor(role_label: int, num_role_classes: int) -> bool:
    """Return whether a grouped role should enter the persistent anchor bank.

    The current transition-pointer branch is only tuned for the 6-role
    `put_block_back` grouping, where the persistent anchors should keep the
    initial slot prototypes.
    """
    if num_role_classes == len(PUT_BLOCK_BACK_ROLE_NAMES):
        return role_label in PUT_BLOCK_BACK_PERSISTENT_ANCHOR_ROLE_IDS
    return False
