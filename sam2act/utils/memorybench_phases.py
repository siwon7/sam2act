from typing import Dict, List


MEMORYBENCH_PHASE_NAMES: List[str] = [
    "pickup_first_carry",
    "intermediate_place_or_state_change",
    "task_interaction",
    "regrasp_return_finalize",
]


# Coarse 4-phase mapping shared across MemoryBench tasks.
# The labels are intentionally broad so they can supervise multitask phase
# structure without depending on exact per-task node identities.
_MEMORYBENCH_PHASE_SEQUENCES: Dict[str, List[int]] = {
    "put_block_back": [0, 0, 0, 1, 2, 2, 2, 3, 3, 3, 3, 3],
    "reopen_drawer": [0, 0, 1, 1, 2, 2, 2, 3, 3, 3],
    "rearrange_block": [0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 3],
}


def get_memorybench_phase_label(task: str, keypoint_idx: int, num_keypoints: int) -> int:
    """Map a MemoryBench keypoint index to a shared coarse phase label.

    For known tasks we use an explicit task-specific phase sequence. For
    unexpected keypoint counts or unknown tasks we fall back to quartile-style
    bucketing so replay generation remains robust.
    """
    if keypoint_idx < 0:
        raise ValueError(f"keypoint_idx must be non-negative, got {keypoint_idx}")
    if num_keypoints <= 0:
        raise ValueError(f"num_keypoints must be positive, got {num_keypoints}")

    phase_sequence = _MEMORYBENCH_PHASE_SEQUENCES.get(task)
    if phase_sequence and len(phase_sequence) == num_keypoints:
        return int(phase_sequence[min(keypoint_idx, num_keypoints - 1)])

    # Fallback for tasks or keypoint counts we have not explicitly annotated.
    capped_idx = min(keypoint_idx, num_keypoints - 1)
    phase = int((4 * capped_idx) / num_keypoints)
    return min(phase, len(MEMORYBENCH_PHASE_NAMES) - 1)
