from typing import Dict, List, Tuple

import numpy as np


MAX_MEMORYBENCH_GRAPH_NODES = 12

GRAPH_MODE_IGNORE = -1
GRAPH_MODE_NEW = 0
GRAPH_MODE_REVISIT = 1


# Conservative graph supervision for MemoryBench.
# The first implementation focuses on put_block_back, where we already have
# reliable step semantics and retrieval analysis. Other tasks can be annotated
# later without changing the training codepath.
_PUT_BLOCK_BACK_GRAPH: Dict[int, Tuple[int, List[int]]] = {
    0: (GRAPH_MODE_NEW, []),
    1: (GRAPH_MODE_REVISIT, [0]),
    2: (GRAPH_MODE_NEW, []),
    3: (GRAPH_MODE_REVISIT, [2]),
    4: (GRAPH_MODE_NEW, []),
    5: (GRAPH_MODE_NEW, []),
    6: (GRAPH_MODE_REVISIT, [5]),
    7: (GRAPH_MODE_REVISIT, [2, 3]),
    8: (GRAPH_MODE_REVISIT, [2, 3, 7]),
    9: (GRAPH_MODE_REVISIT, [1, 7, 8]),
    10: (GRAPH_MODE_REVISIT, [0, 1, 9]),
    11: (GRAPH_MODE_REVISIT, [0, 1, 10]),
}


def _empty_ref_mask(max_nodes: int = MAX_MEMORYBENCH_GRAPH_NODES) -> np.ndarray:
    return np.zeros((max_nodes,), dtype=np.float32)


def get_memorybench_graph_targets(
    task: str,
    keypoint_idx: int,
    num_keypoints: int,
    max_nodes: int = MAX_MEMORYBENCH_GRAPH_NODES,
) -> Tuple[int, int, np.ndarray]:
    """Return graph retrieval supervision for a MemoryBench keypoint.

    Returns:
        mode_label:
            GRAPH_MODE_NEW / GRAPH_MODE_REVISIT / GRAPH_MODE_IGNORE
        ref_valid:
            1 when ref_mask supervision is meaningful, 0 otherwise
        ref_mask:
            multi-hot mask over semantic node ids to revisit
    """
    if keypoint_idx < 0:
        raise ValueError(f"keypoint_idx must be non-negative, got {keypoint_idx}")
    if num_keypoints <= 0:
        raise ValueError(f"num_keypoints must be positive, got {num_keypoints}")

    ref_mask = _empty_ref_mask(max_nodes=max_nodes)

    if task == "put_block_back" and num_keypoints == 12:
        mode_label, ref_nodes = _PUT_BLOCK_BACK_GRAPH.get(
            keypoint_idx, (GRAPH_MODE_IGNORE, [])
        )
        for ref_node in ref_nodes:
            if 0 <= ref_node < max_nodes:
                ref_mask[ref_node] = 1.0
        ref_valid = int(len(ref_nodes) > 0)
        return int(mode_label), ref_valid, ref_mask

    # Conservative fallback for tasks that do not yet have an explicit graph.
    return GRAPH_MODE_IGNORE, 0, ref_mask
