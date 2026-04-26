# Spatial Graph Memory Design

## Goal

Stage 2 only experiment for SAM2Act on MemoryBench `put_block_back`.

Target problem:
- current stage2 memory is queue/FIFO-like
- repeated visits to the same 3D bottlenecks create redundant memory entries
- cross-attention remains dense over all stored entries

This branch replaces the plain queue behavior with a lightweight spatial graph memory that is still compatible with the original SAM2Act per-view heatmap-guided memory backbone.

## Design

Three modules were added.

### 1. Spatial-aware memory aggregation

File:
- `sam2act/utils/spatial_graph_memory.py`

Implementation:
- `SpatialGraphMemoryBank.update(...)`

Behavior:
- each memory node stores:
  - `memory`
  - `memory_pos`
  - `pos3d`
  - `last_obs_idx`
  - `count`
- when a new keyframe is written:
  - find nearest existing node in 3D
  - if distance `< epsilon`, merge with EMA
  - else append as a new node

Current write position policy:
- training: GT local waypoint if `spatial_graph_write_use_gt_train=True`
- eval: predicted local waypoint from current `trans`

### 2. Sparse spatial masking for memory attention

Files:
- `sam2act/utils/spatial_graph_memory.py`
- `sam2act/mvt/sam2_train/modeling/sam/transformer.py`
- `sam2act/mvt/sam2_train/modeling/memory_attention.py`

Implementation:
- `build_spatial_attention_bias(...)`
- additive `attn_mask` support in SAM2 attention path

Behavior:
- current query position is compared against stored node 3D positions
- nodes outside the spatial radius receive a large negative additive bias
- if no node is within threshold, the nearest node is kept to avoid all-`-inf` attention

Current query position policy:
- training: GT local waypoint of current observation in the sequential batch
- eval: previous predicted waypoint as proxy for current end-effector location

### 3. Spatial graph contrastive loss

Files:
- `sam2act/utils/spatial_graph_memory.py`
- `sam2act/models/sam2act_agent.py`

Implementation:
- `spatial_graph_contrastive_loss(hidden_states, positions, temperature, distance_threshold)`

Behavior:
- pooled memory-conditioned hidden states are contrasted inside each temporal sequence
- positives: steps whose 3D waypoints are spatially close
- negatives: same-sequence steps outside the threshold
- safe fallback returns zero loss when there are no positives

## Integration points

- `sam2act/mvt/config.py`
  - stage2 spatial graph toggles and thresholds
- `sam2act/config.py`
  - auxiliary contrastive loss weights
- `sam2act/mvt/mvt_sam2.py`
  - forwards new config fields into the stage2 MVT
- `sam2act/mvt/mvt_sam2_single.py`
  - memory bank changed from per-step dict entries to per-view node lists
  - read path applies sparse spatial bias
  - write path performs EMA node coarsening
  - pooled hidden state exported as `spatial_graph_hidden`
- `sam2act/models/sam2act_agent.py`
  - stage2 total loss becomes:
    - `trans_loss + lambda * spatial_graph_contrastive_loss`
- `sam2act/train_plus.py`
  - rank0-first replay initialization for DDP
- `sam2act/utils/dataset.py`
  - replay exists only if `init_frame_to_idx.pkl` exists

## Experimental choices

- base repo: `sam2act_baseline`
- stage1 init: dirty `put_block_back` `model_38.pth`
- stage2 only
- `num_maskmem=11`
- 2 GPU smoke run

## Important caveat

This branch intentionally keeps the original SAM2Act heatmap-guided per-view memory write/read structure.

What changed is:
- memory entries are spatially merged
- memory attention becomes spatially sparse
- hidden states receive an explicit spatial topology auxiliary loss

This is therefore a direct stage2 memory-architecture experiment, not a new preprocessed task graph pipeline.
