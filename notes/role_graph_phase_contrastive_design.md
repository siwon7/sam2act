# Role Graph + Phase + Contrastive Design

## Branch

- repo: `/home/cv25/siwon/sam2act_role_graph_memory`
- branch: `exp/role-graph-phase-contrastive`

## Why This Branch Exists

The previous `graph-retrieval-pointer` branch proved that stage2 can fit a
retrieval probe numerically while still collapsing at rollout time. The main
problem was that the learned probe was not changing *which memory tokens the
model attends to* in a structured way.

This branch reworks the idea around three principles:

1. **grouped role graph instead of exact node id**
2. **phase supervision instead of pure node classification**
3. **additive attention bias on memory read instead of post-hoc memory scaling**

It keeps SAM2Act+'s heatmap-based memory bank, but adds event/phase structure on
top of it.

## Grouped Role Graph

For `put_block_back`, local keyframe inspection showed that the 12 discovered
keyframes repeat a small number of spatial/event roles.

Implemented grouped roles:

1. `initial_slot_low`  -> keyframes `{0, 11}`
2. `initial_slot_high` -> keyframes `{1, 10}`
3. `center_high`       -> keyframes `{2, 4, 7, 9}`
4. `center_low`        -> keyframes `{3, 8}`
5. `button_high`       -> keyframes `{5}`
6. `button_low`        -> keyframes `{6}`

Current late-phase reference targets:

- step 7/8  -> `{center_high, center_low}`
- step 9    -> `{initial_slot_high, center_high}`
- step 10   -> `{initial_slot_low, initial_slot_high}`
- step 11   -> `{initial_slot_low, initial_slot_high}`

Replay fields added:

- `role_label`
- `role_ref_valid`
- `role_ref_mask`
- `anchor_use_label`

File:

- `sam2act/utils/memorybench_role_graph.py`

## Implemented Losses

Stage2 now supports the following memory-structure losses in addition to
`trans_loss`.

### 1. Phase Auxiliary Loss

- head: `phase_graph_logits`
- target: `phase_label`
- loss: CE

Purpose:
- learn where the rollout is in the task graph
- reduce phase confusion around interaction and return

### 2. Role Graph Loss

- head: `role_graph_logits`
- target: `role_label`
- loss: CE

Purpose:
- teach the feature to identify grouped event roles rather than raw step ids

### 3. Retrieval Reference Loss

- head: `role_ref_logits`
- target: `role_ref_mask`
- valid mask: `role_ref_valid`
- loss: BCE-with-logits

Purpose:
- supervise which role groups should be recalled at late revisit steps

### 4. Anchor-Use Loss

- head: `anchor_use_logits`
- target: `anchor_use_label`
- loss: BCE-with-logits

Purpose:
- teach the model *when* persistent early anchors should matter

### 5. Role Contrastive Loss

- source embedding: pooled SAM2 memory-write feature
- projection: `role_contrast_proj`
- target label: `role_label`
- loss: supervised contrastive

Purpose:
- align memory-write embeddings by event role, not just by spatial proximity

## Additive Attention Bias

This branch does **not** apply graph supervision as a hard mask.

Instead, memory read uses a **soft additive bias** on the cross-attention logits:

- `role_ref_logits` raise/lower scores for memory entries tagged with matching
  role groups
- `anchor_use_logits` raise/lower scores for persistent anchor entries

Implementation path:

- `sam2act/mvt/sam2_train/modeling/memory_attention.py`
- `sam2act/mvt/sam2_train/modeling/sam/transformer.py`
- `sam2act/mvt/mvt_sam2_single.py`

Important detail:
- this is a true qk-logit bias via `attn_mask` into
  `scaled_dot_product_attention`, not multiplicative scaling of memory values

## Persistent Anchor Memory

This branch keeps the existing persistent anchor idea and ties it into the new
losses.

Current behavior:

- first `persistent_anchor_max_steps` observations are written into an anchor
  bank
- anchor memories stay available outside the FIFO memory window
- `anchor_use_label` supervises when those anchors should be emphasized

## Runtime Outputs Added

`mvt_sam2_single.py` now exports:

- `role_graph_logits`
- `phase_graph_logits`
- `role_ref_logits`
- `anchor_use_logits`
- `role_contrast_embeddings`

## Current Smoke Findings

Local smoke (`1 epoch`, `10 train iterations`, `1 eval episode`) completed
successfully end-to-end after fixing a contrastive half-precision overflow bug.

Observed training summary:

- `trans_loss`: `5.3843`
- `phase_aux_loss`: `6.3898`
- `role_graph_loss`: `8.7456`
- `role_ref_loss`: `2.0034`
- `anchor_use_loss`: `1.8762`
- `role_contrastive_loss`: `3.4562`
- `role_ref_top3_acc`: `0.90`

Smoke eval:

- `put_block_back`: `0.0`
- episode length: `25`

Interpretation:

- the new path is live and no longer collapses immediately from an obvious
  implementation bug
- but the weights and/or supervision balance are not yet good enough to produce
  positive rollout behavior

This branch should now be treated as a **working experimental baseline** for
further tuning, not a finished method.
