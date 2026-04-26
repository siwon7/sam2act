# Spatial Graph Smoke 2026-04-26

## Repo

- repo: `sam2act_spatial_graph_memory`
- branch: `exp/spatial-graph-memory`

## Run

- run dir:
  - `sam2act/runs/sam2act_sam2act_pbb_spatial_graph_smoke_20260426_retry4`
- script:
  - `scripts/run_put_block_back_spatial_graph_smoke.sh`
- task:
  - `put_block_back`
- init checkpoint:
  - dirty stage1 `model_38.pth`
- setup:
  - stage2 only
  - `num_maskmem=11`
  - 2 GPU DDP
  - `demo=20`
  - `epochs=1`
  - `train_iter=12000`
  - actual epoch length observed: `500` updates

## Key fixes needed before the run worked

1. `data_memory` symlink was missing in the new repo
2. local SAM2 checkpoint symlink was missing
3. replay existence check was too loose
4. DDP replay init needed rank0-first creation
5. `SpatialGraphMemoryBank.update()` call had a keyword mismatch bug

## Train summary

Observed training snapshots:
- step `100`: `total_loss ≈ 5.1168`, `trans_loss ≈ 5.0754`
- step `200`: `total_loss ≈ 5.0604`, `trans_loss ≈ 5.0221`
- step `300`: `total_loss ≈ 4.9217`, `trans_loss ≈ 4.8781`
- step `400`: `total_loss ≈ 4.8099`, `trans_loss ≈ 4.7606`
- step `500`: `total_loss ≈ 4.5415`, `trans_loss ≈ 4.5070`

Final logged summary:
- `total_loss = 4.8901`
- `trans_loss = 4.8486`
- `spatial_graph_aux_loss = 0.8285`
- `spatial_graph_aux_top3_acc = 0.0743`

## Eval

Result file:
- `sam2act/runs/sam2act_sam2act_pbb_spatial_graph_smoke_20260426_retry4/eval/sam2act_sam2act_pbb_spatial_graph_smoke_20260426_retry4_smoke5/model_plus_last/eval_results.csv`

Result:
- `put_block_back = 0.0%`
- average episode length `25.0`
- total transitions `125`

## Interpretation

What worked:
- code path is valid end-to-end
- DDP training runs
- replay creation and stage2 checkpoint handoff work
- spatial coarsening + sparse attention mask + contrastive aux all run without device mismatch

What did not work:
- rollout success did not improve
- all 5 smoke eval episodes timed out
- `spatial_graph_aux_top3_acc` remained low, suggesting the auxiliary signal is not yet shaping a useful retrieval structure

## Recommended next ablations

1. reduce spatial graph pressure
   - try `spatial_graph_contrastive_loss_weight = 0.01`

2. soften sparsity
   - keep coarsening
   - reduce the attention mask magnitude so it becomes a weaker bias

3. compare write position policies
   - GT write during train vs predicted write during train

4. compare masking vs coarsening separately
   - coarsening only
   - masking only
   - both together
