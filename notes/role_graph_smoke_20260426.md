# Smoke Result: 2026-04-26

## Repo

- repo: `/home/cv25/siwon/sam2act_role_graph_memory`
- branch: `exp/role-graph-phase-contrastive`

## Goal

Verify that the new role-graph / phase / contrastive / additive-bias path
actually trains and evaluates without collapsing from a code bug.

## Stage1 Start Point

Used dirty stage1 checkpoint:

- `/home/cv25/siwon/sam2act/sam2act/runs/sam2act_memorybench_put_block_back_freshdebug_20260405/model_38.pth`

## Smoke Train Setup

- task: `put_block_back`
- stage2 only
- `num_maskmem=11`
- `bs=12`
- `epochs=1`
- `train_iter=120`  -> `10` train iterations
- `same_trans_aug_per_seq=True`
- `phase_aux_loss_weight=0.1`
- `role_graph_loss_weight=0.05`
- `role_ref_loss_weight=0.02`
- `anchor_use_loss_weight=0.01`
- `role_contrastive_loss_weight=0.01`
- `role_graph_bias_scale=0.05`
- `anchor_use_bias_scale=0.02`
- `persistent_anchor_enabled=True`

Run dir:

- `/home/cv25/siwon/sam2act_role_graph_memory/sam2act/runs/sam2act_pbb_role_graph_smoke2_20260426`

## First Bug Found

First smoke failed in the contrastive loss:

- location: `sam2act/models/sam2act_agent.py`
- issue: masking the similarity matrix with `-1e9` under half precision caused
  overflow

Fix:

- cast embeddings to `float32`
- change diagonal fill from `-1e9` to `-1e4`

## Smoke Train Outcome

Training completed.

Final printed summary:

- `total_loss`: `6.5540`
- `trans_loss`: `5.3843`
- `phase_aux_loss`: `6.3898`
- `phase_aux_acc`: `0.20`
- `role_graph_loss`: `8.7456`
- `role_graph_acc`: `0.10`
- `role_ref_loss`: `2.0034`
- `role_ref_top3_acc`: `0.90`
- `anchor_use_loss`: `1.8762`
- `anchor_use_acc`: `0.525`
- `role_contrastive_loss`: `3.4562`

## Smoke Eval Outcome

Eval command:

- `1` episode
- `headless`
- `put_block_back`

Result:

- score: `0.0`
- episode length: `25`

Eval file:

- `/home/cv25/siwon/sam2act_role_graph_memory/sam2act/runs/sam2act_pbb_role_graph_smoke2_20260426/eval/smoke/pbb_role_graph_smoke2/model_plus_last/eval_results.csv`

## What This Means

Good:

- replay with new role fields builds correctly
- model initializes from stage1 checkpoint
- additive attention bias path runs
- role/phase/ref/anchor/contrastive losses all backpropagate
- headless eval also runs

Not good:

- rollout quality is still poor
- current weights are too weak or too misbalanced to improve policy behavior

So this smoke should be read as:

- **implementation path verified**
- **algorithm not yet tuned**
