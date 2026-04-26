# Transition Pointer Smoke 2026-04-26

## Repo

- `/home/cv25/siwon/sam2act_transition_pointer_memory`
- branch: `exp/slot-mode-pointer`

## Smoke Goal

Verify that the new-vs-revisit path is live end-to-end with:

- grouped role labels
- visit-mode loss
- grouped role-ref supervision
- prototype-first persistent anchors
- revisit-gated additive memory bias

## Command

Script:

- `scripts/run_put_block_back_transition_pointer_smoke.sh`

Key settings:

- task: `put_block_back`
- stage1 init: dirty `model_38.pth`
- `num_maskmem=11`
- `epochs=1`
- `train_iter=120`
- `phase_aux_loss_weight=0.05`
- `role_graph_loss_weight=0.02`
- `visit_mode_loss_weight=0.05`
- `role_ref_loss_weight=0.01`
- `anchor_use_loss_weight=0.0`
- `role_contrastive_loss_weight=0.0`
- `role_graph_bias_scale=0.02`
- `anchor_use_bias_scale=0.01`

## Smoke Result

Run:

- `sam2act_pbb_transition_pointer_smoke_20260426_180732`

Eval file:

- `/home/cv25/siwon/sam2act_transition_pointer_memory/sam2act/runs/sam2act_pbb_transition_pointer_smoke_20260426_180732/eval/smoke/pbb_transition_pointer_smoke_20260426_180732/model_plus_last/eval_results.csv`

Result:

- `put_block_back = 0.0`
- episode length `25`

## Final Train Summary

- `total_loss = 5.9638`
- `trans_loss = 5.2313`
- `phase_aux_loss = 3.7098`
- `role_graph_loss = 10.2954`
- `visit_mode_loss = 6.2113`
- `role_ref_loss = 3.0570`
- `role_ref_top3_acc = 0.74`
- `phase_aux_acc = 0.3417`
- `role_graph_acc = 0.1167`
- `visit_mode_acc = 0.5`

## Interpretation

The good news:

- the new path trains and evaluates end-to-end
- it does not collapse immediately from a code-path bug
- replay regeneration with the new schema works
- the grouped role/reference bias path is active

The bad news:

- rollout is still `0.0`
- `visit_mode_acc = 0.5` is weak
- grouped role accuracy is still very low at this short horizon

So this smoke validates the implementation path, but not the method quality.

## Runs Left Running

Two longer single-GPU runs were launched after this smoke:

1. `sam2act_pbb_transition_pointer_balanced_20260426`
   - GPU `0`
   - `phase=0.05`, `role=0.02`, `visit=0.05`, `role_ref=0.01`

2. `sam2act_pbb_transition_pointer_soft_20260426`
   - GPU `1`
   - `phase=0.05`, `role=0.01`, `visit=0.02`, `role_ref=0.01`

They use:

- `epochs=5`
- `train_iter=1000`
- `eval_episodes=5`
- replay reuse (`REFRESH_REPLAY=0`)

## Next Readouts

For the longer runs, the most important indicators are:

1. whether `visit_mode_acc` rises above the smoke level
2. whether `role_ref_top3_acc` remains high without hurting `trans_loss`
3. whether `5ep` smoke rises above `0.0`
