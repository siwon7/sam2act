# cv11 Server Handoff: Transition Pointer Branch

## Repo

- worktree: `/home/cv25/siwon/sam2act_transition_pointer_memory`
- branch: `exp/slot-mode-pointer`

## What This Branch Adds

Relative to the earlier role-graph branch, this branch adds:

- explicit `visit_mode_label` (`new` vs `revisit`)
- `visit_mode_loss`
- revisit-gated additive memory bias
- persistent anchors keyed by first prototype occurrence instead of first
  timestep only

This branch is intended for the observation:

> MemoryBench requires both new-location behavior and revisit behavior.

## Main Notes

- design note:
  - `/home/cv25/siwon/sam2act_transition_pointer_memory/notes/transition_pointer_memory_design.md`
- smoke note:
  - `/home/cv25/siwon/sam2act_transition_pointer_memory/notes/transition_pointer_smoke_20260426.md`

## Scripts

### Smoke

- `/home/cv25/siwon/sam2act_transition_pointer_memory/scripts/run_put_block_back_transition_pointer_smoke.sh`

### Longer Single-GPU Run

- `/home/cv25/siwon/sam2act_transition_pointer_memory/scripts/run_put_block_back_transition_pointer_train.sh`

Useful environment overrides:

- `GPU`
- `RUN_NAME`
- `REFRESH_REPLAY`
- `EPOCHS`
- `TRAIN_ITER`
- `EVAL_EPISODES`
- `PHASE_W`
- `ROLE_W`
- `VISIT_W`
- `ROLE_REF_W`
- `ANCHOR_W`
- `CONTRAST_W`
- `ROLE_BIAS`
- `ANCHOR_BIAS`

## Current Local Outcome

Smoke runs end-to-end successfully, but the latest smoke score is still `0.0`.

That means:

- the code path is live
- replay regeneration is correct
- the new loss heads are wired correctly
- the method still needs tuning

## Suggested First cv11 Runs

1. rerun the smoke once with `REFRESH_REPLAY=1`
2. if it matches local behavior, run the single-GPU trainer with:
   - `EPOCHS=5`
   - `TRAIN_ITER=1000`
3. compare:
   - a balanced setting
   - a softer `visit_mode` setting

## Important Files

- grouped targets:
  - `sam2act/utils/memorybench_role_graph.py`
- replay schema:
  - `sam2act/utils/dataset.py`
- memory logic:
  - `sam2act/mvt/mvt_sam2_single.py`
- losses:
  - `sam2act/models/sam2act_agent.py`
