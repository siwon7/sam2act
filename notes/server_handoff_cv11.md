# cv11 Server Handoff

## Repo

- worktree: `/home/cv25/siwon/sam2act_role_graph_memory`
- branch: `exp/role-graph-phase-contrastive`

## What Is In This Branch

- grouped `put_block_back` role graph labels
- MemoryBench `phase_label`
- persistent anchor memory bank
- additive graph-biased memory attention
- role graph / role ref / anchor use / contrastive auxiliary losses

Main note:

- `/home/cv25/siwon/sam2act_role_graph_memory/notes/role_graph_phase_contrastive_design.md`

Smoke note:

- `/home/cv25/siwon/sam2act_role_graph_memory/notes/role_graph_smoke_20260426.md`

## Fastest Local Smoke Script

- `/home/cv25/siwon/sam2act_role_graph_memory/scripts/run_put_block_back_role_graph_smoke.sh`

This script:

- links `data_memory`
- links the SAM2 base checkpoint
- seeds stage2 from dirty stage1 `model_38`
- optionally regenerates local `replay_temporal_memory`
- runs `1` epoch / `10` train iterations
- runs `1` eval episode

## Smoke Result So Far

The path is now live end-to-end, but not yet tuned:

- training completes
- eval completes
- latest smoke score: `0.0`
- failure mode: timeout at length `25`, not immediate step-1 crash

That means:

- implementation bugs in the role/phase/ref/anchor/contrastive path are mostly
  cleared
- the next step on cv11 should be weight tuning or longer training, not another
  rewrite from scratch

## Suggested First cv11 Runs

### A. Reproduce the local smoke with more compute stability

Use the smoke script as-is first, then increase:

- `epochs`
- `train_iter`
- `eval_episodes`

### B. 8-GPU tuning starting from this branch

Recommended first tuning knobs:

- `peract.phase_aux_loss_weight`
- `peract.role_graph_loss_weight`
- `peract.role_ref_loss_weight`
- `peract.anchor_use_loss_weight`
- `peract.role_contrastive_loss_weight`
- `mvt.role_graph_bias_scale`
- `mvt.anchor_use_bias_scale`

## Notes

- this branch is experimental and intentionally separate from `sam2act_upstream_main`
- for `put_block_back`, the current implementation assumes the grouped 6-role
  mapping documented in `memorybench_role_graph.py`
- if replay schema changes again, regenerate local replay with `--refresh_replay`
