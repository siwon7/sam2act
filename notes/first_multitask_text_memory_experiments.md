# First Multitask Text-Memory Experiments

## Scope

This branch is for experiment scaffolding only. Core `sam2act/` package changes
should come later, after the run surface and note structure stabilize.

## Immediate Goal

Use a single 4-GPU stage2 launcher that can cover three experiment families:

1. **Baseline multitask MemoryBench stage2**
2. **`mem11` window extension where it helps**
3. **Future coarse phase aux loss and text/task memory gate experiments**

The launcher should stay usable even before exact final config flag names are
implemented.

## First Experiments

### Exp 1: Backward-compatible multitask baseline

- tasks: `(put_block_back,reopen_drawer,rearrange_block)`
- 4 GPUs
- stage2 only
- no extra experimental flags

Purpose:
- verify that the multitask run surface and checkpoint handoff work cleanly
- establish a baseline before new losses or gates are enabled

### Exp 2: `mem11`-only multitask scaffold

- same as Exp 1
- set `ENABLE_MEM11=1`
- keep all other future flags disabled

Purpose:
- keep the launcher compatible with `put_block_back`-style long temporal recall
- avoid baking task-specific graph logic into the run script

### Exp 3: coarse phase aux loss placeholder

- same as Exp 2
- set `ENABLE_COARSE_PHASE_AUX=1`
- pass concrete option names through:
  - `COARSE_PHASE_EXP_CFG_OPTS`
  - `COARSE_PHASE_MVT_CFG_OPTS`

Expected future semantics:
- shared coarse phase labels across tasks
- phase auxiliary loss improves phase disambiguation without hardcoding exact
  keypoint graphs per task

### Exp 4: text/task memory gate placeholder

- same as Exp 3
- set `ENABLE_TEXT_TASK_GATE=1`
- pass concrete option names through:
  - `TEXT_TASK_GATE_EXP_CFG_OPTS`
  - `TEXT_TASK_GATE_MVT_CFG_OPTS`

Expected future semantics:
- task/language-conditioned routing over memory candidates
- supports persistent task anchors without overfitting to a single task

## Practical Notes

- `put_block_back` benefits from `mem11`; `reopen_drawer` and
  `rearrange_block` may not need the same value.
- This scaffold therefore treats `mem11` as a launcher convenience, not as a
  final multitask design decision.
- Future exact flag names should be added in core code, while the launcher
  remains stable by forwarding them through `EXTRA_*` and `*_OPTS` variables.

## Success Checks

- run starts cleanly from a stage1 checkpoint without editing core code
- resulting command line is easy to inspect from the log
- new phase/gate flags can be turned on by environment variables alone
