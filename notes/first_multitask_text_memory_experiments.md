# First Multitask Text-Memory Experiments

## Scope

This branch is for experiment scaffolding only. Core `sam2act/` package changes
should come later, after the run surface and note structure stabilize.

## Immediate Goal

Use a single 4-GPU stage2 launcher that can cover four experiment families:

1. **Baseline multitask MemoryBench stage2**
2. **`mem11` window extension where it helps**
3. **Coarse phase aux loss and text/task memory gate experiments**
4. **Persistent anchor memory experiments**

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

### Exp 4: text/task memory gate

- same as Exp 3
- set `ENABLE_TEXT_TASK_GATE=1`
- pass concrete option names through:
  - `TEXT_TASK_GATE_EXP_CFG_OPTS`
  - `TEXT_TASK_GATE_MVT_CFG_OPTS`

Expected future semantics:
- task/language-conditioned routing over memory candidates
- supports persistent task anchors without overfitting to a single task

### Exp 5: persistent anchor memory

- same as Exp 4
- set `ENABLE_PERSISTENT_ANCHOR=1`
- default opts:
  - `persistent_anchor_enabled True`
  - `persistent_anchor_max_steps 2`

Expected semantics:
- preserve early task-critical states in a dedicated anchor bank
- allow late return phases to access initial evidence even when recent-window
  attention is biased toward an intermediate anchor such as the button phase
- stay generic across tasks by defaulting to the first `k` key steps rather
  than hardcoding task-specific node ids inside the launcher

## Practical Notes

- `put_block_back` benefits from `mem11`; `reopen_drawer` and
  `rearrange_block` may not need the same value.
- This scaffold therefore treats `mem11` as a launcher convenience, not as a
  final multitask design decision.
- Persistent anchors are intentionally separated from the FIFO window. This
  follows the observation that `put_block_back` often over-attends to the
  `move_over_button` phase even when teacher-forced memory exposes the initial
  slot in the bank.
- Future exact flag names should be added in core code, while the launcher
  remains stable by forwarding them through `EXTRA_*` and `*_OPTS` variables.

## Success Checks

- run starts cleanly from a stage1 checkpoint without editing core code
- resulting command line is easy to inspect from the log
- new phase/gate flags can be turned on by environment variables alone
- persistent anchor memory can be enabled without changing the launcher
