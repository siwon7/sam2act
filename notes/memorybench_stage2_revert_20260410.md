## MemoryBench Stage-2 Revert Record

Date: 2026-04-10

Decision:
- Revert only the stage-two baseline-sensitive files to `origin/main`.
- Keep stage-one checkpoints and stage-one code path unchanged.

Why:
- RLBench evaluation is working, so the common backbone and basic action path are not the main issue.
- MemoryBench remains low even after environment fixes and `num_maskmem=10` retry.
- The highest-risk local divergence from upstream was the stage-two memory path rewrite.

Reverted files:
- `sam2act/mvt/mvt_sam2_single.py`
- `sam2act/mvt/mvt_sam2.py`
- `sam2act/models/sam2act_agent.py`
- `sam2act/mvt/config.py`

Not reverted in this step:
- `sam2act/eval.py`
- `sam2act/train_plus.py`
- `sam2act/utils/rvt_utils.py`

Reason they were not reverted now:
- They are useful for stable local evaluation and checkpoint diagnostics.
- They are less likely than the memory-path rewrite to be the direct cause of the MemoryBench regression.

Compatibility follow-up:
- Added `hasattr(...)` guards around `set_task_vocab` in `sam2act/train.py`, `sam2act/train_plus.py`, and `sam2act/eval.py`.
- Added a `hasattr(...)` guard around `set_eval_task` in `sam2act/eval.py`.
- These guards are only to let the reverted upstream baseline agent run under the current local runner without reintroducing the stage-two memory-path rewrite.
- Added a minimal `use_memory=True` inference compatibility patch in `sam2act/models/sam2act_agent.py` so MemoryBench stage-two eval can read waypoint and RGC outputs from the correct stage-two path during rollout.

Observed MemoryBench results before this revert:
- `freshdebug_fix` full 25ep: `24.0`
- `freshdebug_fix` after extra stage-two training to epoch 35: `20.0`
- `freshdebug_fix_mem10` full 25ep: `20.0`

Next step:
- Re-run MemoryBench `put_block_back` stage-two from the existing stage-one checkpoint using the reverted baseline-sensitive code.
- Evaluate again under the verified environment:
  - `DISPLAY=:3`
  - regular `CoppeliaSim`
  - `headless=true`
  - `eval_datafolder=./data_memory/test`

Follow-up findings:
- The reverted upstream stage-two `use_memory=True` eval path was crashing because `get_q()` returned `rot_q/grip_q/collision_q = None` while `act()` still passed them into `get_pred()`.
- Minimal local compatibility fix: keep stage-two eval decoding the waypoint from `mvt2`, but recover `rot/grip/collision` logits when the output dict contains them.
- After that fix, `5ep` eval runs again for reverted stage-two checkpoints.

Recent verified evals:
- `runs/sam2act_memorybench_put_block_back_stage2revert_20260410/model_plus_last.pth`: `5ep = 20.0`
- `runs/sam2act_memorybench_put_block_back/model_plus_last.pth`: `5ep = 0.0` under the same current eval path

Current reproduction run:
- Created a fresh, paper-faithful stage-two rerun directory from the reference `put_block_back` stage-one checkpoint:
  - `runs/sam2act_memorybench_put_block_back_repro20_20260410`
- This rerun uses:
  - reference `stage1` checkpoint from `runs/sam2act_memorybench_put_block_back/model_last.pth`
  - `stage2` config with `epochs: 20`
  - `num_maskmem: 9`
  - `bs: 10`
  - `fresh-start` stage-two training

Current status:
- `stage2revert` eval now runs again after the stage-two inference compatibility fix.
- `stage2revert` latest checkpoint `5ep` smoke on `put_block_back` is currently `20.0`.
- The faithful reproduction path is being continued with `epochs=20` rather than the accidental `35` inherited from the debug copy.
