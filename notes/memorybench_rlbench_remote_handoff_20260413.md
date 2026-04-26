# MemoryBench / RLBench / Remote Migration Handoff

Date: 2026-04-13

Authoring context:
- Local workspace: `/home/cv25/siwon/sam2act`
- Remote workspace: `cvlab-dgx:/home/cvlab-dgx/siwon`
- Main task under investigation: `MemoryBench put_block_back`

## 1. Executive summary

This document records:
- what was changed locally while debugging `SAM2Act` and `SAM2Act+`
- what was tried for `MemoryBench`
- what worked and did not work
- what exact RLBench evaluation setup was verified
- what was migrated to `cvlab-dgx`
- what remains unresolved

Current high-level conclusion:
- `RLBench` is reproducible on the current codebase and on the remote server when evaluated under the verified display + regular CoppeliaSim setup.
- `MemoryBench put_block_back` is still not reproduced to paper-level performance under the current local setup, even after stage-one and stage-two retraining attempts.
- The current evidence points more strongly to a `MemoryBench stage-two / data / environment / public-repo gap` problem than to a simple common-backbone failure.

## 2. Verified RLBench setup

The RLBench setup that was actually verified to work is:
- `conda env`: `sam2act5090` locally, `sam2act` remotely
- `DISPLAY=:3` locally, `DISPLAY=:0` remotely
- regular `CoppeliaSim_Edu_V4_1_0_Ubuntu20_04`
- not the `_headless` root
- run from `sam2act/sam2act`
- explicit config arguments:
  - `--exp_cfg_path configs/sam2act.yaml`
  - `--mvt_cfg_path mvt/configs/sam2act.yaml`

Verified local RLBench reference:
- `runs/sam2act_rlbench/model_89.pth`
- `verify_model89_smoke_d3.log` gave `insert_onto_square_peg = 100.0`
- `eval_full_model89_d3` produced `average ~= 87.11`

Verified remote RLBench reference:
- `cvlab-dgx:/home/cvlab-dgx/siwon/sam2act/sam2act/runs/sam2act_rlbench/model_89.pth`
- remote smoke run on `insert_onto_square_peg` succeeded with `100.0`

Meaning:
- shared model load path, renderer, PyRep, RLBench bindings, and action rollout are functioning under the verified setup.

## 3. MemoryBench main findings

### 3.1 Environment mattered

MemoryBench evaluation quality depended on simulator setup:
- `_headless` CoppeliaSim root gave `0.0`
- `DISPLAY + regular CoppeliaSim + headless flag` improved to non-zero results

This was first a real blocker:
- `libsimExtCustomUI` / segfault happened under the wrong CoppeliaSim root
- once switched to the regular install with a display, the crash path disappeared

### 3.2 Stage-two behavior improved over stage-one, but not enough

Observed behavior:
- `stage1` was often near `0`
- `stage2` could reach `20~24` in some runs

Meaning:
- the memory-enabled stage was not totally dead
- but it did not converge to the expected `SAM2Act+` MemoryBench quality

### 3.3 The failure pattern was not a simple random 4-way return-slot error

Video inspection suggested:
- the policy could often pick, lift, move, and sometimes press the button
- failures often happened in the later phase
- it sometimes hovered after the button or failed to cleanly transition to the final return subgoal

Interpretation:
- not just `which initial slot was correct`
- more likely `phase transition / procedural state tracking / memory recall timing`

### 3.4 Increasing `num_maskmem` was constrained by the data

Tried:
- `num_maskmem = 12`, `bs = 13`

Result:
- training appeared to hang before meaningful progress

Root cause:
- not a model-shape hardcode
- the temporal replay sampler required `num_obs = num_maskmem + 1`
- for `put_block_back`, almost all sequences were shorter than `13`
- therefore the sampler kept skipping candidates and effectively stalled

Consequence:
- `num_maskmem = 12` is not viable under the current replay data/sampler combination without changing sampler logic

## 4. MemoryBench experiments and outcomes

### 4.1 Fresh/debug stage-two variants

Observed runs included:
- `freshdebug_fix`
- `freshdebug_fix_mem10`
- `stage2revert`
- `repro20`
- `cleanrepro`

Representative results:
- `freshdebug_fix` full 25ep: `24.0`
- `freshdebug_fix` after further stage-two training: `20.0`
- `freshdebug_fix_mem10` full 25ep: `20.0`
- `stage2revert` 5ep: `20.0`
- `stage2revert` 25ep on `model_plus_8`: `20.0`
- `cleanrepro` full 25ep latest: `8.0`
- `cleanrepro` `model_plus_15` 5ep: `20.0`
- `cleanrepro` `model_plus_19` 5ep: `0.0`

Interpretation:
- re-running cleanly did not recover the expected paper-level result
- later stage-two checkpoints could even degrade

### 4.2 Stage-one / stage-two clean rerun conclusion

A clean rerun path was attempted:
- new `stage1`
- then new `stage2`

Current conclusion from that path:
- the latest clean reproduction line still underperformed badly
- therefore the issue is not explained by a single stale checkpoint alone

## 5. Method ideas explored for future MemoryBench work

These were not all kept in the final baseline path, but they were the main ideas explored:

1. `shared_fifo`
- plain shared-memory baseline
- intended as the multitask lower bound

2. `routing`
- task-aware retrieval from shared memory
- hypothesis: interference is mostly at read time

3. `dual_memory`
- split between recent/local memory and longer/event-like memory
- hypothesis: short-horizon and long-horizon memory should not share the same bank

4. `full`
- routing + dual-memory + event write + pruning style ideas

Additional conceptual insights considered:
- fixed update path rather than overly learnable recurrent propagation
- auxiliary visual memory loss
- streaming recurrent training

These ideas were discussed because:
- the public `SAM2Act+` MemoryBench path looked weak under current reproduction
- the behavior suggested memory/state-transition failure rather than generic manipulation failure

## 6. Important local code changes that remain

The following files still have meaningful local modifications relative to the repo baseline.

### 6.1 `sam2act/models/sam2act_agent.py`

Purpose:
- restore stage-two evaluation compatibility when `use_memory=True`

Key change:
- `get_q()` no longer returns `rot_q/grip_q/collision_q = None` blindly for the memory path
- `get_pred()` now reconstructs `rot/grip/collision` if missing

Why:
- reverted or upstream-like stage-two MemoryBench rollout was crashing during eval because action decoding expected those logits

Impact:
- this is primarily an inference compatibility fix
- not intended as a method change

### 6.2 `sam2act/eval.py`

Purpose:
- make evaluation more robust locally

Key changes:
- safer device resolution
- `hasattr(...)` guards for `set_task_vocab` and `set_eval_task`
- direct `imageio` video writing instead of the older ffmpeg shell path
- allow video recording in eval loop
- fallback handling for `Value(...)` creation

Why:
- needed to keep both current local runners and reverted agents evaluable

### 6.3 `sam2act/train.py`

Purpose:
- safer stage-one training and resume behavior

Key changes:
- `--use-memory-data`
- `--fresh-start`
- rank-0-first replay creation for DDP
- fallback from full resume to model-only resume when optimizer state mismatches
- `set_task_vocab` compatibility guard

Why:
- stage-one training was vulnerable to replay generation races and optimizer-group mismatch on resumed runs

### 6.4 `sam2act/train_plus.py`

Purpose:
- safer stage-two training and more explicit handoff logging

Key changes:
- DDP-safe replay initialization
- `--fresh-start`
- `set_task_vocab` compatibility guard
- explicit logging that stage-two keeps memory modules fresh

Important note:
- stage-two still intentionally excludes:
  - `memory_attention`
  - `memory_encoder`
  - `maskmem_tpos_enc`

Meaning:
- stage-two memory modules are freshly initialized by design

### 6.5 `sam2act/utils/rvt_utils.py`

Purpose:
- improve checkpoint load diagnostics

Key changes:
- `_load_state_dict_with_report`
- missing/unexpected key prefix summaries

Why:
- useful for seeing whether stage-one/stage-two checkpoint contracts drifted

### 6.6 `sam2act/mvt/renderer.py`

Purpose:
- local renderer flexibility

Key changes:
- support for `three_views`
- safer `torch.cuda.empty_cache()` guard

### 6.7 `sam2act/libs/point-renderer/point_renderer/renderer.py`

Key change:
- removed `@torch.jit.script` decoration on `_prep_render_batch_inputs`

Reason:
- local compatibility / build behavior

### 6.8 `sam2act/utils/custom_rlbench_env.py`

Purpose:
- support richer RLBench-side debugging / GT extraction

Key changes:
- store current demo
- `extract_obs`
- `get_ground_truth_action`

This is useful for debugging and comparison, not specifically the MemoryBench regression.

## 7. Important reverted path

A stage-two-sensitive revert was recorded in:
- `notes/memorybench_stage2_revert_20260410.md`

That revert targeted the highest-risk MemoryBench divergence:
- `sam2act/mvt/mvt_sam2_single.py`
- `sam2act/mvt/mvt_sam2.py`
- `sam2act/models/sam2act_agent.py`
- `sam2act/mvt/config.py`

Conclusion from that revert path:
- even after reverting the baseline-sensitive stage-two path and patching eval compatibility, MemoryBench was still poor
- so the issue could not be blamed only on the new local memory-path rewrite

## 8. Remote migration to `cvlab-dgx`

### 8.1 What was copied

Copied code-first repo set to `~/siwon`:
- `ELSA-Robotics-Challenge`
- `IsaacLab-Scanbot`
- `MemoryVLA`
- `MindExplore`
- `PyRep`
- `RLBench-peract`
- `RLBench-stepjam`
- `RMBench`
- `RVT`
- `TGM-VLA`
- `YARR-peract`
- `mrest-rlbench`
- `peract`
- `pytorch3d-5090`
- `robomme_benchmark`
- `robot-colosseum`
- `rvt_colosseum`
- `sam2act`

Excluded during sync:
- `.git`
- local codex metadata
- `runs`
- `logs`
- `data`
- `data_memory`
- `replay`
- checkpoints
- videos
- large binary artifacts

### 8.2 What was added on remote

Added:
- `CoppeliaSim_Edu_V4_1_0_Ubuntu20_04`
- remote activation script:
  - `~/siwon/bin/cvlab_dgx_sam2act_activate.sh`
- RLBench setup note:
  - `~/siwon/bin/cvlab_dgx_rlbench_eval_setup.txt`
- migration bundle:
  - `~/siwon/migration_bundle_20260413`

### 8.3 Remote fixes needed for runtime

1. Fixed `libcoppeliaSim.so.1` symlink inside remote `CoppeliaSim_Edu`
2. Rebuilt `sam2act/sam2act/libs/point-renderer/point_renderer/_C.so` on the remote host

Why:
- the local-built extension required newer GLIBC than the remote host provided

### 8.4 Remote verification

Verified:
- `sam2act` imports with PyRep after environment activation
- RLBench smoke eval works remotely
- MemoryBench smoke eval runs remotely once data and the run folder are copied

Observed remote MemoryBench result for current `cleanrepro`:
- `model_plus_15`, `5ep`, `put_block_back`: `0.0`

## 9. Environment and package migration status

Exported env specs:
- `elsa-robotics-challenge.yml`
- `ibrl.yml`
- `kdd2026.yml`
- `sam2act.yml`
- `sam2act118.yml`
- `sam2act5090.yml`
- `sam2act5090w.yml`
- `sigir26.yml`
- `tgm-vla.yml`

Remote creation status observed:
- created:
  - `elsa-robotics-challenge`
  - `kdd2026`
- already existed:
  - `sam2act`
- failures observed:
  - `ibrl`
  - `sam2act118`

Why those failed:
- package/channel mismatch on remote
- examples:
  - `cuda-nvtx=12.1.105`
  - `pytorch3d=0.7.8=py310_cu118_pyt231`
  - `pytorch-cuda=11.8`
  - `fvcore`
  - `iopath`

Meaning:
- the bundle is good as a migration starting point
- some envs still require manual channel tuning or relaxed specs

## 10. Current best operational commands on remote

Base shell:
```bash
ssh -p 52023 cvlab-dgx@10.150.21.73
conda activate sam2act
source ~/siwon/bin/cvlab_dgx_sam2act_activate.sh
cd ~/siwon/sam2act/sam2act
```

RLBench smoke:
```bash
python eval.py \
  --model-folder runs/sam2act_rlbench \
  --eval-datafolder ./data/test \
  --tasks insert_onto_square_peg \
  --eval-episodes 1 \
  --episode-length 25 \
  --device 0 \
  --headless \
  --model-name model_89.pth \
  --exp_cfg_path configs/sam2act.yaml \
  --mvt_cfg_path mvt/configs/sam2act.yaml
```

MemoryBench smoke:
```bash
python eval.py \
  --tasks put_block_back \
  --model-folder runs/sam2act_memorybench_put_block_back_cleanrepro_20260411 \
  --model-name model_plus_15.pth \
  --eval-datafolder ./data_memory/test \
  --eval-episodes 5 \
  --episode-length 25 \
  --device 0 \
  --headless \
  --exp_cfg_path runs/sam2act_memorybench_put_block_back_cleanrepro_20260411/exp_cfg_plus.yaml \
  --mvt_cfg_path runs/sam2act_memorybench_put_block_back_cleanrepro_20260411/mvt_cfg_plus.yaml
```

## 11. What is still unresolved

1. Why the public/local `SAM2Act+` MemoryBench path does not reproduce expected `put_block_back` performance.
2. Whether the main gap is:
   - public-repo vs paper-internal code
   - replay/data construction mismatch
   - evaluation mismatch
   - stage-two memory-learning instability
3. Which MemoryBench checkpoints are genuinely best under stable repeated evaluation.

## 12. Recommended next actions

1. Keep RLBench as the verified control task.
2. For MemoryBench, compare on the same environment:
   - reference run
   - `cleanrepro` `model_plus_15`
   - `cleanrepro` `model_plus_last`
3. Save videos for the exact same seed set across checkpoints.
4. If method development resumes, do it on a branch or a clearly separated config path, not inside the baseline path.
5. For remote env migration, fix failed envs one by one only if they are actually needed.

