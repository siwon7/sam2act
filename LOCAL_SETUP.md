# Upstream Clean Setup

This workspace is a clean `origin/main` worktree of `sam2act` at commit `3be785b`.

Paths:
- repo root: `/home/cv25/siwon/sam2act_upstream_main`
- code root: `/home/cv25/siwon/sam2act_upstream_main/sam2act`

Linked resources:
- `sam2act/data` -> local RLBench dataset
- `sam2act/data_memory` -> local MemoryBench dataset
- `sam2act/replay_temporal` -> local replay buffer
- `sam2act/replay_temporal_memory` -> local memory replay buffer
- `sam2act/mvt/sam2_train/checkpoints/sam2.1_hiera_base_plus.pt` -> local SAM2 checkpoint

Execution helpers:
- `scripts/env_upstream.sh`
- `scripts/run_train_plus_upstream.sh`
- `scripts/run_eval_upstream.sh`

Notes:
- This uses the existing `sam2act5090` conda env for practicality.
- The code path is clean upstream; the Python environment is not a brand-new env.
- `DISPLAY` defaults to `:3` because this machine has a live TigerVNC session there.

Examples:

```bash
/home/cv25/siwon/sam2act_upstream_main/scripts/run_train_plus_upstream.sh \
  --exp_cfg_opts "tasks put_block_back" \
  --exp_cfg_path configs/sam2act_plus.yaml \
  --mvt_cfg_path mvt/configs/sam2act_plus.yaml
```

```bash
/home/cv25/siwon/sam2act_upstream_main/scripts/run_eval_upstream.sh \
  --model-folder runs/sam2act_plus_put_block_back \
  --eval-datafolder ./data_memory/test \
  --tasks put_block_back \
  --eval-episodes 5 \
  --log-name smoke/1 \
  --device 0 \
  --headless \
  --model-name model_plus_19.pth
```
