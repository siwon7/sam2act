#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CODE_ROOT="$UPSTREAM_ROOT/sam2act"
RUN_NAME="${RUN_NAME:-upstream_put_block_back_stage2_20260415}"
RUN_DIR="$CODE_ROOT/runs/sam2act_${RUN_NAME}"
STAGE1_SRC="${STAGE1_SRC:-/home/cv25/siwon/sam2act/sam2act/runs/sam2act_memorybench_put_block_back_cleanrepro_20260411/model_last.pth}"
SMOKE_EPISODES="${SMOKE_EPISODES:-5}"
FULL_EPISODES="${FULL_EPISODES:-25}"

mkdir -p "$RUN_DIR"
cp -f "$STAGE1_SRC" "$RUN_DIR/model_last.pth"

"$SCRIPT_DIR/run_train_plus_upstream.sh" \
  --log-dir runs \
  --exp_cfg_path configs/sam2act_plus.yaml \
  --mvt_cfg_path mvt/configs/sam2act_plus.yaml \
  --exp_cfg_opts "tasks put_block_back exp_name ${RUN_NAME} wandb False"

"$SCRIPT_DIR/run_eval_upstream.sh" \
  --model-folder "runs/sam2act_${RUN_NAME}" \
  --eval-datafolder ./data_memory/test \
  --tasks put_block_back \
  --eval-episodes "$SMOKE_EPISODES" \
  --log-name "upstream_stage2_smoke_${SMOKE_EPISODES}_$(date +%Y%m%d_%H%M%S)" \
  --device 0 \
  --headless \
  --model-name model_plus_last.pth

"$SCRIPT_DIR/run_eval_upstream.sh" \
  --model-folder "runs/sam2act_${RUN_NAME}" \
  --eval-datafolder ./data_memory/test \
  --tasks put_block_back \
  --eval-episodes "$FULL_EPISODES" \
  --log-name "upstream_stage2_full_${FULL_EPISODES}_$(date +%Y%m%d_%H%M%S)" \
  --device 0 \
  --headless \
  --model-name model_plus_last.pth
