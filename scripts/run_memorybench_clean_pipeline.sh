#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CODE_ROOT="$UPSTREAM_ROOT/sam2act"

TASK="${TASK:-put_block_back}"
STAGE1_RUN_NAME="${STAGE1_RUN_NAME:-upstream_memorybench_${TASK}_stage1_$(date +%Y%m%d)}"
STAGE2_RUN_NAME="${STAGE2_RUN_NAME:-upstream_memorybench_${TASK}_stage2_$(date +%Y%m%d)}"
SMOKE_EPISODES="${SMOKE_EPISODES:-5}"
FULL_EPISODES="${FULL_EPISODES:-25}"

STAGE1_RUN_DIR="$CODE_ROOT/runs/sam2act_${STAGE1_RUN_NAME}"
STAGE2_RUN_DIR="$CODE_ROOT/runs/sam2act_${STAGE2_RUN_NAME}"

"$SCRIPT_DIR/run_train_stage1_upstream.sh" \
  --log-dir runs \
  --exp_cfg_path configs/sam2act.yaml \
  --mvt_cfg_path mvt/configs/sam2act.yaml \
  --exp_cfg_opts "tasks ${TASK} exp_name ${STAGE1_RUN_NAME} wandb False"

mkdir -p "$STAGE2_RUN_DIR"
cp -f "$STAGE1_RUN_DIR/model_last.pth" "$STAGE2_RUN_DIR/model_last.pth"

"$SCRIPT_DIR/run_train_plus_upstream.sh" \
  --log-dir runs \
  --exp_cfg_path configs/sam2act_plus.yaml \
  --mvt_cfg_path mvt/configs/sam2act_plus.yaml \
  --exp_cfg_opts "tasks ${TASK} exp_name ${STAGE2_RUN_NAME} wandb False"

"$SCRIPT_DIR/run_eval_upstream.sh" \
  --model-folder "runs/sam2act_${STAGE2_RUN_NAME}" \
  --eval-datafolder ./data_memory/test \
  --tasks "${TASK}" \
  --eval-episodes "$SMOKE_EPISODES" \
  --log-name "clean_stage2_smoke_${SMOKE_EPISODES}_$(date +%Y%m%d_%H%M%S)" \
  --device 0 \
  --headless \
  --model-name model_plus_last.pth

"$SCRIPT_DIR/run_eval_upstream.sh" \
  --model-folder "runs/sam2act_${STAGE2_RUN_NAME}" \
  --eval-datafolder ./data_memory/test \
  --tasks "${TASK}" \
  --eval-episodes "$FULL_EPISODES" \
  --log-name "clean_stage2_full_${FULL_EPISODES}_$(date +%Y%m%d_%H%M%S)" \
  --device 0 \
  --headless \
  --model-name model_plus_last.pth
