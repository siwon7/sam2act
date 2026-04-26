#!/usr/bin/env bash
set -euo pipefail

TASK="${1:-put_block_back}"
STAGE1_RUN_DIR="${2:-/home/cv25/siwon/sam2act/sam2act/runs/sam2act_memorybench_put_block_back_stage2revert_20260410}"
LR2X_EXP="${3:-mb_stage2_lr2x_20260421}"
AUTO_EVAL_EPISODES="${4:-5}"

REPO_ROOT="/home/cv25/siwon/sam2act"
LOG_ROOT="$REPO_ROOT/logs"
SELF_LOG="$LOG_ROOT/continue_memorybench_stage2_4gpu_after_lr2x_$(date '+%Y%m%d_%H%M%S').log"

mkdir -p "$LOG_ROOT"

wait_for_lr2x() {
  while pgrep -af "train_plus.py .*exp_name ${LR2X_EXP}" >/dev/null; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] waiting for ${LR2X_EXP} to finish" | tee -a "$SELF_LOG"
    sleep 120
  done
}

run_smoke_eval() {
  local run_dir="$1"
  local exp_name="$2"

  set +u
  source /home/cv25/miniconda3/etc/profile.d/conda.sh
  conda activate sam2act5090
  set -u

  export DISPLAY=:3
  export XAUTHORITY=/home/cv25/.Xauthority
  export COPPELIASIM_ROOT=/home/cv25/siwon/coppeliasim/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
  export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
  export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:/home/cv25/miniconda3/envs/sam2act5090/lib:/home/cv25/miniconda3/envs/sam2act5090/lib/python3.10/site-packages/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export PYTHONUNBUFFERED=1
  export CUDA_VISIBLE_DEVICES=3

  cd "$REPO_ROOT/sam2act"
  local log_name="${exp_name}_smoke${AUTO_EVAL_EPISODES}_$(date '+%Y%m%d_%H%M%S')"
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] smoke eval start ${exp_name}" | tee -a "$SELF_LOG"
  python eval.py \
    --model-folder "$run_dir" \
    --model-name model_plus_last.pth \
    --eval-datafolder ./data_memory/test \
    --tasks "$TASK" \
    --eval-episodes "$AUTO_EVAL_EPISODES" \
    --log-name "$log_name" \
    --device 0 \
    --headless \
    2>&1 | tee -a "$SELF_LOG"
}

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] orchestrator start" | tee -a "$SELF_LOG"
wait_for_lr2x
run_smoke_eval "$REPO_ROOT/sam2act/runs/sam2act_${LR2X_EXP}" "$LR2X_EXP"

/home/cv25/siwon/sam2act/scripts/run_memorybench_stage2_4gpu_ablation.sh \
  "$TASK" "$STAGE1_RUN_DIR" "mb_stage2_base_20260421" "10" "1.25e-06" "20" "false" "true" "$AUTO_EVAL_EPISODES" \
  2>&1 | tee -a "$SELF_LOG"

/home/cv25/siwon/sam2act/scripts/run_memorybench_stage2_4gpu_ablation.sh \
  "$TASK" "$STAGE1_RUN_DIR" "mb_stage2_bs2x_20260421" "20" "1.25e-06" "20" "false" "true" "$AUTO_EVAL_EPISODES" \
  2>&1 | tee -a "$SELF_LOG"

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] orchestrator finished" | tee -a "$SELF_LOG"
