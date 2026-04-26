#!/usr/bin/env bash
set -euo pipefail

TASK="${1:?task required}"
STAGE1_RUN_DIR="${2:?stage1 run dir required}"
AUTO_EVAL_EPISODES="${3:-5}"
SUMMARY_LOG="/home/cv25/siwon/sam2act/logs/memorybench_stage2_4gpu_chain_$(date '+%Y%m%d_%H%M%S').log"

REPO_ROOT="/home/cv25/siwon/sam2act"
CODE_ROOT="$REPO_ROOT/sam2act"

declare -a VARIANTS=(
  "mb_stage2_base_20260421 10 1.25e-06 20 false true"
  "mb_stage2_lr2x_20260421 10 2.5e-06 20 false true"
  "mb_stage2_bs2x_20260421 20 1.25e-06 20 false true"
)

for variant in "${VARIANTS[@]}"; do
  read -r EXP_NAME BS LR EPOCHS SAME_AUG TRANS_AUG <<<"$variant"
  echo "=== $(date '+%Y-%m-%d %H:%M:%S %Z') start $EXP_NAME ===" | tee -a "$SUMMARY_LOG"
  /home/cv25/siwon/sam2act/scripts/run_memorybench_stage2_4gpu_ablation.sh \
    "$TASK" "$STAGE1_RUN_DIR" "$EXP_NAME" "$BS" "$LR" "$EPOCHS" "$SAME_AUG" "$TRANS_AUG" "$AUTO_EVAL_EPISODES"
  echo "=== $(date '+%Y-%m-%d %H:%M:%S %Z') finish $EXP_NAME ===" | tee -a "$SUMMARY_LOG"
done
