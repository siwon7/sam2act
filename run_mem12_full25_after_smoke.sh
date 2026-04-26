#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="/home/cv25/siwon/sam2act_memdebug/sam2act/runs/sam2act_mb_stage2_mem12_bs13_20260422"
EVAL_CSV="$RUN_DIR/eval/mb_stage2_mem12_bs13_20260422_smoke5_$(date +%Y%m%d)_*.csv"
# wait for training to finish and smoke eval results to appear
while true; do
  if compgen -G "$RUN_DIR/eval/mb_stage2_mem12_bs13_20260422_smoke5_*/model_plus_last/eval_results.csv" > /dev/null; then
    break
  fi
  sleep 60
done
cd /home/cv25/siwon/sam2act_memdebug/sam2act
DISPLAY=:3 CUDA_VISIBLE_DEVICES=3 conda run -n sam2act5090 python eval.py \
  --tasks put_block_back \
  --model-folder runs/sam2act_mb_stage2_mem12_bs13_20260422 \
  --model-name model_plus_last.pth \
  --eval-datafolder ./data_memory/test \
  --eval-episodes 25 \
  --log-name mem12_bs13_full25_gpu3_display3_20260422 \
  --device 0 \
  --headless
