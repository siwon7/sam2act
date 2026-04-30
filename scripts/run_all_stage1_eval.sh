#!/bin/bash
set -e

cd /home/cv11/project/siwon/sam2act_dirty/sam2act

export COPPELIASIM_ROOT=/home/cv11/project/siwon/coppeliasim/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:${LD_LIBRARY_PATH}"

SCRIPT=/home/cv11/project/siwon/sam2act_multipeak_graph/scripts/eval_stage1_heatmap.py
OUTDIR=/home/cv11/project/siwon/sam2act_multipeak_graph/results/stage1_eval

MODELS=(
  "runs/sam2act_memorybench_put_block_back"
  "runs/sam2act_memorybench_put_block_back_cleanrepro_20260411"
  "runs/sam2act_memorybench_put_block_back_freshdebug_fix_20260408"
  "runs/sam2act_memorybench_put_block_back_freshdebug_fix_mem10_20260409"
  "runs/sam2act_memorybench_put_block_back_freshdebug_fix_mem12_20260409"
  "runs/sam2act_memorybench_put_block_back_fullhandoff_20260414"
  "runs/sam2act_memorybench_put_block_back_repro20_20260410"
  "runs/sam2act_memorybench_put_block_back_freshdebug_fix_repro_4gpu_20260426_202600"
)

for model_dir in "${MODELS[@]}"; do
  name=$(basename "$model_dir")
  echo "=== $name ==="
  conda run -n sam2act5090 \
    python "$SCRIPT" \
    --model-path "$model_dir/model_last.pth" \
    --task put_block_back \
    --data-root /hdd3/siwon_data/sam2act/data_memory/test \
    --repo-code-root . \
    --episodes 3 \
    --output-dir "$OUTDIR" \
    --device 0 2>&1 | grep -E "KF[0-9]+ \[AMBIG\]|KF9 |  chose|  Ambig|  Non"
  echo ""
done
