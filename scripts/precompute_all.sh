#!/bin/bash
cd /home/cv11/project/siwon/sam2act_dirty/sam2act
export COPPELIASIM_ROOT=/home/cv11/project/siwon/coppeliasim/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:${LD_LIBRARY_PATH}"

SCRIPT=/home/cv11/project/siwon/sam2act_multipeak_graph/scripts/precompute_multipeak_targets.py
OUTDIR=/home/cv11/project/siwon/sam2act_multipeak_graph/sam2act/data_memory/multipeak_targets

for task in put_block_back rearrange_block reopen_drawer; do
  for mode in intra cross; do
    echo "=== $task / $mode ==="
    conda run -n sam2act5090 python "$SCRIPT" \
      --data-root /hdd3/siwon_data/sam2act/data_memory/train \
      --task "$task" --mode "$mode" \
      --output-dir "$OUTDIR" 2>&1 | tail -5
    echo ""
  done
done
