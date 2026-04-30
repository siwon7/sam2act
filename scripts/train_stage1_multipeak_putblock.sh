#!/bin/bash
# Stage1 training with multi-peak labels for put_block_back
# Compare against baseline 89-epoch models in sam2act_dirty

set -e

cd /home/cv11/project/siwon/sam2act_multipeak_graph/sam2act

export COPPELIASIM_ROOT=/home/cv11/project/siwon/coppeliasim/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:${LD_LIBRARY_PATH}"
export WANDB_MODE="offline"

# 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --nproc_per_node=4 --nnodes=1 \
  train.py \
  --use-memory-data \
  --refresh_replay \
  --exp_cfg_opts "tasks put_block_back exp_name stage1_multipeak_putblock wandb False" \
  --exp_cfg_path configs/sam2act.yaml \
  --mvt_cfg_path mvt/configs/sam2act.yaml
