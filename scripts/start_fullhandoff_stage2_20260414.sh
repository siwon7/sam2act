#!/usr/bin/env bash
set -eo pipefail

source /home/cv25/miniconda3/etc/profile.d/conda.sh
conda activate sam2act5090

cd /home/cv25/siwon/sam2act/sam2act

exec torchrun --standalone --nproc_per_node=4 train_plus.py \
  --fresh-start \
  --device 0,1,2,3 \
  --exp_cfg_path runs/sam2act_memorybench_put_block_back_fullhandoff_20260414/exp_cfg_plus.yaml \
  --mvt_cfg_path runs/sam2act_memorybench_put_block_back_fullhandoff_20260414/mvt_cfg_plus.yaml
