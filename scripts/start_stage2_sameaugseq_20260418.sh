#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CODE_ROOT="$UPSTREAM_ROOT/sam2act"

RUN_NAME="memorybench_put_block_back_stage2_sameaugseq_20260418"
STAGE1_RUN_NAME="memorybench_put_block_back_stage1_clean_e2e_20260416"

STAGE1_RUN_DIR="$CODE_ROOT/runs/sam2act_${STAGE1_RUN_NAME}"
STAGE2_RUN_DIR="$CODE_ROOT/runs/sam2act_${RUN_NAME}"

mkdir -p "$STAGE2_RUN_DIR"
cp -f "$STAGE1_RUN_DIR/model_last.pth" "$STAGE2_RUN_DIR/model_last.pth"

set +u
source "$SCRIPT_DIR/env_upstream.sh"
set -u
cd "$CODE_ROOT"

exec torchrun --standalone --nproc_per_node=4 train_plus.py \
  --log-dir runs \
  --exp_cfg_path configs/sam2act_plus.yaml \
  --mvt_cfg_path mvt/configs/sam2act_plus.yaml \
  --exp_cfg_opts "tasks put_block_back exp_name ${RUN_NAME} wandb False peract.same_trans_aug_per_seq True"
