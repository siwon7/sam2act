#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/cv25/siwon/sam2act_text_latent_prototype"
CODE_ROOT="$REPO_ROOT/sam2act"
LOG_ROOT="$REPO_ROOT/logs"

EXP_NAME="${1:-sam2act_pbb_text_latent_proto_smoke_$(date +%Y%m%d_%H%M%S)}"
STAGE1_CKPT="${2:-/home/cv25/siwon/sam2act/sam2act/runs/sam2act_memorybench_put_block_back_freshdebug_20260405/model_38.pth}"
DEMO_COUNT="${DEMO_COUNT:-20}"
EPOCHS="${EPOCHS:-1}"
BS="${BS:-12}"
TRAIN_ITER="${TRAIN_ITER:-12000}"
RUN_NAME="sam2act_${EXP_NAME}"

mkdir -p "$LOG_ROOT" "$CODE_ROOT/runs/$RUN_NAME"
cp -f "$STAGE1_CKPT" "$CODE_ROOT/runs/$RUN_NAME/model_last.pth"

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
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH="$CODE_ROOT/libs/YARR:$REPO_ROOT"

cd "$CODE_ROOT"

REFRESH_REPLAY="${REFRESH_REPLAY:-1}"
LOCAL_REPLAY_ROOT="$CODE_ROOT/replay_temporal_memory"
LOCAL_REPLAY_TASK_DIR="$LOCAL_REPLAY_ROOT/replay_train/put_block_back"

if [ "$REFRESH_REPLAY" = "1" ]; then
  if [ -L "$LOCAL_REPLAY_ROOT" ]; then
    rm -f "$LOCAL_REPLAY_ROOT"
  fi
  rm -rf "$LOCAL_REPLAY_TASK_DIR"
fi
mkdir -p "$LOCAL_REPLAY_ROOT/replay_train"

torchrun --standalone --nproc_per_node=2 train_plus.py \
  --device 0,1 \
  --log-dir runs \
  --exp_cfg_path configs/sam2act_plus.yaml \
  --mvt_cfg_path mvt/configs/sam2act_plus.yaml \
  --exp_cfg_opts "tasks put_block_back exp_name ${EXP_NAME} wandb False demo ${DEMO_COUNT} bs ${BS} epochs ${EPOCHS} train_iter ${TRAIN_ITER} num_workers 0 peract.lr 1.25e-06 peract.same_trans_aug_per_seq False peract.transform_augmentation True peract.latent_revisit_loss_weight 0.05 peract.latent_proto_align_loss_weight 0.05 peract.latent_proto_usage_loss_weight 0.01 peract.latent_proto_match_thresh 0.05" \
  --mvt_cfg_opts "num_maskmem 11 use_memory True latent_proto_enabled True latent_proto_num_prototypes 8 latent_proto_hidden_dim 128 latent_proto_bias_scale 0.3 latent_proto_use_text True graph_retrieval_enabled False graph_node_classes 0 memory_gate_enabled False persistent_anchor_enabled False" \
  2>&1 | tee "$LOG_ROOT/${EXP_NAME}_train.log"

python eval.py \
  --model-folder "$CODE_ROOT/runs/${RUN_NAME}" \
  --model-name model_plus_last.pth \
  --eval-datafolder ./data_memory/test \
  --tasks put_block_back \
  --eval-episodes 5 \
  --log-name "${RUN_NAME}_smoke5" \
  --device 0 \
  --headless \
  2>&1 | tee "$LOG_ROOT/${EXP_NAME}_eval.log"
