#!/usr/bin/env bash
set -euo pipefail

set +u
source /home/cv25/miniconda3/etc/profile.d/conda.sh
conda activate sam2act5090
set -u

ROOT=/home/cv25/siwon/sam2act_transition_pointer_memory
REPO=$ROOT/sam2act
BASELINE=/home/cv25/siwon/sam2act/sam2act

RUN_NAME=${RUN_NAME:-pbb_transition_pointer_$(date +%Y%m%d_%H%M%S)}
RUN_DIR=$REPO/runs/sam2act_${RUN_NAME}
GPU=${GPU:-0}
REFRESH_REPLAY=${REFRESH_REPLAY:-0}
EPOCHS=${EPOCHS:-5}
TRAIN_ITER=${TRAIN_ITER:-1000}
EVAL_EPISODES=${EVAL_EPISODES:-5}

PHASE_W=${PHASE_W:-0.05}
ROLE_W=${ROLE_W:-0.02}
VISIT_W=${VISIT_W:-0.05}
ROLE_REF_W=${ROLE_REF_W:-0.01}
ANCHOR_W=${ANCHOR_W:-0.0}
CONTRAST_W=${CONTRAST_W:-0.0}
ROLE_BIAS=${ROLE_BIAS:-0.02}
ANCHOR_BIAS=${ANCHOR_BIAS:-0.01}

cd "$REPO"
export PYTHONPATH="$ROOT:$REPO"

ln -sfn "$BASELINE/data_memory" ./data_memory
mkdir -p ./mvt/sam2_train/checkpoints
ln -sfn \
  "$BASELINE/mvt/sam2_train/checkpoints/sam2.1_hiera_base_plus.pt" \
  ./mvt/sam2_train/checkpoints/sam2.1_hiera_base_plus.pt

mkdir -p "$RUN_DIR"
ln -sfn \
  "$BASELINE/runs/sam2act_memorybench_put_block_back_freshdebug_20260405/model_38.pth" \
  "$RUN_DIR/model_last.pth"

if [[ "$REFRESH_REPLAY" == "1" ]]; then
  rm -rf "$REPO/replay_temporal_memory/replay_train/put_block_back"
  REFRESH_FLAG="--refresh_replay"
else
  REFRESH_FLAG=""
fi

CUDA_VISIBLE_DEVICES=${GPU} \
torchrun --standalone --nproc_per_node=1 train_plus.py \
  ${REFRESH_FLAG} \
  --device 0 \
  --log-dir "$REPO/runs" \
  --exp_cfg_path configs/sam2act_plus.yaml \
  --mvt_cfg_path mvt/configs/sam2act_plus.yaml \
  --exp_cfg_opts "tasks put_block_back exp_name ${RUN_NAME} wandb False bs 12 epochs ${EPOCHS} num_workers 0 train_iter ${TRAIN_ITER} peract.same_trans_aug_per_seq True peract.phase_aux_loss_weight ${PHASE_W} peract.phase_aux_num_classes 4 peract.role_graph_loss_weight ${ROLE_W} peract.visit_mode_loss_weight ${VISIT_W} peract.role_ref_loss_weight ${ROLE_REF_W} peract.anchor_use_loss_weight ${ANCHOR_W} peract.role_contrastive_loss_weight ${CONTRAST_W}" \
  --mvt_cfg_opts "num_maskmem 11 role_graph_enabled True role_graph_num_classes 6 phase_graph_num_classes 4 role_graph_bias_scale ${ROLE_BIAS} anchor_use_bias_scale ${ANCHOR_BIAS} persistent_anchor_enabled True persistent_anchor_max_steps 2 persistent_anchor_prepend True"

export COPPELIASIM_ROOT=/home/cv25/siwon/coppeliasim/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:/home/cv25/miniconda3/envs/sam2act5090/lib:/home/cv25/miniconda3/envs/sam2act5090/lib/python3.10/site-packages/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

xvfb-run -a -s "-screen 0 1400x900x24" python eval.py \
  --model-folder "$RUN_DIR" \
  --model-name model_plus_last.pth \
  --eval-datafolder ./data_memory/test \
  --tasks put_block_back \
  --eval-episodes "$EVAL_EPISODES" \
  --log-name smoke/${RUN_NAME} \
  --device 0 \
  --headless
