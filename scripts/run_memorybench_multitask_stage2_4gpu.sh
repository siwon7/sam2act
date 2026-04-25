#!/usr/bin/env bash
set -euo pipefail

# 4-GPU stage2 scaffold for multitask MemoryBench experiments.
# This script only relies on backward-compatible train_plus.py arguments and
# exposes future experimental knobs via EXTRA_* opt passthroughs.
#
# Usage:
#   ./scripts/run_memorybench_multitask_stage2_4gpu.sh \
#     "(put_block_back,reopen_drawer,rearrange_block)" \
#     /abs/path/to/stage1_run_dir_or_model_last.pth \
#     multitask_mem11_phaseaux_v0
#
# Optional env vars:
#   BS=12
#   LR=1.25e-06
#   EPOCHS=20
#   NUM_MASKMEM=11
#   SAME_AUG_PER_SEQ=False
#   TRANSFORM_AUG=True
#   AUTO_EVAL_EPISODES=5
#   DISPLAY_ID=:3
#   CUDA_DEVICES=0,1,2,3
#   ENABLE_MEM11=1
#   ENABLE_COARSE_PHASE_AUX=1
#   ENABLE_TEXT_TASK_GATE=1
#   ENABLE_PERSISTENT_ANCHOR=1
#   COARSE_PHASE_EXP_CFG_OPTS="peract.phase_aux_loss_weight 0.5"
#   COARSE_PHASE_MVT_CFG_OPTS="coarse_phase_classes 4"
#   TEXT_TASK_GATE_EXP_CFG_OPTS="peract.task_gate_loss_weight 0.2"
#   TEXT_TASK_GATE_MVT_CFG_OPTS="task_gate_mode text"
#   PERSISTENT_ANCHOR_MVT_CFG_OPTS="persistent_anchor_enabled True persistent_anchor_max_steps 2"
#   EXTRA_EXP_CFG_OPTS="..."
#   EXTRA_MVT_CFG_OPTS="..."
#   EXTRA_TRAIN_ARGS="..."

TASKS="${1:?tasks string required, e.g. (put_block_back,reopen_drawer,rearrange_block)}"
STAGE1_SRC="${2:?stage1 run dir or model checkpoint required}"
EXP_NAME="${3:?exp_name required}"

REPO_ROOT="/home/cv25/siwon/sam2act_multitask_txt_memory"
CODE_ROOT="$REPO_ROOT/sam2act"
LOG_ROOT="$REPO_ROOT/logs"
RUN_DIR="$CODE_ROOT/runs/${EXP_NAME}"
LOG_FILE="$LOG_ROOT/${EXP_NAME}_stage2.log"

BS="${BS:-12}"
LR="${LR:-1.25e-06}"
EPOCHS="${EPOCHS:-20}"
NUM_MASKMEM="${NUM_MASKMEM:-11}"
SAME_AUG_PER_SEQ="${SAME_AUG_PER_SEQ:-False}"
TRANSFORM_AUG="${TRANSFORM_AUG:-True}"
AUTO_EVAL_EPISODES="${AUTO_EVAL_EPISODES:-0}"
DISPLAY_ID="${DISPLAY_ID:-:3}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
ENABLE_MEM11="${ENABLE_MEM11:-0}"
ENABLE_COARSE_PHASE_AUX="${ENABLE_COARSE_PHASE_AUX:-0}"
ENABLE_TEXT_TASK_GATE="${ENABLE_TEXT_TASK_GATE:-0}"
ENABLE_PERSISTENT_ANCHOR="${ENABLE_PERSISTENT_ANCHOR:-0}"
COARSE_PHASE_EXP_CFG_OPTS="${COARSE_PHASE_EXP_CFG_OPTS:-}"
COARSE_PHASE_MVT_CFG_OPTS="${COARSE_PHASE_MVT_CFG_OPTS:-}"
TEXT_TASK_GATE_EXP_CFG_OPTS="${TEXT_TASK_GATE_EXP_CFG_OPTS:-}"
TEXT_TASK_GATE_MVT_CFG_OPTS="${TEXT_TASK_GATE_MVT_CFG_OPTS:-}"
PERSISTENT_ANCHOR_MVT_CFG_OPTS="${PERSISTENT_ANCHOR_MVT_CFG_OPTS:-persistent_anchor_enabled True persistent_anchor_max_steps 2}"
EXTRA_EXP_CFG_OPTS="${EXTRA_EXP_CFG_OPTS:-}"
EXTRA_MVT_CFG_OPTS="${EXTRA_MVT_CFG_OPTS:-}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

mkdir -p "$LOG_ROOT" "$RUN_DIR"

append_opts() {
  local current="$1"
  local extra="$2"
  if [[ -n "$extra" ]]; then
    if [[ -n "$current" ]]; then
      printf '%s %s' "$current" "$extra"
    else
      printf '%s' "$extra"
    fi
  else
    printf '%s' "$current"
  fi
}

resolve_stage1_ckpt() {
  local src="$1"
  if [[ -f "$src" ]]; then
    printf '%s' "$src"
    return 0
  fi
  if [[ -f "$src/model_last.pth" ]]; then
    printf '%s/model_last.pth' "$src"
    return 0
  fi
  return 1
}

STAGE1_CKPT="$(resolve_stage1_ckpt "$STAGE1_SRC")" || {
  echo "Could not resolve stage1 checkpoint from: $STAGE1_SRC" >&2
  exit 1
}

cp -f "$STAGE1_CKPT" "$RUN_DIR/model_last.pth"

EXP_CFG_OPTS="tasks ${TASKS} exp_name ${EXP_NAME} wandb False bs ${BS} epochs ${EPOCHS} peract.lr ${LR} peract.same_trans_aug_per_seq ${SAME_AUG_PER_SEQ} peract.transform_augmentation ${TRANSFORM_AUG}"
MVT_CFG_OPTS="num_maskmem ${NUM_MASKMEM}"

if [[ "$ENABLE_MEM11" == "1" ]]; then
  MVT_CFG_OPTS="$(append_opts "$MVT_CFG_OPTS" "num_maskmem 11")"
fi

if [[ "$ENABLE_COARSE_PHASE_AUX" == "1" ]]; then
  # Intentionally no assumed flag names here. Once implemented, pass the
  # corresponding names via COARSE_PHASE_EXP_CFG_OPTS / COARSE_PHASE_MVT_CFG_OPTS.
  EXP_CFG_OPTS="$(append_opts "$EXP_CFG_OPTS" "$COARSE_PHASE_EXP_CFG_OPTS")"
  MVT_CFG_OPTS="$(append_opts "$MVT_CFG_OPTS" "$COARSE_PHASE_MVT_CFG_OPTS")"
fi

if [[ "$ENABLE_TEXT_TASK_GATE" == "1" ]]; then
  # Same idea: keep the scaffold backward-compatible and inject future flags via env.
  EXP_CFG_OPTS="$(append_opts "$EXP_CFG_OPTS" "$TEXT_TASK_GATE_EXP_CFG_OPTS")"
  MVT_CFG_OPTS="$(append_opts "$MVT_CFG_OPTS" "$TEXT_TASK_GATE_MVT_CFG_OPTS")"
fi

if [[ "$ENABLE_PERSISTENT_ANCHOR" == "1" ]]; then
  MVT_CFG_OPTS="$(append_opts "$MVT_CFG_OPTS" "$PERSISTENT_ANCHOR_MVT_CFG_OPTS")"
fi

EXP_CFG_OPTS="$(append_opts "$EXP_CFG_OPTS" "$EXTRA_EXP_CFG_OPTS")"
MVT_CFG_OPTS="$(append_opts "$MVT_CFG_OPTS" "$EXTRA_MVT_CFG_OPTS")"

set +u
source /home/cv25/miniconda3/etc/profile.d/conda.sh
conda activate sam2act5090
set -u

export DISPLAY="$DISPLAY_ID"
export XAUTHORITY=/home/cv25/.Xauthority
export COPPELIASIM_ROOT=/home/cv25/siwon/coppeliasim/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:/home/cv25/miniconda3/envs/sam2act5090/lib:/home/cv25/miniconda3/envs/sam2act5090/lib/python3.10/site-packages/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$CODE_ROOT"

{
  echo "=== $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
  echo "tasks=$TASKS"
  echo "stage1_ckpt=$STAGE1_CKPT"
  echo "exp_name=$EXP_NAME"
  echo "bs=$BS lr=$LR epochs=$EPOCHS num_maskmem=$NUM_MASKMEM"
  echo "enable_mem11=$ENABLE_MEM11 enable_coarse_phase_aux=$ENABLE_COARSE_PHASE_AUX enable_text_task_gate=$ENABLE_TEXT_TASK_GATE"
  echo "enable_persistent_anchor=$ENABLE_PERSISTENT_ANCHOR"
  echo "exp_cfg_opts=$EXP_CFG_OPTS"
  echo "mvt_cfg_opts=$MVT_CFG_OPTS"
  echo "extra_train_args=$EXTRA_TRAIN_ARGS"
} | tee "$LOG_FILE"

read -r -a EXTRA_TRAIN_ARGV <<< "$EXTRA_TRAIN_ARGS"

torchrun --standalone --nproc_per_node=4 train_plus.py \
  --device 0,1,2,3 \
  --log-dir runs \
  --exp_cfg_path configs/sam2act_plus.yaml \
  --mvt_cfg_path mvt/configs/sam2act_plus.yaml \
  --exp_cfg_opts "$EXP_CFG_OPTS" \
  --mvt_cfg_opts "$MVT_CFG_OPTS" \
  "${EXTRA_TRAIN_ARGV[@]}" \
  2>&1 | tee -a "$LOG_FILE"

if [[ "$AUTO_EVAL_EPISODES" -gt 0 ]]; then
  EVAL_LOG_NAME="${EXP_NAME}_smoke${AUTO_EVAL_EPISODES}_$(date '+%Y%m%d_%H%M%S')"
  echo "=== $(date '+%Y-%m-%d %H:%M:%S %Z') smoke eval start ===" | tee -a "$LOG_FILE"
  python eval.py \
    --model-folder "$RUN_DIR" \
    --model-name model_plus_last.pth \
    --eval-datafolder ./data_memory/test \
    --tasks "$TASKS" \
    --eval-episodes "$AUTO_EVAL_EPISODES" \
    --log-name "$EVAL_LOG_NAME" \
    --device 3 \
    --headless \
    2>&1 | tee -a "$LOG_FILE"
fi
