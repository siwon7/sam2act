#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:?variant required: shared_fifo | routing | dual_memory | full}"
TASKS="${2:?comma-separated memorybench tasks required}"
EXP_NAME="${3:?exp_name required}"

REPO_ROOT="/home/cv25/siwon/sam2act"
CODE_ROOT="$REPO_ROOT/sam2act"
LOG_ROOT="$REPO_ROOT/logs"
STATUS_LOG="$LOG_ROOT/${EXP_NAME}_status.log"

case "$VARIANT" in
  shared_fifo)
    MVT_CFG="mvt/configs/sam2act_plus_mt_shared_fifo.yaml"
    ;;
  routing)
    MVT_CFG="mvt/configs/sam2act_plus_mt_routing.yaml"
    ;;
  dual_memory)
    MVT_CFG="mvt/configs/sam2act_plus_mt_dual_memory.yaml"
    ;;
  full)
    MVT_CFG="mvt/configs/sam2act_plus_mt_full.yaml"
    ;;
  *)
    echo "Unknown variant: $VARIANT" >&2
    exit 1
    ;;
esac

source /home/cv25/miniconda3/etc/profile.d/conda.sh
conda activate sam2act5090

export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export COPPELIASIM_ROOT=/home/cv25/siwon/coppeliasim/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:/home/cv25/miniconda3/envs/sam2act5090/lib:/home/cv25/miniconda3/envs/sam2act5090/lib/python3.10/site-packages/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$CODE_ROOT"

mkdir -p "$LOG_ROOT"

{
  echo "=== $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
  echo "prepare memorybench replay"
} | tee -a "$STATUS_LOG"

python "$REPO_ROOT/scripts/prepare_memorybench.py" 2>&1 | tee -a "$STATUS_LOG"

replay_task_dir="$CODE_ROOT/replay_temporal_memory/replay_train/${TASKS%%,*}"
refresh_flag=""
if [[ ! -d "$replay_task_dir" ]]; then
  refresh_flag="--refresh_replay"
fi

{
  echo "=== $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
  echo "variant=${VARIANT}"
  echo "tasks=${TASKS}"
  echo "refresh_flag=${refresh_flag:-none}"
} | tee -a "$STATUS_LOG"

torchrun --nproc_per_node=4 --nnodes=1 train.py \
  --device 0,1,2,3 \
  --use-memory-data \
  --fresh-start \
  $refresh_flag \
  --exp_cfg_path configs/sam2act.yaml \
  --mvt_cfg_path mvt/configs/sam2act.yaml \
  --exp_cfg_opts "tasks ${TASKS} exp_name ${EXP_NAME} wandb False" \
  2>&1 | tee "$REPO_ROOT/logs/${EXP_NAME}_stage1.log"

torchrun --nproc_per_node=4 --nnodes=1 train_plus.py \
  --device 0,1,2,3 \
  --fresh-start \
  --exp_cfg_path configs/sam2act_plus.yaml \
  --mvt_cfg_path "${MVT_CFG}" \
  --exp_cfg_opts "tasks ${TASKS} exp_name ${EXP_NAME} wandb False" \
  2>&1 | tee "$REPO_ROOT/logs/${EXP_NAME}_stage2.log"
