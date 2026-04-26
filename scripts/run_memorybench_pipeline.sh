#!/usr/bin/env bash
set -eo pipefail

REPO_ROOT="/home/cv25/siwon/sam2act"
CODE_ROOT="$REPO_ROOT/sam2act"
LOG_ROOT="$REPO_ROOT/logs"
STATUS_LOG="$LOG_ROOT/memorybench_pipeline.log"

TASKS=(
  "put_block_back"
  "rearrange_block"
  "reopen_drawer"
)

mkdir -p "$LOG_ROOT"

source /home/cv25/miniconda3/etc/profile.d/conda.sh
conda activate sam2act5090

export DISPLAY=:3
export XAUTHORITY=/home/cv25/.Xauthority
export COPPELIASIM_ROOT=/home/cv25/siwon/coppeliasim/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:/home/cv25/miniconda3/envs/sam2act5090/lib:/home/cv25/miniconda3/envs/sam2act5090/lib/python3.10/site-packages/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1,2,3

cd "$CODE_ROOT"

{
  echo "=== $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
  echo "prepare memorybench"
} | tee -a "$STATUS_LOG"

python "$REPO_ROOT/scripts/prepare_memorybench.py" 2>&1 | tee -a "$STATUS_LOG"

for task in "${TASKS[@]}"; do
  exp_name="memorybench_${task}"
  run_dir="$CODE_ROOT/runs/sam2act_${exp_name}"
  stage1_log="$LOG_ROOT/${exp_name}_stage1.log"
  stage2_log="$LOG_ROOT/${exp_name}_stage2.log"
  replay_task_dir="$CODE_ROOT/replay_temporal_memory/replay_train/${task}"

  {
    echo "=== $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
    echo "task=${task}"
    echo "stage1 start"
  } | tee -a "$STATUS_LOG"

  refresh_flag=""
  if [[ ! -d "$replay_task_dir" ]]; then
    refresh_flag="--refresh_replay"
  fi

  torchrun --nproc_per_node=4 --nnodes=1 \
    train.py \
    --device 0,1,2,3 \
    --use-memory-data \
    --fresh-start \
    $refresh_flag \
    --exp_cfg_path configs/sam2act.yaml \
    --mvt_cfg_path mvt/configs/sam2act.yaml \
    --exp_cfg_opts "tasks ${task} exp_name ${exp_name} wandb False" \
    2>&1 | tee "$stage1_log"

  {
    echo "=== $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
    echo "task=${task}"
    echo "stage1 done"
    echo "stage2 start"
  } | tee -a "$STATUS_LOG"

  torchrun --nproc_per_node=4 --nnodes=1 \
    train_plus.py \
    --device 0,1,2,3 \
    --fresh-start \
    --exp_cfg_path configs/sam2act_plus.yaml \
    --mvt_cfg_path mvt/configs/sam2act_plus.yaml \
    --exp_cfg_opts "tasks ${task} exp_name ${exp_name} wandb False" \
    2>&1 | tee "$stage2_log"

  {
    echo "=== $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
    echo "task=${task}"
    echo "stage2 done"
  } | tee -a "$STATUS_LOG"
done

{
  echo "=== $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
  echo "memorybench pipeline finished"
} | tee -a "$STATUS_LOG"
