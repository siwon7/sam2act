#!/usr/bin/env bash
set -euo pipefail

REPLAY_SESSION="${1:?replay session required}"
TRAIN_SESSION="${2:?train session required}"
WATCHDOG_SESSION="${3:?watchdog session required}"
STATUS_LOG="${4:?status log required}"

REPO_ROOT="/home/cv25/siwon/sam2act"
CODE_ROOT="$REPO_ROOT/sam2act"
INTEGRITY_LOG="$REPO_ROOT/logs/sam2act_replay_integrity.log"
TRAIN_LOG="$REPO_ROOT/logs/sam2act_cleanretrain.log"
WATCHDOG_LOG="$REPO_ROOT/logs/sam2act_cleanretrain_watchdog.log"

mkdir -p "$(dirname "$STATUS_LOG")"

{
  echo "=== $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
  echo "waiting for replay session: $REPLAY_SESSION"
} >> "$STATUS_LOG"

while tmux has-session -t "$REPLAY_SESSION" 2>/dev/null; do
  sleep 60
done

{
  echo "=== $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
  echo "replay session finished, running full integrity check"
} >> "$STATUS_LOG"

source /home/cv25/miniconda3/etc/profile.d/conda.sh
conda activate sam2act5090

python "$REPO_ROOT/scripts/check_replay_integrity.py" \
  --replay-root "$CODE_ROOT/replay_temporal/replay_train" \
  --limit-per-task 100000000 \
  --fail-on-corrupt \
  2>&1 | tee -a "$INTEGRITY_LOG"

{
  echo "=== $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
  echo "integrity check passed, starting training"
} >> "$STATUS_LOG"

tmux new-session -d -s "$TRAIN_SESSION" \
  "source /home/cv25/miniconda3/etc/profile.d/conda.sh && \
   conda activate sam2act5090 && \
   export COPPELIASIM_ROOT=/home/cv25/siwon/coppeliasim/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04 && \
   export QT_QPA_PLATFORM_PLUGIN_PATH=/home/cv25/siwon/coppeliasim/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04 && \
   export LD_LIBRARY_PATH=/home/cv25/siwon/coppeliasim/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04:/home/cv25/miniconda3/envs/sam2act5090/lib:/home/cv25/miniconda3/envs/sam2act5090/lib/python3.10/site-packages/nvidia/cudnn/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH} && \
   export PYTHONUNBUFFERED=1 && \
   export WANDB_MODE=offline && \
   cd $CODE_ROOT && \
   torchrun --nproc_per_node=4 --nnodes=1 train.py \
     --device 0,1,2,3 \
     --exp_cfg_path configs/sam2act.yaml \
     --mvt_cfg_path mvt/configs/sam2act.yaml \
     --exp_cfg_opts 'exp_name cleanreplay bs 12 wandb False' \
     2>&1 | tee $TRAIN_LOG"

tmux new-session -d -s "$WATCHDOG_SESSION" \
  "bash $REPO_ROOT/scripts/watch_training.sh $TRAIN_SESSION $TRAIN_LOG $WATCHDOG_LOG 60"

{
  echo "=== $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
  echo "training and watchdog started"
} >> "$STATUS_LOG"
