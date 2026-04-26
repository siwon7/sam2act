#!/usr/bin/env bash
set -euo pipefail

TASK="${1:?task required, e.g. put_block_back}"
STAGE1_RUN_DIR="${2:?stage1 run dir required}"
EXP_NAME="${3:?exp_name required}"
BS="${4:?global batch size required}"
LR="${5:?learning rate required}"
EPOCHS="${6:-20}"
NUM_MASKMEM="${7:?num_maskmem required}"
AUTO_EVAL_EPISODES="${8:-5}"
ATTN_EPISODE="${9:-0}"

REPO_ROOT="/home/cv25/siwon/sam2act_memdebug"
CODE_ROOT="$REPO_ROOT/sam2act"
LOG_ROOT="$REPO_ROOT/logs"
RUN_DIR="$CODE_ROOT/runs/sam2act_${EXP_NAME}"
LOG_FILE="$LOG_ROOT/${EXP_NAME}_stage2.log"
EVAL_LOG_NAME="${EXP_NAME}_smoke${AUTO_EVAL_EPISODES}_$(date '+%Y%m%d_%H%M%S')"
ATTN_STEM="${EXP_NAME}_attn_ep${ATTN_EPISODE}_$(date '+%Y%m%d_%H%M%S')"

mkdir -p "$LOG_ROOT" "$RUN_DIR"

if [[ ! -f "$STAGE1_RUN_DIR/model_last.pth" ]]; then
  echo "Missing stage1 checkpoint: $STAGE1_RUN_DIR/model_last.pth" >&2
  exit 1
fi

cp -f "$STAGE1_RUN_DIR/model_last.pth" "$RUN_DIR/model_last.pth"

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
export CUDA_VISIBLE_DEVICES=0,1,2,3

cd "$CODE_ROOT"

echo "=== $(date '+%Y-%m-%d %H:%M:%S %Z') ===" | tee "$LOG_FILE"
echo "task=$TASK exp_name=$EXP_NAME bs=$BS lr=$LR epochs=$EPOCHS num_maskmem=$NUM_MASKMEM auto_eval_episodes=$AUTO_EVAL_EPISODES attn_episode=$ATTN_EPISODE" | tee -a "$LOG_FILE"

torchrun --standalone --nproc_per_node=4 train_plus.py \
  --device 0,1,2,3 \
  --fresh-start \
  --log-dir runs \
  --exp_cfg_path configs/sam2act_plus.yaml \
  --mvt_cfg_path mvt/configs/sam2act_plus.yaml \
  --exp_cfg_opts "tasks ${TASK} exp_name ${EXP_NAME} wandb False bs ${BS} epochs ${EPOCHS} peract.lr ${LR} peract.same_trans_aug_per_seq False peract.transform_augmentation True" \
  --mvt_cfg_opts "num_maskmem ${NUM_MASKMEM}" \
  2>&1 | tee -a "$LOG_FILE"

if [[ "$AUTO_EVAL_EPISODES" -gt 0 ]]; then
  echo "=== $(date '+%Y-%m-%d %H:%M:%S %Z') smoke eval start ===" | tee -a "$LOG_FILE"
  python eval.py \
    --model-folder "$RUN_DIR" \
    --model-name model_plus_last.pth \
    --eval-datafolder ./data_memory/test \
    --tasks "$TASK" \
    --eval-episodes "$AUTO_EVAL_EPISODES" \
    --log-name "$EVAL_LOG_NAME" \
    --device 3 \
    --headless \
    2>&1 | tee -a "$LOG_FILE"
fi

if [[ "$ATTN_EPISODE" -ge 0 ]]; then
  echo "=== $(date '+%Y-%m-%d %H:%M:%S %Z') attention analysis start ===" | tee -a "$LOG_FILE"
  ATTN_JSON="$LOG_ROOT/${ATTN_STEM}.json"
  ATTN_HTML="$LOG_ROOT/${ATTN_STEM}.html"
  ATTN_CSV="$LOG_ROOT/${ATTN_STEM}_top3.csv"
  ATTN_MD="$LOG_ROOT/${ATTN_STEM}_top3.md"

  python /home/cv25/siwon/sam2act/scripts/analyze_memorybench_attention.py \
    --model-path "$RUN_DIR/model_plus_last.pth" \
    --task "$TASK" \
    --episode "$ATTN_EPISODE" \
    --device 3 \
    --repo-code-root "$CODE_ROOT" \
    --data-root "$CODE_ROOT/data_memory/test" \
    --output-json "$ATTN_JSON" \
    --output-html "$ATTN_HTML" \
    2>&1 | tee -a "$LOG_FILE"

  python /home/cv25/siwon/sam2act/scripts/summarize_memorybench_attention.py \
    --input-json "$ATTN_JSON" \
    --output-csv "$ATTN_CSV" \
    --output-md "$ATTN_MD" \
    --topk 3 \
    2>&1 | tee -a "$LOG_FILE"

  echo "attention_json=$ATTN_JSON" | tee -a "$LOG_FILE"
  echo "attention_html=$ATTN_HTML" | tee -a "$LOG_FILE"
  echo "attention_csv=$ATTN_CSV" | tee -a "$LOG_FILE"
  echo "attention_md=$ATTN_MD" | tee -a "$LOG_FILE"
fi
