#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/stage2_oracle_env.sh"

GPU="${GPU:-7}"
EPISODES="${EPISODES:-10}"
TASKS="${TASKS:-put_block_back,rearrange_block,reopen_drawer}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/logs}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

declare -A MODELS=(
  [put_block_back]="/hdd3/siwon_ckpt/sam2act/runs/sam2act_memorybench_put_block_back/model_last.pth"
  [rearrange_block]="/hdd3/siwon_ckpt/sam2act/runs/sam2act_memorybench_rearrange_block/model_last.pth"
  [reopen_drawer]="/hdd3/siwon_ckpt/sam2act/runs/sam2act_memorybench_reopen_drawer/model_last.pth"
)

IFS=',' read -r -a TASK_ARRAY <<< "${TASKS}"
for task in "${TASK_ARRAY[@]}"; do
  task="$(echo "${task}" | xargs)"
  [[ -n "${task}" ]] || continue
  model="${MODELS[${task}]:-}"
  if [[ -z "${model}" ]]; then
    echo "Unknown task: ${task}" >&2
    exit 2
  fi
  if [[ ! -f "${model}" ]]; then
    echo "Missing model: ${model}" >&2
    exit 1
  fi

  echo "=== Stage1 collapse probe: ${task} on GPU${GPU} ==="
  CUDA_VISIBLE_DEVICES="${GPU}" "${SAM2ACT_ENV_PYTHON}" "${SCRIPT_DIR}/stage1_collapse_probe.py" \
    --device 0 \
    --tasks "${task}" \
    --episodes "${EPISODES}" \
    --model "${model}" \
    --out-dir "${OUT_DIR}" \
    --output-stem "dirty_${task}_model_last_gpu${GPU}_${STAMP}"
done
