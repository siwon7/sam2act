#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_ROOT="${LOG_ROOT:-/home/cv11/project/siwon/sam2act_stage2_oracle_ceiling/logs}"
GPU_LIST="${GPU_LIST:-4,5,6}"
EXP_SUFFIX="${EXP_SUFFIX:-$(date '+%Y%m%d_%H%M%S')}"

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
TASKS=(put_block_back rearrange_block reopen_drawer)
PORTS=(29740 29741 29742)

if (( ${#GPUS[@]} < ${#TASKS[@]} )); then
  echo "GPU_LIST must provide at least ${#TASKS[@]} ids; got ${GPU_LIST}" >&2
  exit 2
fi

mkdir -p "${LOG_ROOT}"

for idx in "${!TASKS[@]}"; do
  task="${TASKS[$idx]}"
  gpu="${GPUS[$idx]}"
  log_file="${LOG_ROOT}/stage2_oracle_${task}_${EXP_SUFFIX}.log"
  pid_file="${LOG_ROOT}/stage2_oracle_${task}_${EXP_SUFFIX}.pid"
  echo "[launch] ${task} on GPU ${gpu}; log=${log_file}"
  setsid -f bash -c 'pid_file="$1"; shift; echo "$$" > "${pid_file}"; exec "$@"' _ "${pid_file}" \
    "${SCRIPT_DIR}/run_stage2_oracle_train.sh" \
    --task "${task}" \
    --gpu "${gpu}" \
    --master-port "${PORTS[$idx]}" \
    --exp-suffix "${EXP_SUFFIX}" \
    > "${log_file}" 2>&1
  sleep 0.2
  echo "[launch] pid=$(cat "${pid_file}")"
done
