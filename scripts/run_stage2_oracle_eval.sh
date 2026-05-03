#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/stage2_oracle_env.sh
source "${SCRIPT_DIR}/stage2_oracle_env.sh"

TASK="${TASK:-put_block_back}"
GPU="${GPU:-7}"
MODEL_FOLDER="${MODEL_FOLDER:-/hdd3/siwon_ckpt/sam2act/runs/sam2act_stage2_oracle_put_block_back_dirty_stage1_smoke}"
MODEL_NAME="${MODEL_NAME:-model_plus_last.pth}"
EVAL_DATAFOLDER="${EVAL_DATAFOLDER:-/hdd3/siwon_data/sam2act/data_memory/test}"
START_EPISODE="${START_EPISODE:-0}"
EVAL_EPISODES="${EVAL_EPISODES:-1}"
EPISODE_LENGTH="${EPISODE_LENGTH:-12}"
LOG_NAME="${LOG_NAME:-oracle_stage1_smoke}"
ORACLE_STAGE1=1
SAVE_VIDEO=0
USE_XVFB="${USE_XVFB:-1}"

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_stage2_oracle_eval.sh [options]

Options:
  --task TASK
  --gpu ID
  --model-folder DIR
  --model-name FILE
  --eval-datafolder DIR
  --start-episode N
  --eval-episodes N
  --episode-length N
  --log-name NAME
  --no-oracle-stage1
  --no-xvfb
  --save-video
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="${2:?missing --task value}"; shift 2 ;;
    --task=*) TASK="${1#*=}"; shift ;;
    --gpu) GPU="${2:?missing --gpu value}"; shift 2 ;;
    --gpu=*) GPU="${1#*=}"; shift ;;
    --model-folder) MODEL_FOLDER="${2:?missing --model-folder value}"; shift 2 ;;
    --model-folder=*) MODEL_FOLDER="${1#*=}"; shift ;;
    --model-name) MODEL_NAME="${2:?missing --model-name value}"; shift 2 ;;
    --model-name=*) MODEL_NAME="${1#*=}"; shift ;;
    --eval-datafolder) EVAL_DATAFOLDER="${2:?missing --eval-datafolder value}"; shift 2 ;;
    --eval-datafolder=*) EVAL_DATAFOLDER="${1#*=}"; shift ;;
    --start-episode) START_EPISODE="${2:?missing --start-episode value}"; shift 2 ;;
    --start-episode=*) START_EPISODE="${1#*=}"; shift ;;
    --eval-episodes) EVAL_EPISODES="${2:?missing --eval-episodes value}"; shift 2 ;;
    --eval-episodes=*) EVAL_EPISODES="${1#*=}"; shift ;;
    --episode-length) EPISODE_LENGTH="${2:?missing --episode-length value}"; shift 2 ;;
    --episode-length=*) EPISODE_LENGTH="${1#*=}"; shift ;;
    --log-name) LOG_NAME="${2:?missing --log-name value}"; shift 2 ;;
    --log-name=*) LOG_NAME="${1#*=}"; shift ;;
    --no-oracle-stage1) ORACLE_STAGE1=0; shift ;;
    --no-xvfb) USE_XVFB=0; shift ;;
    --save-video) SAVE_VIDEO=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -f "${MODEL_FOLDER}/${MODEL_NAME}" ]]; then
  echo "missing model: ${MODEL_FOLDER}/${MODEL_NAME}" >&2
  exit 1
fi

cd "${SAM2ACT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export QT_X11_NO_MITSHM=1
export LIBGL_DRIVERS_PATH="${LIBGL_DRIVERS_PATH:-/usr/lib/x86_64-linux-gnu/dri}"
export MESA_LOADER_DRIVER_OVERRIDE="${MESA_LOADER_DRIVER_OVERRIDE:-swrast}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
if [[ -f /usr/lib/x86_64-linux-gnu/libffi.so.7 ]]; then
  export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libffi.so.7${LD_PRELOAD:+:${LD_PRELOAD}}"
fi
if [[ "${USE_XVFB}" == "1" ]]; then
  unset QT_QPA_PLATFORM
else
  export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
fi

cmd=(
  python eval.py
  --tasks "${TASK}"
  --model-folder "${MODEL_FOLDER}"
  --model-name "${MODEL_NAME}"
  --eval-datafolder "${EVAL_DATAFOLDER}"
  --start-episode "${START_EPISODE}"
  --eval-episodes "${EVAL_EPISODES}"
  --episode-length "${EPISODE_LENGTH}"
  --device 0
  --headless
  --log-name "${LOG_NAME}"
)

if [[ "${ORACLE_STAGE1}" == "1" ]]; then
  cmd+=(--oracle-stage1)
fi
if [[ "${SAVE_VIDEO}" == "1" ]]; then
  cmd+=(--save-video)
fi

printf '[stage2-oracle-eval] CUDA_VISIBLE_DEVICES=%q ' "${GPU}"
if [[ "${USE_XVFB}" == "1" ]]; then
  printf 'xvfb-run -a '
fi
printf '%q ' "${cmd[@]}"
printf '\n'
if [[ "${USE_XVFB}" == "1" ]]; then
  exec xvfb-run -a -s "-screen 0 1280x1024x24" "${cmd[@]}"
else
  exec "${cmd[@]}"
fi
