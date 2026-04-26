#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-full25}"

REPO_ROOT="/home/cv11/project/siwon/sam2act"
CODE_ROOT="$REPO_ROOT/sam2act"
RUN_DIR="$CODE_ROOT/runs/sam2act_memorybench_put_block_back_stage2revert_20260410"
ENV_ROOT="/home/cv11/anaconda3/envs/sam2act5090"

case "$MODE" in
  full25)
    MODEL_NAME="model_plus_last.pth"
    EVAL_EPISODES=25
    LOG_NAME="cv11_stage2revert_full25_$(date '+%Y%m%d_%H%M%S')"
    ;;
  smoke5)
    MODEL_NAME="model_plus_31.pth"
    EVAL_EPISODES=5
    LOG_NAME="cv11_stage2revert_smoke5_$(date '+%Y%m%d_%H%M%S')"
    ;;
  *)
    echo "Usage: $0 [full25|smoke5]" >&2
    exit 1
    ;;
esac

if [[ ! -f "$RUN_DIR/$MODEL_NAME" ]]; then
  echo "Missing checkpoint: $RUN_DIR/$MODEL_NAME" >&2
  exit 1
fi

set +u
source /home/cv11/anaconda3/etc/profile.d/conda.sh
conda activate sam2act5090
set -u

export COPPELIASIM_ROOT="/home/cv11/project/siwon/coppeliasim/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:$ENV_ROOT/lib:$ENV_ROOT/lib/python3.10/site-packages/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="/lib/x86_64-linux-gnu/libffi.so.7"
export PYOPENGL_PLATFORM="egl"
export MESA_GL_VERSION_OVERRIDE="4.1"
export LIBGL_DRIVERS_PATH="/usr/lib/x86_64-linux-gnu/dri"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

XVFB_DISPLAY=":102"
XVFB_PID=""

cleanup() {
  if [[ -n "$XVFB_PID" ]] && kill -0 "$XVFB_PID" 2>/dev/null; then
    kill "$XVFB_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT

if ! DISPLAY="$XVFB_DISPLAY" xdpyinfo >/dev/null 2>&1; then
  if ! command -v Xvfb >/dev/null 2>&1; then
    echo "Missing Xvfb and no usable display is available." >&2
    exit 1
  fi
  Xvfb "$XVFB_DISPLAY" -screen 0 1280x720x24 >/tmp/cv11_eval_success_combo.xvfb.log 2>&1 &
  XVFB_PID=$!
  sleep 2
fi

export DISPLAY="$XVFB_DISPLAY"
unset XAUTHORITY

cd "$CODE_ROOT"

echo "[cv11_eval_success_combo] mode=$MODE model=$MODEL_NAME eval_episodes=$EVAL_EPISODES"
EXTRA_ARGS=()
if [[ -n "${MVT_CFG_OPTS:-}" ]]; then
  EXTRA_ARGS+=(--mvt_cfg_opts "$MVT_CFG_OPTS")
  echo "[cv11_eval_success_combo] mvt_cfg_opts=$MVT_CFG_OPTS"
fi
python eval.py \
  --model-folder "$RUN_DIR" \
  --model-name "$MODEL_NAME" \
  --eval-datafolder ./data_memory/test \
  --tasks put_block_back \
  --eval-episodes "$EVAL_EPISODES" \
  --log-name "$LOG_NAME" \
  --device 0 \
  --headless \
  "${EXTRA_ARGS[@]}"

EVAL_CSV="$RUN_DIR/eval/$LOG_NAME/${MODEL_NAME%.pth}/eval_results.csv"
if [[ -f "$EVAL_CSV" ]]; then
  echo "[cv11_eval_success_combo] eval_results_csv=$EVAL_CSV"
  cat "$EVAL_CSV"
fi
