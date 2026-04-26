#!/usr/bin/env bash
set -eo pipefail

if [[ -f "/home/cv25/miniconda3/etc/profile.d/conda.sh" ]]; then
  source /home/cv25/miniconda3/etc/profile.d/conda.sh
elif [[ -f "/home/cv25/anaconda3/etc/profile.d/conda.sh" ]]; then
  source /home/cv25/anaconda3/etc/profile.d/conda.sh
else
  echo "missing conda.sh under /home/cv25/miniconda3 or /home/cv25/anaconda3" >&2
  exit 1
fi
conda activate sam2act5090

export SAM2ACT_UPSTREAM_ROOT="/home/cv25/siwon/sam2act_upstream_main"
export SAM2ACT_UPSTREAM_CODE_ROOT="$SAM2ACT_UPSTREAM_ROOT/sam2act"
export PYTHONPATH="$SAM2ACT_UPSTREAM_CODE_ROOT:$SAM2ACT_UPSTREAM_ROOT:${PYTHONPATH:-}"
export COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-/home/cv25/siwon/coppeliasim/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${QT_QPA_PLATFORM_PLUGIN_PATH:-$COPPELIASIM_ROOT}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$COPPELIASIM_ROOT:/home/cv25/siwon/coppeliasim/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04_headless"
export DISPLAY="${DISPLAY:-:3}"

cd "$SAM2ACT_UPSTREAM_CODE_ROOT"
