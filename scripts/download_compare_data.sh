#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-help}"
TARGET_ROOT="${TARGET_ROOT:-/home/cv25/siwon/data_compare}"

mkdir -p "$TARGET_ROOT"

download_peract_checkpoint() {
  local target="$TARGET_ROOT/peract"
  mkdir -p "$target"
  (cd /home/cv25/siwon/peract && bash scripts/quickstart_download.sh)
}

download_rlbench_demos_hf() {
  local target="$TARGET_ROOT/rlbench_18_tasks"
  mkdir -p "$target"
  python - <<'PY'
import os
import sys
target = os.environ.get("TARGET_ROOT", "/home/cv25/siwon/data_compare")
print("RLBench demo mirror: hqfang/rlbench-18-tasks")
print("Suggested command:")
print(f"  huggingface-cli download hqfang/rlbench-18-tasks --repo-type dataset --local-dir {target}/rlbench_18_tasks")
print("If you prefer the original PerAct dataset, use the Google Drive release described in peract/README.md.")
PY
}

download_colosseum_dataset() {
  local target="$TARGET_ROOT/colosseum"
  mkdir -p "$target"
  python - <<'PY'
import os
target = os.environ.get("TARGET_ROOT", "/home/cv25/siwon/data_compare")
print("Colosseum dataset: colosseum/colosseum-challenge")
print("Suggested command:")
print(f"  huggingface-cli download colosseum/colosseum-challenge --repo-type dataset --local-dir {target}/colosseum")
print("The dataset is organized as tar files per task / variation.")
PY
}

download_mrest_assets() {
  echo "MREST-RLBench blocker:"
  echo "  The repo references iamlab-cmu/mrest-env-deps for task_ttms, but that repository was not publicly available at the time of this pass."
  echo "  There is no robust public download path to automate here."
}

case "$ACTION" in
  peract-checkpoint)
    download_peract_checkpoint
    ;;
  rlbench-demos)
    download_rlbench_demos_hf
    ;;
  colosseum-dataset)
    download_colosseum_dataset
    ;;
  mrest-assets)
    download_mrest_assets
    ;;
  all)
    download_peract_checkpoint
    download_rlbench_demos_hf
    download_colosseum_dataset
    download_mrest_assets
    ;;
  help|*)
    cat <<EOF
Usage:
  $0 peract-checkpoint
  $0 rlbench-demos
  $0 colosseum-dataset
  $0 mrest-assets
  $0 all

Env:
  TARGET_ROOT=/home/cv25/siwon/data_compare

Notes:
  - PerAct checkpoint download runs the repo's own quickstart script.
  - RLBench demo mirror uses the public HF mirror referenced in sam2act docs.
  - Colosseum dataset uses the public HF challenge dataset.
  - MREST currently has a public dependency blocker.
EOF
    ;;
esac

