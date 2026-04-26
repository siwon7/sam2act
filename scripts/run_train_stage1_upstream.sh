#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env_upstream.sh"

exec torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-4}" \
  train.py \
  "$@"
