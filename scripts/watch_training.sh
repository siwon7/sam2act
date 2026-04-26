#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${1:?session name required}"
TRAIN_LOG="${2:?train log path required}"
WATCHDOG_LOG="${3:?watchdog log path required}"
INTERVAL="${4:-60}"

mkdir -p "$(dirname "$WATCHDOG_LOG")"

while tmux has-session -t "$SESSION_NAME" 2>/dev/null; do
  {
    echo "=== $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
    if [[ -f "$TRAIN_LOG" ]]; then
      grep -E "total loss:|\\[Finish\\]|Rank \\[0\\], Epoch" "$TRAIN_LOG" | tail -n 5 || true
    fi
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
    echo
  } >> "$WATCHDOG_LOG"
  sleep "$INTERVAL"
done

{
  echo "=== $(date '+%Y-%m-%d %H:%M:%S %Z') ==="
  echo "session ended: $SESSION_NAME"
} >> "$WATCHDOG_LOG"
