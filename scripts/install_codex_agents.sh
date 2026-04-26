#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$REPO_ROOT/.codex.local"
DEST_DIR="$REPO_ROOT/.codex"

if [[ ! -d "$SRC_DIR" ]]; then
  echo "missing source directory: $SRC_DIR" >&2
  exit 1
fi

if [[ ! -d "$DEST_DIR" ]]; then
  echo "destination is not a writable directory: $DEST_DIR" >&2
  echo "this workspace currently exposes .codex as a read-only mount, so copy manually in an environment where .codex is writable" >&2
  exit 2
fi

cp "$SRC_DIR/config.toml" "$DEST_DIR/config.toml"
mkdir -p "$DEST_DIR/agents"
cp "$SRC_DIR"/agents/*.toml "$DEST_DIR/agents/"

echo "installed sam2act Codex agents into $DEST_DIR"
