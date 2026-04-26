#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-$HOME/project/siwon}"
HUB_DIR="${2:-$ROOT_DIR/sam2act_stage2_hub}"
REMOTE_URL="${REMOTE_URL:-https://github.com/siwon7/sam2act.git}"

echo "[stage2-worktrees] root: $ROOT_DIR"
echo "[stage2-worktrees] hub:  $HUB_DIR"

mkdir -p "$ROOT_DIR"

if [[ ! -d "$HUB_DIR/.git" ]]; then
  echo "[stage2-worktrees] cloning hub repo from $REMOTE_URL"
  git clone "$REMOTE_URL" "$HUB_DIR"
fi

cd "$HUB_DIR"
git fetch --all --prune

declare -a BRANCHES=(
  "exp/dirty-baseline:sam2act_stage2_baseline"
  "exp/spatial-graph-memory:sam2act_stage2_spatial_graph"
  "exp/text-latent-prototype-memory:sam2act_stage2_text_latent_proto"
)

for spec in "${BRANCHES[@]}"; do
  branch="${spec%%:*}"
  dir_name="${spec##*:}"
  target_dir="$ROOT_DIR/$dir_name"

  if [[ -e "$target_dir" ]]; then
    echo "[stage2-worktrees] skip existing $target_dir"
    continue
  fi

  echo "[stage2-worktrees] adding worktree $target_dir <- $branch"
  git worktree add "$target_dir" "$branch"
done

cat <<'EOF'

[stage2-worktrees] ready

Recommended worktrees:
- sam2act_stage2_baseline
  Dirty practical baseline plus teacher-forced attention/keyframe analysis scripts.
- sam2act_stage2_spatial_graph
  Stage2-only spatial graph memory branch with teacher-forced node inspection.
- sam2act_stage2_text_latent_proto
  Text-conditioned latent prototype retrieval branch (non-manual routing attempt).

Note:
- This does not touch any existing /home/cv11/project/siwon/sam2act working tree.
- Update later with:
    cd "$HUB_DIR" && git fetch --all --prune
EOF
