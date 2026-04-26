# cv11 Stage2-Ready Branches

This note is the shortest handoff for using the recent teacher-forcing diagnostics and stage2 memory experiments on `cv11` without touching the working `sam2act` tree.

## Rule

Do not modify the in-progress `sam2act` tree on `cv11`.

Use a separate clone/worktree bundle instead.

## One-command setup

From any clone of the `siwon7/sam2act` repo:

```bash
bash scripts/setup_cv11_stage2_worktrees.sh /home/cv11/project/siwon /home/cv11/project/siwon/sam2act_stage2_hub
```

This creates:

- `sam2act_stage2_baseline`
- `sam2act_stage2_spatial_graph`
- `sam2act_stage2_text_latent_proto`

All three come from GitHub branches, so later updates are just `git fetch`.

## What each branch is for

### `exp/dirty-baseline`

Use this when you want:
- the practical dirty stage2 baseline
- keyframe plots
- teacher-forced attention analysis
- rollout attention summaries

Useful scripts:
- `scripts/analyze_memorybench_attention_teacher_forced.py`
- `scripts/summarize_teacher_forced_attention_batch.py`
- `scripts/visualize_memorybench_keyframes_episodewise.py`

Why keep it:
- best branch for diagnosis and comparison
- no graph-memory modifications

### `exp/spatial-graph-memory`

Use this when you want:
- the current graph-memory line
- dual-bank spatial graph memory
- gripper-aware summary-node merge
- teacher-forced node/edge inspection

Useful scripts:
- `scripts/run_put_block_back_spatial_graph_smoke.sh`
- `scripts/inspect_spatial_graph_teacher_forced_nodes.py`
- `scripts/analyze_memorybench_attention_teacher_forced.py`

Current status:
- implementation path works
- `5ep smoke` has stayed `0.0%`
- still useful because it contains the cleanest teacher-forced graph diagnostics

### `exp/text-latent-prototype-memory`

Use this when you want:
- a non-manual branch
- text-conditioned latent prototype retrieval
- soft retrieve-vs-explore style routing

Useful scripts:
- `scripts/run_put_block_back_latent_proto_smoke.sh`
- `scripts/analyze_memorybench_attention_teacher_forced.py`

Current status:
- smoke path works
- current smoke result is also `0.0%`
- worth keeping as the cleanest non-manual routing attempt

## What to run first on cv11

### 1. Teacher-forced graph construction sanity check

In `sam2act_stage2_spatial_graph`:

```bash
conda run -n sam2act5090 python scripts/inspect_spatial_graph_teacher_forced_nodes.py \
  --task put_block_back \
  --episodes 1 \
  --output-dir logs/teacher_forced_nodes_cv11
```

Why:
- checks whether repeated GT keyframes naturally merge into sensible summary nodes
- avoids conflating rollout drift with graph construction quality

### 2. Teacher-forced attention summary baseline

In `sam2act_stage2_baseline`:

```bash
conda run -n sam2act5090 python scripts/analyze_memorybench_attention_teacher_forced.py \
  --model-path sam2act/runs/<stage2_run>/model_plus_last.pth \
  --task put_block_back \
  --episode 0 \
  --device 0
```

Why:
- quickest way to compare "what the model can read under clean memory"

### 3. Spatial graph smoke

In `sam2act_stage2_spatial_graph`:

```bash
bash scripts/run_put_block_back_spatial_graph_smoke.sh
```

Why:
- verifies the stage2 graph-memory path still runs end-to-end on the server

## Current recommendation

If you only keep two branches active on `cv11`, keep:

1. `exp/dirty-baseline`
2. `exp/spatial-graph-memory`

Reason:
- baseline branch gives the strongest diagnostics
- spatial-graph branch contains the teacher-forced node/edge inspector and the latest graph-memory implementation
