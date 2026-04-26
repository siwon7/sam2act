# SAM2Act-MT++: MemoryBench Multitask Method and Ablation Plan

This note defines the current multitask memory method implemented in the repo, the intended ablation table, and the exact execution recipes.

## Method Summary

We extend `SAM2Act+` for multitask MemoryBench with a task-conditioned dual-level memory design.

Core components:

1. Shared local memory
   - Original recent-memory path.
   - Stores dense recent SAM2 memory tokens.
   - Preserves the original short-horizon memory behavior of SAM2Act+.

2. Event memory
   - Additional sparse episodic memory bank.
   - Stores memory slots for key task events.
   - Intended to preserve long-horizon task-critical state, such as initial object location or task-specific scene state.

3. Task-conditioned routing
   - Uses task embeddings to modulate memory readout.
   - Can bias retrieval toward task-relevant memory channels.
   - Can optionally fuse local and event memories with task-conditioned weights.

4. Task-conditioned event writing
   - Uses task embeddings and heatmap statistics to score whether a memory slot should be promoted into event memory.
   - Keeps event memory sparse and task-relevant.

5. Adaptive event consolidation
   - Event memory does not rely on strict FIFO eviction.
   - When enabled, the lowest-importance event slot is pruned first.

## Contribution Framing

Recommended framing:

`Multitask failure on MemoryBench is not only a data-sharing problem, but a memory lifecycle problem: what to write, what to retain, and what to retrieve.`

This gives three clean contributions:

1. Task-conditioned memory routing for multitask spatial recall.
2. Dual-level memory for recent spatial context and sparse episodic events.
3. Adaptive event consolidation beyond fixed FIFO memory.

## Implemented Method Variants

The following stage-2 variants are exposed through MVT config presets.

### 1. Shared FIFO baseline

File:
- `sam2act/mvt/configs/sam2act_plus_mt_shared_fifo.yaml`

Behavior:
- Original shared memory only.
- No task routing.
- No event memory.
- This is the multitask baseline.

### 2. Routing-only ablation

File:
- `sam2act/mvt/configs/sam2act_plus_mt_routing.yaml`

Behavior:
- Shared memory only.
- Task-conditioned memory routing enabled.
- No event memory.
- Tests whether task-conditioned retrieval alone helps.

### 3. Dual-memory ablation

File:
- `sam2act/mvt/configs/sam2act_plus_mt_dual_memory.yaml`

Behavior:
- Local memory + event memory enabled.
- Task-conditioned local/event fusion enabled.
- Event writes are dense, not selectively routed.
- Event pruning is FIFO-like.
- Tests whether the dual-level structure helps on its own.

### 4. Full method

File:
- `sam2act/mvt/configs/sam2act_plus_mt_full.yaml`

Behavior:
- Local memory + event memory.
- Task-conditioned routing.
- Heatmap-aware task-conditioned event writing.
- Adaptive event pruning by importance.
- This is the main method.

## Current Config Switches

Defined in:
- `sam2act/mvt/config.py`

Switches:
- `enable_task_memory_routing`
- `enable_dual_memory`
- `enable_event_memory`
- `enable_event_write`
- `enable_local_event_fusion`
- `enable_adaptive_event_prune`
- `enable_heatmap_event_features`

These switches are consumed in:
- `sam2act/mvt/mvt_sam2.py`
- `sam2act/mvt/mvt_sam2_single.py`

## Ablation Table

Recommended first table:

1. Single-task SAM2Act+ per task
2. Multitask shared FIFO memory
3. Multitask + task-conditioned routing
4. Multitask + dual memory
5. Multitask + full method

Recommended extended ablation:

1. Shared FIFO
2. + task routing
3. + dual memory
4. + event write
5. + adaptive prune
6. + heatmap-aware event features

## Execution Recipes

All commands are run from repo root:
- `/home/cv25/siwon/sam2act`

### Scripted multitask run

Use:

```bash
bash scripts/run_memorybench_mt_ablation.sh <variant> <tasks> <exp_name>
```

Examples:

```bash
bash scripts/run_memorybench_mt_ablation.sh shared_fifo put_block_back,rearrange_block mt_fifo_pb_rb
bash scripts/run_memorybench_mt_ablation.sh routing put_block_back,rearrange_block mt_routing_pb_rb
bash scripts/run_memorybench_mt_ablation.sh dual_memory put_block_back,rearrange_block mt_dualmem_pb_rb
bash scripts/run_memorybench_mt_ablation.sh full put_block_back,rearrange_block mt_full_pb_rb
```

The script will:

1. Train stage 1 with `configs/sam2act.yaml`
2. Train stage 2 with the selected multitask memory preset
3. Save logs in:
   - `logs/<exp_name>_stage1.log`
   - `logs/<exp_name>_stage2.log`

### Manual stage-2-only ablation run

If stage 1 already exists in the same run folder:

```bash
cd /home/cv25/siwon/sam2act/sam2act
torchrun --nproc_per_node=4 --nnodes=1 train_plus.py \
  --device 0,1,2,3 \
  --fresh-start \
  --exp_cfg_path configs/sam2act_plus.yaml \
  --mvt_cfg_path mvt/configs/sam2act_plus_mt_full.yaml \
  --exp_cfg_opts "tasks put_block_back,rearrange_block exp_name mt_full_pb_rb wandb False"
```

### Eval

Per-task eval should still be run one task at a time for clean reporting:

```bash
cd /home/cv25/siwon/sam2act/sam2act
CUDA_VISIBLE_DEVICES=0 python eval.py \
  --tasks put_block_back \
  --model-folder runs/sam2act_mt_full_pb_rb \
  --model-name model_plus_last.pth \
  --eval-datafolder ./data_memory/test \
  --eval-episodes 5 \
  --episode-length 25 \
  --device 0 \
  --log-name eval_put_block_back_5ep \
  --headless
```

Repeat for `rearrange_block` and other tasks.

## Important Experimental Rules

1. Use fresh run names.
2. Do not mix old single-task checkpoints with new multitask results.
3. Before eval, confirm no stray GPU jobs.
4. Report per-task results and average.
5. Prefer at least:
   - 5-episode smoke eval during development
   - 25-episode full eval for final table

## Current Implementation Scope

Already implemented:

1. Task-conditioned routing scaffold
2. Dual local/event memory banks
3. Event write scoring path
4. Adaptive importance-based event pruning
5. Config-driven ablation toggles

Not yet implemented:

1. Merge-based consolidation of similar event slots
2. Scene-cleared write masking
3. Auxiliary event sparsity losses
4. Automatic best-checkpoint selection across all stage-2 epochs

## Recommended Next Steps

1. Verify that the new toggles do not break single-task stage-2 training.
2. Run the multitask shared FIFO baseline.
3. Run routing-only.
4. Run the full method.
5. Compare:
   - per-task success
   - average success
   - run-to-run variance
   - training stability
