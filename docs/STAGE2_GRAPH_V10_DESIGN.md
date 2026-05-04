# SAM2Act Stage2 Graph V10 Design

Date: 2026-05-04
Branch: `exp/v9-topk-selector`

## Goal

V9 showed that the Stage1 heatmap top1 can collapse to a single direction even
when other valid directions exist. V10 keeps the original Stage1 heatmap loss
available, but changes the interface between Stage1, Stage2, and memory so the
pipeline can preserve candidates instead of committing to one peak too early.

The design splits the problem into:

1. Candidate preservation: Stage1 must expose top-K 3D peaks, not only top1.
2. Candidate selection: Stage2 should receive a candidate packet and select the
   semantically valid crop.
3. Memory write policy: SAM2Act+ memory should not be forced to store only the
   raw collapsed Stage1 heatmap when top-K evidence is useful.

## Implemented Variants

### V10a: Graph Candidate Selector

`stage2_candidate_mode=selector`

Stage1 extracts K non-maximum-suppressed 3D heatmap peaks. The new
`Stage2CandidateGraphSelector` scores each candidate as a graph node using:

- candidate 3D location;
- fused heatmap score;
- per-view heatmap support;
- rank one-hot;
- proprio;
- mean language embedding;
- pairwise relative geometry between candidates.

By default, training still uses the GT/oracle Stage2 crop
(`stage2_candidate_train_crop=gt`) and adds the selector CE loss. This keeps the
old oracle Stage2 ceiling experiment stable while learning whether the selector
can choose the right candidate. To force Stage2 training to use the selected
candidate crop, set `stage2_candidate_train_crop=selector`. To teacher-force the
nearest candidate branch, set `stage2_candidate_train_crop=nearest_gt`.

### V10b: K-Crop Stage2 Branches

`stage2_candidate_mode=kcrop`

Stage2 is run once per Stage1 candidate. The selected branch is then gathered
per sample and exposed through the normal `out["mvt2"]`, `out["wpt_local1"]`,
and `out["rev_trans"]` interface, so the existing agent loss and prediction
code can still run.

Training branch selection:

- `stage2_kcrop_train_pick=target`: train the branch nearest to GT.
- `stage2_kcrop_train_pick=selector`: train the branch chosen by the graph
  selector.

This is heavier than V10a. With `graph_peak_topk=3`, the Stage2 MVT forward is
approximately tripled.

### V10c: Memory Heatmap Write Policy

`stage2_memory_write_mode=topk_soft` or `selected`

SAM2Act+ currently writes memory inside `MVT_SAM2_Single` immediately after the
Stage1 heatmap is produced. That means the wrapper-level Stage2 selector cannot
yet decide what gets written to memory without a larger pipeline restructure.

This implementation adds a practical V10c hook at the actual memory write site:

- `stage1`: original raw Stage1 logits.
- `topk_soft`: per-view top-K normalized heatmap, optionally gaussian-expanded.
- `weighted_topk`: alias for `topk_soft`.
- `selected`: per-view top1 heatmap, optionally gaussian-expanded.

So V10c now tests whether preserving top-K heatmap evidence inside memory helps
avoid raw Stage1 collapse dominating memory attention. A future V10d can move
memory writing after wrapper-level candidate selection to write the Stage2
selected 3D candidate instead.

## Option Surface

All options are available through `mvt_cfg_opts`, so train/eval scripts and
manual runs can share the same flags.

| option | default | values | purpose |
|---|---:|---|---|
| `use_graph_peak_select` | `False` | bool | Enables selector aux loss and candidate packet. Auto-enabled by `--v10a/--v10b` script shortcuts. |
| `graph_peak_topk` | `3` | int | Number of Stage1 3D candidates. |
| `graph_peak_select_loss_weight` | `0.5` | float | CE weight for selector target. |
| `graph_peak_insert_gt_train` | `True` | bool | Insert GT as final candidate for selector ceiling training. |
| `graph_peak_positive_radius` | `0.05` | float | Candidate is covered if within this local-cube radius. |
| `graph_peak_nms_dist` | `0.05` | float | NMS distance for Stage1 peak extraction. |
| `stage2_candidate_mode` | `top1` | `top1`, `selector`, `kcrop` | Main V10 mode. |
| `stage2_candidate_train_crop` | `gt` | `gt`, `selector`, `nearest_gt` | Single-crop training policy for V10a. |
| `stage2_kcrop_train_pick` | `target` | `target`, `selector` | Branch selection policy for V10b training. |
| `stage2_candidate_insert_gt_train` | `True` | bool | GT insertion for V10 candidate packet. |
| `stage2_memory_write_mode` | `stage1` | `stage1`, `topk_soft`, `weighted_topk`, `selected` | V10c memory mask write policy. |
| `stage2_memory_write_topk` | `3` | int | Per-view top-K for V10c. |
| `stage2_memory_write_temperature` | `0.25` | float | Softmax temperature for V10c top-K weights. |
| `stage2_memory_write_sigma` | `1.5` | float | Gaussian sigma in image pixels; use `0` for sparse masks. |

## Run Commands

Dry-run first:

```bash
scripts/run_stage2_oracle_train.sh --task put_block_back --v10a --dry-run
scripts/run_stage2_oracle_train.sh --task put_block_back --v10b --dry-run
scripts/run_stage2_oracle_train.sh --task put_block_back --use-memory --v10c-memory topk_soft --dry-run
```

V10a selector ceiling run:

```bash
GPU=0,1,2,3 NPROC=4 MASTER_PORT=29791 \
scripts/run_stage2_oracle_train.sh \
  --task put_block_back \
  --exp-name stage2_v10a_selector_put_block_back \
  --v10a \
  --graph-peak-topk 3 \
  --stage2-candidate-train-crop gt
```

V10a selected-crop stress run:

```bash
GPU=0,1,2,3 NPROC=4 MASTER_PORT=29792 \
scripts/run_stage2_oracle_train.sh \
  --task put_block_back \
  --exp-name stage2_v10a_selectedcrop_put_block_back \
  --v10a \
  --stage2-candidate-train-crop selector
```

V10b K-crop run:

```bash
GPU=0,1,2,3 NPROC=4 MASTER_PORT=29793 \
scripts/run_stage2_oracle_train.sh \
  --task put_block_back \
  --exp-name stage2_v10b_kcrop_put_block_back \
  --v10b \
  --graph-peak-topk 3 \
  --stage2-kcrop-train-pick target
```

V10c memory write run:

```bash
GPU=0,1,2,3 NPROC=4 MASTER_PORT=29794 \
scripts/run_stage2_oracle_train.sh \
  --task put_block_back \
  --exp-name stage2_v10c_memory_topk_put_block_back \
  --use-memory \
  --v10c-memory topk_soft \
  --stage2-memory-write-topk 3 \
  --stage2-memory-write-temperature 0.25 \
  --stage2-memory-write-sigma 1.5
```

Pure Stage1 top-K coverage without GT insertion:

```bash
scripts/run_stage2_oracle_train.sh \
  --task put_block_back \
  --v10a \
  --no-graph-peak-insert-gt
```

## Files Changed

- `sam2act/mvt/graph_peak_selector.py`
  - Keeps V9 `TopK3DPeakSelector`.
  - Adds V10 `Stage2CandidateGraphSelector`.
- `sam2act/mvt/mvt_sam2.py`
  - Builds a Stage2 candidate packet.
  - Adds V10a selector crop and V10b K-crop branch routing.
- `sam2act/mvt/mvt_sam2_single.py`
  - Adds V10c memory heatmap formatting at the SAM2 memory write site.
- `sam2act/mvt/config.py`
  - Adds shared config defaults for all V10 modes.
- `sam2act/mvt/configs/sam2act.yaml`
  - Exposes the common V10 defaults in the run config.
- `scripts/run_stage2_oracle_train.sh`
  - Adds `--v10a`, `--v10b`, and V10c memory options.

## Interpretation

V10a answers whether the correct candidate is present and learnably selectable.
V10b answers whether Stage2 can still solve the action when multiple Stage1
crops are preserved and one branch is selected. V10c answers whether memory
attention improves when it stores a top-K-preserving mask instead of a raw
collapsed Stage1 heatmap.

If V10a selector accuracy is high but rollout remains bad, the issue is likely
the single-crop handoff or memory write timing. If V10b improves rollout,
single top1 crop commitment is the bottleneck. If V10c improves SAM2Act+
memory runs, the memory representation was amplifying collapse.
