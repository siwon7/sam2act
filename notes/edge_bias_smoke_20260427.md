# Edge-Bias-Only Stage2 Smoke

Date: 2026-04-27

Branch:
- `exp/edge-bias-only`

## Goal

Keep SAM2Act stage2 memory architecture as intact as possible and only add a
small graph-style bias to the memory attention score:

```text
score_i = q_t k_i + lambda * b_i^edge
```

The purpose of this branch is to test whether a very small amount of structured
retrieval prior helps without introducing node merging, contrastive losses,
extra phase heads, or explicit graph supervision.

## What changed

### Kept unchanged

- temporal memory bank write path
- SAM2 memory encoder
- per-view memory bank structure
- base trans loss

### Added

Per-memory-step additive bias with four components:

1. temporal bonus
   - slightly favors recent memory
2. revisit bonus
   - favors memory whose stored 3D target is close to current query 3D point
3. transition bonus
   - favors memory near the previously most-attended memory step
4. grip bonus
   - weak bonus for similar gripper-open state

Implementation detail:
- the bias is expanded to all tokens of a memory step and added directly to the
  cross-attention score before softmax
- no hard mask is used

## Query / memory metadata

Training:
- current query position uses `wpt_local`
- current grip uses `proprio[..., 2]`

Evaluation:
- current query position uses the previous predicted waypoint
- current grip uses current `proprio[..., 2]`

Memory entries store:
- encoded memory tokens
- encoded memory positional embedding
- query position used when the memory was written
- gripper-open scalar
- observation index

## Smoke setup

Script:
- `scripts/run_put_block_back_edge_bias_smoke.sh`

Settings:
- task: `put_block_back`
- dirty stage1 checkpoint: `model_38.pth`
- `num_maskmem=11`
- 2 GPU DDP
- `demo=20`
- `train_iter=1200`
- effective training loop length: `50` updates

Bias scales:
- temporal: `0.15`
- revisit: `0.35`
- transition: `0.20`
- grip: `0.05`

## Result

Run:
- `sam2act/runs/sam2act_sam2act_pbb_edge_bias_smoke_retry_20260427_022826`

Training:
- finished normally
- final log:
  - `total_loss = 5.2202`
  - `trans_loss = 5.2202`

Evaluation:
- `5ep smoke = 0.0%`
- all 5 episodes timed out at length `25`

CSV:
- `sam2act/runs/sam2act_sam2act_pbb_edge_bias_smoke_retry_20260427_022826/eval/sam2act_sam2act_pbb_edge_bias_smoke_retry_20260427_022826_smoke5/model_plus_last/eval_results.csv`

## Interpretation

This is a cleaner first ablation than the previous graph-memory branches:

- no manual node labels
- no node merge
- no contrastive auxiliary
- no explicit new/revisit head

So the failure is informative:

- a weak additive edge bias alone is not enough to lift policy success
- but it also did not cause the immediate one-step collapse seen in some
  earlier graph-heavy branches

## Next candidates

1. stronger revisit scale, keep others weak
2. disable temporal bonus, keep only revisit + transition
3. text-conditioned bias term instead of hand-fixed scalar components
