# Stage1 Collapse Probe on GPU7

## Question

The target question is whether the current failure mode comes from the
stage2-oracle training method or from the stage1 heatmap used by the normal
SAM2Act pipeline.

The GPU7 probe checks the stage1 waypoint that would be used as the stage2
crop center. It does not evaluate the final stage2 action. A collapse is only
bad when the collapsed top1/crop is not the GT or a valid task-equivalent
direction.

## Setup

- Probe script: `scripts/stage1_collapse_probe.py`
- Dirty checkpoint launcher: `scripts/run_stage1_collapse_probe_gpu7.sh`
- Dirty checkpoints:
  - `sam2act_memorybench_put_block_back/model_last.pth`
  - `sam2act_memorybench_rearrange_block/model_last.pth`
  - `sam2act_memorybench_reopen_drawer/model_last.pth`
- Episodes: `0..9`
- Correct crop threshold: `0.050m` from GT
- Peak status from median p1/p2 ratio:
  - `alive <= 3`
  - `weak <= 20`
  - `faint <= 100`
  - `collapsed > 100`

Raw result files:

- `logs/stage1_collapse_probe_dirty_put_block_back_model_last_gpu7_20260504_pos2_072558.md`
- `logs/stage1_collapse_probe_dirty_rearrange_block_model_last_gpu7_20260504_pos2_072558.md`
- `logs/stage1_collapse_probe_dirty_reopen_drawer_model_last_gpu7_20260504_pos2_072558.md`

## Dirty Checkpoint Findings

| task | KF | MP | peak | top1 gt/alt/tie/off | correct/wrong | collapsed ok/wrong/total | score gt/alt/tie | GT/p1 | alt/GT | main cause |
|---|---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---|
| put_block_back | 0 | 0/10 | collapsed | 5/0/0/5 | 4/6 | 4/5/9 | 0/0/0 | 0.333-0.344-1.000 | - | mixed single-step: GT collapse in 4 eps, wrong/off in 6 eps |
| put_block_back | 2 | 10/10 | collapsed | 7/3/0/0 | 7/3 | 6/0/6 | 7/1/2 | 0.302-0.864-1.000 | 0.000-0.001-2.599 | mostly OK, but some alt/tie errors |
| put_block_back | 4 | 10/10 | collapsed | 0/10/0/0 | 0/10 | 0/10/10 | 0/10/0 | 0.000-0.000-0.000 | inf-inf-inf | heatmap favors alt candidate |
| put_block_back | 6 | 0/10 | collapsed | 4/0/0/6 | 0/10 | 0/7/7 | 0/0/0 | 0.318-0.344-0.538 | - | GT direction/far crop plus off-target predictions |
| put_block_back | 9 | 10/10 | collapsed | 0/10/0/0 | 0/10 | 0/8/8 | 0/10/0 | 0.000-0.000-0.000 | 4868.006-10230.830-inf | heatmap favors alt candidate |
| rearrange_block | 0 | 0/10 | collapsed | 10/0/0/0 | 0/10 | 0/10/10 | 0/0/0 | 0.334-0.340-0.391 | - | GT direction but crop is consistently far, about 0.212m |
| rearrange_block | 5 | 0/10 | collapsed | 10/0/0/0 | 9/1 | 8/1/9 | 0/0/0 | 0.198-0.966-1.000 | - | mostly OK, one far-crop outlier |
| rearrange_block | 7 | 0/10 | faint | 5/0/0/5 | 4/6 | 4/1/5 | 0/0/0 | 0.327-0.336-0.586 | - | mixed GT/far/off behavior |
| rearrange_block | 8 | 0/10 | collapsed | 0/0/0/10 | 0/10 | 0/10/10 | 0/0/0 | 0.000-0.001-0.001 | - | GT score suppressed |
| reopen_drawer | 0 | 10/10 | collapsed | 0/0/0/10 | 0/10 | 0/10/10 | 10/0/0 | 0.000-0.000-0.001 | 0.000-0.000-0.000 | GT score suppressed/off-target top1 |
| reopen_drawer | 5 | 0/10 | collapsed | 1/0/0/9 | 0/10 | 0/10/10 | 0/0/0 | 0.248-0.329-0.398 | - | mostly off-target single-step predictions |
| reopen_drawer | 6 | 0/10 | weak | 10/0/0/0 | 4/6 | 1/0/1 | 0/0/0 | 0.192-0.338-0.484 | - | GT direction but many far crops |
| reopen_drawer | 8 | 10/10 | collapsed | 0/9/0/1 | 0/10 | 0/8/8 | 1/9/0 | 0.000-0.000-0.333 | 0.711-1453.387-inf | heatmap favors alt candidate |

## Interpretation

This does not show that the stage2-oracle training method is broken.

The dirty checkpoint has a stage1 prediction problem: at several keyframes the
stage1 heatmap collapses to the wrong target, suppresses the GT score, or
selects a crop that points in the GT direction but is too far from the GT
position. In normal rollout this becomes a pipeline ceiling because stage2
receives the stage1 crop. In the oracle stage2 training runs, the stage2 crop
is forced to GT, so these dirty stage1 top1 errors are bypassed during
training.

Practical read:

- Benign collapse: top1 is collapsed but still lands on GT or a valid
  equivalent direction.
- Bad collapse: top1 is collapsed to alt/off or the crop is far from GT.
- Dirty-specific issue: if the latest stage1 checkpoint fixes these rows.
- Method/label issue: if the latest stage1 checkpoint repeats the same rows
  with the same GT suppression or alt preference.

## Next GPU7 Experiment

Run the same probe against the latest numeric checkpoint from the active
stage1 v7 training, not `model_last.pth`.

Targets:

- `runs/sam2act_stage1_v7_spatial_only_3task/model_33.pth`
- `runs/sam2act_stage1_v7_spatial_only_3task/model_34.pth`

Outputs:

- `logs/stage1_collapse_probe_stage1_v7_model_33_gpu7_20260504_073352.md`
- `logs/stage1_collapse_probe_stage1_v7_model_33_gpu7_20260504_073352.csv`
- `logs/stage1_collapse_probe_stage1_v7_model_33_gpu7_20260504_073352_per_episode.csv`
- `logs/stage1_collapse_probe_stage1_v7_model_34_gpu7_20260504_073821.md`
- `logs/stage1_collapse_probe_stage1_v7_model_34_gpu7_20260504_073821.csv`
- `logs/stage1_collapse_probe_stage1_v7_model_34_gpu7_20260504_073821_per_episode.csv`

## Latest Numeric Checkpoint Result

`model_33.pth` was the latest numeric checkpoint when the first comparison
started. `model_34.pth` appeared while the probe was running, so GPU7 was run
again on `model_34.pth`. The latest numeric checkpoint does not simply fix the
dirty behavior. Some dirty failures persist, some improve, and reopen_drawer
exposes additional wrong-direction rows.

| task | KF | dirty verdict | model_34 verdict | read |
|---|---:|---|---|---|
| put_block_back | 4 | wrong-collapse-mp, alt 10/10 | wrong-collapse-mp, alt 10/10 | persistent alt-collapse |
| put_block_back | 6 | wrong-collapse-single | wrong, GT direction/far crop 10/10 | persistent far-crop, collapse ratio improved |
| put_block_back | 9 | wrong-collapse-mp, alt 10/10 | wrong, alt 10/10 | persistent alt preference, collapse ratio improved |
| rearrange_block | 0 | wrong-collapse-single | ok | fixed in model_34 |
| rearrange_block | 8 | wrong-collapse-single | mixed | improved but still 7/10 wrong |
| reopen_drawer | 0 | wrong-collapse-mp/off | wrong, alt-favored 10/10 | persistent wrong target, no longer collapsed |
| reopen_drawer | 5 | wrong-collapse-single | ok | fixed in model_34 |
| reopen_drawer | 6 | mixed | mixed | similar, 5/10 correct |
| reopen_drawer | 8 | wrong-collapse-mp | mixed | no longer collapsed, still 9/10 wrong |
| reopen_drawer | 2 | ok-collapse | wrong | new/latest issue: off/tie against ambiguous targets |
| reopen_drawer | 3 | ok-collapse | mixed | new/latest issue: 6/10 wrong despite strong GT score |
| reopen_drawer | 4 | ok-collapse | wrong | new/latest issue: alt/tie/far behavior |

Problem rows from latest numeric `model_34`:

| task | KF | MP | peak | top1 correct/wrong | score gt/alt/tie | GT/p1 | alt/GT | cause |
|---|---:|:---:|:---:|:---:|:---:|---:|---:|---|
| put_block_back | 2 | 10/10 | alive | 0/10 | 5/4/1 | 0.231-0.329-0.831 | 0.292-0.843-2.552 | heatmap_favors_alt_candidate:4,top1_wrong_despite_gt_score:3 |
| put_block_back | 4 | 10/10 | collapsed | 0/10 | 0/10/0 | 0.000-0.000-0.000 | 126073.905-1316096.000-inf | heatmap_favors_alt_candidate:10 |
| put_block_back | 6 | 0/10 | weak | 0/10 | 0/0/0 | 0.158-0.326-0.358 | - | gt_direction_but_far_crop:10 |
| put_block_back | 9 | 10/10 | weak | 0/10 | 0/10/0 | 0.023-0.056-0.134 | 8.187-19.898-50.424 | heatmap_favors_alt_candidate:10 |
| rearrange_block | 5 | 0/10 | weak | 3/7 | 0/0/0 | 0.236-0.471-0.994 | - | gt_direction_but_far_crop:7 |
| rearrange_block | 8 | 0/10 | alive | 3/7 | 0/0/0 | 0.238-0.451-0.709 | - | gt_direction_but_far_crop:5,single_step_direction_wrong:2 |
| reopen_drawer | 0 | 10/10 | alive | 0/10 | 0/10/0 | 0.000-0.000-0.000 | 7189.445-16952.389-58165.895 | heatmap_favors_alt_candidate:10 |
| reopen_drawer | 2 | 10/10 | alive | 0/10 | 0/0/10 | 0.956-0.990-1.000 | 0.975-0.987-0.990 | gt_alt_scores_tied_but_top1_wrong:10 |
| reopen_drawer | 3 | 10/10 | alive | 4/6 | 10/0/0 | 0.874-0.914-0.926 | 0.370-0.377-0.385 | top1_wrong_despite_gt_score:6 |
| reopen_drawer | 4 | 10/10 | weak | 0/10 | 4/4/2 | 0.118-0.283-0.386 | 0.156-1.021-2.475 | heatmap_favors_alt_candidate:4,top1_wrong_despite_gt_score:4 |
| reopen_drawer | 8 | 10/10 | weak | 1/9 | 0/4/6 | 0.324-0.345-0.473 | 0.842-1.103-1.467 | gt_alt_scores_tied_but_top1_wrong:5,heatmap_favors_alt_candidate:4 |

## Conclusion

The dirty checkpoint is not the only problem. The active stage1 training
checkpoint still has persistent wrong-target rows. Therefore the immediate
ceiling in normal rollout is stage1 top1/crop quality, not stage2-oracle
training. The oracle stage2 runs are still valid as a ceiling check because
they bypass these stage1 crop errors during training.

Comparison rule:

- If put_block_back KF4/KF9 and reopen_drawer KF8 stop preferring alt, the
  dirty checkpoint is the main problem.
- If rearrange_block KF0 remains GT-direction-but-far, inspect waypoint
  extraction/crop target semantics for that keyframe.
- If reopen_drawer KF0 and rearrange_block KF8 still suppress GT score, inspect
  label projection, visibility, and heatmap target construction.
- If the same failure rows persist across dirty and latest stage1, treat this
  as a stage1 objective/data semantics issue rather than a single checkpoint
  issue.

Observed outcome for latest numeric `model_34`:

- put_block_back KF4/KF9 persisted, so this is not just dirty checkpoint noise.
- put_block_back KF6 persisted as GT-direction-but-far crop.
- rearrange_block KF0 is fixed, while KF5/KF8 still show far-crop behavior.
- reopen_drawer KF0 persists and KF2/KF3/KF4/KF8 need a vision/target
  semantics check.
