# V9 Top-K Selector Run

Date: 2026-05-04
Branch: `exp/v9-topk-selector`

## Code status

- Added a lightweight V9 selector module:
  - `sam2act/mvt/graph_peak_selector.py`
- Connected `use_graph_peak_select=True` through:
  - `sam2act/mvt/mvt_sam2.py`
  - `sam2act/models/sam2act_agent.py`
  - `sam2act/train_plus.py`
  - `scripts/run_stage2_oracle_train.sh`
- Training mode is V9a:
  - extract Stage1 3D top-K candidates
  - insert GT candidate during training
  - train selector CE over candidates
  - keep normal Stage2 action loss on GT crop
- Eval mode uses selector argmax as the Stage2 crop center when oracle Stage1 is not forced.

## Smoke

Command: 1 iteration on GPU0, `put_block_back`, dirty stage1 `model_45.pth`.

Result:

```text
total_loss: 5.3244
trans_loss: 3.8874
graph_peak_select_loss: 1.0998
graph_peak_select_acc: 0.20
graph_peak_coverage: 1.00
graph_peak_pred_topk_coverage: 1.00
```

This confirms the new selector parameters load as missing checkpoint keys and
then train normally.

## Top-K audit

GPU7 audit completed for dirty stage1 `model_45.pth`.

Files:

- `sam2act/logs/stage1_collapse_probe_v9_topk_audit_model45_gpu7_20260504_175240.md`
- `sam2act/logs/stage1_collapse_probe_v9_topk_audit_model45_gpu7_20260504_175240.csv`
- `sam2act/logs/stage1_collapse_probe_v9_topk_audit_model45_gpu7_20260504_175240_per_episode.csv`

Key readout:

- `put_block_back`: recoverable top1 failures exist at KF0, KF1, KF6, KF8.
- `put_block_back`: KF4 and KF9 are not recoverable by topK=3 because GT is not in topK.
- `rearrange_block`: KF0 is fully recoverable by topK; KF8 mostly is not.
- `reopen_drawer`: KF3 is recoverable by topK; KF0 and KF4 are not.

## Active training

Started V9a full training on GPUs 0,1,2,3 and resumed from hdd4 after the
storage migration:

```text
task: put_block_back
stage1: /hdd4/siwon/checkpoints/sam2act/runs/sam2act_stage1_v7_spatial_only_3task/model_45.pth
run_dir: /hdd4/siwon/checkpoints/sam2act/runs/sam2act_stage2_v9a_topk_selector_put_block_back_model45_gpus0123_20260504_175941
log: /home/cv11/project/siwon/sam2act_stage2_oracle_ceiling/logs/stage2_v9a_topk_selector_put_block_back_model45_gpus0123_hdd4_resume_20260504_211228.log
pid: 78959
epochs: 20
topK: 3
selector loss weight: 1.0
```

At the latest check during epoch 2, progress had passed step 600/4000 with
`total_loss ~= 3.87` and `trans_loss ~= 3.64`.
