# Stage2 Oracle Ceiling Pipeline

This branch is for measuring the stage2 ceiling when the stage1 crop center is assumed correct.

## What Is Oracle Here

In the current non-memory stage2 code path, training already uses the ground-truth waypoint as the stage1 location:

- `mvt/mvt_sam2.py`: during training, `wpt_local_stage_one = wpt_local`.
- The point cloud for `mvt2` is cropped with `trans_pc(... loc=wpt_local_stage_one_noisy ...)`.
- The target for `mvt2` is the GT waypoint transformed into that local crop.

So the baseline oracle stage2 experiment is:

- load a dirty stage1 checkpoint as initialization;
- train stage2 with `stage_two=True`;
- keep `use_memory=False`;
- keep `use_multipeak=False`, so the target heatmap is the single GT heatmap.

This isolates whether stage2 can learn the correct local refinement when stage1 is not allowed to choose the wrong crop center during training.

## Why Dirty Stage1 Can Be Used

`train_plus.py` looks for `model_last.pth` in the target stage2 run directory. If it exists, it loads that checkpoint with `load_agent_only_model(...)` and then starts saving stage2 checkpoints as `model_plus_*.pth`.

The run script symlinks the dirty stage1 checkpoint into the new stage2 run dir:

```bash
/hdd4/siwon/checkpoints/sam2act/runs/sam2act_stage2_oracle_<task>_dirty_stage1/model_last.pth \
  -> /hdd4/siwon/checkpoints/sam2act/runs/sam2act_memorybench_<task>/model_last.pth
```

That means the dirty stage1 model can seed oracle stage2 training immediately.

## Commands

Prepare one task without training:

```bash
bash scripts/run_stage2_oracle_train.sh --task put_block_back --prepare-only
```

Smoke test on GPU 4:

```bash
bash scripts/run_stage2_oracle_train.sh \
  --task put_block_back \
  --gpu 4 \
  --exp-suffix smoke \
  --epochs 1 \
  --train-iter 10 \
  --bs 10 \
  --num-workers 0
```

`demo_uniform_temporal` samples `num_maskmem + 1` observations per sequence. With the default `num_maskmem=9`, `bs` must be a multiple of 10.

Full single-task run:

```bash
bash scripts/run_stage2_oracle_train.sh --task put_block_back --gpu 4
```

Launch the three dirty-stage1 oracle runs detached on GPUs 4, 5, and 6:

```bash
bash scripts/launch_stage2_oracle_3task_detached.sh
```

Logs for detached jobs go under:

```bash
/home/cv11/project/siwon/sam2act_stage2_oracle_ceiling/logs/
```

Stage2 checkpoints go under:

```bash
/hdd4/siwon/checkpoints/sam2act/runs/sam2act_stage2_oracle_<task>_dirty_stage1[_suffix]/
```

## Forced Heatmap Variants

For the first ceiling run, forced heatmap code is not required because non-memory stage2 training already uses the GT crop center and a single GT heatmap.

For rollout ceiling, this branch adds `eval.py --oracle-stage1`. RLBench eval resets to a demo, reads the demo keypoint action for the current policy step, injects the keypoint translation as `oracle_stage1_wpt`, and forces the stage1 heatmap/crop to that location. The final action is still predicted by stage2.

Smoke run:

```bash
bash scripts/run_stage2_oracle_eval.sh \
  --task put_block_back \
  --gpu 7 \
  --model-folder /hdd4/siwon/checkpoints/sam2act/runs/sam2act_stage2_oracle_put_block_back_dirty_stage1_smoke \
  --model-name model_plus_last.pth \
  --eval-datafolder /hdd4/siwon/datasets/sam2act/data_memory/test \
  --eval-episodes 1 \
  --episode-length 12 \
  --log-name oracle_stage1_smoke
```

The eval launcher uses `xvfb-run` by default because CoppeliaSim vision sensors need an X-backed GL context on this machine.

The useful next variants are:

- Offline oracle eval: run `mvt2` with the GT crop center instead of `mvt1` top1, then measure the local heatmap error.
- Memory oracle: if `use_memory=True`, write memory from GT heatmaps only. This checks whether correct memory can remove the ceiling caused by wrong stage1 heatmaps being written into memory.

The important distinction is training versus rollout. Training can be oracle now. Standard rollout eval still crops stage2 around stage1 top1, so a dirty stage1 wrong-collapse can remain a rollout ceiling unless eval is also oracle-forced.
