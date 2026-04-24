# cv11 Server Handoff

## Repo

- worktree: `/home/cv25/siwon/sam2act_multitask_txt_memory`
- branch: `exp/persistent-anchor-memory`

## What Is In This Branch

- shared MemoryBench `phase_label` replay field
- stage2 `phase_aux_loss`
- text-gated memory scaffold
- persistent anchor memory bank
- 4-GPU launcher for multitask stage2 experiments

## Basic Launcher

Script:

- `/home/cv25/siwon/sam2act_multitask_txt_memory/scripts/run_memorybench_multitask_stage2_4gpu.sh`

Usage:

```bash
cd /home/cv25/siwon/sam2act_multitask_txt_memory

ENABLE_COARSE_PHASE_AUX=1 \
ENABLE_PERSISTENT_ANCHOR=1 \
COARSE_PHASE_EXP_CFG_OPTS="peract.phase_aux_loss_weight 0.5 peract.phase_aux_num_classes 4" \
AUTO_EVAL_EPISODES=5 \
./scripts/run_memorybench_multitask_stage2_4gpu.sh \
  "(put_block_back,reopen_drawer,rearrange_block)" \
  /abs/path/to/stage1_run_dir_or_model_last.pth \
  multitask_phaseaux_anchor_v1
```

## Useful Flag Sets

### Phase Aux Only

```bash
ENABLE_COARSE_PHASE_AUX=1
COARSE_PHASE_EXP_CFG_OPTS="peract.phase_aux_loss_weight 0.5 peract.phase_aux_num_classes 4"
```

### Persistent Anchor Only

```bash
ENABLE_PERSISTENT_ANCHOR=1
PERSISTENT_ANCHOR_MVT_CFG_OPTS="persistent_anchor_enabled True persistent_anchor_max_steps 2"
```

### Text Gate

```bash
ENABLE_TEXT_TASK_GATE=1
TEXT_TASK_GATE_MVT_CFG_OPTS="memory_gate_enabled True memory_gate_mode both memory_gate_use_text True"
```

### Put Block Back–style Long Window

```bash
ENABLE_MEM11=1
BS=12
NUM_MASKMEM=11
```

## Suggested Order

1. multitask baseline
2. phase aux only
3. persistent anchor only
4. phase aux + persistent anchor
5. phase aux + persistent anchor + text gate

## Notes

- `put_block_back` benefits from `mem11`; multitask runs may need to keep
  `NUM_MASKMEM=11` only if the temporal sampler and task mix can support it.
- persistent anchors default to the first 2 sequence steps and are kept outside
  the FIFO memory window.
- this branch is for experimentation, not the clean upstream baseline.
