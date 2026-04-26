# SAM2Act-MT++ Detailed Research Plan

This document is the working research plan for extending `SAM2Act` from single-task MemoryBench to a stronger multitask memory framework.

It is written against the current code already implemented in this repo.

Relevant implementation anchors:

- [sam2act/mvt/config.py](/home/cv25/siwon/sam2act/sam2act/mvt/config.py)
- [sam2act/mvt/mvt_sam2.py](/home/cv25/siwon/sam2act/sam2act/mvt/mvt_sam2.py)
- [sam2act/mvt/mvt_sam2_single.py](/home/cv25/siwon/sam2act/sam2act/mvt/mvt_sam2_single.py)
- [sam2act/mvt/configs/sam2act_plus_mt_shared_fifo.yaml](/home/cv25/siwon/sam2act/sam2act/mvt/configs/sam2act_plus_mt_shared_fifo.yaml)
- [sam2act/mvt/configs/sam2act_plus_mt_routing.yaml](/home/cv25/siwon/sam2act/sam2act/mvt/configs/sam2act_plus_mt_routing.yaml)
- [sam2act/mvt/configs/sam2act_plus_mt_dual_memory.yaml](/home/cv25/siwon/sam2act/sam2act/mvt/configs/sam2act_plus_mt_dual_memory.yaml)
- [sam2act/mvt/configs/sam2act_plus_mt_full.yaml](/home/cv25/siwon/sam2act/sam2act/mvt/configs/sam2act_plus_mt_full.yaml)
- [scripts/run_memorybench_mt_ablation.sh](/home/cv25/siwon/sam2act/scripts/run_memorybench_mt_ablation.sh)

---

## 1. Problem Statement

### 1.1 Immediate practical problem

`SAM2Act+` works on `MemoryBench` only in the original single-task setting, and current reproduction is unstable or weak.

Observed failure modes:

1. Old crash issue from the simulator/plugin side was a separate environment problem.
2. Once crashes were isolated, evaluation still remained low or unstable.
3. MemoryBench checkpoints are much more fragile than RLBench multitask checkpoints.
4. Current code had to be made backward-compatible with older checkpoints after task-routing modules were introduced.
5. MemoryBench reproduction is confounded by stage-1/stage-2 checkpoint quality, replay setup, and evaluation variance.

### 1.2 Research problem

Even if single-task reproduction becomes stable again, `MemoryBench` remains too small and too weakly explored as a multitask memory benchmark.

The core research question is:

`Can SAM2Act learn shared multitask manipulation while preserving task-specific episodic recall through explicit memory lifecycle control?`

That question is stronger than:

- "Can joint training help?"
- "Can task conditioning help?"

It targets the actual mechanism:

- what to write
- what to retain
- what to retrieve

---

## 2. Current Thesis

### 2.1 Main claim

`Multitask failure on MemoryBench is not only a data-sharing problem. It is a memory lifecycle problem.`

Shared memory alone is too weak because tasks differ in:

- which past state matters
- how long it must be retained
- whether retrieval should emphasize recent local context or sparse task events

### 2.2 Method hypothesis

`Task-conditioned dual-level memory with adaptive event retention should outperform both shared FIFO memory and routing-only memory on multitask long-horizon manipulation.`

---

## 3. Current Method

### 3.1 Method name

Recommended working title:

`SAM2Act-MT++: Task-Conditioned Dual-Level Memory with Adaptive Event Consolidation`

Alternative title:

`Beyond FIFO Memory for Multitask Manipulation`

### 3.2 Method overview

The implemented method already supports four progressively stronger stage-2 variants:

1. `shared_fifo`
2. `routing`
3. `dual_memory`
4. `full`

These are controlled by new MVT config toggles.

### 3.3 Memory architecture

The method adds two memory levels:

- `local memory`
  - recent dense memory entries
  - preserves short-term spatial context

- `event memory`
  - sparse task-relevant entries
  - intended to preserve longer-horizon episodic state

### 3.4 Memory routing

Task identity is used to:

- gate memory channels
- bias local versus event retrieval
- score event writes

### 3.5 Event retention

Event memory can be pruned in two ways:

- FIFO-like pruning when adaptive pruning is off
- importance-based pruning when adaptive pruning is on

### 3.6 Heatmap-aware event features

The full method uses summary statistics from predicted heatmaps:

- peak confidence
- entropy
- mean
- standard deviation

These act as a lightweight spatial event descriptor.

---

## 4. High-Level Architecture Diagram

```text
Input Observation Sequence
        |
        v
  SAM2 Image Encoder
        |
        v
  SAM2Act Coarse Branch
        |
        +------------------------------+
        |                              |
        v                              v
  Local Memory Bank              Event Memory Bank
  recent dense slots             sparse episodic slots
        |                              |
        +--------------+---------------+
                       |
                       v
         Task-Conditioned Read Routing
         - memory channel gate
         - local/event fusion weights
                       |
                       v
            Memory-Aware Coarse Features
                       |
                       v
              Translation Heatmap Output
                       |
                       +----------------------+
                       |                      |
                       v                      v
            Heatmap Summary          Event Write Score
                       |                      |
                       +----------+-----------+
                                  |
                                  v
                    Event Memory Write / Retention
```

---

## 5. Variant Ladder and Intended Meaning

### 5.1 Variant 1: Shared FIFO

Preset:

- [sam2act_plus_mt_shared_fifo.yaml](/home/cv25/siwon/sam2act/sam2act/mvt/configs/sam2act_plus_mt_shared_fifo.yaml)

Meaning:

- pure multitask shared-memory baseline
- no task-conditioned routing
- no event memory

Scientific role:

- answers whether simple joint replay with original memory is enough

### 5.2 Variant 2: Routing

Preset:

- [sam2act_plus_mt_routing.yaml](/home/cv25/siwon/sam2act/sam2act/mvt/configs/sam2act_plus_mt_routing.yaml)

Meaning:

- task-conditioned memory channel routing only
- still one shared memory level

Scientific role:

- isolates the benefit of task-conditioned retrieval

### 5.3 Variant 3: Dual Memory

Preset:

- [sam2act_plus_mt_dual_memory.yaml](/home/cv25/siwon/sam2act/sam2act/mvt/configs/sam2act_plus_mt_dual_memory.yaml)

Meaning:

- local memory plus event memory
- task-conditioned local/event fusion
- dense event writes
- non-adaptive pruning

Scientific role:

- tests whether the dual-level structure helps even before learned event selection

### 5.4 Variant 4: Full

Preset:

- [sam2act_plus_mt_full.yaml](/home/cv25/siwon/sam2act/sam2act/mvt/configs/sam2act_plus_mt_full.yaml)

Meaning:

- task routing
- dual memory
- task-conditioned event writing
- heatmap-aware event features
- adaptive event pruning

Scientific role:

- complete proposed method

---

## 6. Detailed Contribution Breakdown

### 6.1 Contribution 1: Task-conditioned memory routing

What is new:

- task identity is no longer passive metadata
- it actively gates memory retrieval

Why it matters:

- multitask memory interference is likely task-dependent
- different tasks should not read the same memory channels in the same way

Where it lives in code:

- [mvt_sam2_single.py](/home/cv25/siwon/sam2act/sam2act/mvt/mvt_sam2_single.py#L258)
- [mvt_sam2_single.py](/home/cv25/siwon/sam2act/sam2act/mvt/mvt_sam2_single.py#L604)

### 6.2 Contribution 2: Dual-level memory

What is new:

- memory is explicitly split into recent dense context and sparse event context

Why it matters:

- local context is useful for short-horizon action refinement
- episodic events are useful for delayed recall of earlier task state

Where it lives in code:

- [mvt_sam2_single.py](/home/cv25/siwon/sam2act/sam2act/mvt/mvt_sam2_single.py#L251)
- [mvt_sam2_single.py](/home/cv25/siwon/sam2act/sam2act/mvt/mvt_sam2_single.py#L699)
- [mvt_sam2_single.py](/home/cv25/siwon/sam2act/sam2act/mvt/mvt_sam2_single.py#L717)

### 6.3 Contribution 3: Adaptive event consolidation

What is new:

- event memory can drop low-importance entries instead of strictly using FIFO

Why it matters:

- SAM2Act explicitly notes fixed memory window as a limitation
- not all historical slots are equally valuable

Where it lives in code:

- [mvt_sam2_single.py](/home/cv25/siwon/sam2act/sam2act/mvt/mvt_sam2_single.py#L665)

### 6.4 Contribution 4: Heatmap-aware event writing

What is new:

- event write decisions can use heatmap summary signals

Why it matters:

- spatially decisive steps are more likely to be true memory events
- supports the link between action localization and event selection

Where it lives in code:

- [mvt_sam2_single.py](/home/cv25/siwon/sam2act/sam2act/mvt/mvt_sam2_single.py#L652)
- [mvt_sam2_single.py](/home/cv25/siwon/sam2act/sam2act/mvt/mvt_sam2_single.py#L811)

---

## 7. Why This Method is Plausible

### 7.1 Why shared FIFO may fail

In multitask MemoryBench:

- one task may need recent manipulation state
- another may need an earlier scene configuration
- another may need both

Shared FIFO memory makes all tasks pay the same retention policy.

### 7.2 Why routing-only may be insufficient

Routing-only changes how memory is read, but not:

- what enters memory
- how long it remains

That likely helps, but it does not fully solve delayed recall.

### 7.3 Why dual memory is promising

Dual memory lets the model keep:

- a stable recent context path
- a sparse episodic path

This is the most direct architectural response to long-horizon manipulation with delayed recall.

### 7.4 Why adaptive event pruning is important

If event memory is sparse but still FIFO, it can still overwrite the wrong slots.

Adaptive pruning is therefore not optional if event memory is meant to be meaningful.

---

## 8. Connection to Referenced Papers

### 8.1 SAM2Act

Used for:

- base architecture
- MemoryBench formulation
- identifying fixed memory window as a limitation

### 8.2 MemoryVLA

Used for:

- thinking about memory as retrieval + fusion + consolidation
- motivating non-FIFO memory handling

### 8.3 ReMem-VLA

Used for:

- dual-level memory intuition
- short-term versus long-term memory split

### 8.4 RMBench

Used for:

- framing the problem as a memory-architecture question
- strengthening the paper narrative beyond a 3-task benchmark

### 8.5 BridgeVLA

Used for:

- spatially aligned event signals
- heatmap-aware memory decisions

### 8.6 MemoAct

Used for:

- supporting the short-term versus long-term memory story

### 8.7 CORAL

Used for:

- motivating task specialization as a competitor baseline

### 8.8 HSC-VLA

Used for:

- future direction on scene-clearing or clutter-aware writing

---

## 9. Exact Experimental Plan

### 9.1 Phase 0: Finish clean reproduction

Goal:

- stabilize MemoryBench single-task reproduction first

Required outputs:

- fresh `put_block_back` stage1 and stage2
- fresh `rearrange_block` stage1 and stage2
- clean 5-episode and 25-episode evals

Why:

- if the baseline is unstable, multitask results are hard to interpret

### 9.2 Phase 1: Multitask baseline

Run:

- `shared_fifo`

Tasks:

- first pair: `put_block_back,rearrange_block`

Outputs:

- stage1 training log
- stage2 training log
- per-task 5-episode smoke eval
- per-task 25-episode full eval

Question answered:

- does naïve multitask shared memory work at all

### 9.3 Phase 2: Routing ablation

Run:

- `routing`

Question answered:

- does task-conditioned retrieval improve over shared FIFO

### 9.4 Phase 3: Dual-memory ablation

Run:

- `dual_memory`

Question answered:

- does the two-level memory structure help even before learned event selection

### 9.5 Phase 4: Full method

Run:

- `full`

Question answered:

- do write scoring, heatmap-aware event signals, and adaptive pruning produce the best multitask result

---

## 10. Main Ablation Table

```text
Table 1. MemoryBench Multitask Main Results

1. Single-task SAM2Act+ per task
2. Multitask Shared FIFO
3. Multitask + Task Routing
4. Multitask + Dual Memory
5. Multitask + Full Method
```

Recommended metrics:

- per-task success rate
- average success rate
- standard deviation across repeats

### 10.0 Performance comparison philosophy

Performance comparison must answer four different questions, not just one.

1. `Does multitask hurt or help relative to single-task?`
2. `If multitask helps, is the gain coming from joint training alone or from memory design?`
3. `Does the proposed method improve the hard memory-heavy tasks, or only average performance?`
4. `Are gains stable enough to be believable, or are they just evaluation variance?`

Therefore the paper should not rely on a single average score.

The comparison must always include:

- single-task upper/reference line
- multitask shared baseline
- multitask proposed variants
- per-task breakdown
- repeated runs or repeated evals when possible

### 10.0.1 Primary comparison axes

The main performance story should be organized along these axes.

#### Axis A. Single-task versus multitask

This is the first and most important question.

```text
single-task SAM2Act+ per task
vs
multitask shared_fifo
vs
multitask full
```

Interpretation:

- if `shared_fifo` is much worse than single-task, multitask interference is real
- if `full` closes most of the gap, the method successfully addresses multitask memory interference
- if `full` exceeds single-task on some tasks, that is the strongest result

#### Axis B. Shared memory versus routed memory

```text
shared_fifo
vs
routing
```

Interpretation:

- isolates the value of task-conditioned retrieval
- answers whether memory selection alone matters

#### Axis C. Routing versus dual memory

```text
routing
vs
dual_memory
```

Interpretation:

- tests whether structural separation of local and event memory adds value beyond routing

#### Axis D. Dual memory versus full method

```text
dual_memory
vs
full
```

Interpretation:

- isolates the benefit of selective event writing, heatmap-aware event signals, and adaptive pruning

### 10.0.2 What "better performance" should mean

The ideal claim is not only:

`full has higher average success`

The stronger claim is:

`full improves memory-heavy tasks while reducing the multitask gap to single-task performance`

That requires showing:

- mean performance
- per-task performance
- retained or improved difficult-task performance

### 10.0.3 Performance interpretation template

Use this template during experiments:

```text
Case 1:
single-task > shared_fifo, and full > shared_fifo
=> multitask interference exists, and the method helps

Case 2:
single-task > shared_fifo, but full ~= shared_fifo
=> the method is not yet solving the real bottleneck

Case 3:
full > single-task on average
=> strong claim: shared multitask memory is beneficial, not only non-destructive

Case 4:
average improves but hard tasks do not
=> likely shallow gain; method may be helping easy tasks only
```

### 10.1 Extended ablation table

```text
Table 2. Component Ablation

A. Shared FIFO
B. + task-conditioned routing
C. + dual memory
D. + event write
E. + adaptive prune
F. + heatmap-aware event features
```

### 10.2 Analysis table

```text
Table 3. Memory Behavior Analysis

- local/event usage ratio
- number of event writes per episode
- retained event count
- sensitivity to event memory size

### 10.3 Proposed result tables in detail

#### Table A. Core MemoryBench performance

Columns:

- method
- put_block_back
- rearrange_block
- reopen_drawer if available
- average
- std

Rows:

1. single-task SAM2Act+
2. multitask shared_fifo
3. multitask routing
4. multitask dual_memory
5. multitask full

Purpose:

- this is the headline table

#### Table B. Multitask gap table

Columns:

- task
- single-task score
- multitask shared_fifo score
- multitask full score
- gap(shared_fifo - single-task)
- gap(full - single-task)

Purpose:

- quantifies how much the proposed method closes the multitask gap

#### Table C. Stability / reproducibility table

Columns:

- method
- eval repeat 1
- eval repeat 2
- eval repeat 3
- mean
- std

Purpose:

- addresses variance and trustworthiness
- especially important for MemoryBench where success rates can fluctuate a lot

#### Table D. Memory behavior table

Columns:

- method
- avg event writes / episode
- avg retained event slots
- avg local/event fusion ratio
- avg event-memory usage on success episodes
- avg event-memory usage on failure episodes

Purpose:

- demonstrates that the method is actually using the new memory mechanisms

### 10.4 What numbers would count as a meaningful win

Because the benchmark is small and unstable, the paper should define practical success criteria early.

Recommended thresholds:

1. `shared_fifo` should be worse than single-task on at least one hard task
   - otherwise the memory-interference story is weak

2. `full` should beat `shared_fifo` by a visible margin
   - recommended target: at least `+5 to +10 absolute success points` on average
   - or a similarly strong gain on the hardest task

3. `full` should reduce variance relative to weaker methods
   - not mandatory, but highly valuable

4. `full` should not only improve one easy task
   - the hard task improvement matters more than average-only gain

### 10.5 Comparison to external baselines

There are three baseline families to compare against.

#### Family 1. Original SAM2Act baselines

- single-task stage1
- single-task stage2
- any reproduced MemoryBench single-task baseline

Purpose:

- proves the proposed multitask method remains competitive with the original single-task recipe

#### Family 2. Internal ablation baselines

- shared_fifo
- routing
- dual_memory

Purpose:

- proves each design component matters

#### Family 3. External literature baselines

Potential families:

- task specialization baselines inspired by `CORAL`
- memory hierarchy baselines inspired by `MemoryVLA` or `MemoAct`
- same-platform long-horizon RLBench baselines where feasible

Purpose:

- situates the method beyond an internal-only ablation story

### 10.6 Same-platform performance comparison plan

If same-platform RLBench extensions are added, performance comparison should be organized differently from MemoryBench.

Recommended RLBench comparison axes:

1. `single-task RLBench` versus `multitask RLBench`
2. `downloaded/reference checkpoint` versus `freshly retrained checkpoint`
3. original `SAM2Act` versus `SAM2Act-MT++`

Recommended metrics:

- task success rate
- suite average
- if possible, zero-shot/generalization split

Purpose:

- shows that the proposed memory design does not only work on a tiny synthetic memory benchmark
- establishes broader platform consistency

### 10.7 External benchmark comparison plan

If `RMBench` or `RoboMME` is added later:

The paper should not immediately try to beat all foundation-model methods in breadth.

Instead, it should ask:

```text
Does the same memory-lifecycle design principle transfer?
```

Recommended comparison:

- original adapted SAM2Act baseline
- shared memory multitask baseline
- proposed full method

That is enough for a strong transfer story.

### 10.8 Failure-driven performance analysis

Not all failures mean the same thing.

The paper should classify failure cases into:

1. `write failure`
   - the relevant event never entered event memory

2. `retention failure`
   - event was written but later pruned or overwhelmed

3. `retrieval failure`
   - event remained in memory but was not used at decision time

4. `execution failure`
   - memory retrieval looked correct, but manipulation still failed

This turns performance comparison into a mechanism story instead of a pure scoreboard.

### 10.9 Go / no-go criteria for the paper direction

The method is worth pushing as the main paper direction if at least two of the following are true:

1. `full > shared_fifo` by a clear margin
2. `full > routing`
3. `full` reduces the gap to single-task
4. hard tasks improve substantially
5. event-memory analysis shows meaningful usage patterns

If these are not true, then the right fallback paper is:

- benchmark/reproduction analysis
- routing-only method
- or a stronger baseline paper around multitask memory interference
```

---

## 11. Evaluation Plan Beyond MemoryBench

MemoryBench alone is too small to carry the full paper.

### 11.1 Same-platform extension

Most realistic same-platform expansion:

- RLBench-based evaluation protocol
- Colosseum
- PerAct / RLBench task-set extensions

Why:

- same simulator family
- same manipulation domain
- better platform consistency for comparison

### 11.2 External memory benchmark

Best candidate:

- `RMBench`

Strong secondary candidate:

- `RoboMME`

Why:

- stronger benchmark framing
- broader memory taxonomy

### 11.3 Final benchmark mix

Recommended final paper benchmark mix:

1. MemoryBench
2. RLBench extension on same platform
3. RMBench or RoboMME if feasible

---

## 12. Execution Recipes

### 12.1 Scripted multitask training

```bash
bash scripts/run_memorybench_mt_ablation.sh shared_fifo put_block_back,rearrange_block mt_fifo_pb_rb
bash scripts/run_memorybench_mt_ablation.sh routing put_block_back,rearrange_block mt_routing_pb_rb
bash scripts/run_memorybench_mt_ablation.sh dual_memory put_block_back,rearrange_block mt_dualmem_pb_rb
bash scripts/run_memorybench_mt_ablation.sh full put_block_back,rearrange_block mt_full_pb_rb
```

### 12.2 Stage-2-only manual run

```bash
cd /home/cv25/siwon/sam2act/sam2act
torchrun --nproc_per_node=4 --nnodes=1 train_plus.py \
  --device 0,1,2,3 \
  --fresh-start \
  --exp_cfg_path configs/sam2act_plus.yaml \
  --mvt_cfg_path mvt/configs/sam2act_plus_mt_full.yaml \
  --exp_cfg_opts "tasks put_block_back,rearrange_block exp_name mt_full_pb_rb wandb False"
```

### 12.3 Eval

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

Repeat for each task separately.

---

## 13. Code Readiness Checklist

### 13.1 Already implemented

1. config toggles for all variants
2. task-conditioned routing scaffold
3. dual local/event memory banks
4. event write scoring path
5. adaptive event pruning path
6. variant presets
7. ablation launch script

### 13.2 Verified so far

1. script syntax
2. config merge behavior
3. Python compile
4. routing parameters remain trainable
5. stage1 MemoryBench path in the ablation script

### 13.3 Not yet verified under training load

1. full stage1 to stage2 handoff with new variants
2. actual tensor behavior during MemoryBench multitask training
3. convergence of event write and dual-memory fusion
4. effect on final success rate

---

## 14. Known Risks

### 14.1 Risk 1: baseline instability

If fresh single-task MemoryBench is still unstable, the multitask story will remain hard to interpret.

Mitigation:

- finish single-task reproduction first
- log and compare fresh checkpoints only

### 14.2 Risk 2: full method behaves like routing-only early on

Because initialization is conservative, event memory may be underused early in training.

Mitigation:

- inspect event write statistics during training
- compare `dual_memory` and `full`

### 14.3 Risk 3: event memory is too weakly supervised

Write scores currently depend on task embedding and heatmap summary without explicit event supervision.

Mitigation:

- first test whether implicit learning is enough
- add sparsity or event auxiliary losses only if necessary

### 14.4 Risk 4: benchmark weakness

MemoryBench alone may not be sufficient for a strong paper.

Mitigation:

- add RLBench same-platform extension
- add RMBench or RoboMME if feasible

---

## 15. Future Method Extensions

These are not required for the first paper version, but they are reasonable upgrades.

### 15.1 Merge-based event consolidation

Instead of dropping the lowest-importance event directly:

- merge highly similar events
- preserve coverage while saving slots

### 15.2 Scene-cleared writing

Task-conditioned spatial masking before write scoring:

- suppress distractor-heavy writes
- reduce memory pollution

### 15.3 Auxiliary event sparsity loss

Encourage event memory to stay sparse but informative.

### 15.4 Task-specific memory heads baseline

Strong competitor baseline:

- shared backbone
- task-specific memory adapters

---

## 16. Proposed Paper Structure

### 16.1 Introduction

- MemoryBench multitask is underexplored
- shared memory is insufficient
- memory lifecycle control is needed

### 16.2 Related Work

- memory in robotic manipulation
- memory in VLA models
- long-horizon benchmark design
- multitask interference

### 16.3 Method

- SAM2Act background
- task-conditioned routing
- dual-level memory
- heatmap-aware event writing
- adaptive event consolidation

### 16.4 Experiments

- single-task reproduction
- multitask MemoryBench
- ablations
- same-platform extension
- optional external memory benchmark

### 16.5 Analysis

- event usage
- retention behavior
- failure cases

---

## 17. Tomorrow Reading Order

Top priority:

1. `MemoryVLA`
2. `ReMem-VLA`
3. `RMBench`

Second priority:

4. `RoboMME`
5. `BridgeVLA`
6. `MemoAct`

Third priority:

7. `CORAL`
8. `HSC-VLA`
9. `MindExplore`

---

## 18. Immediate Next Actions

1. Finish single-task fresh MemoryBench reproduction.
2. Run `shared_fifo` and `full` smoke multitask training.
3. Verify logs, tensor behavior, and event-memory usage.
4. Expand to full ablation ladder.
5. Add same-platform benchmark extension.
6. Prepare the first paper result table.

---

## 19. Compact One-Page Summary Diagram

```text
Problem:
  MemoryBench multitask is weak and unstable.

Hypothesis:
  Shared FIFO memory is not enough.

Method:
  Task-conditioned dual-level memory
    -> local recent memory
    -> sparse event memory
    -> heatmap-aware event writing
    -> adaptive event pruning

Ablation ladder:
  shared_fifo -> routing -> dual_memory -> full

Expected outcome:
  better multitask recall
  better delayed state retrieval
  stronger long-horizon manipulation

Paper strength:
  MemoryBench main result
  RLBench same-platform extension
  optional RMBench / RoboMME external validation
```
