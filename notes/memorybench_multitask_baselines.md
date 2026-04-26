# MemoryBench Multitask Baselines for SAM2Act

This note proposes five multitask baselines that are implementable in the current `sam2act` codebase and each targets a different failure mode of memory-based multitask learning.

Context from the current repo:

- Stage 1 uses temporal replay without memory-enabled inference/training.
- Stage 2 turns on memory and warm-starts from the stage-1 checkpoint in the same run folder.
- The current memory bank is sequence-local and is reset per sequence in `sam2act/mvt/mvt_sam2_single.py`.
- The data loader can already ingest multiple tasks, but the training and evaluation recipe is orchestrated as single-task runs.

The baselines below are ordered from lowest integration cost to highest research novelty.

## 1. Joint Replay + Shared Memory

### Core idea

Train stage 1 and stage 2 on a single replay buffer built from multiple MemoryBench tasks, with one fully shared model and one fully shared memory stack.

### Why it matters

This is the minimum credible multitask baseline. Without it, every stronger method can be dismissed as benefitting only from task-specific engineering.

### Why it is not trivial in this repo

- The data path can already accept multiple tasks, but the run orchestration and checkpoint handoff are single-task oriented.
- Stage 2 currently assumes one task-specific warm-start path and one task-specific run folder.

### Minimal implementation

- Allow `exp_cfg.tasks` to contain a comma-separated list for MemoryBench in both `train.py` and `train_plus.py`.
- Introduce a multitask run name such as `memorybench_mt_pb_rb_rd`.
- Reuse the existing temporal replay loader with `sample_distribution_mode=task_uniform`.
- Evaluate per task and average.

### Novelty level

Low by itself, but it is the necessary base table for the paper.

### Key risk

Shared memory may collapse into the dominant task and hurt all others through interference.

## 2. Task-Conditioned Memory Routing

### Core idea

Keep one shared backbone, but add lightweight task-conditioned gating on top of the memory retrieval path so each task can softly select which memory channels or slots matter.

### Why it matters

The current code reads `tasks` in the agent update path but does not use task identity for memory control. This is the cleanest way to attack memory interference without abandoning parameter sharing.

### Mechanism

- Encode task identity as a learned task embedding.
- Use that embedding to produce gates for memory attention, memory encoder output, or memory-slot scoring.
- The image backbone remains shared; only the routing policy changes by task.

### Minimal implementation

- Add a task vocabulary from `exp_cfg.tasks`.
- Thread task ids through replay to the model forward.
- Inject a small MLP gate into the memory path before or inside `memory_attention`.

### Novelty level

Medium to high. It is more than joint training, but still simple enough to implement and ablate.

### Key risk

If the gate is too strong, this degenerates into soft task-specific branches and loses the benefit of shared memory.

## 3. Shared Backbone + Task-Specific Memory Heads

### Core idea

Train one shared stage-1 backbone across all tasks, then attach a small task-specific memory adapter for stage 2 while keeping the visual encoder and action decoder mostly shared.

### Why it matters

This baseline tests whether the real bottleneck is shared perception or shared episodic recall. It gives a direct answer to whether memory interference is the central issue.

### Mechanism

- One shared model for image encoding and coarse action prediction.
- A small per-task adapter around memory attention or memory token projection.
- Memory adapters are selected by task id at stage 2.

### Minimal implementation

- Keep current stage-1 joint training shared.
- Replace the single memory block with a bank of lightweight adapters.
- Dispatch the matching adapter using the task id.

### Novelty level

Medium. This is a strong structural baseline and produces a useful ablation against fully shared memory.

### Key risk

If each adapter is too large, the method becomes hard to compare fairly against single-task models.

## 4. Task-Aware Prototype Memory

### Core idea

Augment the episodic memory bank with a small set of learned task prototypes. During retrieval, the model attends over both the current episode memory and the task prototype memory.

### Why it matters

Current memory is strictly episode-local. That means there is no persistent task-level knowledge in the memory path. Prototype memory adds reusable task structure without storing cross-episode raw trajectories.

### Mechanism

- Learn a small set of persistent memory tokens per task.
- At inference, combine prototype tokens with the per-episode memory bank.
- Let the model attend over both sources.

### Minimal implementation

- Add task-specific learnable tokens near `maskmem_tpos_enc` or as an additional memory source to `memory_attention`.
- Condition retrieval on the current task embedding.
- Compare against shared-memory and adapter baselines.

### Novelty level

High. This moves beyond ordinary multitask training and argues for a new type of task memory.

### Key risk

Prototype memory may simply act like extra parameters unless the ablation isolates its effect from model size.

## 5. Curriculum Stage-2 Memory Mixing

### Core idea

Do not switch from single-task stage 1 to fully joint stage 2 in one step. Instead, use a curriculum where stage 2 starts with single-task memory episodes, then progressively mixes cross-task batches while controlling memory interference.

### Why it matters

The current repo already separates stage 1 and stage 2. That creates a natural place to introduce a curriculum, and it is especially relevant when memory modules are newly initialized at stage 2.

### Mechanism

- Start stage 2 from a shared multitask stage-1 checkpoint.
- Use a schedule on task mixing or sampling temperature.
- Increase the proportion of mixed-task batches over time.

### Minimal implementation

- Extend `sample_distribution_mode` or batch construction with a stage-2 curriculum schedule.
- Log per-task success and loss throughout the schedule.
- Compare against immediate fully joint stage 2.

### Novelty level

High if framed properly. It uses the repo's two-stage design as a core algorithmic lever instead of treating it as a training detail.

### Key risk

If the curriculum only improves optimization smoothness, reviewers may see it as a training trick unless the interference story is well supported.

## Recommended order

If the goal is a paper and not just a working multitask run, the strongest progression is:

1. Joint Replay + Shared Memory
2. Task-Conditioned Memory Routing
3. Shared Backbone + Task-Specific Memory Heads
4. Task-Aware Prototype Memory
5. Curriculum Stage-2 Memory Mixing

## Current recommendation

The highest-value first target is `Task-Conditioned Memory Routing`.

Why:

- It directly addresses the current missing ingredient: task identity is not used to control memory.
- It preserves the core SAM2Act story instead of replacing it.
- It is easy to compare against both fully shared and task-specific alternatives.
- It gives a clean ablation axis for the paper: no routing vs soft routing vs task-specific memory.

## Code touch points for the top candidate

If we implement `Task-Conditioned Memory Routing` first, the likely minimum touch set is:

- `sam2act/models/sam2act_agent.py`
  - Convert `replay_sample["tasks"]` from passive metadata into a task id tensor.
  - Pass the task id or task embedding into the network forward.

- `sam2act/mvt/mvt_sam2.py`
  - Extend the forward signature to accept task conditioning alongside `lang_emb`.
  - Thread the task embedding into the memory-enabled stage.

- `sam2act/mvt/mvt_sam2_single.py`
  - Inject a small task-conditioned gate before `memory_attention`, on memory-slot weights, or on the memory features returned by `sam2_forward_with_memory`.
  - Keep the gate lightweight so the comparison against shared-memory remains fair.

- `sam2act/train.py` and `sam2act/train_plus.py`
  - Build and persist a multitask vocabulary for the current run.
  - Support multitask MemoryBench run names and checkpoint handoff.

- `sam2act/eval.py`
  - Load the same task vocabulary and evaluate each MemoryBench task under the correct task id.

## First ablation table for the paper

The first strong table should be:

1. Single-task SAM2Act+ per task
2. Joint Replay + Shared Memory
3. Joint Replay + Task-Conditioned Memory Routing
4. Shared Backbone + Task-Specific Memory Heads

This table answers the reviewer's first question:

Is the gain coming from joint training itself, from better memory routing, or from giving each task its own memory block?
