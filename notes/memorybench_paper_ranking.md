# SAM2Act MemoryBench Multitask Directions: Paper Ranking

This note ranks the main multitask directions for `sam2act` by paper-worthiness, novelty, and fit to the current codebase.

Current repo constraints that matter:

- MemoryBench stage 1 and stage 2 are organized as single-task runs.
- The loader can already ingest multiple tasks.
- The memory bank is sequence-local and reset each sequence.
- `tasks` metadata is available in the agent update path but is not used to control memory behavior.
- Full evaluation currently reproduces as a low and unstable baseline, so the method must improve both capability and experimental clarity.

## Rank 1. Task-Conditioned Memory Routing

### Core claim

Multitask failure in MemoryBench is not just a data-sharing problem. It is a memory selection problem. A shared policy improves when memory retrieval is conditioned on task identity.

### Why it is novel enough

- The current code shares visual features and language inputs but does not use task identity to control the memory path.
- This gives a direct mechanism-level hypothesis rather than “we trained on more tasks.”
- It is more novel than joint replay and cleaner than introducing a fully separate model per task.

### Why it fits sam2act

- `tasks` already exists in `sam2act/models/sam2act_agent.py`.
- The memory path is localized enough that a routing gate can be introduced without rewriting the whole architecture.
- It preserves the core SAM2Act+ identity: a shared visual policy with a memory module.

### Minimum implementation

- Build a task-id vocabulary for the active multitask run.
- Thread task ids from `SAM2Act_Agent.update()` into `MVT_SAM2.forward()`.
- Add a small task embedding and gate on memory attention or memory feature fusion.
- Keep stage 1 shared and apply the routing mainly in stage 2.

### Best ablation

- Shared memory, no routing
- Shared memory, learned soft routing
- Task-specific memory heads

### Likely reviewer criticism

- “This is just task conditioning.”
- “The gain might come from extra parameters rather than better memory.”

### How to defend it

- Keep the router tiny.
- Match parameter count against a task-specific adapter baseline.
- Show reduced interference, not just higher mean success.

## Rank 2. Task-Aware Prototype Memory

### Core claim

Episode-local memory is insufficient for multitask manipulation. A small persistent task prototype memory improves retrieval by injecting reusable task structure.

### Why it is novel enough

- Current SAM2Act memory is reset per sequence, so it is not really a task-level memory system.
- Prototype memory extends the repository's memory story in a conceptually new way.
- This is a stronger conceptual contribution than a training-only trick.

### Why it fits sam2act

- The existing memory interface already has explicit memory tokens and positional encodings.
- Prototype memory can be added as an extra memory source rather than rewriting the backbone.

### Minimum implementation

- Add learned task prototype tokens.
- Concatenate or fuse prototype tokens with episodic memory before memory attention.
- Use task identity to select the right prototype set.

### Best ablation

- Shared episodic memory only
- Shared episodic memory + shared prototypes
- Shared episodic memory + task-specific prototypes

### Likely reviewer criticism

- “This may just be more parameters.”
- “Prototype tokens may act like a lookup table rather than real memory.”

### How to defend it

- Hold parameter count fixed where possible.
- Show that prototypes help especially on long-horizon, memory-heavy tasks.
- Compare against task-specific memory adapters.

## Rank 3. Shared Backbone + Task-Specific Memory Heads

### Core claim

The main interference point is memory, not perception. Sharing the backbone while separating the memory head is the right multitask decomposition.

### Why it is useful

- It is a very strong structural baseline.
- If this wins, it proves that fully shared memory is the wrong bias for MemoryBench.
- It is clean to implement and valuable even if it is not the final paper method.

### Why it is less novel

- Reviewers may see it as a modularization baseline rather than a main contribution.
- It can look like “partial parameter sharing” unless the paper has a strong interference analysis.

### Minimum implementation

- Joint multitask stage 1 shared across all tasks.
- Replace one shared memory stack with a small bank of task-specific memory adapters.
- Dispatch by task id in stage 2.

### Best ablation

- Fully shared memory
- Task-specific memory heads
- Task-conditioned routed memory

### Likely reviewer criticism

- “You just gave each task its own branch.”
- “This is an engineering compromise, not a new memory mechanism.”

### How to defend it

- Keep the task-specific memory module very small.
- Show it is a diagnostic baseline and not the sole novelty claim.

## Rank 4. Curriculum Stage-2 Memory Mixing

### Core claim

Multitask memory should not be turned on abruptly. A stage-2 curriculum that gradually mixes tasks reduces interference and stabilizes memory learning.

### Why it is interesting

- It uses the repo's existing two-stage structure as an algorithmic lever.
- It may be strong empirically if stage 2 optimization is fragile.

### Why it ranks lower

- It is more vulnerable to the “training trick” criticism.
- If the paper's core novelty is just scheduling, reviewers may want a stronger mechanism.

### Minimum implementation

- Joint multitask stage-1 checkpoint.
- Stage-2 batch schedule or sampling curriculum from single-task-dominant to mixed-task.
- Per-task tracking across the curriculum.

### Best ablation

- Immediate joint stage 2
- Curriculum stage 2
- Curriculum stage 2 + routed memory

### Likely reviewer criticism

- “This is optimization engineering.”
- “Would a simpler LR or warmup change do the same?”

### How to defend it

- Position it as a secondary method or stabilizer, not the flagship idea.

## Rank 5. Joint Replay + Shared Memory

### Core claim

A single shared multitask replay and a single shared memory policy can solve multiple MemoryBench tasks.

### Why it must exist

- This is the baseline every other idea needs.
- Without it, any stronger method is under-anchored.

### Why it is not enough as a paper

- The novelty is weak.
- If it works, it is still mostly a recipe contribution.
- If it fails, it only tells us interference exists.

### Minimum implementation

- Multitask task list in `train.py` and `train_plus.py`
- Shared run naming and checkpoint handoff
- Per-task eval breakdown

### Best ablation

- Single-task SAM2Act+
- Joint replay + shared memory

### Likely reviewer criticism

- “This is just the obvious baseline.”

## Recommendation

If the goal is the strongest paper direction, do this order:

1. Establish `Joint Replay + Shared Memory` as the baseline.
2. Make `Task-Conditioned Memory Routing` the main method.
3. Use `Shared Backbone + Task-Specific Memory Heads` as the strongest competing decomposition.
4. Add `Task-Aware Prototype Memory` only if time remains and the routing result plateaus.

## Best overall paper framing

`SAM2Act-MT: Task-Conditioned Memory Routing for Multitask Long-Horizon Manipulation`

This framing works because:

- It directly fixes a real gap in the current code.
- It gives a mechanism, not just a recipe.
- It keeps the original SAM2Act story intact.
- It supports clean ablations against both shared and task-specific memory baselines.
