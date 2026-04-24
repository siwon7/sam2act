# Multitask Text-Memory Plan

## Goal

Improve MemoryBench stage2 by adding task-aware memory structure beyond plain FIFO attention.

Primary targets:
- reduce button-anchor bias in `put_block_back`
- support multitask memory policies across
  - `put_block_back`
  - `reopen_drawer`
  - `rearrange_block`

## Branch Roles

- `main`
  - dirty baseline and accumulated practical tooling
- `exp/memory-aux-loss`
  - current graph-node auxiliary loss experiments
- `exp/multitask-txt-memory`
  - research branch for task/text-conditioned memory routing

## Current Findings

- plain stage2 `trans_loss` is too weak for phase/node disambiguation
- `put_block_back` shows strong retrieval bias toward button phase (`step 5`)
- increasing `num_maskmem` was necessary but not sufficient
- teacher-forced analysis shows retrieval can see early states, but still ranks the wrong anchor highest

## Research Directions

### 1. Coarse Phase Supervision

Replace exact keypoint node prediction with coarse phase labels:
- phase 0: pickup / first carry
- phase 1: center placement
- phase 2: button interaction
- phase 3: regrasp / return

Why:
- easier to generalize across tasks
- less brittle than exact 12-node supervision

### 2. Task/Text-Conditioned Memory Routing

Use task identity or language embedding to modulate memory read/write:
- task embedding from `lang_goal_embs` or task id
- route memory retrieval differently per task
- let each task keep a small persistent memory subset

Candidate design:
- task-conditioned gate on memory candidates
- per-task memory anchor tokens
- text-conditioned phase head

### 3. Persistent Anchor Memory

Store important early-state memory outside the FIFO window:
- initial slot for `put_block_back`
- drawer state anchor for `reopen_drawer`
- block arrangement anchor for `rearrange_block`

### 4. Graph-Guided Candidate Retrieval

Before cross-attention, select memory candidates using:
- current phase logits
- task-conditioned prior
- temporal distance
- similarity to current feature

## Minimal Experiment Order

1. exact node auxiliary loss on `put_block_back`
2. coarse phase auxiliary loss on `put_block_back`
3. coarse phase auxiliary loss on all 3 MemoryBench tasks
4. add task/text-conditioned memory gate
5. compare against persistent-anchor variant

## Success Criteria

- `put_block_back`:
  - final return phase references `0/1/9/10` more often than button phase
  - smoke and full25 improve over `mem11` baseline
- multitask:
  - no severe regression on `reopen_drawer`
  - memory attention shifts to task-relevant anchors per task
