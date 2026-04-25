# Multitask Text-Memory Design

## Why This Branch Exists

The original MemoryBench stage2 setup uses a FIFO temporal memory window and a
translation-centric loss. In `put_block_back`, this was not enough:

- default `num_maskmem=9` cannot expose the earliest slot evidence to the last
  return steps;
- even with `mem11`, teacher-forced analysis still showed a strong retrieval
  bias toward the button-transition phase;
- rollout failures mixed **phase confusion** and **slot recall failure**.

This branch extends stage2 toward a multitask-friendly memory design without
breaking the existing code path.

## Implemented Ideas

### 1. Shared Coarse Phase Labels

Replay now stores `phase_label` for MemoryBench temporal samples.

Current mapping uses four coarse phases shared across tasks:

1. `pickup_first_carry`
2. `intermediate_place_or_state_change`
3. `task_interaction`
4. `regrasp_return_finalize`

Why:
- exact keypoint-node supervision works for single-task diagnosis but does not
  generalize well across tasks;
- coarse phases provide a shared graph vocabulary for multitask stage2.

## 2. Phase Auxiliary Loss

Stage2 can now optimize:

`total_loss = trans_loss + phase_aux_loss_weight * phase_aux_loss`

where `phase_aux_loss` is CE over replay `phase_label`.

Why:
- current stage2 is under-supervised for task flow;
- phase supervision directly targets the observed failure mode where the policy
  enters the wrong phase, especially around interaction and return transitions.

## 3. Text-Gated Memory

Memory retrieval can be modulated by language-conditioned gates.

Current scaffold supports:
- text-conditioned gate from `lang_emb`
- optional task-conditioned gate hook (`task_cond`) for later work

Gate application points:
- before memory attention (`memory` mode)
- after fusion (`fusion` mode)
- both (`both`)

Why:
- text will not identify the exact slot, but it can bias which memory type is
  useful for the task.

## 4. Persistent Anchor Memory

This branch adds a separate anchor bank on top of the FIFO window.

Current default behavior:
- first `persistent_anchor_max_steps` observations are stored in an anchor bank
- anchor memories are always available during later retrieval
- anchors receive a dedicated positional encoding

Why:
- the FIFO window alone forgets early task-critical evidence or underweights it
- `put_block_back` specifically needs the initial slot evidence to remain
  available at the end of the sequence
- this follows the anchor-memory lesson from recent memory-policy work

## 5. Graph Retrieval Probe and Soft Bias

This branch now adds a conservative graph-retrieval mechanism on top of the
memory bank.

Replay stores:

- `graph_mode_label`: whether the current step should behave like a `new`
  subgoal or a `revisit` step
- `graph_ref_mask`: a multi-hot mask over candidate prior keypoint nodes
- `graph_ref_valid`: whether a retrieval target is defined for the step

For `put_block_back`, the current mapping is explicit and conservative:

- step `9` prefers `{1, 7, 8}`
- step `10` prefers `{0, 1, 9}`
- step `11` prefers `{0, 1, 10}`

Model side:

- a retrieval query is produced from current visual features
- `retrieval_mode_logits` predicts `new` vs `revisit`
- `retrieval_ref_logits` scores prior graph nodes
- stored memory entries carry a `node_tag`
- retrieval scores induce a **soft multiplicative bias** on matching memory
  entries instead of hard pruning

Why soft bias first:

- skeptical review showed that hard graph retrieval can easily harden the wrong
  prior and collapse to a shortcut
- teacher-forced analysis suggested the real issue is not total lack of early
  evidence, but biased retrieval anchors
- soft bias lets us diagnose and nudge retrieval without forcing brittle exact
  routing

Current limitation:

- only `put_block_back` has explicit graph targets today
- the retrieval module is meant as a probe and gentle prior, not yet a fully
  hard pointer policy

## Current Config Knobs

### peract

- `peract.phase_aux_loss_weight`
- `peract.phase_aux_num_classes`
- `peract.phase_aux_label_key`
- `peract.graph_retrieval_mode_loss_weight`
- `peract.graph_retrieval_ref_loss_weight`
- `peract.graph_retrieval_mode_label_key`
- `peract.graph_retrieval_ref_valid_key`
- `peract.graph_retrieval_ref_label_key`

### mvt

- `memory_gate_enabled`
- `memory_gate_mode`
- `memory_gate_use_text`
- `memory_gate_use_task`
- `memory_gate_hidden_dim`
- `memory_gate_task_cond_dim`
- `persistent_anchor_enabled`
- `persistent_anchor_max_steps`
- `persistent_anchor_prepend`
- `graph_retrieval_enabled`
- `graph_retrieval_num_classes`
- `graph_retrieval_hidden_dim`
- `graph_retrieval_bias_scale`

## Recommended First Experiments

### A. Multitask baseline

- no phase aux
- no gate
- no persistent anchor

Purpose:
- verify stage2 and checkpoint handoff still work

### B. Add phase aux only

- `phase_aux_loss_weight > 0`
- `phase_aux_num_classes = 4`

Purpose:
- test whether shared flow supervision reduces phase confusion

### C. Add persistent anchor only

- `persistent_anchor_enabled True`
- `persistent_anchor_max_steps 2`

Purpose:
- test whether preserving early task evidence helps final return phases

### D. Add phase aux + persistent anchor

Purpose:
- combine flow supervision and persistent early-state access

### E. Add text gate on top

Purpose:
- test whether language-conditioned routing improves which memory bank entries
  are emphasized

### F. Add graph retrieval probe

- `graph_retrieval_enabled True`
- `graph_retrieval_num_classes 12`
- non-zero `graph_retrieval_mode_loss_weight`
- non-zero `graph_retrieval_ref_loss_weight`
- modest `graph_retrieval_bias_scale` such as `0.25 ~ 0.5`

Purpose:
- test whether explicit revisit/new supervision and soft retrieval bias reduce
  the button-anchor collapse observed in `put_block_back`

## Not Yet Implemented

These are the next ideas, not part of the current commit:

- explicit `task_cond` routing from task id or learned task embedding
- anchor-reference loss that directly rewards attending to expected anchor
  states during return phases
- past-observation prediction / reconstruction auxiliary objective
- task-specific persistent anchor policies beyond "first k steps"
- hard pointer-style retrieval with phase-gated candidate pruning

## Branching Strategy

Keep this branch as the reusable multitask scaffold.

Future branches should fork from here:

- `exp/anchor-ref-loss`
- `exp/past-observation-pred`
- `exp/task-cond-gate`
- `exp/task-specific-anchor-policy`

That keeps the scaffold stable while making each research idea traceable.
