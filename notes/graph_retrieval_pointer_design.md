# Graph Retrieval Pointer Design

## Problem

`put_block_back` behaves like a mixed policy:

1. some steps move to a **new** subgoal
2. some steps should **revisit** one of several prior semantic nodes

The previous stage2 variants were weak for two different reasons:

- pure translation loss under-supervised task flow
- node classification auxiliaries solved an easy shortcut task but did not
  improve retrieval

Teacher-forced analysis also showed that the model can still over-anchor on the
button-transition state even when early evidence is present.

## Retrieved Ideas From Recent Work

### ReMem-VLA

- emphasizes recurrent memory summaries instead of plain bank retrieval
- adds an auxiliary past-observation prediction objective

Takeaway for this codebase:
- auxiliary supervision should shape retrieval, not only classify phases

Source:
- https://arxiv.org/abs/2603.12942

### MemoryVLA

- separates working memory from a longer-term perceptual-cognitive memory bank
- retrieves decision-relevant entries instead of attending to everything equally

Takeaway for this codebase:
- keep FIFO recent memory, but add a separate persistent anchor path and
  retrieval prior

Sources:
- https://arxiv.org/abs/2508.19236
- https://github.com/shihao1895/MemoryVLA

### RMBench / Mem-0

- highlights anchor memory and subtask structure as critical for
  memory-dependent manipulation

Takeaway for this codebase:
- preserve early anchor states
- explicitly model revisit-style decisions instead of treating all actions as
  fresh heatmap generation

Sources:
- https://arxiv.org/abs/2603.01229
- https://github.com/robotwin-Platform/rmbench

## Design Choice

We intentionally implemented a **soft graph retrieval probe**, not a hard graph
pointer policy yet.

Current branch adds:

- replay labels for `new` vs `revisit`
- replay multi-hot expected reference nodes
- current-step retrieval query
- `retrieval_mode_logits`
- `retrieval_ref_logits`
- node-tagged memory entries
- soft multiplicative bias on matched memory entries

## Why Not Hard Masking Yet

A skeptical review of the design suggested hard masking would be risky because:

- the node labels are task-specific and partially teacher-defined
- the model could learn a brittle shortcut rather than robust retrieval
- teacher-forced analysis already showed strong anchor collapse to the wrong
  phase node

So this branch uses:

- diagnostics
- soft retrieval nudging
- explicit logging

before moving to hard pointer routing.

## Next Logical Steps

1. run retrieval probe on `put_block_back`
2. compare rollout vs teacher-forced retrieval stats
3. if the probe helps:
   - add phase-gated candidate pruning
   - add pointer/copy head for revisit steps
4. if it does not help:
   - move to anchor-reference loss and past-observation prediction
