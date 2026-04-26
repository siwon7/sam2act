# MemoryBench Reproduce Next Steps

Date: 2026-04-13

Focus:
- `MemoryBench`
- task: `put_block_back`
- model family: `SAM2Act+`

## Current factual state

What is already verified:
- RLBench works locally and remotely under the verified setup
- MemoryBench environment crashes were fixed
- MemoryBench evaluation can now run locally and remotely
- current clean reruns still do not reproduce the expected score

Representative current results:
- `cleanrepro model_plus_last` full 25ep: `8.0`
- `cleanrepro model_plus_15` 5ep: `20.0`
- `cleanrepro model_plus_19` 5ep: `0.0`
- remote `cleanrepro model_plus_15` 5ep: `0.0`

## Interpretation

The remaining problem is not:
- basic RLBench runtime
- PyRep import
- point renderer import
- CoppeliaSim linking

The remaining problem is more likely one of:
1. public-repo vs paper-internal MemoryBench gap
2. replay/data mismatch
3. stage-two memory learning instability
4. evaluation-seed / simulator nondeterminism interacting with a weak checkpoint

## High-value next steps

### 1. Re-evaluate the exact same checkpoint on fixed seed sets

Goal:
- separate checkpoint quality from eval variance

Action:
- choose 10 fixed episode seeds
- evaluate these exact seeds for:
  - `model_plus_15`
  - `model_plus_19`
  - `model_plus_last`

Why:
- current 5ep and 25ep results vary enough that checkpoint ranking is unclear

### 2. Save video for the same seed list

Goal:
- compare behavior phase-by-phase

Action:
- same fixed seeds
- same task
- same environment
- video on

Why:
- failure mode appears concentrated in later subgoal transitions

### 3. Compare local vs remote on the same seed list

Goal:
- detect whether the environment difference still matters

Action:
- run the same checkpoint and same seeds locally and on `cvlab-dgx`

Why:
- current local `model_plus_15` 5ep was `20.0`
- remote `model_plus_15` 5ep came out `0.0`
- that difference must be explained before trusting either score

### 4. Audit replay construction, not just train/eval code

Goal:
- verify that the replay underlying stage-two really matches the public recipe

Action:
- compare:
  - replay size
  - episode/keyframe counts
  - temporal sampling contract
  - `put_block_back` trajectory lengths

Why:
- `num_maskmem=12` already exposed that the dataset/sampler contract is a real bottleneck

### 5. Keep future method work separate from the baseline path

Rule:
- baseline reproduction path should stay clean
- new ideas should go into separate config names or branches

Why:
- the current investigation already showed how hard it is to reason about regressions when method edits and reproduction fixes are mixed

## Concrete recommended experiment order

1. fixed-seed eval on local
- `model_plus_15`
- `model_plus_19`
- `model_plus_last`

2. fixed-seed eval on remote
- same checkpoints
- same seeds

3. video comparison for the best and worst checkpoint

4. replay integrity summary for `put_block_back`

5. only after that, decide whether to:
- continue reproduction
- or switch to a new method line

## What not to do next

Avoid:
- changing `num_maskmem` again before replay/seed variance is pinned down
- mixing new memory methods into the baseline path
- trusting a single 5-episode result as the final ranking

## Suggested deliverables for the next debugging round

1. one CSV with checkpoint-by-seed success
2. one folder with matched videos
3. one replay integrity note for `put_block_back`
4. one local-vs-remote comparison note

