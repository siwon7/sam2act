# Comparison Benchmarks For SAM2Act-MT

This note focuses on benchmarks and datasets that are either on the same RLBench / CoppeliaSim / PyRep stack as SAM2Act, or are close enough to be useful for a broader comparison section.

## What Is Most Relevant

If the goal is to make `SAM2Act-MT` look less like a 3-task MemoryBench trick, the strongest same-platform additions are:

1. `Colosseum`
2. `MREST-RLBench`
3. `PerAct / RLBench task-set evaluation`

`RMBench` and `RoboMME` are still valuable, but they are better treated as broader memory-benchmark comparisons rather than same-platform comparisons.

## Same-Platform Candidates

| Candidate | Platform | Tasks | Code | Data | Memory relevance | Multitask relevance | Compatibility with SAM2Act | Setup difficulty | Priority |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RLBench | PyRep + CoppeliaSim | 100 tasks total; task-set splits 10/25/50/95 or MT15/30/55/100 | Official repo exists | Demo datasets used by many papers | Medium | High | Exact platform match | Medium | 1 |
| PerAct RLBench benchmark | RLBench fork + YARR | 18 tasks in the paper; 100/25/25 demos | Official repo exists | Pre-generated RLBench demos and pretrained ckpts | Medium | High | Exact platform match | Medium | 2 |
| Colosseum | PyRep + RLBench + CoppeliaSim | 20 RLBench tasks + 14 variation factors | Official repo exists | HF challenge dataset + generation scripts | High | High | Exact simulator stack, stronger generalization stress test | Medium-High | 1 |
| MREST-RLBench | RLBench + PyRep + CoppeliaSim | 4 precise tasks | Official repo exists | No public standalone dataset found; task updates rely on external env-deps | Medium | Low-Medium | Exact stack, but task count is small and dependency is missing | High | 2 |
| RVT-Colosseum | Same stack as Colosseum | 20 tasks | Official repo exists | Uses Colosseum challenge dataset | Medium | High | Very close, useful as a baseline implementation | Medium | 3 |

## Broader But Useful Candidates

| Candidate | Platform | Tasks | Why It Matters |
| --- | --- | --- | --- |
| RMBench | Memory-dependent robotic manipulation benchmark | Multiple tasks, benchmark-style | Stronger memory benchmark framing than MemoryBench and good for architecture ablations |
| RoboMME | Robotic memory benchmark for generalist/VLA policies | 16 tasks with memory taxonomy | Good external benchmark if we want to argue memory generalization beyond RLBench |
| MIKASA-Robo | Memory-intensive tabletop manipulation | 32 tasks | Good if we want a broader manipulation-memory benchmark, but less aligned with RLBench stack |

## Repository Status In This Workspace

Cloned and available locally:

- [`RLBench-stepjam`](/home/cv25/siwon/RLBench-stepjam)
- [`peract`](/home/cv25/siwon/peract)
- [`PyRep`](/home/cv25/siwon/PyRep)
- [`RLBench-peract`](/home/cv25/siwon/RLBench-peract)
- [`YARR-peract`](/home/cv25/siwon/YARR-peract)
- [`robot-colosseum`](/home/cv25/siwon/robot-colosseum)
- [`rvt_colosseum`](/home/cv25/siwon/rvt_colosseum)
- [`mrest-rlbench`](/home/cv25/siwon/mrest-rlbench)

## Data Status

### RLBench / PerAct

- Pre-generated RLBench demos for train / val / test exist and are widely used by PerAct.
- The PerAct README says the full RLBench demo set is about 116 GB.
- The repo also provides pretrained checkpoints through a release zip.

### Colosseum

- The Colosseum dataset is hosted on Hugging Face as task/variation tar files.
- The repo provides scripts for data collection and a dataset download workflow.
- This is the best same-platform benchmark candidate for a paper-level comparison section.

### MREST-RLBench

- The repo documents 4 precise RLBench tasks.
- The task scene files are referenced through `mrest-env-deps`, but the referenced repo was not publicly available when checked.
- This is a real blocker for full reproduction, so it should be treated as a partial setup candidate rather than a fully reproducible benchmark today.

## Recommendation

For the paper, I would use:

1. `MemoryBench` as the core problem setting.
2. `Colosseum` as the strongest same-platform extension.
3. `PerAct / RLBench` as the canonical RLBench baseline and data reference.
4. `MREST-RLBench` only if we are willing to describe the dependency gap clearly.
5. `RMBench` as the broader memory-focused benchmark.

## Why This Mix Works

- `Colosseum` adds same-platform scale and variation factors.
- `PerAct / RLBench` gives a stable, standard benchmark baseline that reviewers already recognize.
- `MREST-RLBench` adds a more precise control setting, but its public reproducibility is weaker.
- `RMBench` lets us say the method is not only good on 3 MemoryBench tasks, but also on a more explicit memory benchmark.

