# Comparison Setup Notes

This note records the repos, env recommendations, data sources, and blockers for the comparison stack around RLBench / CoppeliaSim.

## Cloned Repos

- [`RLBench-stepjam`](/home/cv25/siwon/RLBench-stepjam)
- [`PyRep`](/home/cv25/siwon/PyRep)
- [`peract`](/home/cv25/siwon/peract)
- [`RLBench-peract`](/home/cv25/siwon/RLBench-peract)
- [`YARR-peract`](/home/cv25/siwon/YARR-peract)
- [`robot-colosseum`](/home/cv25/siwon/robot-colosseum)
- [`rvt_colosseum`](/home/cv25/siwon/rvt_colosseum)
- [`mrest-rlbench`](/home/cv25/siwon/mrest-rlbench)

## Recommended Environment Layout

Use Python 3.8 for the comparison stack unless a repo explicitly says otherwise.

Suggested envs:

- `rlbench38`
- `peract38`
- `colosseum38`
- `mrest38`

The common simulator dependencies are:

- CoppeliaSim 4.1
- PyRep
- RLBench

For our current `sam2act5090` env, keep it separate; do not try to merge the stacks.

## Installation Order

1. Install CoppeliaSim 4.1 and export:
   - `COPPELIASIM_ROOT`
   - `LD_LIBRARY_PATH`
   - `QT_QPA_PLATFORM_PLUGIN_PATH`
2. Install `PyRep`.
3. Install the RLBench fork required by the target repo.
4. Install the target repo in editable mode.
5. Install data or checkpoints.

## Repo-Specific Notes

### RLBench-stepjam

- Official benchmark repo.
- Useful for canonical task sets and baseline data generation.
- Install guide is the standard PyRep + RLBench workflow.

### PerAct

- Uses a RLBench fork plus YARR fork.
- The README recommends Python 3.8 and gives a pre-generated RLBench dataset path.
- The repo includes a quickstart checkpoint download script.

### Colosseum

- Built on PyRep + RLBench.
- Adds 20 RLBench tasks and 14 variation factors.
- Has a Hugging Face challenge dataset with tar files per task / variation.
- The repo includes `collect_dataset.sh` and `collect_dataset_cluster.sh`.

### RVT-Colosseum

- RVT baseline for Colosseum.
- Useful because the code path is close to our RVT-style stack.
- The README points to the same RLBench / Colosseum dataset assumptions.

### MREST-RLBench

- Uses 4 precise tasks on the RLBench stack.
- The repo points to `mrest-env-deps` for new `task_ttms`.
- That referenced repo was not publicly available when checked, so full reproduction is blocked.

## Data Sources

### RLBench / PerAct demos

- PerAct README says the full pre-generated demo set for train / val / test is about 116 GB.
- It also says the demos are split by tasks and not bundled as a single zip.
- Use the Google Drive release or a mirror if available.

### Colosseum dataset

- Public Hugging Face dataset:
  - `colosseum/colosseum-challenge`
- Data is organized as tar files for the 20 tasks and their variations.

### MREST

- No public standalone dataset was found during this pass.
- The dependency on `mrest-env-deps` appears to be the main blocker.

## What Was Actually Cloned

All of the following are present locally:

- `/home/cv25/siwon/PyRep`
- `/home/cv25/siwon/RLBench-stepjam`
- `/home/cv25/siwon/RLBench-peract`
- `/home/cv25/siwon/YARR-peract`
- `/home/cv25/siwon/peract`
- `/home/cv25/siwon/robot-colosseum`
- `/home/cv25/siwon/rvt_colosseum`
- `/home/cv25/siwon/mrest-rlbench`

## Current Blockers

1. `mrest-env-deps` was not publicly available at the checked URL.
2. The active `sam2act5090` training job is occupying all GPUs, so I avoided smoke-evaluating these comparison repos in this pass.
3. The comparison stack is older and sensitive to exact simulator / Python / torch versions, so mixing it with `sam2act5090` is not recommended.

## Next Practical Step

The best next action is to run a lightweight smoke check for:

- `RLBench-stepjam`
- `PerAct`
- `Colosseum`

and only then decide whether `MREST-RLBench` is worth manual patching with a local replacement for `mrest-env-deps`.

