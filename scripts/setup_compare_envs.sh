#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/cv25/siwon}"

cat <<EOF
Comparison env plan for RLBench/CoppeliaSim stack

Recommended env names:
  - rlbench38
  - peract38
  - colosseum38
  - mrest38

Common base:
  - Python 3.8
  - CoppeliaSim 4.1
  - PyRep
  - RLBench (or the repo-specific fork)

Cloned repos already present:
  - $ROOT/PyRep
  - $ROOT/RLBench-stepjam
  - $ROOT/RLBench-peract
  - $ROOT/YARR-peract
  - $ROOT/peract
  - $ROOT/robot-colosseum
  - $ROOT/rvt_colosseum
  - $ROOT/mrest-rlbench

Suggested install order:
  1. conda create -n <env> python=3.8 pip
  2. install CoppeliaSim 4.1 and export COPPELIASIM_ROOT / LD_LIBRARY_PATH / QT_QPA_PLATFORM_PLUGIN_PATH
  3. pip install -r requirements.txt in PyRep
  4. pip install -e repo under test

Repo-specific notes:
  - PerAct uses RLBench-peract + YARR-peract
  - Colosseum uses RLBench + PyRep and has HF dataset tar files
  - MREST-RLBench depends on external mrest-env-deps task_ttms that was not public at check time

This script is documentation-first. If you want it to actively create conda skeletons, add a --apply flag and wire conda commands in here.
EOF

