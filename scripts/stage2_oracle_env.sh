#!/usr/bin/env bash
# Source this file before running oracle stage-two jobs from this worktree.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "source this file instead of executing it directly" >&2
  exit 2
fi

_ORACLE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SAM2ACT_REPO_ROOT="${SAM2ACT_REPO_ROOT:-$(cd "${_ORACLE_SCRIPT_DIR}/.." && pwd)}"
export SAM2ACT_ROOT="${SAM2ACT_ROOT:-${SAM2ACT_REPO_ROOT}/sam2act}"
export SAM2ACT_ENV_PYTHON="${SAM2ACT_ENV_PYTHON:-/home/cv11/anaconda3/envs/sam2act5090/bin/python3.10}"

if [[ -x "${SAM2ACT_ENV_PYTHON}" ]]; then
  export PATH="$(dirname "${SAM2ACT_ENV_PYTHON}"):${PATH}"
fi

export PYTHONNOUSERSITE=1
export TRANSFORMERS_NO_TF=1
export TRANSFORMERS_NO_FLAX=1
export USE_TF=0
export TF_CPP_MIN_LOG_LEVEL=3
export BITSANDBYTES_NOWELCOME=1
export TOKENIZERS_PARALLELISM=false
export MPLBACKEND=Agg

export COPPELIASIM_ROOT="${COPPELIASIM_ROOT:-/home/cv11/project/siwon/coppeliasim/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04}"
export QT_QPA_PLATFORM_PLUGIN_PATH="${COPPELIASIM_ROOT}"
export LD_LIBRARY_PATH="${COPPELIASIM_ROOT}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

_DIRTY_SAM2ACT_ROOT="/home/cv11/project/siwon/sam2act_dirty/sam2act"
_SYNC_SAM2ACT_ROOT="/home/cv11/project/siwon/.sync_unpack_20260426/sam2act_dirty/sam2act/sam2act"
_TGM_VLA_ROOT="/home/cv11/project/siwon/TGM-VLA/tgm_vla"
_LOCAL_PYREP_ROOT="${SAM2ACT_ROOT}/libs/PyRep"
_DIRTY_PYREP_ROOT="${_DIRTY_SAM2ACT_ROOT}/libs/PyRep"
_SYNC_PYREP_ROOT="${_SYNC_SAM2ACT_ROOT}/libs/PyRep"
_TGM_PYREP_ROOT="${_TGM_VLA_ROOT}/libs/PyRep"

_has_pyrep_ext() {
  local root="$1"
  [[ -d "${root}/pyrep/backend" ]] || return 1
  find -L "${root}/pyrep/backend" -maxdepth 1 -type f -name '_sim_cffi*.so' | grep -q .
}

_PYREP_ROOT="${_LOCAL_PYREP_ROOT}"
for _candidate in "${_LOCAL_PYREP_ROOT}" "${_DIRTY_PYREP_ROOT}" "${_SYNC_PYREP_ROOT}" "${_TGM_PYREP_ROOT}"; do
  if _has_pyrep_ext "${_candidate}"; then
    _PYREP_ROOT="${_candidate}"
    break
  fi
done

_prepend_pythonpath() {
  local entry="$1"
  [[ -d "${entry}" ]] || return 0
  case ":${PYTHONPATH:-}:" in
    *":${entry}:"*) ;;
    *) export PYTHONPATH="${entry}${PYTHONPATH:+:${PYTHONPATH}}" ;;
  esac
}

_prepend_pythonpath "${SAM2ACT_REPO_ROOT}"
_prepend_pythonpath "${SAM2ACT_ROOT}"
_prepend_pythonpath "${_PYREP_ROOT}"
_prepend_pythonpath "${SAM2ACT_ROOT}/libs/RLBench"
_prepend_pythonpath "${SAM2ACT_ROOT}/libs/YARR"
_prepend_pythonpath "${SAM2ACT_ROOT}/libs/peract"

unset _ORACLE_SCRIPT_DIR _DIRTY_SAM2ACT_ROOT _SYNC_SAM2ACT_ROOT _TGM_VLA_ROOT
unset _LOCAL_PYREP_ROOT _DIRTY_PYREP_ROOT _SYNC_PYREP_ROOT _TGM_PYREP_ROOT _PYREP_ROOT _candidate
