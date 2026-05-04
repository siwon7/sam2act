#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/stage2_oracle_env.sh
source "${SCRIPT_DIR}/stage2_oracle_env.sh"

TASK=""
GPU="${GPU:-4}"
NPROC="${NPROC:-1}"
MASTER_PORT="${MASTER_PORT:-$((29700 + RANDOM % 200))}"
RUNS_ROOT="${RUNS_ROOT:-/hdd4/siwon/checkpoints/sam2act/runs}"
DATA_ROOT="${DATA_ROOT:-/hdd4/siwon/datasets/sam2act}"
SAM2_CKPT="${SAM2_CKPT:-/hdd4/siwon/assets/sam2act_backbone/checkpoints/sam2.1_hiera_base_plus.pt}"
EXP_NAME=""
EXP_SUFFIX=""
STAGE1_RUN_DIR=""
STAGE1_CKPT=""
BS="${BS:-10}"
EPOCHS="${EPOCHS:-20}"
TRAIN_ITER="${TRAIN_ITER:-160000}"
DEMO="${DEMO:-100}"
NUM_WORKERS="${NUM_WORKERS:-5}"
LR="${LR:-1.25e-06}"
NUM_MASKMEM="${NUM_MASKMEM:-9}"
USE_MEMORY="${USE_MEMORY:-False}"
USE_MULTIPEAK="${USE_MULTIPEAK:-False}"
USE_GRAPH_PEAK_SELECT="${USE_GRAPH_PEAK_SELECT:-False}"
GRAPH_PEAK_TOPK="${GRAPH_PEAK_TOPK:-3}"
GRAPH_PEAK_SELECT_LOSS_WEIGHT="${GRAPH_PEAK_SELECT_LOSS_WEIGHT:-1.0}"
GRAPH_PEAK_INSERT_GT_TRAIN="${GRAPH_PEAK_INSERT_GT_TRAIN:-True}"
GRAPH_PEAK_POSITIVE_RADIUS="${GRAPH_PEAK_POSITIVE_RADIUS:-0.05}"
GRAPH_PEAK_NMS_DIST="${GRAPH_PEAK_NMS_DIST:-0.05}"
STAGE2_CANDIDATE_MODE="${STAGE2_CANDIDATE_MODE:-top1}"
STAGE2_CANDIDATE_TRAIN_CROP="${STAGE2_CANDIDATE_TRAIN_CROP:-gt}"
STAGE2_KCROP_TRAIN_PICK="${STAGE2_KCROP_TRAIN_PICK:-target}"
STAGE2_CANDIDATE_INSERT_GT_TRAIN="${STAGE2_CANDIDATE_INSERT_GT_TRAIN:-True}"
STAGE2_MEMORY_WRITE_MODE="${STAGE2_MEMORY_WRITE_MODE:-stage1}"
STAGE2_MEMORY_WRITE_TOPK="${STAGE2_MEMORY_WRITE_TOPK:-3}"
STAGE2_MEMORY_WRITE_TEMPERATURE="${STAGE2_MEMORY_WRITE_TEMPERATURE:-0.25}"
STAGE2_MEMORY_WRITE_SIGMA="${STAGE2_MEMORY_WRITE_SIGMA:-1.5}"
SAMPLE_MODE="${SAMPLE_MODE:-demo_uniform_temporal}"
FRESH_START=1
PREPARE_ONLY=0
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_stage2_oracle_train.sh --task TASK [options]

Tasks:
  put_block_back | rearrange_block | reopen_drawer

Options:
  --gpu IDS              CUDA_VISIBLE_DEVICES value. Default: 4
  --nproc N             torchrun processes. Default: 1
  --master-port PORT     torchrun rendezvous port. Default: random 29700-29899
  --runs-root DIR        Checkpoint root. Default: /hdd4/siwon/checkpoints/sam2act/runs
  --data-root DIR        Replay/data root. Default: /hdd4/siwon/datasets/sam2act
  --sam2-ckpt FILE       SAM2 backbone checkpoint. Default: /hdd4/siwon/assets/sam2act_backbone/checkpoints/sam2.1_hiera_base_plus.pt
  --stage1-run-dir DIR   Source dirty stage1 run dir. Default is task-specific memorybench run.
  --stage1-ckpt FILE     Source stage1 checkpoint. Default: STAGE1_RUN_DIR/model_last.pth
  --exp-name NAME        Stage2 run exp_name. Default: stage2_oracle_TASK_dirty_stage1
  --exp-suffix SUFFIX    Appended to exp_name.
  --bs N                 Batch size per process. Default: 10
  --epochs N             Epochs. Default: 20
  --train-iter N         train_iter before train_plus division. Default: 160000
  --lr FLOAT             Per-sample LR before train_plus scaling. Default: 1.25e-06
  --demo N               Number of demos. Default: 100
  --num-workers N        Dataloader workers. Default: 5
  --sample-mode MODE     Replay sampling mode. Default: demo_uniform_temporal
  --use-memory           Train memory stage2 path instead of non-memory oracle crop.
  --use-multipeak        Keep stage1 multipeak labels enabled. Default is off.
  --use-graph-peak-select
                         Enable V9 top-K selector aux loss and selector crop at eval.
  --graph-peak-topk N     Top-K Stage1 3D candidates. Default: 3
  --graph-peak-select-loss-weight FLOAT
                         Selector CE weight. Default: 1.0
  --no-graph-peak-insert-gt
                         Do not insert GT into top-K during training.
  --v10a                  Enable V10a selector crop mode.
  --v10b                  Enable V10b K-crop Stage2 branch mode.
  --stage2-candidate-mode MODE
                         top1 | selector | kcrop. Default: top1
  --stage2-candidate-train-crop MODE
                         gt | selector | nearest_gt. Default: gt
  --stage2-kcrop-train-pick MODE
                         target | selector. Default: target
  --no-stage2-candidate-insert-gt
                         Do not insert GT into Stage2 candidate packet during training.
  --v10c-memory MODE      Memory write mode: stage1 | topk_soft | selected.
  --stage2-memory-write-topk N
                         V10c top-K per-view memory write count. Default: 3
  --stage2-memory-write-temperature FLOAT
                         V10c soft top-K temperature. Default: 0.25
  --stage2-memory-write-sigma FLOAT
                         V10c gaussian sigma in image pixels. Use 0 for sparse.
  --resume-plus          Resume model_plus_last.pth if it exists.
  --prepare-only         Create links/config handoff, then exit.
  --dry-run              Print command after preparation, then exit.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="${2:?missing --task value}"; shift 2 ;;
    --task=*) TASK="${1#*=}"; shift ;;
    --gpu) GPU="${2:?missing --gpu value}"; shift 2 ;;
    --gpu=*) GPU="${1#*=}"; shift ;;
    --nproc) NPROC="${2:?missing --nproc value}"; shift 2 ;;
    --nproc=*) NPROC="${1#*=}"; shift ;;
    --master-port) MASTER_PORT="${2:?missing --master-port value}"; shift 2 ;;
    --master-port=*) MASTER_PORT="${1#*=}"; shift ;;
    --runs-root) RUNS_ROOT="${2:?missing --runs-root value}"; shift 2 ;;
    --runs-root=*) RUNS_ROOT="${1#*=}"; shift ;;
    --data-root) DATA_ROOT="${2:?missing --data-root value}"; shift 2 ;;
    --data-root=*) DATA_ROOT="${1#*=}"; shift ;;
    --sam2-ckpt) SAM2_CKPT="${2:?missing --sam2-ckpt value}"; shift 2 ;;
    --sam2-ckpt=*) SAM2_CKPT="${1#*=}"; shift ;;
    --stage1-run-dir) STAGE1_RUN_DIR="${2:?missing --stage1-run-dir value}"; shift 2 ;;
    --stage1-run-dir=*) STAGE1_RUN_DIR="${1#*=}"; shift ;;
    --stage1-ckpt) STAGE1_CKPT="${2:?missing --stage1-ckpt value}"; shift 2 ;;
    --stage1-ckpt=*) STAGE1_CKPT="${1#*=}"; shift ;;
    --exp-name) EXP_NAME="${2:?missing --exp-name value}"; shift 2 ;;
    --exp-name=*) EXP_NAME="${1#*=}"; shift ;;
    --exp-suffix) EXP_SUFFIX="${2:?missing --exp-suffix value}"; shift 2 ;;
    --exp-suffix=*) EXP_SUFFIX="${1#*=}"; shift ;;
    --bs) BS="${2:?missing --bs value}"; shift 2 ;;
    --bs=*) BS="${1#*=}"; shift ;;
    --epochs) EPOCHS="${2:?missing --epochs value}"; shift 2 ;;
    --epochs=*) EPOCHS="${1#*=}"; shift ;;
    --train-iter) TRAIN_ITER="${2:?missing --train-iter value}"; shift 2 ;;
    --train-iter=*) TRAIN_ITER="${1#*=}"; shift ;;
    --lr) LR="${2:?missing --lr value}"; shift 2 ;;
    --lr=*) LR="${1#*=}"; shift ;;
    --demo) DEMO="${2:?missing --demo value}"; shift 2 ;;
    --demo=*) DEMO="${1#*=}"; shift ;;
    --num-workers) NUM_WORKERS="${2:?missing --num-workers value}"; shift 2 ;;
    --num-workers=*) NUM_WORKERS="${1#*=}"; shift ;;
    --sample-mode) SAMPLE_MODE="${2:?missing --sample-mode value}"; shift 2 ;;
    --sample-mode=*) SAMPLE_MODE="${1#*=}"; shift ;;
    --num-maskmem) NUM_MASKMEM="${2:?missing --num-maskmem value}"; shift 2 ;;
    --num-maskmem=*) NUM_MASKMEM="${1#*=}"; shift ;;
    --use-memory) USE_MEMORY=True; shift ;;
    --use-multipeak) USE_MULTIPEAK=True; shift ;;
    --use-graph-peak-select) USE_GRAPH_PEAK_SELECT=True; shift ;;
    --graph-peak-topk) GRAPH_PEAK_TOPK="${2:?missing --graph-peak-topk value}"; shift 2 ;;
    --graph-peak-topk=*) GRAPH_PEAK_TOPK="${1#*=}"; shift ;;
    --graph-peak-select-loss-weight) GRAPH_PEAK_SELECT_LOSS_WEIGHT="${2:?missing --graph-peak-select-loss-weight value}"; shift 2 ;;
    --graph-peak-select-loss-weight=*) GRAPH_PEAK_SELECT_LOSS_WEIGHT="${1#*=}"; shift ;;
    --no-graph-peak-insert-gt) GRAPH_PEAK_INSERT_GT_TRAIN=False; STAGE2_CANDIDATE_INSERT_GT_TRAIN=False; shift ;;
    --v10a) STAGE2_CANDIDATE_MODE=selector; USE_GRAPH_PEAK_SELECT=True; shift ;;
    --v10b) STAGE2_CANDIDATE_MODE=kcrop; USE_GRAPH_PEAK_SELECT=True; shift ;;
    --stage2-candidate-mode) STAGE2_CANDIDATE_MODE="${2:?missing --stage2-candidate-mode value}"; shift 2 ;;
    --stage2-candidate-mode=*) STAGE2_CANDIDATE_MODE="${1#*=}"; shift ;;
    --stage2-candidate-train-crop) STAGE2_CANDIDATE_TRAIN_CROP="${2:?missing --stage2-candidate-train-crop value}"; shift 2 ;;
    --stage2-candidate-train-crop=*) STAGE2_CANDIDATE_TRAIN_CROP="${1#*=}"; shift ;;
    --stage2-kcrop-train-pick) STAGE2_KCROP_TRAIN_PICK="${2:?missing --stage2-kcrop-train-pick value}"; shift 2 ;;
    --stage2-kcrop-train-pick=*) STAGE2_KCROP_TRAIN_PICK="${1#*=}"; shift ;;
    --no-stage2-candidate-insert-gt) STAGE2_CANDIDATE_INSERT_GT_TRAIN=False; shift ;;
    --v10c-memory) STAGE2_MEMORY_WRITE_MODE="${2:?missing --v10c-memory value}"; shift 2 ;;
    --v10c-memory=*) STAGE2_MEMORY_WRITE_MODE="${1#*=}"; shift ;;
    --stage2-memory-write-topk) STAGE2_MEMORY_WRITE_TOPK="${2:?missing --stage2-memory-write-topk value}"; shift 2 ;;
    --stage2-memory-write-topk=*) STAGE2_MEMORY_WRITE_TOPK="${1#*=}"; shift ;;
    --stage2-memory-write-temperature) STAGE2_MEMORY_WRITE_TEMPERATURE="${2:?missing --stage2-memory-write-temperature value}"; shift 2 ;;
    --stage2-memory-write-temperature=*) STAGE2_MEMORY_WRITE_TEMPERATURE="${1#*=}"; shift ;;
    --stage2-memory-write-sigma) STAGE2_MEMORY_WRITE_SIGMA="${2:?missing --stage2-memory-write-sigma value}"; shift 2 ;;
    --stage2-memory-write-sigma=*) STAGE2_MEMORY_WRITE_SIGMA="${1#*=}"; shift ;;
    --resume-plus) FRESH_START=0; shift ;;
    --prepare-only) PREPARE_ONLY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    put_block_back|rearrange_block|reopen_drawer)
      if [[ -n "${TASK}" ]]; then
        echo "task already set to ${TASK}" >&2
        exit 2
      fi
      TASK="$1"
      shift
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${TASK}" in
  put_block_back|rearrange_block|reopen_drawer) ;;
  "")
    echo "--task is required" >&2
    usage >&2
    exit 2
    ;;
  *)
    echo "unsupported task: ${TASK}" >&2
    exit 2
    ;;
esac

if [[ "${SAMPLE_MODE}" == "demo_uniform_temporal" ]]; then
  NUM_OBS=$((NUM_MASKMEM + 1))
  if (( BS % NUM_OBS != 0 )); then
    echo "bs must be divisible by num_maskmem + 1 (${NUM_OBS}) for demo_uniform_temporal; got bs=${BS}" >&2
    exit 2
  fi
fi
if (( TRAIN_ITER < BS * NPROC )); then
  echo "train-iter must be >= bs * nproc so train_plus runs at least one iteration; got train_iter=${TRAIN_ITER}, bs=${BS}, nproc=${NPROC}" >&2
  exit 2
fi
if [[ "${STAGE2_CANDIDATE_MODE}" != "top1" ]]; then
  USE_GRAPH_PEAK_SELECT=True
fi

if [[ -z "${EXP_NAME}" ]]; then
  EXP_NAME="stage2_oracle_${TASK}_dirty_stage1"
fi
if [[ -n "${EXP_SUFFIX}" ]]; then
  EXP_NAME="${EXP_NAME}_${EXP_SUFFIX}"
fi

if [[ -z "${STAGE1_RUN_DIR}" ]]; then
  STAGE1_RUN_DIR="${RUNS_ROOT}/sam2act_memorybench_${TASK}"
fi
if [[ -z "${STAGE1_CKPT}" ]]; then
  STAGE1_CKPT="${STAGE1_RUN_DIR}/model_last.pth"
fi

RUN_DIR="${RUNS_ROOT}/sam2act_${EXP_NAME}"
EXP_CFG_PATH="${SAM2ACT_ROOT}/configs/sam2act.yaml"
MVT_CFG_PATH="${SAM2ACT_ROOT}/mvt/configs/sam2act.yaml"

ensure_link() {
  local target="$1"
  local link="$2"
  if [[ ! -e "${target}" ]]; then
    echo "missing target for ${link}: ${target}" >&2
    exit 1
  fi
  if [[ -L "${link}" ]]; then
    ln -sfn "${target}" "${link}"
  elif [[ -e "${link}" ]]; then
    echo "keeping existing path: ${link}"
  else
    ln -s "${target}" "${link}"
  fi
}

mkdir -p "${RUN_DIR}"
ensure_link "${DATA_ROOT}/data" "${SAM2ACT_ROOT}/data"
ensure_link "${DATA_ROOT}/data_memory" "${SAM2ACT_ROOT}/data_memory"
ensure_link "${DATA_ROOT}/replay_temporal_memory" "${SAM2ACT_ROOT}/replay_temporal_memory"
ensure_link "${RUNS_ROOT}" "${SAM2ACT_ROOT}/runs"
mkdir -p "${SAM2ACT_ROOT}/mvt/sam2_train/checkpoints"
ensure_link "${SAM2_CKPT}" "${SAM2ACT_ROOT}/mvt/sam2_train/checkpoints/sam2.1_hiera_base_plus.pt"

if [[ ! -f "${STAGE1_CKPT}" ]]; then
  echo "missing stage1 checkpoint: ${STAGE1_CKPT}" >&2
  exit 1
fi
ln -sfn "${STAGE1_CKPT}" "${RUN_DIR}/model_last.pth"

for cfg in exp_cfg.yaml mvt_cfg.yaml args.yaml; do
  if [[ -f "${STAGE1_RUN_DIR}/${cfg}" ]]; then
    cp -p "${STAGE1_RUN_DIR}/${cfg}" "${RUN_DIR}/stage1_${cfg}"
  fi
done
{
  echo "task=${TASK}"
  echo "stage1_run_dir=${STAGE1_RUN_DIR}"
  echo "stage1_ckpt=${STAGE1_CKPT}"
  echo "stage2_run_dir=${RUN_DIR}"
  echo "prepared_at=$(date '+%Y-%m-%d %H:%M:%S %Z')"
} > "${RUN_DIR}/stage1_handoff.txt"

EXP_CFG_OPTS="tasks ${TASK} exp_name ${EXP_NAME} bs ${BS} epochs ${EPOCHS} train_iter ${TRAIN_ITER} demo ${DEMO} num_workers ${NUM_WORKERS} sample_distribution_mode ${SAMPLE_MODE} wandb False peract.lr ${LR}"
MVT_CFG_OPTS="stage_two True use_memory ${USE_MEMORY} use_multipeak ${USE_MULTIPEAK} num_maskmem ${NUM_MASKMEM} use_graph_peak_select ${USE_GRAPH_PEAK_SELECT} graph_peak_topk ${GRAPH_PEAK_TOPK} graph_peak_select_loss_weight ${GRAPH_PEAK_SELECT_LOSS_WEIGHT} graph_peak_insert_gt_train ${GRAPH_PEAK_INSERT_GT_TRAIN} graph_peak_positive_radius ${GRAPH_PEAK_POSITIVE_RADIUS} graph_peak_nms_dist ${GRAPH_PEAK_NMS_DIST} stage2_candidate_mode ${STAGE2_CANDIDATE_MODE} stage2_candidate_train_crop ${STAGE2_CANDIDATE_TRAIN_CROP} stage2_kcrop_train_pick ${STAGE2_KCROP_TRAIN_PICK} stage2_candidate_insert_gt_train ${STAGE2_CANDIDATE_INSERT_GT_TRAIN} stage2_memory_write_mode ${STAGE2_MEMORY_WRITE_MODE} stage2_memory_write_topk ${STAGE2_MEMORY_WRITE_TOPK} stage2_memory_write_temperature ${STAGE2_MEMORY_WRITE_TEMPERATURE} stage2_memory_write_sigma ${STAGE2_MEMORY_WRITE_SIGMA}"

TRAIN_CMD=(
  torchrun
  --nproc_per_node="${NPROC}"
  --nnodes=1
  --master_port="${MASTER_PORT}"
  train_plus.py
  --device "${GPU}"
  --log-dir "${RUNS_ROOT}"
  --exp_cfg_path "${EXP_CFG_PATH}"
  --mvt_cfg_path "${MVT_CFG_PATH}"
  --exp_cfg_opts "${EXP_CFG_OPTS}"
  --mvt_cfg_opts "${MVT_CFG_OPTS}"
)
if [[ "${FRESH_START}" == "1" ]]; then
  TRAIN_CMD+=(--fresh-start)
fi

echo "[stage2-oracle] repo=${SAM2ACT_REPO_ROOT}"
echo "[stage2-oracle] branch=$(git -C "${SAM2ACT_REPO_ROOT}" branch --show-current)"
echo "[stage2-oracle] task=${TASK}"
echo "[stage2-oracle] stage1=${STAGE1_CKPT}"
echo "[stage2-oracle] run_dir=${RUN_DIR}"
echo "[stage2-oracle] master_port=${MASTER_PORT}"
echo "[stage2-oracle] use_memory=${USE_MEMORY} use_multipeak=${USE_MULTIPEAK}"
echo "[stage2-oracle] use_graph_peak_select=${USE_GRAPH_PEAK_SELECT} topk=${GRAPH_PEAK_TOPK} insert_gt_train=${GRAPH_PEAK_INSERT_GT_TRAIN}"
echo "[stage2-oracle] stage2_candidate_mode=${STAGE2_CANDIDATE_MODE} train_crop=${STAGE2_CANDIDATE_TRAIN_CROP} kcrop_pick=${STAGE2_KCROP_TRAIN_PICK}"
echo "[stage2-oracle] memory_write=${STAGE2_MEMORY_WRITE_MODE} memory_topk=${STAGE2_MEMORY_WRITE_TOPK} memory_temp=${STAGE2_MEMORY_WRITE_TEMPERATURE} memory_sigma=${STAGE2_MEMORY_WRITE_SIGMA}"

if [[ "${PREPARE_ONLY}" == "1" ]]; then
  echo "[stage2-oracle] prepare-only done"
  exit 0
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '[stage2-oracle] command: CUDA_VISIBLE_DEVICES=%q ' "${GPU}"
  printf '%q ' "${TRAIN_CMD[@]}"
  printf '\n'
  exit 0
fi

cd "${SAM2ACT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU}"
exec "${TRAIN_CMD[@]}"
