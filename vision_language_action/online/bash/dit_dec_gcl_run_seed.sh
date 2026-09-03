#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SCRIPT_PATH="./lerobot_lsy/src/lerobot/scripts/clare.py"
PYTHON_BIN="${PYTHON_BIN:-python}"

SEED="${SEED:-42}"
GPU_ID="${GPU_ID:-}"
JOB_PREFIX="${JOB_PREFIX:-dit_flow_mt_cl_gcl}"
JOB_SUFFIX="${JOB_SUFFIX:-reproduce}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-./outputs/dit_dec_gcl}"
LOG_ROOT="${LOG_ROOT:-./logs/dit_dec_gcl}"
POLICY_PATH="${POLICY_PATH:-./models/dit_flow_mt_libero_90_pretrain}"
PEFT_CFG_PATH="${PEFT_CFG_PATH:-./peft_lsy/peft_config/clare_dit_flow_encoder_adapter}"
BENCHMARK_NAME="${BENCHMARK_NAME:-libero_goal}"
TASK_PREFIX="${TASK_PREFIX:-Libero_Goal_Task}"
DATASET_PREFIX="${DATASET_PREFIX:-continuallearning/libero_goal_image_task}"
MAX_TASK="${MAX_TASK:-9}"
GCL_N_PERCENT="${GCL_N_PERCENT:-50}"
GCL_M_PERCENT="${GCL_M_PERCENT:-30}"
STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-16}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-clare_experiments}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

# 解析参数
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed=*) SEED="${1#*=}"; shift ;;
    --seed) SEED="$2"; shift 2 ;;
    --gpu-id=*) GPU_ID="${1#*=}"; shift ;;
    --gpu-id) GPU_ID="$2"; shift 2 ;;
    --job-prefix=*) JOB_PREFIX="${1#*=}"; shift ;;
    --job-prefix) JOB_PREFIX="$2"; shift 2 ;;
    --job-suffix=*) JOB_SUFFIX="${1#*=}"; shift ;;
    --job-suffix) JOB_SUFFIX="$2"; shift 2 ;;
    --checkpoint-root=*) CHECKPOINT_ROOT="${1#*=}"; shift ;;
    --checkpoint-root) CHECKPOINT_ROOT="$2"; shift 2 ;;
    --log-root=*) LOG_ROOT="${1#*=}"; shift ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    --policy-path=*) POLICY_PATH="${1#*=}"; shift ;;
    --policy-path) POLICY_PATH="$2"; shift 2 ;;
    --peft-cfg-path=*) PEFT_CFG_PATH="${1#*=}"; shift ;;
    --peft-cfg-path) PEFT_CFG_PATH="$2"; shift 2 ;;
    --gcl-n-percent=*) GCL_N_PERCENT="${1#*=}"; shift ;;
    --gcl-n-percent) GCL_N_PERCENT="$2"; shift 2 ;;
    --gcl-m-percent=*) GCL_M_PERCENT="${1#*=}"; shift ;;
    --gcl-m-percent) GCL_M_PERCENT="$2"; shift 2 ;;
    --wandb-enable=*) WANDB_ENABLE="${1#*=}"; shift ;;
    --wandb-enable) WANDB_ENABLE="$2"; shift 2 ;;
    --wandb-project=*) WANDB_PROJECT="${1#*=}"; shift ;;
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    --wandb-entity=*) WANDB_ENTITY="${1#*=}"; shift ;;
    --wandb-entity) WANDB_ENTITY="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -n "$GPU_ID" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-./.cache/huggingface}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-./.cache/huggingface/lerobot}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

# 构建所有任务的repo_id列表，用逗号分隔
ALL_REPO_IDS=""
for ((i=0; i<=MAX_TASK; i++)); do
  if [[ -n "$ALL_REPO_IDS" ]]; then
    ALL_REPO_IDS="${ALL_REPO_IDS},"
  fi
  ALL_REPO_IDS="${ALL_REPO_IDS}${DATASET_PREFIX}_${i}"
done

# 构建所有任务的env task列表
ALL_TASKS=""
for ((i=0; i<=MAX_TASK; i++)); do
  if [[ -n "$ALL_TASKS" ]]; then
    ALL_TASKS="${ALL_TASKS},"
  fi
  ALL_TASKS="${ALL_TASKS}${TASK_PREFIX}_${i}"
done

JOB_NAME="${JOB_PREFIX}_seed_${SEED}_${BENCHMARK_NAME}_gcl_n${GCL_N_PERCENT}_m${GCL_M_PERCENT}_${JOB_SUFFIX}"
OUTPUT_DIR="${CHECKPOINT_ROOT}/${JOB_NAME}"

mkdir -p "$LOG_ROOT"
LOG_FILE="${LOG_ROOT}/${JOB_NAME}.log"

echo "Job: $JOB_NAME"
echo "Output: $OUTPUT_DIR"
echo "Datasets: $ALL_REPO_IDS"
echo "Log: $LOG_FILE"

"$PYTHON_BIN" "$SCRIPT_PATH" \
  "--seed=${SEED}" \
  "--job_name=${JOB_NAME}" \
  "--output_dir=${OUTPUT_DIR}" \
  "--dataset.repo_id=${ALL_REPO_IDS}" \
  "--dataset.use_gcl=true" \
  "--dataset.use_gcl_expand=true" \
  "--dataset.gcl_n_percent=${GCL_N_PERCENT}" \
  "--dataset.gcl_m_percent=${GCL_M_PERCENT}" \
  "--policy.path=${POLICY_PATH}" \
  "--policy.push_to_hub=false" \
  "--batch_size=${BATCH_SIZE}" \
  "--num_workers=${NUM_WORKERS}" \
  "--steps=${STEPS}" \
  "--env.type=libero" \
  "--env.benchmark=${BENCHMARK_NAME}" \
  "--env.task=${ALL_TASKS}" \
  "--peft_cfg_path=${PEFT_CFG_PATH}" \
  "--wandb.enable=${WANDB_ENABLE}" \
  "--wandb.disable_artifact=true" \
  "--wandb.project=${WANDB_PROJECT}" \
  "--wandb.entity=${WANDB_ENTITY}" \
  2>&1 | tee "$LOG_FILE"
