#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

METHOD="${METHOD:-seqfft}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-42}"
GPU_ID="${GPU_ID:-}"
JOB_SUFFIX="${JOB_SUFFIX:-gcl}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-./outputs/baselines_gcl}"
LOG_ROOT="${LOG_ROOT:-./logs/baselines_gcl}"
POLICY_PATH="${POLICY_PATH:-./models/dit_flow_mt_libero_90_pretrain}"
PEFT_CFG_PATH="${PEFT_CFG_PATH:-}"
BENCHMARK_NAME="${BENCHMARK_NAME:-libero_10}"
TASK_PREFIX="${TASK_PREFIX:-Libero_10_Task}"
DATASET_PREFIX="${DATASET_PREFIX:-continuallearning/libero_10_image_task}"
MAX_TASK="${MAX_TASK:-9}"
GCL_N_PERCENT="${GCL_N_PERCENT:-50}"
GCL_M_PERCENT="${GCL_M_PERCENT:-30}"
GCL_EXPAND="${GCL_EXPAND:-true}"
STEPS="${STEPS:-200000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
REPLAY_BATCH_SIZE="${REPLAY_BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-16}"
REPLAY_NUM_WORKERS="${REPLAY_NUM_WORKERS:-16}"
LOG_STEPS="${LOG_STEPS:-100}"
EVAL_FREQ="${EVAL_FREQ:-0}"
SAVE_FREQ="${SAVE_FREQ:-20000}"
N_EVAL="${N_EVAL:-0}"
BS_EVAL="${BS_EVAL:-0}"
EVAL_MAX_EPISODES_RENDERED="${EVAL_MAX_EPISODES_RENDERED:-0}"
AUTO_RESUME="${AUTO_RESUME:-true}"
RESUME="${RESUME:-false}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-baseline_gcl_experiments}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
POST_PRUNE_STEPS="${POST_PRUNE_STEPS:-20000}"
PRUNE_RATIO="${PRUNE_RATIO:-0.75}"
IGNORE_MODULES="${IGNORE_MODULES:-}"
EWC_LAMBDA="${EWC_LAMBDA:-1000.0}"
EWC_FISHER_BATCHES="${EWC_FISHER_BATCHES:-200}"
EWC_FISHER_BATCH_SIZE="${EWC_FISHER_BATCH_SIZE:-}"
LWF_LAMBDA="${LWF_LAMBDA:-0.1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --method=*) METHOD="${1#*=}"; shift ;;
    --method) METHOD="$2"; shift 2 ;;
    --seed=*) SEED="${1#*=}"; shift ;;
    --seed) SEED="$2"; shift 2 ;;
    --gpu-id=*) GPU_ID="${1#*=}"; shift ;;
    --gpu-id) GPU_ID="$2"; shift 2 ;;
    --task-start=*|--task-start) if [[ "$1" == *=* ]]; then shift; else shift 2; fi ;;
    --task-end=*) MAX_TASK="${1#*=}"; shift ;;
    --task-end) MAX_TASK="$2"; shift 2 ;;
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
    --benchmark-name=*) BENCHMARK_NAME="${1#*=}"; shift ;;
    --benchmark-name) BENCHMARK_NAME="$2"; shift 2 ;;
    --task-prefix=*) TASK_PREFIX="${1#*=}"; shift ;;
    --task-prefix) TASK_PREFIX="$2"; shift 2 ;;
    --dataset-prefix=*) DATASET_PREFIX="${1#*=}"; shift ;;
    --dataset-prefix) DATASET_PREFIX="$2"; shift 2 ;;
    --gcl-n-percent=*) GCL_N_PERCENT="${1#*=}"; shift ;;
    --gcl-n-percent) GCL_N_PERCENT="$2"; shift 2 ;;
    --gcl-m-percent=*) GCL_M_PERCENT="${1#*=}"; shift ;;
    --gcl-m-percent) GCL_M_PERCENT="$2"; shift 2 ;;
    --gcl-expand=*) GCL_EXPAND="${1#*=}"; shift ;;
    --gcl-expand) GCL_EXPAND="$2"; shift 2 ;;
    --wandb-enable=*) WANDB_ENABLE="${1#*=}"; shift ;;
    --wandb-enable) WANDB_ENABLE="$2"; shift 2 ;;
    --wandb-project=*) WANDB_PROJECT="${1#*=}"; shift ;;
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    --wandb-entity=*) WANDB_ENTITY="${1#*=}"; shift ;;
    --wandb-entity) WANDB_ENTITY="$2"; shift 2 ;;
    --auto-resume=*) AUTO_RESUME="${1#*=}"; shift ;;
    --auto-resume) AUTO_RESUME="$2"; shift 2 ;;
    --resume=*) RESUME="${1#*=}"; shift ;;
    --resume) RESUME="$2"; shift 2 ;;
    --checkpoint-path=*) CHECKPOINT_PATH="${1#*=}"; shift ;;
    --checkpoint-path) CHECKPOINT_PATH="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$METHOD" in
  seqfft)
    SCRIPT_PATH="./lerobot_lsy/src/lerobot/scripts/train_eval_multi.py"
    JOB_PREFIX="seqfft_gcl"
    ;;
  seqlora)
    SCRIPT_PATH="./lerobot_lsy/src/lerobot/scripts/train_peft.py"
    PEFT_CFG_PATH="${PEFT_CFG_PATH:-./peft_lsy/peft_config/seqlora_dit_flow_adapter}"
    JOB_PREFIX="seqlora_gcl"
    ;;
  packnet)
    SCRIPT_PATH="./lerobot_lsy/src/lerobot/scripts/packnet.py"
    JOB_PREFIX="packnet_gcl"
    ;;
  er|er_reservoir)
    SCRIPT_PATH="./lerobot_lsy/src/lerobot/scripts/er.py"
    JOB_PREFIX="${METHOD}_gcl"
    BATCH_SIZE="${BATCH_SIZE:-16}"
    ;;
  ewc)
    SCRIPT_PATH="./lerobot_lsy/src/lerobot/scripts/ewc.py"
    JOB_PREFIX="ewc_gcl"
    ;;
  lwf)
    SCRIPT_PATH="./lerobot_lsy/src/lerobot/scripts/lwf.py"
    JOB_PREFIX="lwf_gcl"
    ;;
  l2p_adapter)
    SCRIPT_PATH="./lerobot_lsy/src/lerobot/scripts/l2p_adapter.py"
    PEFT_CFG_PATH="${PEFT_CFG_PATH:-./peft_lsy/peft_config/l2p_dit_flow_adapter}"
    JOB_PREFIX="l2p_adapter_gcl"
    ;;
  dualprompt_adapter)
    SCRIPT_PATH="./lerobot_lsy/src/lerobot/scripts/dualprompt_adapter.py"
    PEFT_CFG_PATH="${PEFT_CFG_PATH:-./peft_lsy/peft_config/dualprompt_dit_flow_adapter}"
    JOB_PREFIX="dualprompt_adapter_gcl"
    ;;
  *)
    echo "ERROR: GCL launcher for method '${METHOD}' has no implementation in this CLARE checkout." >&2
    exit 90
    ;;
esac

if [[ -n "$GPU_ID" ]]; then export CUDA_VISIBLE_DEVICES="$GPU_ID"; fi
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-./.cache/huggingface}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-./data}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

ALL_REPO_IDS=""
ALL_TASKS=""
for ((i=0; i<=MAX_TASK; i++)); do
  [[ -n "$ALL_REPO_IDS" ]] && ALL_REPO_IDS="${ALL_REPO_IDS},"
  [[ -n "$ALL_TASKS" ]] && ALL_TASKS="${ALL_TASKS},"
  ALL_REPO_IDS="${ALL_REPO_IDS}${DATASET_PREFIX}_${i}"
  ALL_TASKS="${ALL_TASKS}${TASK_PREFIX}_${i}"
done

JOB_NAME="${JOB_PREFIX}_seed_${SEED}_${BENCHMARK_NAME}_gcl_n${GCL_N_PERCENT}_m${GCL_M_PERCENT}_${JOB_SUFFIX}"
OUTPUT_DIR="${CHECKPOINT_ROOT}/${JOB_NAME}"
mkdir -p "$LOG_ROOT"
LOG_FILE="${LOG_ROOT}/${JOB_NAME}.log"

if [[ "$AUTO_RESUME" == "true" && -z "$CHECKPOINT_PATH" && -e "${OUTPUT_DIR}/checkpoints/last" ]]; then
  RESUME="true"
  CHECKPOINT_PATH="${OUTPUT_DIR}/checkpoints/last"
fi

cmd=(

  "$PYTHON_BIN" "$SCRIPT_PATH"
  "--seed=${SEED}"
  "--job_name=${JOB_NAME}"
  "--output_dir=${OUTPUT_DIR}"
  "--dataset.repo_id=${ALL_REPO_IDS}"
  "--dataset.use_gcl=true"
  "--dataset.use_gcl_expand=${GCL_EXPAND}"
  "--dataset.gcl_n_percent=${GCL_N_PERCENT}"
  "--dataset.gcl_m_percent=${GCL_M_PERCENT}"
  "--policy.path=${POLICY_PATH}"
  "--policy.push_to_hub=false"
  "--batch_size=${BATCH_SIZE}"
  "--num_workers=${NUM_WORKERS}"
  "--steps=${STEPS}"
  "--env.type=libero"
  "--env.benchmark=${BENCHMARK_NAME}"
  "--env.task=${ALL_TASKS}"
  "--eval.batch_size=${BS_EVAL}"
  "--eval.n_episodes=${N_EVAL}"
  "--eval.max_episodes_rendered=${EVAL_MAX_EPISODES_RENDERED}"
  "--eval_freq=${EVAL_FREQ}"
  "--save_freq=${SAVE_FREQ}"
  "--log_freq=${LOG_STEPS}"
  "--wandb.enable=${WANDB_ENABLE}"
  "--wandb.disable_artifact=true"
  "--wandb.project=${WANDB_PROJECT}"
  "--wandb.entity=${WANDB_ENTITY}"
)

if [[ "$RESUME" == "true" ]]; then
  if [[ -z "$CHECKPOINT_PATH" ]]; then
    echo "ERROR: RESUME=true but CHECKPOINT_PATH is empty" >&2
    exit 91
  fi
  cmd+=("--resume=true" "--config_path=${CHECKPOINT_PATH}/pretrained_model/train_config.json")
fi

if [[ "$METHOD" == "seqlora" ]]; then
  cmd+=("--peft_cfg_path=${PEFT_CFG_PATH}" "--merge_back_to_policy=true")
elif [[ "$METHOD" == "l2p_adapter" || "$METHOD" == "dualprompt_adapter" ]]; then
  cmd+=("--peft_cfg_path=${PEFT_CFG_PATH}")
elif [[ "$METHOD" == "ewc" ]]; then
  cmd+=("--ewc_lambda=${EWC_LAMBDA}" "--ewc_fisher_batches=${EWC_FISHER_BATCHES}")
  [[ -n "$EWC_FISHER_BATCH_SIZE" ]] && cmd+=("--ewc_fisher_batch_size=${EWC_FISHER_BATCH_SIZE}")
elif [[ "$METHOD" == "lwf" ]]; then
  cmd+=("--lwf_lambda=${LWF_LAMBDA}")
elif [[ "$METHOD" == "packnet" ]]; then
  cmd+=("--current_task=0" "--post_prune_steps=${POST_PRUNE_STEPS}" "--prune_ratio=${PRUNE_RATIO}" "--ignore_modules=${IGNORE_MODULES}")
elif [[ "$METHOD" == "er" || "$METHOD" == "er_reservoir" ]]; then
  cmd+=(
    "--replay_dataset.repo_id=${DATASET_PREFIX}_0"
    "--replay_dataset_repo_ids=${ALL_REPO_IDS}"
    "--replay_batch_size=${REPLAY_BATCH_SIZE}"
    "--replay_num_workers=${REPLAY_NUM_WORKERS}"
  )
fi

echo "Job: $JOB_NAME"
echo "Output: $OUTPUT_DIR"
echo "Datasets: $ALL_REPO_IDS"
echo "Log: $LOG_FILE"
printf 'Command: %q ' "${cmd[@]}"
printf '\n'
"${cmd[@]}" 2>&1 | tee "$LOG_FILE"
