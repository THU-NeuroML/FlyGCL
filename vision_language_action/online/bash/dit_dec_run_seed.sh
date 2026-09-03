#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash/dit_dec_run_seed.sh [options] [-- extra clare.py args...]

Options:
  --seed INT                     Seed to run. Default: 42
  --task INT                     Run a single task.
  --task-start INT               First task in a contiguous range. Default: 0
  --task-end INT                 Last task in a contiguous range. Default: 9
  --gpu-id ID                    Physical GPU id for CUDA_VISIBLE_DEVICES.
  --job-prefix STR               Job name prefix. Default: dit_flow_mt_cl
  --job-suffix STR               Job name suffix appended after threshold slug.
  --source-job-suffix STR        Suffix used to locate the predecessor adapter for the
                                 first task in the range. Defaults to --job-suffix.
  --checkpoint-root PATH         Root directory for outputs. Default: ./outputs
  --source-checkpoint-root PATH  Root directory used to locate the predecessor adapter
                                 for the first task in the range. Defaults to
                                 --checkpoint-root.
  --log-root PATH                Root directory for launcher logs.
                                 Default: ./logs/dit_dec
  --policy-path PATH             Base pretrained policy path.
  --peft-cfg-path PATH           CLARE adapter config for the first task when no
                                 predecessor adapter is provided.
  --initial-adapter-path PATH    Explicit adapter path for the first task in the range.
  --benchmark-name STR           LIBERO benchmark passed to --env.benchmark.
                                 Default: libero_10
  --task-prefix STR              Task name prefix before the numeric task id.
                                 Default: Libero_10_Task
  --dataset-prefix STR           Dataset repo prefix before the numeric task id.
                                 Default: continuallearning/libero_10_image_task
  --max-task INT                 Maximum allowed task index. Default: 9
  --env-task-override CSV        Explicit comma-separated task list forwarded to
                                 --env.task. Useful for cross-suite sequences.
  --expand-threshold FLOAT       Expansion threshold. Default: 2.5
  --expand-threshold-slug STR    Threshold slug used in job names. Defaults to the
                                 threshold with '.' replaced by '_'.
  --wandb-enable BOOL            true or false. Default: false
  --wandb-project STR            Default: clare_experiments
  --wandb-entity STR             Default: 
  --show-command                 Print the full python command before each task.
  --help                         Show this message.

Any arguments after '--' are forwarded to clare.py unchanged.
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SCRIPT_PATH="./lerobot_lsy/src/lerobot/scripts/clare.py"
PEFT_CFG_PATH="${PEFT_CFG_PATH:-./peft_lsy/peft_config/clare_dit_flow_encoder_adapter}"

SEED="${SEED:-42}"
TASK_START="${TASK_START:-0}"
TASK_END="${TASK_END:-9}"
GPU_ID="${GPU_ID:-}"
JOB_PREFIX="${JOB_PREFIX:-dit_flow_mt_cl}"
JOB_SUFFIX="${JOB_SUFFIX:-reproduce}"
SOURCE_JOB_SUFFIX="${SOURCE_JOB_SUFFIX:-}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-./outputs}"
SOURCE_CHECKPOINT_ROOT="${SOURCE_CHECKPOINT_ROOT:-}"
LOG_ROOT="${LOG_ROOT:-./logs/dit_dec}"
POLICY_PATH="${POLICY_PATH:-./models/dit_flow_mt_libero_90_pretrain}"
INITIAL_ADAPTER_PATH="${INITIAL_ADAPTER_PATH:-}"
BENCHMARK_NAME="${BENCHMARK_NAME:-libero_10}"
TASK_PREFIX="${TASK_PREFIX:-Libero_10_Task}"
DATASET_PREFIX="${DATASET_PREFIX:-continuallearning/libero_10_image_task}"
MAX_TASK="${MAX_TASK:-9}"
ENV_TASK_OVERRIDE="${ENV_TASK_OVERRIDE:-}"
EXPAND_THRESHOLD="${EXPAND_THRESHOLD:-2.5}"
EXPAND_THRESHOLD_SLUG="${EXPAND_THRESHOLD_SLUG:-}"
SHOW_COMMAND=false

BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-16}"
STEPS="${STEPS:-20000}"
LOG_STEPS="${LOG_STEPS:-100}"
N_EVAL="${N_EVAL:-100}"
BS_EVAL="${BS_EVAL:-50}"
EVAL_MAX_EPISODES_RENDERED="${EVAL_MAX_EPISODES_RENDERED:-100}"
EVAL_FREQ="${EVAL_FREQ:-200000}"
DETECT_DISTRIBUTION_SHIFT_STEPS="${DETECT_DISTRIBUTION_SHIFT_STEPS:-200}"
DETECT_DISTRIBUTION_SHIFT_BATCH_SIZE="${DETECT_DISTRIBUTION_SHIFT_BATCH_SIZE:-32}"
DETECT_DISTRIBUTION_SHIFT_NUM_WORKERS="${DETECT_DISTRIBUTION_SHIFT_NUM_WORKERS:-16}"
DETECT_DISTRIBUTION_SHIFT_LOG_FREQ="${DETECT_DISTRIBUTION_SHIFT_LOG_FREQ:-10}"
TRAIN_DISCRIMINATORS_STEPS="${TRAIN_DISCRIMINATORS_STEPS:-2000}"
TRAIN_DISCRIMINATORS_BATCH_SIZE="${TRAIN_DISCRIMINATORS_BATCH_SIZE:-32}"
TRAIN_DISCRIMINATORS_NUM_WORKERS="${TRAIN_DISCRIMINATORS_NUM_WORKERS:-16}"
TRAIN_DISCRIMINATORS_LOG_FREQ="${TRAIN_DISCRIMINATORS_LOG_FREQ:-50}"
TRAIN_DISCRIMINATORS_EVAL_FREQ="${TRAIN_DISCRIMINATORS_EVAL_FREQ:-2000}"
TRAIN_DISCRIMINATORS_SAVE_FREQ="${TRAIN_DISCRIMINATORS_SAVE_FREQ:-2000}"

WANDB_ENABLE="${WANDB_ENABLE:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-clare_experiments}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed)
      SEED="$2"
      shift 2
      ;;
    --seed=*)
      SEED="${1#*=}"
      shift
      ;;
    --task)
      TASK_START="$2"
      TASK_END="$2"
      shift 2
      ;;
    --task=*)
      TASK_START="${1#*=}"
      TASK_END="${1#*=}"
      shift
      ;;
    --task-start)
      TASK_START="$2"
      shift 2
      ;;
    --task-start=*)
      TASK_START="${1#*=}"
      shift
      ;;
    --task-end)
      TASK_END="$2"
      shift 2
      ;;
    --task-end=*)
      TASK_END="${1#*=}"
      shift
      ;;
    --gpu-id)
      GPU_ID="$2"
      shift 2
      ;;
    --gpu-id=*)
      GPU_ID="${1#*=}"
      shift
      ;;
    --job-prefix)
      JOB_PREFIX="$2"
      shift 2
      ;;
    --job-prefix=*)
      JOB_PREFIX="${1#*=}"
      shift
      ;;
    --job-suffix)
      JOB_SUFFIX="$2"
      shift 2
      ;;
    --job-suffix=*)
      JOB_SUFFIX="${1#*=}"
      shift
      ;;
    --source-job-suffix)
      SOURCE_JOB_SUFFIX="$2"
      shift 2
      ;;
    --source-job-suffix=*)
      SOURCE_JOB_SUFFIX="${1#*=}"
      shift
      ;;
    --checkpoint-root)
      CHECKPOINT_ROOT="$2"
      shift 2
      ;;
    --checkpoint-root=*)
      CHECKPOINT_ROOT="${1#*=}"
      shift
      ;;
    --source-checkpoint-root)
      SOURCE_CHECKPOINT_ROOT="$2"
      shift 2
      ;;
    --source-checkpoint-root=*)
      SOURCE_CHECKPOINT_ROOT="${1#*=}"
      shift
      ;;
    --log-root)
      LOG_ROOT="$2"
      shift 2
      ;;
    --log-root=*)
      LOG_ROOT="${1#*=}"
      shift
      ;;
    --policy-path)
      POLICY_PATH="$2"
      shift 2
      ;;
    --policy-path=*)
      POLICY_PATH="${1#*=}"
      shift
      ;;
    --peft-cfg-path)
      PEFT_CFG_PATH="$2"
      shift 2
      ;;
    --peft-cfg-path=*)
      PEFT_CFG_PATH="${1#*=}"
      shift
      ;;
    --initial-adapter-path)
      INITIAL_ADAPTER_PATH="$2"
      shift 2
      ;;
    --initial-adapter-path=*)
      INITIAL_ADAPTER_PATH="${1#*=}"
      shift
      ;;
    --benchmark-name)
      BENCHMARK_NAME="$2"
      shift 2
      ;;
    --benchmark-name=*)
      BENCHMARK_NAME="${1#*=}"
      shift
      ;;
    --task-prefix)
      TASK_PREFIX="$2"
      shift 2
      ;;
    --task-prefix=*)
      TASK_PREFIX="${1#*=}"
      shift
      ;;
    --dataset-prefix)
      DATASET_PREFIX="$2"
      shift 2
      ;;
    --dataset-prefix=*)
      DATASET_PREFIX="${1#*=}"
      shift
      ;;
    --max-task)
      MAX_TASK="$2"
      shift 2
      ;;
    --max-task=*)
      MAX_TASK="${1#*=}"
      shift
      ;;
    --env-task-override)
      ENV_TASK_OVERRIDE="$2"
      shift 2
      ;;
    --env-task-override=*)
      ENV_TASK_OVERRIDE="${1#*=}"
      shift
      ;;
    --expand-threshold)
      EXPAND_THRESHOLD="$2"
      shift 2
      ;;
    --expand-threshold=*)
      EXPAND_THRESHOLD="${1#*=}"
      shift
      ;;
    --expand-threshold-slug)
      EXPAND_THRESHOLD_SLUG="$2"
      shift 2
      ;;
    --expand-threshold-slug=*)
      EXPAND_THRESHOLD_SLUG="${1#*=}"
      shift
      ;;
    --wandb-enable)
      WANDB_ENABLE="$2"
      shift 2
      ;;
    --wandb-enable=*)
      WANDB_ENABLE="${1#*=}"
      shift
      ;;
    --wandb-project)
      WANDB_PROJECT="$2"
      shift 2
      ;;
    --wandb-project=*)
      WANDB_PROJECT="${1#*=}"
      shift
      ;;
    --wandb-entity)
      WANDB_ENTITY="$2"
      shift 2
      ;;
    --wandb-entity=*)
      WANDB_ENTITY="${1#*=}"
      shift
      ;;
    --show-command)
      SHOW_COMMAND=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$SOURCE_JOB_SUFFIX" ]]; then
  SOURCE_JOB_SUFFIX="$JOB_SUFFIX"
fi

if [[ -z "$SOURCE_CHECKPOINT_ROOT" ]]; then
  SOURCE_CHECKPOINT_ROOT="$CHECKPOINT_ROOT"
fi

if [[ -z "$EXPAND_THRESHOLD_SLUG" ]]; then
  EXPAND_THRESHOLD_SLUG="${EXPAND_THRESHOLD//./_}"
fi

if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --seed must be a non-negative integer." >&2
  exit 2
fi

if [[ ! "$TASK_START" =~ ^[0-9]+$ || ! "$TASK_END" =~ ^[0-9]+$ || ! "$MAX_TASK" =~ ^[0-9]+$ ]]; then
  echo "ERROR: task indices and --max-task must be non-negative integers." >&2
  exit 2
fi

if (( TASK_START < 0 || TASK_END > MAX_TASK || TASK_START > TASK_END )); then
  echo "ERROR: task range must satisfy 0 <= start <= end <= ${MAX_TASK}." >&2
  exit 2
fi

if [[ ! -e "$POLICY_PATH" ]]; then
  echo "ERROR: policy path not found: $POLICY_PATH" >&2
  exit 3
fi

if [[ ! -e "$PEFT_CFG_PATH" ]]; then
  echo "ERROR: PEFT config path not found: $PEFT_CFG_PATH" >&2
  exit 3
fi

if [[ "$WANDB_ENABLE" == "true" && -z "$WANDB_ENTITY" ]]; then
  echo "ERROR: WANDB_ENABLE=true but WANDB_ENTITY is empty." >&2
  exit 6
fi

if [[ -n "$GPU_ID" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ -z "${MUJOCO_EGL_DEVICE_ID:-}" ]]; then
  export MUJOCO_EGL_DEVICE_ID="${CUDA_VISIBLE_DEVICES%%,*}"
else
  export MUJOCO_EGL_DEVICE_ID
fi

export HF_HOME="${HF_HOME:-./.cache/huggingface}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-./.cache/huggingface/lerobot}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

log_info() {
  printf '%s\n' "$*" | tee -a "$MASTER_LOG"
}

log_task_info() {
  printf '%s\n' "$*" | tee -a "$MASTER_LOG" "$task_log"
}

run_with_clean_stdout_logs() {
  local cmd_string="$1"
  set +e
  bash -lc "$cmd_string" > >(tee -a "$MASTER_LOG" "$task_log")
  local cmd_exit_code=$?
  set -e
  return "$cmd_exit_code"
}

build_job_name() {
  local seed="$1"
  local task="$2"
  local suffix="$3"
  local suffix_part=""
  if [[ -n "$suffix" ]]; then
    suffix_part="_${suffix}"
  fi

  printf "%s_seed_%s_%s_task_%s_encoder_mlp_adapter_threshold_%s%s" \
    "$JOB_PREFIX" \
    "$seed" \
    "$BENCHMARK_NAME" \
    "$task" \
    "$EXPAND_THRESHOLD_SLUG" \
    "$suffix_part"
}

build_output_dir() {
  local root="$1"
  local seed="$2"
  local task="$3"
  local suffix="$4"
  printf "%s/%s" "$root" "$(build_job_name "$seed" "$task" "$suffix")"
}

build_env_task_list() {
  local end_task="$1"
  if [[ -n "$ENV_TASK_OVERRIDE" ]]; then
    printf "%s" "$ENV_TASK_OVERRIDE"
    return 0
  fi

  local tasks=()
  local i
  for ((i = 0; i <= end_task; i++)); do
    tasks+=("${TASK_PREFIX}_${i}")
  done
  local joined=""
  local task_name
  for task_name in "${tasks[@]}"; do
    if [[ -n "$joined" ]]; then
      joined+=","
    fi
    joined+="$task_name"
  done
  printf "%s" "$joined"
}

resolve_adapter_path() {
  local task="$1"
  if (( task == 0 )); then
    return 1
  fi

  if (( task == TASK_START )) && [[ -n "$INITIAL_ADAPTER_PATH" ]]; then
    printf "%s" "$INITIAL_ADAPTER_PATH"
    return 0
  fi

  if (( task == TASK_START )); then
    printf "%s/checkpoints/last/adapter" \
      "$(build_output_dir "$SOURCE_CHECKPOINT_ROOT" "$SEED" "$((task - 1))" "$SOURCE_JOB_SUFFIX")"
    return 0
  fi

  printf "%s/checkpoints/last/adapter" \
    "$(build_output_dir "$CHECKPOINT_ROOT" "$SEED" "$((task - 1))" "$JOB_SUFFIX")"
}

TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
RUN_LOG_ROOT="${LOG_ROOT}/seed_${SEED}/run_${TIMESTAMP}"
mkdir -p "$RUN_LOG_ROOT"
MASTER_LOG="${RUN_LOG_ROOT}/seed_${SEED}_tasks_${TASK_START}_${TASK_END}.log"

log_info "Repo root: $REPO_ROOT"
log_info "Launcher log: $MASTER_LOG"
log_info "Seed: $SEED"
log_info "Task range: ${TASK_START}-${TASK_END}"
log_info "Benchmark: $BENCHMARK_NAME"
log_info "Task prefix: $TASK_PREFIX"
log_info "Dataset prefix: $DATASET_PREFIX"
if [[ -n "$ENV_TASK_OVERRIDE" ]]; then
  log_info "Env task override: $ENV_TASK_OVERRIDE"
fi
log_info "Checkpoint root: $CHECKPOINT_ROOT"
log_info "Source checkpoint root: $SOURCE_CHECKPOINT_ROOT"
log_info "Job suffix: ${JOB_SUFFIX:-<empty>}"
log_info "Source job suffix: ${SOURCE_JOB_SUFFIX:-<empty>}"
log_info "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
log_info "MUJOCO_EGL_DEVICE_ID: $MUJOCO_EGL_DEVICE_ID"

task=""
for ((task = TASK_START; task <= TASK_END; task++)); do
  job_name="$(build_job_name "$SEED" "$task" "$JOB_SUFFIX")"
  output_dir="$(build_output_dir "$CHECKPOINT_ROOT" "$SEED" "$task" "$JOB_SUFFIX")"
  env_tasks="$(build_env_task_list "$task")"
  task_log="${RUN_LOG_ROOT}/task_${task}.log"

  mkdir -p "$(dirname "$output_dir")"

  cmd=(
    python "$SCRIPT_PATH"
    "--seed=${SEED}"
    "--job_name=${job_name}"
    "--output_dir=${output_dir}"
    "--dataset.repo_id=${DATASET_PREFIX}_${task}"
    "--policy.path=${POLICY_PATH}"
    "--policy.push_to_hub=false"
    "--batch_size=${BATCH_SIZE}"
    "--num_workers=${NUM_WORKERS}"
    "--steps=${STEPS}"
    "--env.type=libero"
    "--env.benchmark=${BENCHMARK_NAME}"
    "--env.task=${env_tasks}"
    "--eval.batch_size=${BS_EVAL}"
    "--eval.n_episodes=${N_EVAL}"
    "--eval.max_episodes_rendered=${EVAL_MAX_EPISODES_RENDERED}"
    "--eval_freq=${EVAL_FREQ}"
    "--save_freq=${STEPS}"
    "--log_freq=${LOG_STEPS}"
    "--expand_threshold=${EXPAND_THRESHOLD}"
    "--detect_distribution_shift_steps=${DETECT_DISTRIBUTION_SHIFT_STEPS}"
    "--detect_distribution_shift_batch_size=${DETECT_DISTRIBUTION_SHIFT_BATCH_SIZE}"
    "--detect_distribution_shift_num_workers=${DETECT_DISTRIBUTION_SHIFT_NUM_WORKERS}"
    "--detect_distribution_shift_log_freq=${DETECT_DISTRIBUTION_SHIFT_LOG_FREQ}"
    "--train_discriminators_steps=${TRAIN_DISCRIMINATORS_STEPS}"
    "--train_discriminators_batch_size=${TRAIN_DISCRIMINATORS_BATCH_SIZE}"
    "--train_discriminators_num_workers=${TRAIN_DISCRIMINATORS_NUM_WORKERS}"
    "--train_discriminators_log_freq=${TRAIN_DISCRIMINATORS_LOG_FREQ}"
    "--train_discriminators_eval_freq=${TRAIN_DISCRIMINATORS_EVAL_FREQ}"
    "--train_discriminators_save_freq=${TRAIN_DISCRIMINATORS_SAVE_FREQ}"
    "--wandb.enable=${WANDB_ENABLE}"
    "--wandb.disable_artifact=true"
    "--wandb.project=${WANDB_PROJECT}"
    "--wandb.entity=${WANDB_ENTITY}"
  )

  if (( task == TASK_START )) && [[ -n "$INITIAL_ADAPTER_PATH" ]]; then
    if [[ ! -d "$INITIAL_ADAPTER_PATH" ]]; then
      echo "ERROR: initial adapter not found for task ${task}: $INITIAL_ADAPTER_PATH" >&2
      exit 7
    fi
    cmd+=("--peft_weight_path=${INITIAL_ADAPTER_PATH}")
  elif (( task == 0 )); then
    cmd+=("--peft_cfg_path=${PEFT_CFG_PATH}")
  else
    adapter_path="$(resolve_adapter_path "$task")"
    if [[ ! -d "$adapter_path" ]]; then
      echo "ERROR: predecessor adapter not found for task ${task}: $adapter_path" >&2
      exit 7
    fi
    cmd+=("--peft_weight_path=${adapter_path}")
  fi

  if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    cmd+=("${EXTRA_ARGS[@]}")
  fi

  log_task_info "==== Running seed ${SEED}, task ${task} ===="
  log_task_info "Output dir: $output_dir"
  log_task_info "Task log: $task_log"
  if (( task == TASK_START )) && [[ -n "$INITIAL_ADAPTER_PATH" ]]; then
    log_task_info "Adapter source: $INITIAL_ADAPTER_PATH"
  elif (( task > 0 )); then
    log_task_info "Adapter source: $(resolve_adapter_path "$task")"
  fi

  cmd_string=""
  printf -v cmd_string '%q ' "${cmd[@]}"
  if [[ "$SHOW_COMMAND" == "true" ]]; then
    log_task_info "Command: ${cmd_string}"
  fi

  run_with_clean_stdout_logs "$cmd_string"
  cmd_exit_code=$?
  if (( cmd_exit_code != 0 )); then
    exit "$cmd_exit_code"
  fi
done

log_info "Completed seed ${SEED}, tasks ${TASK_START}-${TASK_END}."
