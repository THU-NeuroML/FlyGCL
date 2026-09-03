#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash/seqlora_run_seed.sh [options] [-- extra train_peft.py args...]

Options are the same as seqfft_run_seed.sh plus:
  --peft-cfg-path PATH           LoRA config path. Default: ./peft_lsy/peft_config/seqlora_dit_flow_adapter
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
SCRIPT_PATH="./lerobot_lsy/src/lerobot/scripts/train_peft.py"
PEFT_CFG_PATH="${PEFT_CFG_PATH:-./peft_lsy/peft_config/seqlora_dit_flow_adapter}"
SEED="${SEED:-42}"; TASK_START="${TASK_START:-0}"; TASK_END="${TASK_END:-9}"; GPU_ID="${GPU_ID:-}"
JOB_PREFIX="${JOB_PREFIX:-seqlora}"; JOB_SUFFIX="${JOB_SUFFIX:-reproduce}"; SOURCE_JOB_SUFFIX="${SOURCE_JOB_SUFFIX:-}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-./outputs}"; SOURCE_CHECKPOINT_ROOT="${SOURCE_CHECKPOINT_ROOT:-}"
LOG_ROOT="${LOG_ROOT:-./logs/seqlora}"; POLICY_PATH="${POLICY_PATH:-./models/dit_flow_mt_libero_90_pretrain}"; INITIAL_POLICY_PATH="${INITIAL_POLICY_PATH:-}"
BENCHMARK_NAME="${BENCHMARK_NAME:-libero_10}"; TASK_PREFIX="${TASK_PREFIX:-Libero_10_Task}"; DATASET_PREFIX="${DATASET_PREFIX:-continuallearning/libero_10_image_task}"; MAX_TASK="${MAX_TASK:-9}"; ENV_TASK_OVERRIDE="${ENV_TASK_OVERRIDE:-}"
SHOW_COMMAND=false
BATCH_SIZE="${BATCH_SIZE:-32}"; NUM_WORKERS="${NUM_WORKERS:-16}"; STEPS="${STEPS:-20000}"; LOG_STEPS="${LOG_STEPS:-100}"; N_EVAL="${N_EVAL:-100}"; BS_EVAL="${BS_EVAL:-50}"; EVAL_MAX_EPISODES_RENDERED="${EVAL_MAX_EPISODES_RENDERED:-100}"; EVAL_FREQ="${EVAL_FREQ:-$STEPS}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"; WANDB_PROJECT="${WANDB_PROJECT:-baseline_experiments}"; WANDB_ENTITY="${WANDB_ENTITY:-}"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed) SEED="$2"; shift 2 ;; --seed=*) SEED="${1#*=}"; shift ;;
    --task) TASK_START="$2"; TASK_END="$2"; shift 2 ;; --task=*) TASK_START="${1#*=}"; TASK_END="${1#*=}"; shift ;;
    --task-start) TASK_START="$2"; shift 2 ;; --task-start=*) TASK_START="${1#*=}"; shift ;;
    --task-end) TASK_END="$2"; shift 2 ;; --task-end=*) TASK_END="${1#*=}"; shift ;;
    --gpu-id) GPU_ID="$2"; shift 2 ;; --gpu-id=*) GPU_ID="${1#*=}"; shift ;;
    --job-prefix) JOB_PREFIX="$2"; shift 2 ;; --job-prefix=*) JOB_PREFIX="${1#*=}"; shift ;;
    --job-suffix) JOB_SUFFIX="$2"; shift 2 ;; --job-suffix=*) JOB_SUFFIX="${1#*=}"; shift ;;
    --source-job-suffix) SOURCE_JOB_SUFFIX="$2"; shift 2 ;; --source-job-suffix=*) SOURCE_JOB_SUFFIX="${1#*=}"; shift ;;
    --checkpoint-root) CHECKPOINT_ROOT="$2"; shift 2 ;; --checkpoint-root=*) CHECKPOINT_ROOT="${1#*=}"; shift ;;
    --source-checkpoint-root) SOURCE_CHECKPOINT_ROOT="$2"; shift 2 ;; --source-checkpoint-root=*) SOURCE_CHECKPOINT_ROOT="${1#*=}"; shift ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;; --log-root=*) LOG_ROOT="${1#*=}"; shift ;;
    --policy-path) POLICY_PATH="$2"; shift 2 ;; --policy-path=*) POLICY_PATH="${1#*=}"; shift ;;
    --initial-policy-path) INITIAL_POLICY_PATH="$2"; shift 2 ;; --initial-policy-path=*) INITIAL_POLICY_PATH="${1#*=}"; shift ;;
    --peft-cfg-path) PEFT_CFG_PATH="$2"; shift 2 ;; --peft-cfg-path=*) PEFT_CFG_PATH="${1#*=}"; shift ;;
    --benchmark-name) BENCHMARK_NAME="$2"; shift 2 ;; --benchmark-name=*) BENCHMARK_NAME="${1#*=}"; shift ;;
    --task-prefix) TASK_PREFIX="$2"; shift 2 ;; --task-prefix=*) TASK_PREFIX="${1#*=}"; shift ;;
    --dataset-prefix) DATASET_PREFIX="$2"; shift 2 ;; --dataset-prefix=*) DATASET_PREFIX="${1#*=}"; shift ;;
    --max-task) MAX_TASK="$2"; shift 2 ;; --max-task=*) MAX_TASK="${1#*=}"; shift ;;
    --env-task-override) ENV_TASK_OVERRIDE="$2"; shift 2 ;; --env-task-override=*) ENV_TASK_OVERRIDE="${1#*=}"; shift ;;
    --wandb-enable) WANDB_ENABLE="$2"; shift 2 ;; --wandb-enable=*) WANDB_ENABLE="${1#*=}"; shift ;;
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;; --wandb-project=*) WANDB_PROJECT="${1#*=}"; shift ;;
    --wandb-entity) WANDB_ENTITY="$2"; shift 2 ;; --wandb-entity=*) WANDB_ENTITY="${1#*=}"; shift ;;
    --show-command) SHOW_COMMAND=true; shift ;; --help|-h) usage; exit 0 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;; *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -z "$SOURCE_JOB_SUFFIX" ]] && SOURCE_JOB_SUFFIX="$JOB_SUFFIX"; [[ -z "$SOURCE_CHECKPOINT_ROOT" ]] && SOURCE_CHECKPOINT_ROOT="$CHECKPOINT_ROOT"
if [[ ! "$SEED" =~ ^[0-9]+$ || ! "$TASK_START" =~ ^[0-9]+$ || ! "$TASK_END" =~ ^[0-9]+$ || ! "$MAX_TASK" =~ ^[0-9]+$ ]]; then echo "ERROR: seed and task indices must be non-negative integers." >&2; exit 2; fi
if (( TASK_START < 0 || TASK_END > MAX_TASK || TASK_START > TASK_END )); then echo "ERROR: task range must satisfy 0 <= start <= end <= ${MAX_TASK}." >&2; exit 2; fi
[[ -e "$POLICY_PATH" ]] || { echo "ERROR: policy path not found: $POLICY_PATH" >&2; exit 3; }
[[ -e "$PEFT_CFG_PATH" ]] || { echo "ERROR: PEFT config path not found: $PEFT_CFG_PATH" >&2; exit 3; }
if [[ "$WANDB_ENABLE" == "true" && -z "$WANDB_ENTITY" ]]; then echo "ERROR: WANDB_ENABLE=true but WANDB_ENTITY is empty." >&2; exit 6; fi
if [[ -n "$GPU_ID" ]]; then export CUDA_VISIBLE_DEVICES="$GPU_ID"; fi
export MUJOCO_GL="${MUJOCO_GL:-egl}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" HF_HOME="${HF_HOME:-./.cache/huggingface}" HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-./.cache/huggingface/lerobot}" HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${CUDA_VISIBLE_DEVICES%%,*}}"
if [[ -z "${PYTHON_BIN:-}" && -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then PYTHON_BIN="${CONDA_PREFIX}/bin/python"; fi
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"; [[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || { echo "ERROR: could not resolve a runnable python interpreter." >&2; exit 8; }
METHOD_SLUG="lora_r16_a32_merge"
log_info() { printf '%s\n' "$*" | tee -a "$MASTER_LOG"; }
log_task_info() { printf '%s\n' "$*" | tee -a "$MASTER_LOG" "$task_log"; }
run_with_clean_stdout_logs() { local cmd_string="$1"; set +e; bash -lc "$cmd_string" > >(tee -a "$MASTER_LOG" "$task_log") 2> >(tee -a "$MASTER_ERR_LOG" "$task_err_log" >&2); local cmd_exit_code=$?; set -e; return "$cmd_exit_code"; }
build_job_name() { local seed="$1" task="$2" suffix="$3" suffix_part=""; [[ -n "$suffix" ]] && suffix_part="_${suffix}"; printf "%s_seed_%s_%s_task_%s_%s%s" "$JOB_PREFIX" "$seed" "$BENCHMARK_NAME" "$task" "$METHOD_SLUG" "$suffix_part"; }
build_output_dir() { printf "%s/%s" "$1" "$(build_job_name "$2" "$3" "$4")"; }
build_env_task_list() { local end_task="$1"; if [[ -n "$ENV_TASK_OVERRIDE" ]]; then printf "%s" "$ENV_TASK_OVERRIDE"; return 0; fi; local joined="" i=""; for ((i = 0; i <= end_task; i++)); do [[ -n "$joined" ]] && joined+=","; joined+="${TASK_PREFIX}_${i}"; done; printf "%s" "$joined"; }
resolve_policy_path() { local task="$1"; if (( task == TASK_START )) && [[ -n "$INITIAL_POLICY_PATH" ]]; then printf "%s" "$INITIAL_POLICY_PATH"; return 0; fi; if (( task == 0 )); then printf "%s" "$POLICY_PATH"; return 0; fi; if (( task == TASK_START )); then printf "%s/checkpoints/last/pretrained_model" "$(build_output_dir "$SOURCE_CHECKPOINT_ROOT" "$SEED" "$((task - 1))" "$SOURCE_JOB_SUFFIX")"; return 0; fi; printf "%s/checkpoints/last/pretrained_model" "$(build_output_dir "$CHECKPOINT_ROOT" "$SEED" "$((task - 1))" "$JOB_SUFFIX")"; }
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"; RUN_LOG_ROOT="${LOG_ROOT}/seed_${SEED}/run_${TIMESTAMP}"; mkdir -p "$RUN_LOG_ROOT"; MASTER_LOG="${RUN_LOG_ROOT}/seed_${SEED}_tasks_${TASK_START}_${TASK_END}.log"; MASTER_ERR_LOG="${RUN_LOG_ROOT}/seed_${SEED}_tasks_${TASK_START}_${TASK_END}.stderr.log"
log_info "Repo root: $REPO_ROOT"; log_info "Launcher log: $MASTER_LOG"; log_info "Launcher stderr log: $MASTER_ERR_LOG"; log_info "Seed: $SEED"; log_info "Task range: ${TASK_START}-${TASK_END}"; log_info "Benchmark: $BENCHMARK_NAME"; log_info "PEFT config: $PEFT_CFG_PATH"; log_info "Python: $PYTHON_BIN"
for ((task = TASK_START; task <= TASK_END; task++)); do
  job_name="$(build_job_name "$SEED" "$task" "$JOB_SUFFIX")"; output_dir="$(build_output_dir "$CHECKPOINT_ROOT" "$SEED" "$task" "$JOB_SUFFIX")"; env_tasks="$(build_env_task_list "$task")"; current_policy_path="$(resolve_policy_path "$task")"; task_log="${RUN_LOG_ROOT}/task_${task}.log"; task_err_log="${RUN_LOG_ROOT}/task_${task}.stderr.log"
  [[ -d "$current_policy_path" ]] || { echo "ERROR: policy checkpoint not found for task ${task}: $current_policy_path" >&2; exit 7; }
  mkdir -p "$(dirname "$output_dir")"
  cmd=("$PYTHON_BIN" "$SCRIPT_PATH" "--seed=${SEED}" "--job_name=${job_name}" "--output_dir=${output_dir}" "--dataset.repo_id=${DATASET_PREFIX}_${task}" "--policy.path=${current_policy_path}" "--policy.push_to_hub=false" "--batch_size=${BATCH_SIZE}" "--num_workers=${NUM_WORKERS}" "--steps=${STEPS}" "--env.type=libero" "--env.benchmark=${BENCHMARK_NAME}" "--env.task=${env_tasks}" "--eval.batch_size=${BS_EVAL}" "--eval.n_episodes=${N_EVAL}" "--eval.max_episodes_rendered=${EVAL_MAX_EPISODES_RENDERED}" "--eval_freq=${EVAL_FREQ}" "--save_freq=${STEPS}" "--log_freq=${LOG_STEPS}" "--peft_cfg_path=${PEFT_CFG_PATH}" "--merge_back_to_policy=true" "--wandb.enable=${WANDB_ENABLE}" "--wandb.disable_artifact=true" "--wandb.project=${WANDB_PROJECT}" "--wandb.entity=${WANDB_ENTITY}")
  [[ ${#EXTRA_ARGS[@]} -gt 0 ]] && cmd+=("${EXTRA_ARGS[@]}")
  log_task_info "==== Running seed ${SEED}, task ${task} ===="; log_task_info "Output dir: $output_dir"; log_task_info "Policy source: $current_policy_path"; log_task_info "Task log: $task_log"; log_task_info "Task stderr log: $task_err_log"
  cmd_string=""; printf -v cmd_string '%q ' "${cmd[@]}"; [[ "$SHOW_COMMAND" == "true" ]] && log_task_info "Command: ${cmd_string}"
  run_with_clean_stdout_logs "$cmd_string"; cmd_exit_code=$?; (( cmd_exit_code == 0 )) || exit "$cmd_exit_code"
done
log_info "Completed seed ${SEED}, tasks ${TASK_START}-${TASK_END}."
