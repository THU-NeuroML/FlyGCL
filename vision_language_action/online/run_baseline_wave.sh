#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_baseline_wave.sh <start|status|stop|clean-incomplete> [options]
  bash run_baseline_wave.sh <start|status|stop|clean-incomplete> [options]

Automatic queue scheduler for baseline experiments.

Default schedule order:
  For each scenario: 10 -> goal -> spatial -> object
    run methods in pairs: "seqfft seqlora", then "er packnet"
      run all seeds in the method pair concurrently: 2 methods x 3 seeds = 6 sessions
      wait until the whole pair finishes before starting the next pair/scenario

Default usable GPUs are "0 1 2 5 6 7", avoiding GPUs 3 and 4.
Weights & Biases logging is disabled by default; enable it explicitly when needed.

Options:
  --seeds "42 43 44"              Seeds to run for every method/scenario. Default: "42 43 44"
  --gpus "0 1 2 5 6 7"            Physical GPUs to use. Default: "0 1 2 5 6 7"
  --method-pairs "a,b c,d"        Method pairs by wave. Default: "seqfft,seqlora er,packnet"
                                      Extra methods include l2p_adapter and dualprompt_adapter.
  --scenarios "10 goal ..."       Scenarios: 10 goal spatial object. Default: all four
  --task-start INT                 First task. Default: 0
  --task-end INT                   Last task. Default: 9
  --job-suffix STR                 Job suffix. Default: reproduce
  --checkpoint-root PATH           Root for outputs. Default: ./artifacts/outputs/baselines_wave
  --log-root PATH                  Root for logs. Default: ./artifacts/logs/baselines_wave
  --policy-path PATH               Base pretrained policy path.
  --wandb-enable BOOL              true or false. Default: false
  --wandb-project STR              Default: baseline_experiments
  --wandb-entity STR               Default: 
  --poll-seconds INT               Queue polling interval. Default: 120
  --resume BOOL                    Skip completed task outputs and resume from first missing task. Default: true
  --use-gcl BOOL                   Use one GCL run per method/scenario/seed. Default: true
  --gcl-n-percent INT              GCL disjoint percentage. Default: 50
  --gcl-m-percent INT              GCL blurry percentage. Default: 30
  --scheduler-name STR             Name for scheduler pid/stop/log files. Default: default
  --er-reservoir-capacity INT      Fixed episode memory size for er_reservoir. Default: 50
  --dry-run                        Print planned schedule without starting screens.
  --help                           Show this message.

Actions:
  start             Start/resume the automatic queue.
  status            Show running wave sessions.
  stop              Stop matching wave sessions.
  clean-incomplete  Delete incomplete output dirs under the selected methods/scenarios/seeds, preserving completed dirs with multitask_eval_info.json.
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

if [[ $# -lt 1 ]]; then usage >&2; exit 2; fi
ACTION="$1"; shift

SEEDS_STRING="${SEEDS_STRING:-42 43 44}"
GPUS_STRING="${GPUS_STRING:-0 1 2 5 6 7}"
METHOD_PAIRS_STRING="${METHOD_PAIRS_STRING:-seqfft,seqlora er,packnet}"
SCENARIOS_STRING="${SCENARIOS_STRING:-10 goal spatial object}"
TASK_START="${TASK_START:-0}"
TASK_END="${TASK_END:-9}"
JOB_SUFFIX="${JOB_SUFFIX:-reproduce}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-./artifacts/outputs/baselines_wave}"
LOG_ROOT="${LOG_ROOT:-./artifacts/logs/baselines_wave}"
POLICY_PATH="${POLICY_PATH:-./models/dit_flow_mt_libero_90_pretrain}"
if [[ -z "${PYTHON_BIN:-}" && -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
export PYTHON_BIN
WANDB_ENABLE="${WANDB_ENABLE:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-baseline_experiments}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
POLL_SECONDS="${POLL_SECONDS:-120}"
RESUME="${RESUME:-true}"
USE_GCL="${USE_GCL:-true}"
GCL_N_PERCENT="${GCL_N_PERCENT:-50}"
GCL_M_PERCENT="${GCL_M_PERCENT:-30}"
SCHEDULER_NAME="${SCHEDULER_NAME:-default}"
ER_RESERVOIR_CAPACITY="${ER_RESERVOIR_CAPACITY:-${REPLAY_MEMORY_CAPACITY:-50}}"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds) SEEDS_STRING="$2"; shift 2 ;; --seeds=*) SEEDS_STRING="${1#*=}"; shift ;;
    --gpus) GPUS_STRING="$2"; shift 2 ;; --gpus=*) GPUS_STRING="${1#*=}"; shift ;;
    --method-pairs) METHOD_PAIRS_STRING="$2"; shift 2 ;; --method-pairs=*) METHOD_PAIRS_STRING="${1#*=}"; shift ;;
    --methods) METHOD_PAIRS_STRING="$2"; shift 2 ;;
    --methods=*) METHOD_PAIRS_STRING="${1#*=}"; shift ;;
    --scenarios) SCENARIOS_STRING="$2"; shift 2 ;; --scenarios=*) SCENARIOS_STRING="${1#*=}"; shift ;;
    --task-start) TASK_START="$2"; shift 2 ;; --task-start=*) TASK_START="${1#*=}"; shift ;;
    --task-end) TASK_END="$2"; shift 2 ;; --task-end=*) TASK_END="${1#*=}"; shift ;;
    --job-suffix) JOB_SUFFIX="$2"; shift 2 ;; --job-suffix=*) JOB_SUFFIX="${1#*=}"; shift ;;
    --checkpoint-root) CHECKPOINT_ROOT="$2"; shift 2 ;; --checkpoint-root=*) CHECKPOINT_ROOT="${1#*=}"; shift ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;; --log-root=*) LOG_ROOT="${1#*=}"; shift ;;
    --policy-path) POLICY_PATH="$2"; shift 2 ;; --policy-path=*) POLICY_PATH="${1#*=}"; shift ;;
    --wandb-enable) WANDB_ENABLE="$2"; shift 2 ;; --wandb-enable=*) WANDB_ENABLE="${1#*=}"; shift ;;
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;; --wandb-project=*) WANDB_PROJECT="${1#*=}"; shift ;;
    --wandb-entity) WANDB_ENTITY="$2"; shift 2 ;; --wandb-entity=*) WANDB_ENTITY="${1#*=}"; shift ;;
    --poll-seconds) POLL_SECONDS="$2"; shift 2 ;; --poll-seconds=*) POLL_SECONDS="${1#*=}"; shift ;;
    --resume) RESUME="$2"; shift 2 ;; --resume=*) RESUME="${1#*=}"; shift ;;
    --use-gcl) USE_GCL="$2"; shift 2 ;; --use-gcl=*) USE_GCL="${1#*=}"; shift ;;
    --gcl-n-percent) GCL_N_PERCENT="$2"; shift 2 ;; --gcl-n-percent=*) GCL_N_PERCENT="${1#*=}"; shift ;;
    --gcl-m-percent) GCL_M_PERCENT="$2"; shift 2 ;; --gcl-m-percent=*) GCL_M_PERCENT="${1#*=}"; shift ;;
    --scheduler-name) SCHEDULER_NAME="$2"; shift 2 ;; --scheduler-name=*) SCHEDULER_NAME="${1#*=}"; shift ;;
    --er-reservoir-capacity|--replay-memory-capacity) ER_RESERVOIR_CAPACITY="$2"; shift 2 ;;
    --er-reservoir-capacity=*|--replay-memory-capacity=*) ER_RESERVOIR_CAPACITY="${1#*=}"; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ ! -d "${REPO_ROOT}/bash" ]]; then echo "ERROR: expected bash launchers under ${REPO_ROOT}/bash." >&2; exit 3; fi
if ! command -v screen >/dev/null 2>&1; then echo "ERROR: screen is not installed or not on PATH." >&2; exit 3; fi
if [[ ! "$POLL_SECONDS" =~ ^[0-9]+$ || "$POLL_SECONDS" -lt 10 ]]; then echo "ERROR: --poll-seconds must be an integer >= 10." >&2; exit 2; fi
if [[ ! "$ER_RESERVOIR_CAPACITY" =~ ^[0-9]+$ || "$ER_RESERVOIR_CAPACITY" -lt 1 ]]; then echo "ERROR: --er-reservoir-capacity must be a positive integer." >&2; exit 2; fi
if [[ "$RESUME" != "true" && "$RESUME" != "false" ]]; then echo "ERROR: --resume must be true or false." >&2; exit 2; fi
if [[ "$USE_GCL" != "true" && "$USE_GCL" != "false" ]]; then echo "ERROR: --use-gcl must be true or false." >&2; exit 2; fi
if [[ ! "$SCHEDULER_NAME" =~ ^[A-Za-z0-9_.-]+$ ]]; then echo "ERROR: --scheduler-name may only contain letters, numbers, dot, underscore, and dash." >&2; exit 2; fi

read -r -a SEEDS <<<"$SEEDS_STRING"
read -r -a GPUS <<<"$GPUS_STRING"
read -r -a METHOD_PAIRS <<<"$METHOD_PAIRS_STRING"
read -r -a SCENARIOS <<<"$SCENARIOS_STRING"
if [[ ${#GPUS[@]} -eq 0 ]]; then echo "ERROR: no GPUs configured." >&2; exit 2; fi
if [[ ${#SEEDS[@]} -eq 0 ]]; then echo "ERROR: no seeds configured." >&2; exit 2; fi
if [[ ${#METHOD_PAIRS[@]} -eq 0 ]]; then echo "ERROR: no method pairs configured." >&2; exit 2; fi
if [[ ${#SCENARIOS[@]} -eq 0 ]]; then echo "ERROR: no scenarios configured." >&2; exit 2; fi

RUN_LOG_DIR="${LOG_ROOT}/scheduler"
mkdir -p "$RUN_LOG_DIR"
SCHEDULER_LOG="${RUN_LOG_DIR}/run_baseline_wave_${SCHEDULER_NAME}_$(date +%Y-%m-%d_%H-%M-%S).log"
SCHEDULER_PID_FILE="${RUN_LOG_DIR}/run_baseline_wave_${SCHEDULER_NAME}.pid"
STOP_FILE="${RUN_LOG_DIR}/stop_requested_${SCHEDULER_NAME}"

log_msg() {
  local line
  line="[$(date +%Y-%m-%d_%H:%M:%S)] $*"
  echo "$line"
  if [[ "$ACTION" == "start" && "$DRY_RUN" != "true" ]]; then echo "$line" >> "$SCHEDULER_LOG"; fi
}

scenario_args() {
  local scenario="$1"
  case "$scenario" in
    10) echo "libero_10 Libero_10_Task continuallearning/libero_10_image_task" ;;
    goal) echo "libero_goal Libero_Goal_Task continuallearning/libero_goal_image_task" ;;
    spatial) echo "libero_spatial Libero_Spatial_Task continuallearning/libero_spatial_image_task" ;;
    object) echo "libero_object Libero_Object_Task continuallearning/libero_object_image_task" ;;
    *) echo "ERROR: unknown scenario: $scenario" >&2; exit 2 ;;
  esac
}

run_seed_script() {
  local method="$1"
  if [[ "$USE_GCL" == "true" ]]; then
    case "$method" in
      seqfft) echo "seqfft_gcl_run_seed.sh" ;;
      seqlora) echo "seqlora_gcl_run_seed.sh" ;;
      er) echo "er_gcl_run_seed.sh" ;;
      er_reservoir) echo "er_reservoir_gcl_run_seed.sh" ;;
      ewc) echo "ewc_gcl_run_seed.sh" ;;
      lwf) echo "lwf_gcl_run_seed.sh" ;;
      l2p_adapter) echo "l2p_adapter_gcl_run_seed.sh" ;;
      dualprompt_adapter) echo "dualprompt_adapter_gcl_run_seed.sh" ;;
      packnet) echo "packnet_gcl_run_seed.sh" ;;
      *) echo "ERROR: unknown method: $method" >&2; exit 2 ;;
    esac
    return 0
  fi
  case "$method" in
    seqfft) echo "seqfft_run_seed.sh" ;;
    seqlora) echo "seqlora_run_seed.sh" ;;
    er) echo "er_run_seed.sh" ;;
    er_reservoir) echo "er_reservoir_run_seed.sh" ;;
    ewc) echo "ewc_run_seed.sh" ;;
    lwf) echo "lwf_run_seed.sh" ;;
    l2p_adapter) echo "l2p_adapter_run_seed.sh" ;;
    dualprompt_adapter) echo "dualprompt_adapter_run_seed.sh" ;;
    packnet) echo "packnet_run_seed.sh" ;;
    *) echo "ERROR: unknown method: $method" >&2; exit 2 ;;
  esac
}

method_slug() {
  local method="$1"
  case "$method" in
    seqfft) echo "fullft" ;;
    seqlora) echo "lora_r16_a32_merge" ;;
    er) echo "replay50" ;;
    er_reservoir) echo "reservoir${ER_RESERVOIR_CAPACITY}" ;;
    ewc) echo "ewc_l1000_fb200" ;;
    lwf) echo "lwf_vpred_l1" ;;
    l2p_adapter) echo "pool10_top1_key0_1" ;;
    dualprompt_adapter) echo "general1_expert1_proto200" ;;
    packnet) echo "prune0_75_post20000" ;;
    *) echo "ERROR: unknown method: $method" >&2; exit 2 ;;
  esac
}

job_dir_for_task() {
  local method="$1" scenario="$2" seed="$3" task="$4" benchmark task_prefix dataset_prefix slug
  read -r benchmark task_prefix dataset_prefix <<<"$(scenario_args "$scenario")"
  slug="$(method_slug "$method")"
  printf "%s/%s/%s/seed_%s/%s_seed_%s_%s_task_%s_%s_%s" "$CHECKPOINT_ROOT" "$method" "$benchmark" "$seed" "$method" "$seed" "$benchmark" "$task" "$slug" "$JOB_SUFFIX"
}

completed_eval_file() {
  local method="$1" scenario="$2" seed="$3" task="$4"
  if [[ "$USE_GCL" == "true" ]]; then
    local benchmark task_prefix dataset_prefix job_name
    read -r benchmark task_prefix dataset_prefix <<<"$(scenario_args "$scenario")"
    job_name="${method}_gcl_seed_${seed}_${benchmark}_gcl_n${GCL_N_PERCENT}_m${GCL_M_PERCENT}_${JOB_SUFFIX}"
    if [[ "$method" == "seqfft" ]]; then job_name="seqfft_gcl_seed_${seed}_${benchmark}_gcl_n${GCL_N_PERCENT}_m${GCL_M_PERCENT}_${JOB_SUFFIX}"; fi
    if [[ "$method" == "seqlora" ]]; then job_name="seqlora_gcl_seed_${seed}_${benchmark}_gcl_n${GCL_N_PERCENT}_m${GCL_M_PERCENT}_${JOB_SUFFIX}"; fi
    if [[ "$method" == "packnet" ]]; then job_name="packnet_gcl_seed_${seed}_${benchmark}_gcl_n${GCL_N_PERCENT}_m${GCL_M_PERCENT}_${JOB_SUFFIX}"; fi
    printf "%s/%s/%s/seed_%s/%s/multitask_eval_info.json" "$CHECKPOINT_ROOT" "$method" "$benchmark" "$seed" "$job_name"
    return 0
  fi
  printf "%s/multitask_eval_info.json" "$(job_dir_for_task "$method" "$scenario" "$seed" "$task")"
}

first_missing_task() {
  local method="$1" scenario="$2" seed="$3" task
  for ((task = TASK_START; task <= TASK_END; task++)); do
    if [[ ! -f "$(completed_eval_file "$method" "$scenario" "$seed" "$task")" ]]; then
      printf "%s" "$task"
      return 0
    fi
  done
  return 1
}

session_name() {
  local method="$1" scenario="$2" seed="$3" session_method
  session_method="$method"
  if [[ "$method" == "er_reservoir" && "$ER_RESERVOIR_CAPACITY" != "50" ]]; then
    session_method="er_reservoir${ER_RESERVOIR_CAPACITY}"
  fi
  printf "wave-%s-%s-s%s" "$session_method" "$scenario" "$seed"
}

session_exists() {
  local session="$1"
  screen -list | grep -q "[.]${session}[[:space:]]"
}

start_one() {
  local method="$1" scenario="$2" seed="$3" gpu="$4"
  local benchmark task_prefix dataset_prefix script session method_out method_logs cmd cmd_string start_task
  read -r benchmark task_prefix dataset_prefix <<<"$(scenario_args "$scenario")"
  script="$(run_seed_script "$method")"
  session="$(session_name "$method" "$scenario" "$seed")"
  method_out="${CHECKPOINT_ROOT}/${method}/${benchmark}/seed_${seed}"
  method_logs="${LOG_ROOT}/${method}/${benchmark}"

  if session_exists "$session"; then log_msg "Skipping ${session}: already running."; return 0; fi

  start_task="$TASK_START"
  if [[ "$RESUME" == "true" ]]; then
    if ! start_task="$(first_missing_task "$method" "$scenario" "$seed")"; then
      log_msg "Skipping ${session}: tasks ${TASK_START}-${TASK_END} already complete."
      return 2
    fi
  fi

  cmd=(
    bash "${REPO_ROOT}/bash/${script}"
    "--seed=${seed}"
    "--gpu-id=${gpu}"
    "--task-start=${start_task}"
    "--task-end=${TASK_END}"
    "--job-suffix=${JOB_SUFFIX}"
    "--checkpoint-root=${method_out}"
    "--log-root=${method_logs}"
    "--policy-path=${POLICY_PATH}"
    "--benchmark-name=${benchmark}"
    "--task-prefix=${task_prefix}"
    "--dataset-prefix=${dataset_prefix}"
    "--wandb-enable=${WANDB_ENABLE}"
    "--wandb-project=${WANDB_PROJECT}"
    "--wandb-entity=${WANDB_ENTITY}"
    "--gcl-n-percent=${GCL_N_PERCENT}"
    "--gcl-m-percent=${GCL_M_PERCENT}"
  )
  if [[ "$method" == "er_reservoir" ]]; then
    cmd+=(--use-gcl=${USE_GCL})
    cmd+=("--replay-memory-capacity=${ER_RESERVOIR_CAPACITY}")
  fi
  printf -v cmd_string '%q ' "${cmd[@]}"

  if [[ "$DRY_RUN" == "true" ]]; then
    log_msg "DRY-RUN ${session} on GPU ${gpu} tasks ${start_task}-${TASK_END}: ${cmd_string}"
  else
    screen -dmS "$session" bash -lc "$cmd_string"
    sleep 1
    if session_exists "$session"; then
      log_msg "Started ${session} on GPU ${gpu}, tasks ${start_task}-${TASK_END}."
    else
      log_msg "Failed to keep ${session} alive. Check ${method_logs}/seed_${seed}/."
      return 1
    fi
  fi
}

session_log_root() {
  local method="$1" scenario="$2" seed="$3" benchmark task_prefix dataset_prefix
  read -r benchmark task_prefix dataset_prefix <<<"$(scenario_args "$scenario")"
  printf "%s/%s/%s/seed_%s" "$LOG_ROOT" "$method" "$benchmark" "$seed"
}

session_failed() {
  local method="$1" scenario="$2" seed="$3" log_root stderr_file
  log_root="$(session_log_root "$method" "$scenario" "$seed")"
  if [[ ! -d "$log_root" ]]; then
    return 0
  fi
  while IFS= read -r stderr_file; do
    if grep -Eq 'Traceback \(most recent call last\)|(^|[[:space:]])(ERROR|Error|Exception):|FileExistsError|ModuleNotFoundError|NotImplementedError|AttributeError|RuntimeError|ValueError|CUDA out of memory|OutOfMemoryError|Killed|Segmentation fault|fatal:' "$stderr_file" 2>/dev/null; then
      return 0
    fi
  done < <(find "$log_root" -path "*/run_*/*.stderr.log" -type f 2>/dev/null)
  return 1
}

wait_for_jobs() {
  local jobs=("$@")
  local remaining=() job session method scenario seed any_running
  while true; do
    remaining=()
    any_running=false
    for job in "${jobs[@]}"; do
      IFS=: read -r session method scenario seed <<<"$job"
      if session_exists "$session"; then
        remaining+=("$session")
        any_running=true
      elif session_failed "$method" "$scenario" "$seed"; then
        log_msg "ERROR: ${session} failed. Check logs under $(session_log_root "$method" "$scenario" "$seed")."
        return 1
      fi
    done
    if [[ "$any_running" == "false" ]]; then
      log_msg "Current wave finished successfully."
      return 0
    fi
    log_msg "Waiting for ${#remaining[@]} sessions: ${remaining[*]}"
    sleep "$POLL_SECONDS"
  done
}

check_stop_requested() {
  if [[ -f "$STOP_FILE" ]]; then
    log_msg "Stop requested; scheduler will not launch more sessions."
    exit 130
  fi
}

start_queue() {
  rm -f "$STOP_FILE"
  printf '%s\n' "$$" > "$SCHEDULER_PID_FILE"
  trap 'rm -f "$SCHEDULER_PID_FILE"' EXIT
  log_msg "Scheduler name: $SCHEDULER_NAME"
  log_msg "Scheduler log: $SCHEDULER_LOG"
  log_msg "Scenarios: ${SCENARIOS[*]}"
  log_msg "Method pairs: ${METHOD_PAIRS[*]}"
  log_msg "Seeds: ${SEEDS[*]}"
  log_msg "GPUs: ${GPUS[*]}"
  log_msg "WandB enabled: ${WANDB_ENABLE}"
  log_msg "Use GCL: ${USE_GCL} (n=${GCL_N_PERCENT}, m=${GCL_M_PERCENT})"
  log_msg "ER reservoir capacity: ${ER_RESERVOIR_CAPACITY}"

  local scenario method_pair methods method seed gpu_idx sessions session
  for scenario in "${SCENARIOS[@]}"; do
    check_stop_requested
    log_msg "========== Scenario ${scenario} =========="
    for method_pair in "${METHOD_PAIRS[@]}"; do
      check_stop_requested
      IFS=',' read -r -a methods <<<"$method_pair"
      sessions=()
      gpu_idx=0
      log_msg "------ Method pair ${method_pair} on scenario ${scenario} ------"
      for method in "${methods[@]}"; do
        if [[ -z "$method" ]]; then continue; fi
        for seed in "${SEEDS[@]}"; do
          if (( gpu_idx >= ${#GPUS[@]} )); then
            echo "ERROR: not enough GPUs for method pair '${method_pair}' and seeds '${SEEDS_STRING}'. Need more than ${#GPUS[@]} slots." >&2
            exit 2
          fi
          set +e
          start_one "$method" "$scenario" "$seed" "${GPUS[$gpu_idx]}"
          start_status=$?
          set -e
          if (( start_status == 0 )); then
            session="$(session_name "$method" "$scenario" "$seed")"
            sessions+=("${session}:${method}:${scenario}:${seed}")
          elif (( start_status == 2 )); then
            :
          else
            return "$start_status"
          fi
          ((gpu_idx += 1))
        done
      done
      check_stop_requested
      if [[ ${#sessions[@]} -eq 0 ]]; then
        log_msg "No sessions to wait for in method pair ${method_pair} on scenario ${scenario}."
      elif [[ "$DRY_RUN" == "true" ]]; then
        log_msg "DRY-RUN would wait for: ${sessions[*]}"
      else
        wait_for_jobs "${sessions[@]}"
      fi
    done
  done
  log_msg "All scheduled baseline waves finished."
}

show_status() {
  local scenario method_pair methods method seed session
  for scenario in "${SCENARIOS[@]}"; do
    echo "========== Scenario ${scenario} =========="
    for method_pair in "${METHOD_PAIRS[@]}"; do
      IFS=',' read -r -a methods <<<"$method_pair"
      for method in "${methods[@]}"; do
        [[ -z "$method" ]] && continue
        for seed in "${SEEDS[@]}"; do
          session="$(session_name "$method" "$scenario" "$seed")"
          if session_exists "$session"; then
            echo "${session}: RUNNING"
          else
            echo "${session}: STOPPED"
          fi
        done
      done
    done
  done
}

stop_sessions() {
  local scenario method_pair methods method seed session scheduler_pid
  mkdir -p "$RUN_LOG_DIR"
  touch "$STOP_FILE"
  if [[ -f "$SCHEDULER_PID_FILE" ]]; then
    scheduler_pid="$(<"$SCHEDULER_PID_FILE")"
    if [[ -n "$scheduler_pid" ]]; then
      kill "$scheduler_pid" 2>/dev/null || true
      echo "Stopped scheduler process ${scheduler_pid}."
    fi
    rm -f "$SCHEDULER_PID_FILE"
  fi
  for scenario in "${SCENARIOS[@]}"; do
    for method_pair in "${METHOD_PAIRS[@]}"; do
      IFS=',' read -r -a methods <<<"$method_pair"
      for method in "${methods[@]}"; do
        [[ -z "$method" ]] && continue
        for seed in "${SEEDS[@]}"; do
          session="$(session_name "$method" "$scenario" "$seed")"
          if session_exists "$session"; then screen -S "$session" -X quit; echo "Stopped ${session}."; fi
        done
      done
    done
  done
}

clean_incomplete_outputs() {
  local scenario method_pair methods method seed task dir deleted=0
  for scenario in "${SCENARIOS[@]}"; do
    for method_pair in "${METHOD_PAIRS[@]}"; do
      IFS=',' read -r -a methods <<<"$method_pair"
      for method in "${methods[@]}"; do
        [[ -z "$method" ]] && continue
        for seed in "${SEEDS[@]}"; do
          for ((task = TASK_START; task <= TASK_END; task++)); do
            dir="$(job_dir_for_task "$method" "$scenario" "$seed" "$task")"
            [[ -d "$dir" ]] || continue
            if [[ -f "$dir/multitask_eval_info.json" ]]; then
              log_msg "Preserve complete output: ${dir}"
              continue
            fi
            if [[ "$dir" != "$CHECKPOINT_ROOT"/* ]]; then
              echo "ERROR: refusing to delete path outside checkpoint root: $dir" >&2
              exit 8
            fi
            if [[ "$dir" != *"/seed_${seed}/"* || "$dir" != *"_task_${task}_"* || "$dir" != *"_${JOB_SUFFIX}" ]]; then
              echo "ERROR: refusing to delete unexpected output path: $dir" >&2
              exit 8
            fi
            if [[ "$DRY_RUN" == "true" ]]; then
              log_msg "DRY-RUN delete incomplete output: ${dir}"
            else
              log_msg "Delete incomplete output: ${dir}"
              rm -rf -- "$dir"
            fi
            deleted=$((deleted + 1))
          done
        done
      done
    done
  done
  log_msg "Incomplete output cleanup done. Matched ${deleted} directories."
}

case "$ACTION" in
  start) start_queue ;;
  status) show_status ;;
  stop) stop_sessions ;;
  clean-incomplete) clean_incomplete_outputs ;;
  *) echo "Unknown action: $ACTION" >&2; usage >&2; exit 2 ;;
esac
