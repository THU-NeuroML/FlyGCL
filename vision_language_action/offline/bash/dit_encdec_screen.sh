#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash/dit_encdec_screen.sh <start|status|attach|stop> [options] [-- extra launcher args...]

Commands:
  start    Launch one detached screen session per seed.
  status   Show whether each configured screen session is running.
  attach   Attach to the screen session for a single seed.
  stop     Stop the configured screen sessions.

Options:
  --seeds "42 43 44"             Space-separated seed list. Default: "42 43 44"
  --gpus "0 1 2"                Space-separated GPU list. Default: "0 1 2"
  --task-start INT               First task to run. Default: 0
  --task-end INT                 Last task to run. Default: 9
  --job-suffix STR               Output job suffix. Default: reproduce
  --source-job-suffix STR        Suffix used to locate the predecessor adapter for the
                                 first task in the range. Defaults to --job-suffix.
  --checkpoint-base PATH         Base directory for per-seed outputs.
                                 Default: ./outputs/dit_encdec
  --source-checkpoint-base PATH  Base directory for predecessor lookup.
                                 Defaults to --checkpoint-base.
  --log-root PATH                Base log root. Default: ./logs/dit_encdec
  --policy-path PATH             Base pretrained policy path.
  --job-prefix STR               Job prefix forwarded to the seed launcher.
  --peft-cfg-path PATH           CLARE adapter config forwarded to the seed launcher.
  --session-prefix STR           Screen session prefix. Default: dit-encdec
  --seed INT                     Required by attach to select one session.
  --help                         Show this message.

Any arguments after '--' are forwarded to bash/dit_encdec_run_seed.sh.
EOF
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

ACTION="$1"
shift

SEEDS_STRING="${SEEDS_STRING:-42 43 44}"
GPUS_STRING="${GPUS_STRING:-0 1 2}"
TASK_START="${TASK_START:-0}"
TASK_END="${TASK_END:-9}"
JOB_SUFFIX="${JOB_SUFFIX:-reproduce}"
SOURCE_JOB_SUFFIX="${SOURCE_JOB_SUFFIX:-}"
CHECKPOINT_BASE="${CHECKPOINT_BASE:-./outputs/dit_encdec}"
SOURCE_CHECKPOINT_BASE="${SOURCE_CHECKPOINT_BASE:-}"
LOG_ROOT="${LOG_ROOT:-./logs/dit_encdec}"
POLICY_PATH="${POLICY_PATH:-./models/dit_mt_libero_90_pretrain}"
JOB_PREFIX="${JOB_PREFIX:-dit_mt_cl}"
PEFT_CFG_PATH="${PEFT_CFG_PATH:-./peft_lsy/peft_config/clare_dit_mt_encoder_adapter}"
SESSION_PREFIX="${SESSION_PREFIX:-dit-encdec}"
ATTACH_SEED="${ATTACH_SEED:-}"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds)
      SEEDS_STRING="$2"
      shift 2
      ;;
    --seeds=*)
      SEEDS_STRING="${1#*=}"
      shift
      ;;
    --gpus)
      GPUS_STRING="$2"
      shift 2
      ;;
    --gpus=*)
      GPUS_STRING="${1#*=}"
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
    --checkpoint-base)
      CHECKPOINT_BASE="$2"
      shift 2
      ;;
    --checkpoint-base=*)
      CHECKPOINT_BASE="${1#*=}"
      shift
      ;;
    --source-checkpoint-base)
      SOURCE_CHECKPOINT_BASE="$2"
      shift 2
      ;;
    --source-checkpoint-base=*)
      SOURCE_CHECKPOINT_BASE="${1#*=}"
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
    --job-prefix)
      JOB_PREFIX="$2"
      shift 2
      ;;
    --job-prefix=*)
      JOB_PREFIX="${1#*=}"
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
    --session-prefix)
      SESSION_PREFIX="$2"
      shift 2
      ;;
    --session-prefix=*)
      SESSION_PREFIX="${1#*=}"
      shift
      ;;
    --seed)
      ATTACH_SEED="$2"
      shift 2
      ;;
    --seed=*)
      ATTACH_SEED="${1#*=}"
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

if [[ -z "$SOURCE_CHECKPOINT_BASE" ]]; then
  SOURCE_CHECKPOINT_BASE="$CHECKPOINT_BASE"
fi

if ! command -v screen >/dev/null 2>&1; then
  echo "ERROR: screen is not installed or not on PATH." >&2
  exit 3
fi

read -r -a SEEDS <<<"$SEEDS_STRING"
read -r -a GPUS <<<"$GPUS_STRING"

if [[ ${#SEEDS[@]} -eq 0 ]]; then
  echo "ERROR: no seeds configured." >&2
  exit 2
fi

if [[ ${#SEEDS[@]} -ne ${#GPUS[@]} ]]; then
  echo "ERROR: seeds and gpus must have the same length." >&2
  exit 2
fi

session_name() {
  printf "%s-s%s" "$SESSION_PREFIX" "$1"
}

session_exists() {
  local session="$1"
  screen -list | grep -q "[.]${session}[[:space:]]"
}

seed_checkpoint_root() {
  printf "%s/seed_%s" "$CHECKPOINT_BASE" "$1"
}

seed_source_checkpoint_root() {
  printf "%s/seed_%s" "$SOURCE_CHECKPOINT_BASE" "$1"
}

start_sessions() {
  local idx=""
  for idx in "${!SEEDS[@]}"; do
    local seed="${SEEDS[$idx]}"
    local gpu="${GPUS[$idx]}"
    local session
    session="$(session_name "$seed")"

    if session_exists "$session"; then
      echo "Skipping ${session}: already running."
      continue
    fi

    local cmd=(
      bash
      "${REPO_ROOT}/bash/dit_encdec_run_seed.sh"
      "--seed=${seed}"
      "--gpu-id=${gpu}"
      "--job-prefix=${JOB_PREFIX}"
      "--task-start=${TASK_START}"
      "--task-end=${TASK_END}"
      "--job-suffix=${JOB_SUFFIX}"
      "--source-job-suffix=${SOURCE_JOB_SUFFIX}"
      "--checkpoint-root=$(seed_checkpoint_root "$seed")"
      "--source-checkpoint-root=$(seed_source_checkpoint_root "$seed")"
      "--log-root=${LOG_ROOT}"
      "--policy-path=${POLICY_PATH}"
      "--peft-cfg-path=${PEFT_CFG_PATH}"
    )

    if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
      cmd+=("${EXTRA_ARGS[@]}")
    fi

    local cmd_string=""
    printf -v cmd_string '%q ' "${cmd[@]}"

    screen -dmS "$session" bash -lc "$cmd_string"
    sleep 1
    if session_exists "$session"; then
      echo "Started ${session} on GPU ${gpu} for seed ${seed}."
    else
      echo "Failed to keep ${session} alive for seed ${seed}. Check logs under ${LOG_ROOT}/seed_${seed}/." >&2
    fi
  done
}

show_status() {
  local idx=""
  for idx in "${!SEEDS[@]}"; do
    local seed="${SEEDS[$idx]}"
    local gpu="${GPUS[$idx]}"
    local session
    session="$(session_name "$seed")"

    if session_exists "$session"; then
      echo "${session}: RUNNING (seed=${seed}, gpu=${gpu}, output_root=$(seed_checkpoint_root "$seed"))"
    else
      echo "${session}: STOPPED (seed=${seed}, gpu=${gpu}, output_root=$(seed_checkpoint_root "$seed"))"
    fi
  done
}

attach_session() {
  if [[ -z "$ATTACH_SEED" ]]; then
    echo "ERROR: attach requires --seed." >&2
    exit 2
  fi

  local session
  session="$(session_name "$ATTACH_SEED")"
  if ! session_exists "$session"; then
    echo "ERROR: session not running: ${session}" >&2
    exit 4
  fi

  exec screen -r "$session"
}

stop_sessions() {
  local idx=""
  for idx in "${!SEEDS[@]}"; do
    local seed="${SEEDS[$idx]}"
    local session
    session="$(session_name "$seed")"

    if session_exists "$session"; then
      screen -S "$session" -X quit
      echo "Stopped ${session}."
    else
      echo "Skipping ${session}: not running."
    fi
  done
}

case "$ACTION" in
  start)
    start_sessions
    ;;
  status)
    show_status
    ;;
  attach)
    attach_session
    ;;
  stop)
    stop_sessions
    ;;
  *)
    echo "Unknown action: $ACTION" >&2
    usage >&2
    exit 2
    ;;
esac
