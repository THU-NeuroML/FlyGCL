#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

ACTION="${1:-start}"
if [[ $# -gt 0 ]]; then shift; fi

SEEDS_STRING="${SEEDS_STRING:-42 43 44}"
GPUS_STRING="${GPUS_STRING:-0 1 2}"
METHOD_PAIRS_STRING="${METHOD_PAIRS_STRING:-l2p_adapter dualprompt_adapter}"
SCENARIOS_STRING="${SCENARIOS_STRING:-10 goal spatial object}"
SCHEDULER_NAME="${SCHEDULER_NAME:-l2p_dualprompt}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-./artifacts/outputs/l2p_dualprompt_wave}"
LOG_ROOT="${LOG_ROOT:-./artifacts/logs/l2p_dualprompt_wave}"
JOB_SUFFIX="${JOB_SUFFIX:-reproduce}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-baseline_experiments}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
RESUME="${RESUME:-true}"
POLICY_PATH="${POLICY_PATH:-./models/dit_flow_mt_libero_90_pretrain}"
POLL_SECONDS="${POLL_SECONDS:-120}"

exec bash "${REPO_ROOT}/run_baseline_wave.sh" "$ACTION" \
  --seeds "$SEEDS_STRING" \
  --gpus "$GPUS_STRING" \
  --method-pairs "$METHOD_PAIRS_STRING" \
  --scenarios "$SCENARIOS_STRING" \
  --scheduler-name "$SCHEDULER_NAME" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --log-root "$LOG_ROOT" \
  --job-suffix "$JOB_SUFFIX" \
  --wandb-enable "$WANDB_ENABLE" \
  --wandb-project "$WANDB_PROJECT" \
  --wandb-entity "$WANDB_ENTITY" \
  --resume "$RESUME" \
  --policy-path "$POLICY_PATH" \
  --poll-seconds "$POLL_SECONDS" \
  "$@"
