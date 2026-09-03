#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

ER_RESERVOIR_CAPACITY="${ER_RESERVOIR_CAPACITY:-50}"
if [[ ! "$ER_RESERVOIR_CAPACITY" =~ ^[0-9]+$ || "$ER_RESERVOIR_CAPACITY" -lt 1 ]]; then
  echo "ERROR: ER_RESERVOIR_CAPACITY must be a positive integer." >&2
  exit 2
fi

if [[ "$ER_RESERVOIR_CAPACITY" == "50" ]]; then
  SCHEDULER_NAME="${SCHEDULER_NAME:-er_reservoir}"
  JOB_SUFFIX="${JOB_SUFFIX:-reproduce}"
  CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-./outputs/baselines_wave}"
  LOG_ROOT="${LOG_ROOT:-./logs/baselines_wave}"
else
  SCHEDULER_NAME="${SCHEDULER_NAME:-er_reservoir${ER_RESERVOIR_CAPACITY}}"
  JOB_SUFFIX="${JOB_SUFFIX:-reproduce_m${ER_RESERVOIR_CAPACITY}}"
  CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-./outputs/baselines_wave_m${ER_RESERVOIR_CAPACITY}}"
  LOG_ROOT="${LOG_ROOT:-./logs/baselines_wave_m${ER_RESERVOIR_CAPACITY}}"
fi

DRY_RUN_ARGS=()
if [[ "${DRY_RUN:-false}" == "true" ]]; then
  DRY_RUN_ARGS=(--dry-run)
fi

exec bash "$REPO_ROOT/run_baseline_wave.sh" start \
  --scheduler-name "$SCHEDULER_NAME" \
  --method-pairs "er_reservoir" \
  --scenarios "${SCENARIOS:-10 goal spatial object}" \
  --seeds "${SEEDS:-42 43 44}" \
  --gpus "${GPUS:-4 5 6}" \
  --task-start "${TASK_START:-0}" \
  --task-end "${TASK_END:-9}" \
  --job-suffix "$JOB_SUFFIX" \
  --checkpoint-root "$CHECKPOINT_ROOT" \
  --log-root "$LOG_ROOT" \
  --policy-path "${POLICY_PATH:-./models/dit_flow_mt_libero_90_pretrain}" \
  --wandb-enable "${WANDB_ENABLE:-false}" \
  --wandb-project "${WANDB_PROJECT:-baseline_experiments}" \
  --wandb-entity "${WANDB_ENTITY:-}" \
  --resume "${RESUME:-true}" \
  --poll-seconds "${POLL_SECONDS:-120}" \
  --er-reservoir-capacity "$ER_RESERVOIR_CAPACITY" \
  "${DRY_RUN_ARGS[@]}"
