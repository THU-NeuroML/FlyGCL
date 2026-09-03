#!/usr/bin/env bash
set -euo pipefail

SCREEN_SCRIPT="${SCREEN_SCRIPT:?SCREEN_SCRIPT is required}"
METHOD_NAME="${METHOD_NAME:?METHOD_NAME is required}"
SCENARIO_NAME="${SCENARIO_NAME:?SCENARIO_NAME is required}"
DEFAULT_POLICY_PATH="${DEFAULT_POLICY_PATH:-./models/dit_flow_mt_libero_90_pretrain}"
DEFAULT_JOB_PREFIX="${DEFAULT_JOB_PREFIX:-$METHOD_NAME}"
DEFAULT_SESSION_PREFIX="${DEFAULT_SESSION_PREFIX:-$METHOD_NAME-$SCENARIO_NAME}"

case "$SCENARIO_NAME" in
  goal)
    BENCHMARK_NAME="libero_goal"
    TASK_PREFIX="Libero_Goal_Task"
    DATASET_PREFIX="continuallearning/libero_goal_image_task"
    ;;
  object)
    BENCHMARK_NAME="libero_object"
    TASK_PREFIX="Libero_Object_Task"
    DATASET_PREFIX="continuallearning/libero_object_image_task"
    ;;
  spatial)
    BENCHMARK_NAME="libero_spatial"
    TASK_PREFIX="Libero_Spatial_Task"
    DATASET_PREFIX="continuallearning/libero_spatial_image_task"
    ;;
  *)
    echo "ERROR: unknown scenario: $SCENARIO_NAME" >&2
    exit 2
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${REPO_ROOT}/bash/${SCREEN_SCRIPT}" "$1" \
  --checkpoint-base="${CHECKPOINT_BASE:-./outputs/${METHOD_NAME}/${BENCHMARK_NAME}}" \
  --log-root="${LOG_ROOT:-./logs/${METHOD_NAME}/${BENCHMARK_NAME}}" \
  --policy-path="${POLICY_PATH:-$DEFAULT_POLICY_PATH}" \
  --job-prefix="${JOB_PREFIX:-$DEFAULT_JOB_PREFIX}" \
  --session-prefix="${SESSION_PREFIX:-$DEFAULT_SESSION_PREFIX}" \
  "${@:2}" \
  -- \
  --benchmark-name="${BENCHMARK_NAME}" \
  --task-prefix="${TASK_PREFIX}" \
  --dataset-prefix="${DATASET_PREFIX}"
