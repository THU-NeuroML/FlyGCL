#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export BENCHMARK_NAME="${BENCHMARK_NAME:-libero_object}"
export TASK_PREFIX="${TASK_PREFIX:-Libero_Object_Task}"
export DATASET_PREFIX="${DATASET_PREFIX:-continuallearning/libero_object_image_task}"
export MAX_TASK="${MAX_TASK:-9}"

if [[ $# -lt 1 ]]; then
  exec bash "${REPO_ROOT}/bash/dit_dec_screen.sh"
fi

exec bash "${REPO_ROOT}/bash/dit_dec_screen.sh" \
  "$1" \
  --checkpoint-base="${CHECKPOINT_BASE:-./outputs/dit_dec/libero_object}" \
  --log-root="${LOG_ROOT:-./logs/dit_dec/libero_object}" \
  --policy-path="${POLICY_PATH:-./models/dit_flow_mt_libero_90_pretrain}" \
  --job-prefix="${JOB_PREFIX:-dit_flow_mt_cl}" \
  --peft-cfg-path="${PEFT_CFG_PATH:-./peft_lsy/peft_config/clare_dit_flow_encoder_adapter}" \
  --session-prefix="${SESSION_PREFIX:-dit-dec-object}" \
  "${@:2}"
