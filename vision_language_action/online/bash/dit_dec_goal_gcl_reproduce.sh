#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export BENCHMARK_NAME="${BENCHMARK_NAME:-libero_goal}"
export TASK_PREFIX="${TASK_PREFIX:-Libero_Goal_Task}"
export DATASET_PREFIX="${DATASET_PREFIX:-continuallearning/libero_goal_image_task}"
export MAX_TASK="${MAX_TASK:-9}"

exec bash "${REPO_ROOT}/bash/dit_dec_gcl_run_seed.sh" \
  --seed="${SEED:-42}" \
  --job-prefix="${JOB_PREFIX:-dit_flow_mt_cl_gcl}" \
  --job-suffix="${JOB_SUFFIX:-reproduce}" \
  --checkpoint-root="${CHECKPOINT_ROOT:-./outputs/dit_dec_gcl/libero_goal}" \
  --log-root="${LOG_ROOT:-./logs/dit_dec_gcl/libero_goal}" \
  --policy-path="${POLICY_PATH:-./models/dit_flow_mt_libero_90_pretrain}" \
  --peft-cfg-path="${PEFT_CFG_PATH:-./peft_lsy/peft_config/clare_dit_flow_encoder_adapter}" \
  --gcl-n-percent="${GCL_N_PERCENT:-50}" \
  --gcl-m-percent="${GCL_M_PERCENT:-30}" \
  --wandb-enable="${WANDB_ENABLE:-false}" \
  --wandb-project="${WANDB_PROJECT:-clare_experiments}" \
  --wandb-entity="${WANDB_ENTITY:-}" \
  "$@"
