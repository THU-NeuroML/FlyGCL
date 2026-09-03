#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export BENCHMARK_NAME="${BENCHMARK_NAME:-libero_object}"
export TASK_PREFIX="${TASK_PREFIX:-Libero_Object_Task}"
export DATASET_PREFIX="${DATASET_PREFIX:-continuallearning/libero_object_image_task}"
export MAX_TASK="${MAX_TASK:-9}"

exec bash "${REPO_ROOT}/bash/dit_dec_reproduce.sh" \
  --seed="${SEED:-42}" \
  --job-prefix="${JOB_PREFIX:-dit_flow_mt_cl}" \
  --task-start="${TASK_START:-0}" \
  --task-end="${TASK_END:-9}" \
  --job-suffix="${JOB_SUFFIX:-reproduce}" \
  --source-job-suffix="${SOURCE_JOB_SUFFIX:-${JOB_SUFFIX:-reproduce}}" \
  --checkpoint-root="${CHECKPOINT_ROOT:-./outputs/dit_dec/libero_object}" \
  --source-checkpoint-root="${SOURCE_CHECKPOINT_ROOT:-${CHECKPOINT_ROOT:-./outputs/dit_dec/libero_object}}" \
  --log-root="${LOG_ROOT:-./logs/dit_dec/libero_object}" \
  --policy-path="${POLICY_PATH:-./models/dit_flow_mt_libero_90_pretrain}" \
  --peft-cfg-path="${PEFT_CFG_PATH:-./peft_lsy/peft_config/clare_dit_flow_encoder_adapter}" \
  --wandb-enable="${WANDB_ENABLE:-false}" \
  --wandb-project="${WANDB_PROJECT:-clare_experiments}" \
  --wandb-entity="${WANDB_ENTITY:-}" \
  "$@"
