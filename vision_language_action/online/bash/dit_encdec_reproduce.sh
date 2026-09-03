#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

exec bash "${REPO_ROOT}/bash/dit_encdec_run_seed.sh" \
  --seed="${SEED:-42}" \
  --job-prefix="${JOB_PREFIX:-dit_mt_cl}" \
  --task-start="${TASK_START:-0}" \
  --task-end="${TASK_END:-9}" \
  --job-suffix="${JOB_SUFFIX:-reproduce}" \
  --source-job-suffix="${SOURCE_JOB_SUFFIX:-${JOB_SUFFIX:-reproduce}}" \
  --checkpoint-root="${CHECKPOINT_ROOT:-./outputs}" \
  --source-checkpoint-root="${SOURCE_CHECKPOINT_ROOT:-${CHECKPOINT_ROOT:-./outputs}}" \
  --log-root="${LOG_ROOT:-./logs/dit_encdec}" \
  --policy-path="${POLICY_PATH:-./models/dit_mt_libero_90_pretrain}" \
  --peft-cfg-path="${PEFT_CFG_PATH:-./peft_lsy/peft_config/clare_dit_mt_encoder_adapter}" \
  --expand-threshold="${EXPAND_THRESHOLD:-2.5}" \
  --wandb-enable="${WANDB_ENABLE:-false}" \
  --wandb-project="${WANDB_PROJECT:-clare_experiments}" \
  --wandb-entity="${WANDB_ENTITY:-}" \
  "$@"
