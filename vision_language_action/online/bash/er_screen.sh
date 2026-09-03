#!/usr/bin/env bash
METHOD_NAME="er" \
RUN_SEED_SCRIPT="er_run_seed.sh" \
DEFAULT_JOB_PREFIX="er" \
DEFAULT_SESSION_PREFIX="er" \
DEFAULT_CHECKPOINT_BASE="./outputs/er" \
DEFAULT_LOG_ROOT="./logs/er" \
DEFAULT_POLICY_PATH="${POLICY_PATH:-./models/dit_flow_mt_libero_90_pretrain}" \
"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/baseline_screen_template.sh" "$@"
