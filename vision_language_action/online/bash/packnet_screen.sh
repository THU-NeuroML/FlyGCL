#!/usr/bin/env bash
METHOD_NAME="packnet" \
RUN_SEED_SCRIPT="packnet_run_seed.sh" \
DEFAULT_JOB_PREFIX="packnet" \
DEFAULT_SESSION_PREFIX="packnet" \
DEFAULT_CHECKPOINT_BASE="./outputs/packnet" \
DEFAULT_LOG_ROOT="./logs/packnet" \
DEFAULT_POLICY_PATH="${POLICY_PATH:-./models/dit_flow_mt_libero_90_pretrain}" \
"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/baseline_screen_template.sh" "$@"
