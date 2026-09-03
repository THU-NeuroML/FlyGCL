#!/usr/bin/env bash
METHOD_NAME="seqfft" \
RUN_SEED_SCRIPT="seqfft_run_seed.sh" \
DEFAULT_JOB_PREFIX="seqfft" \
DEFAULT_SESSION_PREFIX="seqfft" \
DEFAULT_CHECKPOINT_BASE="./outputs/seqfft" \
DEFAULT_LOG_ROOT="./logs/seqfft" \
DEFAULT_POLICY_PATH="${POLICY_PATH:-./models/dit_flow_mt_libero_90_pretrain}" \
"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/baseline_screen_template.sh" "$@"
