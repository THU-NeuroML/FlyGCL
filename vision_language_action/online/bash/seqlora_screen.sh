#!/usr/bin/env bash
METHOD_NAME="seqlora" \
RUN_SEED_SCRIPT="seqlora_run_seed.sh" \
DEFAULT_JOB_PREFIX="seqlora" \
DEFAULT_SESSION_PREFIX="seqlora" \
DEFAULT_CHECKPOINT_BASE="./outputs/seqlora" \
DEFAULT_LOG_ROOT="./logs/seqlora" \
DEFAULT_PEFT_CFG_PATH="${PEFT_CFG_PATH:-./peft_lsy/peft_config/seqlora_dit_flow_adapter}" \
DEFAULT_POLICY_PATH="${POLICY_PATH:-./models/dit_flow_mt_libero_90_pretrain}" \
"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/baseline_screen_template.sh" "$@"
