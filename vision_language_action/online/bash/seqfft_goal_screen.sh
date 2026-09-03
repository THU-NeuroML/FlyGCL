#!/usr/bin/env bash
METHOD_NAME="seqfft" SCREEN_SCRIPT="seqfft_screen.sh" SCENARIO_NAME="goal" DEFAULT_JOB_PREFIX="seqfft" DEFAULT_SESSION_PREFIX="seqfft-goal" "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/baseline_scenario_wrapper_template.sh" "$@"
