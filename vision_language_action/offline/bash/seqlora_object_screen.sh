#!/usr/bin/env bash
METHOD_NAME="seqlora" SCREEN_SCRIPT="seqlora_screen.sh" SCENARIO_NAME="object" DEFAULT_JOB_PREFIX="seqlora" DEFAULT_SESSION_PREFIX="seqlora-object" "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/baseline_scenario_wrapper_template.sh" "$@"
