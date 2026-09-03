#!/usr/bin/env bash
METHOD_NAME="er" SCREEN_SCRIPT="er_screen.sh" SCENARIO_NAME="spatial" DEFAULT_JOB_PREFIX="er" DEFAULT_SESSION_PREFIX="er-spatial" "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/baseline_scenario_wrapper_template.sh" "$@"
