#!/usr/bin/env bash
METHOD_NAME="packnet" SCREEN_SCRIPT="packnet_screen.sh" SCENARIO_NAME="object" DEFAULT_JOB_PREFIX="packnet" DEFAULT_SESSION_PREFIX="packnet-object" "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/baseline_scenario_wrapper_template.sh" "$@"
