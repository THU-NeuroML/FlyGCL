#!/usr/bin/env bash
METHOD=seqfft exec "$(dirname "$0")/baseline_gcl_run_seed.sh" "$@"
