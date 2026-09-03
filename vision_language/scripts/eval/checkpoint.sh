#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "Usage: $0 CONFIG CHECKPOINT DATA_ROOT OUTPUT_JSON [DEVICE]" >&2
  exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_dir"
python tools/evaluate_checkpoint.py --config "$1" --checkpoint "$2" \
  --data-root "$3" --output "$4" --device "${5:-cuda}"

