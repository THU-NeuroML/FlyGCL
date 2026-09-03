#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 DATA_ROOT OUTPUT_DIR" >&2
  exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
for seed in 0 1 2 42 3407; do
  "$repo_dir/scripts/train/imagenet_r.sh" "$1" "$2" "$seed"
done
python "$repo_dir/tools/summarize_results.py" "$2" \
  --csv "$2/summary.csv" --json "$2/summary.json"

