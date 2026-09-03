#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 DATA_ROOT OUTPUT_DIR [SEED]" >&2
  exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
data_root="$1"
output_dir="$2"
seed="${3:-0}"
cd "$repo_dir"
python main_gcl.py --config-name cifar100/flygcl \
  dataset_root="$data_root" seed="$seed" stream_seed="$seed" \
  hydra.run.dir="$output_dir/seed_$seed"

