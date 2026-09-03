#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

usage() {
  cat <<'EOF'
Usage:
  bash bash/summarize_cl_metrics.sh [options]

Default behavior:
  - scan all results under ./outputs
  - summarize all discovered methods / benchmarks / seeds

Optional narrowing:
  --root PATH            Add a root directory to scan. Can be repeated.
  --scenario-root PATH   Alias of --root for convenience. Can be repeated.
  --help                 Show this message.

Examples:
  bash bash/summarize_cl_metrics.sh
  bash bash/summarize_cl_metrics.sh --root ./outputs/dit_dec/libero_goal
  bash bash/summarize_cl_metrics.sh --root ./outputs/dit_dec/libero_goal --root ./outputs/flyvla_rpresheadema/libero_goal
EOF
}

if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
  usage
  exit 0
fi

ROOT_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root|--scenario-root)
      ROOT_ARGS+=(--root "$2")
      shift 2
      ;;
    *)
      break
      ;;
  esac
done

if [[ ${#ROOT_ARGS[@]} -eq 0 ]]; then
  if [[ -n "${ROOTS:-}" ]]; then
    read -r -a ROOT_LIST <<<"$ROOTS"
    for root in "${ROOT_LIST[@]}"; do
      ROOT_ARGS+=(--root "$root")
    done
  elif [[ -n "${ROOT:-}" ]]; then
    ROOT_ARGS+=(--root "$ROOT")
  fi
fi

python ./lerobot_lsy/src/lerobot/scripts/summarize_cl_metrics.py   "${ROOT_ARGS[@]}"   --seeds ${SEEDS:-42 43 44}   "$@"
