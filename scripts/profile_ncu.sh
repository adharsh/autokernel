#!/usr/bin/env bash
set -euo pipefail

# Run the required Nsight Compute profile for one AutoKernel experiment.
#
# Usage:
#   scripts/profile_ncu.sh a0/1
#   scripts/profile_ncu.sh a0/1 uv run python validate.py
#
# Reports are written under results/experiments/<experiment_id>/ncu/ with slashes
# normalized to underscores. The raw validate.py timing pass remains the source
# for candidate_us; this profile pass is for design evidence and speed-of-light
# analysis.

usage() {
  cat <<'EOF'
Usage: scripts/profile_ncu.sh <experiment_id> [command...]

Runs:
  ncu --set full --target-processes all --kernel-name-base demangled ...

If command is omitted, defaults to:
  uv run python validate.py
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

if [ "$#" -lt 1 ]; then
  usage >&2
  exit 2
fi

EXPERIMENT_ID="$1"
shift

if [ "$#" -eq 0 ]; then
  set -- uv run python validate.py
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${AUTOKERNEL_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}
RESULTS_DIR=${AUTOKERNEL_RESULTS_DIR:-"$ROOT/results"}
EXPERIMENTS_DIR=${AUTOKERNEL_EXPERIMENTS_DIR:-"$RESULTS_DIR/experiments"}
SAFE_ID=${EXPERIMENT_ID//\//_}
EXPERIMENT_DIR="$EXPERIMENTS_DIR/$SAFE_ID"
NCU_DIR="$EXPERIMENT_DIR/ncu"
REPORT_BASENAME="$NCU_DIR/profile"
LOG_PATH="$NCU_DIR/profile.log"

mkdir -p "$NCU_DIR"

echo "experiment_dir=$EXPERIMENT_DIR"
echo "ncu_report=${REPORT_BASENAME}.ncu-rep"
echo "ncu_log=$LOG_PATH"

ncu \
  --set full \
  --target-processes all \
  --kernel-name-base demangled \
  --force-overwrite \
  -o "$REPORT_BASENAME" \
  "$@" \
  > "$LOG_PATH" 2>&1
