#!/usr/bin/env bash
set -euo pipefail

# Run the required Nsight Compute profile for one AutoKernel experiment.
#
# Usage:
#   scripts/profile_ncu.sh a0/1
#   scripts/profile_ncu.sh a0/1 uv run python validate.py
#
# Reports are written under results/experiments/<experiment_id>/ncu/ with slashes
# normalized to underscores. The NCU kernel Duration rows are the source for
# ncu_duration_us and ncu_kernel_count in results/experiments.tsv.

usage() {
  cat <<'EOF'
Usage: scripts/profile_ncu.sh <experiment_id> [command...]

Runs:
  ncu --set full --target-processes all --kernel-name-base demangled ...

If command is omitted, defaults to:
  uv run python scripts/profile_candidate_once.py

The default target warms up the candidate and uses CUDA profiler markers so
Nsight Compute profiles one candidate invocation from the current worktree, not
the full validation loop.
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

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${AUTOKERNEL_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}
TARGET_ROOT=${AUTOKERNEL_PROFILE_TARGET_ROOT:-$PWD}
if [ ! -f "$TARGET_ROOT/validate.py" ]; then
  TARGET_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
fi
RESULTS_DIR=${AUTOKERNEL_RESULTS_DIR:-"$ROOT/results"}
EXPERIMENTS_DIR=${AUTOKERNEL_EXPERIMENTS_DIR:-"$RESULTS_DIR/experiments"}
SAFE_ID=${EXPERIMENT_ID//\//_}
EXPERIMENT_DIR="$EXPERIMENTS_DIR/$SAFE_ID"
NCU_DIR="$EXPERIMENT_DIR/ncu"
REPORT_BASENAME="$NCU_DIR/profile"
LOG_PATH="$NCU_DIR/profile.log"
DETAILS_PATH="$NCU_DIR/details.txt"
PROFILE_FROM_START=${AUTOKERNEL_NCU_PROFILE_FROM_START:-}

if [ "$#" -eq 0 ]; then
  set -- uv run python "$TARGET_ROOT/scripts/profile_candidate_once.py"
  PROFILE_FROM_START=${PROFILE_FROM_START:-off}
fi

mkdir -p "$NCU_DIR"

echo "experiment_dir=$EXPERIMENT_DIR"
echo "ncu_report=${REPORT_BASENAME}.ncu-rep"
echo "ncu_log=$LOG_PATH"
echo "ncu_details=$DETAILS_PATH"

NCU_ARGS=(
  --set full
  --target-processes all
  --kernel-name-base demangled
  --force-overwrite
  -o "$REPORT_BASENAME"
)

if [ -n "$PROFILE_FROM_START" ]; then
  NCU_ARGS+=(--profile-from-start "$PROFILE_FROM_START")
fi

ncu \
  "${NCU_ARGS[@]}" \
  "$@" \
  > "$LOG_PATH" 2>&1

ncu --import "${REPORT_BASENAME}.ncu-rep" --page details > "$DETAILS_PATH"
