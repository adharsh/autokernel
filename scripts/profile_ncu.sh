#!/usr/bin/env bash
set -euo pipefail

# Run an explicit Nsight Compute profile for one AutoKernel experiment.
#
# Usage:
#   scripts/profile_ncu.sh a0/1 basic
#   scripts/profile_ncu.sh a0/1 detailed
#   scripts/profile_ncu.sh a0/1 full
#
# Reports are written under results/experiments/<experiment_id>/ncu/ with slashes
# normalized to underscores.
#
# The profile set is required. Agents must explicitly choose one of:
#   basic    official per-experiment timing profile; required before recording
#   detailed supplemental analysis; required before non-baseline keep rows
#   full     expensive deep analysis for stalls, instruction mix, and codegen
#
# The canonical TSV timing source is ncu/details.txt, produced by the basic
# profile. detailed/full profiles are written under ncu/detailed/ and ncu/full/
# so they do not overwrite the official basic timing profile.

usage() {
  cat <<'EOF'
Usage: scripts/profile_ncu.sh <experiment_id> <basic|detailed|full> [command...]

Runs:
  ncu --set <basic|detailed|full> --target-processes all --kernel-name-base demangled ...

The NCU set argument is mandatory. The agent must choose it explicitly for each
profile command; there is no implicit default.

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

if [ "$#" -lt 2 ]; then
  usage >&2
  exit 2
fi

EXPERIMENT_ID="$1"
NCU_SET="$2"
shift 2

case "$NCU_SET" in
  basic|detailed|full)
    ;;
  *)
    echo "Invalid NCU set '$NCU_SET'. Choose exactly one of: basic, detailed, full." >&2
    usage >&2
    exit 2
    ;;
esac

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
if [ "$NCU_SET" = "basic" ]; then
  REPORT_DIR="$NCU_DIR"
else
  REPORT_DIR="$NCU_DIR/$NCU_SET"
fi
REPORT_BASENAME="$REPORT_DIR/profile"
LOG_PATH="$REPORT_DIR/profile.log"
DETAILS_PATH="$REPORT_DIR/details.txt"
PROFILE_FROM_START=${AUTOKERNEL_NCU_PROFILE_FROM_START:-}

if [ "$#" -eq 0 ]; then
  set -- uv run python "$TARGET_ROOT/scripts/profile_candidate_once.py"
  PROFILE_FROM_START=${PROFILE_FROM_START:-off}
fi

mkdir -p "$REPORT_DIR"

echo "experiment_dir=$EXPERIMENT_DIR"
echo "ncu_set=$NCU_SET"
echo "ncu_report=${REPORT_BASENAME}.ncu-rep"
echo "ncu_log=$LOG_PATH"
echo "ncu_details=$DETAILS_PATH"
if [ "$NCU_SET" != "basic" ]; then
  echo "ncu_supplemental=1"
  echo "record_result.py reads $NCU_DIR/details.txt, so run the basic profile before recording."
fi

NCU_ARGS=(
  --set "$NCU_SET"
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
