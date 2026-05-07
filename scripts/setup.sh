#!/usr/bin/env bash
set -euo pipefail

# Set up local AutoKernel scaffolding.
#
# Handles repeatable repo-local setup:
# - optional uv sync
# - Claude/Codex microbench discovery links
# - root validate.py/reference.py from templates when missing
# - results/experiments.tsv and results/experiments/ initialization
# - calibrated reference timing reminder
# - optional lightweight environment checks
#
# It intentionally does not fill in task-specific benchmark logic.

cd "$(dirname "$0")/.."
ROOT=$(pwd)
TEMPLATE_DIR="$ROOT/docs/templates"
RESULTS_DIR="$ROOT/results"
EXPERIMENTS_DIR="$RESULTS_DIR/experiments"
TSV="$RESULTS_DIR/experiments.tsv"
TSV_HEADER='experiment_id	parent_id	agent_id	commit	timestamp	candidate_us	reference_us	speedup	correctness	peak_vram_mb	status	description'
CODEX_HOME=${CODEX_HOME:-"$HOME/.codex"}

DO_SYNC=0
DO_VERIFY=0
FORCE=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: ./scripts/setup.sh [--sync] [--verify] [--force] [--dry-run]

Options:
  --sync     Run uv python install 3.10 and uv sync.
  --verify   Run lightweight CUDA/Python dependency checks.
  --force    Replace existing symlinks/templates.
  --dry-run  Print actions without changing files.
  -h, --help Show this help.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --sync) DO_SYNC=1 ;;
    --verify) DO_VERIFY=1 ;;
    --force) FORCE=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [ "$DRY_RUN" -eq 0 ]; then
    "$@"
  fi
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "$1 is not on PATH." >&2
    return 1
  fi
}

ensure_symlink() {
  local link="$1"
  local target="$2"

  if [ -e "$link" ] || [ -L "$link" ]; then
    if [ -L "$link" ] && [ "$(readlink "$link")" = "$target" ]; then
      echo "ok: $link"
      return
    fi

    if [ "$FORCE" -eq 0 ]; then
      echo "skip: $link already exists; use --force to replace it"
      return
    fi

    echo "replace: $link"
    if [ "$DRY_RUN" -eq 0 ]; then
      rm -f "$link"
    fi
  fi

  echo "link: $link -> $target"
  if [ "$DRY_RUN" -eq 0 ]; then
    mkdir -p "$(dirname "$link")"
    ln -s "$target" "$link"
  fi
}

copy_template() {
  local name="$1"
  local src="$TEMPLATE_DIR/$name"
  local dst="$ROOT/$name"

  if [ -e "$dst" ]; then
    if [ "$FORCE" -eq 0 ]; then
      echo "skip: $dst already exists; use --force to replace it"
      return
    fi
    echo "replace: $dst"
  fi

  echo "copy: $src -> $dst"
  if [ "$DRY_RUN" -eq 0 ]; then
    cp "$src" "$dst"
  fi
}

setup_microbench_links() {
  ensure_symlink "$ROOT/.claude/agents/microbench.md" "../../agents/microbench.md"
  ensure_symlink "$CODEX_HOME/skills/microbench/SKILL.md" "$ROOT/agents/microbench.md"
}

init_results_tsv() {
  if [ -s "$TSV" ]; then
    echo "ok: $TSV"
    return
  fi

  echo "init: $TSV"
  if [ "$DRY_RUN" -eq 0 ]; then
    mkdir -p "$RESULTS_DIR" "$EXPERIMENTS_DIR"
    printf "%s\n" "$TSV_HEADER" > "$TSV"
  fi
}

init_experiments_dir() {
  echo "ok: $EXPERIMENTS_DIR"
  if [ "$DRY_RUN" -eq 0 ]; then
    mkdir -p "$EXPERIMENTS_DIR"
  fi
}

run_sync() {
  require_command uv || {
    echo "Install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
  }

  run uv python install 3.10
  run uv sync
}

run_checks() {
  local missing=()
  local cmd

  if ! command -v uv >/dev/null 2>&1; then
    if [ "$DRY_RUN" -eq 0 ]; then
      echo "uv is not on PATH. Install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
      exit 1
    fi
    missing+=("uv")
  fi

  for cmd in nvidia-smi ncu nsys; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
      missing+=("$cmd")
    fi
  done

  if [ "${#missing[@]}" -gt 0 ]; then
    echo "missing tools: ${missing[*]}"
  else
    echo "ok: nvidia-smi, ncu, nsys are on PATH"
  fi

  run uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
  run uv run python -c "import triton; print('triton', triton.__version__)"
  run uv run python -c "import cutlass_library, cutlass_cppgen; print('cutlass ok')"
  run uv run python -c "from cuda.bindings import nvrtc; print('cuda-python nvrtc ok')"
}

print_next_steps() {
  cat <<'EOF'

Next steps:
1. Use an AI coding tool or edit by hand to fill root reference.py with
   the trusted implementation.
2. Use an AI coding tool or edit by hand to fill root validate.py with
   exactly one stress benchmark case plus separate correctness-only edge cases.
3. Temporarily make candidate/interface.py call reference.kernel_fn, then run:
   uv run python scripts/calibrate_reference.py
   uv run python validate.py
4. Revert candidate/interface.py to the optimization entry point.
5. Commit validate.py, reference.py, and any task setup so worktrees can see them.
6. Confirm Nsight Compute profiling works:
   scripts/profile_ncu.sh smoke-test
7. Restart Codex/Claude so microbench is discovered.
8. Launch agents:
   ./scripts/agents.sh start
EOF
}

if [ "$DO_SYNC" -eq 1 ]; then
  run_sync
fi

setup_microbench_links
copy_template validate.py
copy_template reference.py
init_results_tsv
init_experiments_dir

if [ "$DO_VERIFY" -eq 1 ]; then
  run_checks
fi

print_next_steps
