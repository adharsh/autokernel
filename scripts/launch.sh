#!/usr/bin/env bash
set -euo pipefail

# Launch one autonomous kernel-optimization agent per GPU.
# Usage:
#   ./scripts/launch.sh            # fresh start
#   ./scripts/launch.sh --resume   # resume from where agents left off

cd "$(dirname "$0")/.."
ROOT=$(pwd)

RESUME=false
if [ "${1:-}" = "--resume" ]; then
  RESUME=true
  echo "Resuming agents from previous sessions..."
fi

# Kill existing agents if not resuming
if [ "$RESUME" = false ] && [ -f .agent_pids ]; then
  echo "Killing existing agents..."
  xargs kill 2>/dev/null < .agent_pids || true
  sleep 1
fi

NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "Detected $NUM_GPUS GPU(s)"

# Find next free agent prefix by checking existing a{N}/* branches
MAX_PREFIX=$(git branch --list 'a*/*' | sed 's|.*a\([0-9]*\)/.*|\1|' | sort -n | tail -1)
if [ -n "$MAX_PREFIX" ]; then
  PREFIX_OFFSET=$((MAX_PREFIX + 1))
else
  PREFIX_OFFSET=0
fi
echo "Agent prefix offset: $PREFIX_OFFSET (agents will be a${PREFIX_OFFSET}..a$((PREFIX_OFFSET + NUM_GPUS - 1)))"

: > .agent_pids  # truncate pid file

for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
  AGENT_ID=$((GPU_ID + PREFIX_OFFSET))
  WORKTREE="$ROOT"
  if [ "$GPU_ID" -gt 0 ]; then
    WORKTREE="$ROOT/worktree-a${AGENT_ID}"
    if [ ! -d "$WORKTREE" ]; then
      git worktree add "$WORKTREE" master
    fi
  fi

  LOG="$ROOT/agent${AGENT_ID}.log"
  echo "Launching agent a${AGENT_ID} on GPU $GPU_ID → $LOG"

  if [ "$RESUME" = true ]; then
    CUDA_VISIBLE_DEVICES=$GPU_ID claude -p \
      --continue \
      --output-format stream-json \
      --verbose \
      --allowedTools "Edit" "Write" "Read" "Bash" "Glob" "Grep" "Agent" \
      --permission-mode bypassPermissions \
      "You were interrupted. Read results.tsv to find your last experiment number and current_base. Check which git branch you're on. Resume the experiment loop from where you left off. Never stop. Never ask the user anything." \
      >> "$LOG" 2>&1 &
  else
    CUDA_VISIBLE_DEVICES=$GPU_ID claude -p \
      --output-format stream-json \
      --verbose \
      --allowedTools "Edit" "Write" "Read" "Bash" "Glob" "Grep" "Agent" \
      --permission-mode bypassPermissions \
      "You are agent $AGENT_ID. AGENT_ID=$AGENT_ID, WORKTREE_PATH=$WORKTREE, CUDA_VISIBLE_DEVICES=$GPU_ID. Read and follow instructions.md in $ROOT. Start the experiment loop now. Never stop. Never ask the user anything." \
      > "$LOG" 2>&1 &
  fi

  echo $! >> .agent_pids
  echo "  PID $!"
done

echo ""
echo "All agents launched. Monitor with:"
echo "  tail -f agent*.log"
echo "  cat results.tsv | column -t -s\$'\\t'"
echo ""
echo "Stop all agents:"
echo "  kill \$(cat .agent_pids)"
echo ""
echo "Resume after killing:"
echo "  ./scripts/launch.sh --resume"
