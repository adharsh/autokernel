#!/usr/bin/env bash
set -euo pipefail

# Launch one autonomous kernel-optimization agent per GPU.
# Usage: ./scripts/launch.sh
# Logs: agent{N}.log per agent, PIDs written to .agent_pids

cd "$(dirname "$0")/.."
ROOT=$(pwd)

NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "Detected $NUM_GPUS GPU(s)"

: > .agent_pids  # truncate pid file

for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
  WORKTREE="$ROOT"
  if [ "$GPU_ID" -gt 0 ]; then
    WORKTREE="$ROOT/worktree-a${GPU_ID}"
    if [ ! -d "$WORKTREE" ]; then
      git worktree add "$WORKTREE" master
    fi
  fi

  LOG="$ROOT/agent${GPU_ID}.log"
  echo "Launching agent $GPU_ID on GPU $GPU_ID → $LOG"

  CUDA_VISIBLE_DEVICES=$GPU_ID claude -p \
    "You are agent $GPU_ID. AGENT_ID=$GPU_ID, WORKTREE_PATH=$WORKTREE, CUDA_VISIBLE_DEVICES=$GPU_ID. Follow instructions.md. Start the experiment loop now. Never stop." \
    --allowedTools "Edit,Write,Read,Bash,Glob,Grep" \
    > "$LOG" 2>&1 &

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
