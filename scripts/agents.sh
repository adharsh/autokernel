#!/usr/bin/env bash
set -euo pipefail

# Manage autonomous kernel-optimization agents.
# Usage:
#   ./scripts/agents.sh start    # fresh start, one agent per GPU
#   ./scripts/agents.sh stop     # kill all running agents
#   ./scripts/agents.sh resume   # resume interrupted agents
#   ./scripts/agents.sh status   # show which agents are alive

cd "$(dirname "$0")/.."
ROOT=$(pwd)
RESULTS_DIR="$ROOT/results"
LOGS_DIR="$RESULTS_DIR/logs"
TSV="$RESULTS_DIR/experiments.tsv"
PID_FILE="$ROOT/.agent_pids"

# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

cmd_stop() {
  if [ ! -f "$PID_FILE" ]; then
    echo "No .agent_pids file found."
    return
  fi
  echo "Killing agents..."
  while IFS=: read -r pid agent_id; do
    kill "$pid" 2>/dev/null || true
  done < "$PID_FILE"
  sleep 1
  # Verify
  while IFS=: read -r pid agent_id; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "  a${agent_id} (PID $pid) still alive, sending SIGKILL..."
      kill -9 "$pid" 2>/dev/null || true
    fi
  done < "$PID_FILE"
  echo "All agents stopped."
}

cmd_status() {
  if [ ! -f "$PID_FILE" ]; then
    echo "No .agent_pids file found. No agents have been launched."
    return
  fi
  echo ""
  while IFS=: read -r pid agent_id; do
    if kill -0 "$pid" 2>/dev/null; then
      state="running"
    else
      state="dead"
    fi
    printf "  a%-4s  PID %-8s  %-8s" "$agent_id" "$pid" "$state"
    if [ -f "$TSV" ]; then
      count=$(awk -F'\t' -v a="$agent_id" '$3 == a' "$TSV" 2>/dev/null | wc -l)
      best=$(awk -F'\t' -v a="$agent_id" '$3 == a && $11 == "keep" { if ($8+0 > max) max=$8+0 } END { if (max > 0) printf "%.2fx", max; else print "n/a" }' "$TSV" 2>/dev/null)
      printf "  | %3d experiments | best %s" "$count" "$best"
    fi
    echo ""
  done < "$PID_FILE"
  echo ""
}

cmd_start() {
  # Kill existing agents first
  if [ -f "$PID_FILE" ]; then
    echo "Killing existing agents..."
    xargs kill 2>/dev/null < "$PID_FILE" || true
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

  mkdir -p "$LOGS_DIR"
  : > "$PID_FILE"

  for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    AGENT_ID=$((GPU_ID + PREFIX_OFFSET))
    WORKTREE="$ROOT"
    if [ "$GPU_ID" -gt 0 ]; then
      WORKTREE="$ROOT/worktree-a${AGENT_ID}"
      if [ ! -d "$WORKTREE" ]; then
        git worktree add "$WORKTREE" master
      fi
    fi

    LOG="$LOGS_DIR/agent${AGENT_ID}.log"
    echo "Launching agent a${AGENT_ID} on GPU $GPU_ID → $LOG"

    CUDA_VISIBLE_DEVICES=$GPU_ID claude -p \
      --output-format stream-json \
      --verbose \
      --allowedTools "Edit" "Write" "Read" "Bash" "Glob" "Grep" "Agent" \
      --permission-mode bypassPermissions \
      "You are agent $AGENT_ID. AGENT_ID=$AGENT_ID, WORKTREE_PATH=$WORKTREE, CUDA_VISIBLE_DEVICES=$GPU_ID. Read and follow instructions.md in $ROOT. Start the experiment loop now. Never stop. Never ask the user anything." \
      > "$LOG" 2>&1 &

    echo "$!:$AGENT_ID" >> "$PID_FILE"
    echo "  PID $!"
  done

  echo ""
  echo "All agents launched. Monitor with:"
  echo "  tail -f $LOGS_DIR/agent*.log"
  echo "  cat $TSV | column -t -s\$'\\t'"
  echo ""
  echo "  ./scripts/agents.sh status"
  echo "  ./scripts/agents.sh stop"
  echo "  ./scripts/agents.sh resume"
}

cmd_resume() {
  NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
  echo "Detected $NUM_GPUS GPU(s)"
  echo "Resuming agents from previous sessions..."

  mkdir -p "$LOGS_DIR"
  : > "$PID_FILE"

  for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    # For resume, we need to find which agent IDs were previously running.
    # Look at existing log files to determine the mapping.
    # Fallback: use GPU_ID directly (works if prefix offset was 0).
    AGENT_ID=$GPU_ID

    # Try to find the most recent log for this GPU slot
    latest_log=$(ls -t "$LOGS_DIR"/agent*.log 2>/dev/null | sed -n "$((GPU_ID + 1))p" || true)
    if [ -n "$latest_log" ]; then
      base=$(basename "$latest_log" .log)
      AGENT_ID=${base#agent}
    fi

    WORKTREE="$ROOT"
    if [ "$GPU_ID" -gt 0 ]; then
      WORKTREE="$ROOT/worktree-a${AGENT_ID}"
    fi

    LOG="$LOGS_DIR/agent${AGENT_ID}.log"
    echo "Resuming agent a${AGENT_ID} on GPU $GPU_ID → $LOG"

    CUDA_VISIBLE_DEVICES=$GPU_ID claude -p \
      --continue \
      --output-format stream-json \
      --verbose \
      --allowedTools "Edit" "Write" "Read" "Bash" "Glob" "Grep" "Agent" \
      --permission-mode bypassPermissions \
      "You were interrupted. Read results/experiments.tsv to find your last experiment number and current_base. Check which git branch you're on. Resume the experiment loop from where you left off. Never stop. Never ask the user anything." \
      >> "$LOG" 2>&1 &

    echo "$!:$AGENT_ID" >> "$PID_FILE"
    echo "  PID $!"
  done

  echo ""
  echo "All agents resumed. Monitor with:"
  echo "  ./scripts/agents.sh status"
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

case "${1:-}" in
  start)  cmd_start ;;
  stop)   cmd_stop ;;
  resume) cmd_resume ;;
  status) cmd_status ;;
  *)
    echo "Usage: ./scripts/agents.sh {start|stop|resume|status}"
    exit 1
    ;;
esac
