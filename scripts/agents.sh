#!/usr/bin/env bash
set -euo pipefail

# Manage autonomous kernel-optimization agents.
# Usage:
#   ./scripts/agents.sh start    # fresh start, one agent per GPU
#   ./scripts/agents.sh stop     # kill all running agents
#   ./scripts/agents.sh resume   # fresh conversations for existing agent worktrees
#   ./scripts/agents.sh status   # show which agents are alive
#   AGENT_CLI=claude ./scripts/agents.sh start

cd "$(dirname "$0")/.."
ROOT=$(pwd)
RESULTS_DIR="$ROOT/results"
LOGS_DIR="$RESULTS_DIR/logs"
TSV="$RESULTS_DIR/experiments.tsv"
EXPERIMENTS_DIR="$RESULTS_DIR/experiments"
REFERENCE_TIMING_PATH="$RESULTS_DIR/reference_timing.json"
PID_FILE="$ROOT/.agent_pids"
TSV_HEADER='experiment_id	parent_id	agent_id	commit	timestamp	ncu_duration_us	ncu_kernel_count	reference_us	speedup	correctness	peak_vram_mb	status	interface_variant	description'

# Agent backend. AGENT_CLI accepts: claude, codex.
# "code" is accepted as a codex alias to avoid accidentally invoking VS Code.
AGENT_CLI=${AGENT_CLI:-codex}

# Claude settings
CLAUDE_BIN=${CLAUDE_BIN:-claude}
CLAUDE_MODEL=claude-opus-4-7
CLAUDE_EFFORT=high
CLAUDE_ALLOWED_TOOLS=(Edit Write Read Bash Glob Grep Agent)

# Codex settings
CODEX_BIN=${CODEX_BIN:-codex}
CODEX_MODEL=gpt-5.5
CODEX_EFFORT=xhigh
CODEX_COMMON_ARGS=(
  --json
  --dangerously-bypass-approvals-and-sandbox
  --model "$CODEX_MODEL"
  -c "model_reasoning_effort=\"$CODEX_EFFORT\""
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Parse a line from PID_FILE. Format: "AGENT_ID:PID".
# Sets: _agent_id, _pid
_parse_pid_line() {
  local line="$1"
  _agent_id="${line%%:*}"
  _pid="${line#*:}"
}

_normalize_agent_cli() {
  case "${AGENT_CLI,,}" in
    claude)
      AGENT_CLI=claude
      ;;
    codex|code)
      AGENT_CLI=codex
      ;;
    *)
      echo "Unsupported AGENT_CLI=$AGENT_CLI. Expected 'claude' or 'codex'." >&2
      exit 1
      ;;
  esac
}

_agent_command() {
  case "$AGENT_CLI" in
    claude) printf "%s\n" "$CLAUDE_BIN" ;;
    codex)  printf "%s\n" "$CODEX_BIN" ;;
  esac
}

_agent_model_summary() {
  case "$AGENT_CLI" in
    claude) printf "model=%s effort=%s\n" "$CLAUDE_MODEL" "$CLAUDE_EFFORT" ;;
    codex)  printf "model=%s effort=%s\n" "$CODEX_MODEL" "$CODEX_EFFORT" ;;
  esac
}

_require_agent_cli() {
  _normalize_agent_cli
  local bin
  bin=$(_agent_command)
  if ! command -v "$bin" >/dev/null 2>&1; then
    echo "AGENT_CLI=$AGENT_CLI requires '$bin' on PATH." >&2
    exit 1
  fi
}

_detect_num_gpus() {
  local num_gpus
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is not on PATH; cannot auto-detect GPUs." >&2
    echo "Run agents on a GPU host with NVIDIA tools available." >&2
    exit 1
  fi
  if ! num_gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader | awk 'NF { count++ } END { print count + 0 }'); then
    echo "Failed to query GPUs with nvidia-smi." >&2
    exit 1
  fi
  if [ "$num_gpus" -le 0 ]; then
    echo "No GPUs detected by nvidia-smi." >&2
    exit 1
  fi
  printf "%s\n" "$num_gpus"
}

_init_results_tsv() {
  mkdir -p "$RESULTS_DIR"
  if [ -s "$TSV" ]; then
    local header
    header=$(head -n 1 "$TSV")
    if [ "$header" != "$TSV_HEADER" ]; then
      echo "results TSV header does not match this run: $TSV" >&2
      echo "Move or remove the existing results directory before launching." >&2
      exit 1
    fi
  else
    printf "%s\n" "$TSV_HEADER" > "$TSV"
  fi
}

_check_reference_calibration() {
  if [ -s "$REFERENCE_TIMING_PATH" ]; then
    return
  fi

  echo "Missing calibrated reference timing: $REFERENCE_TIMING_PATH" >&2
  echo "Run: uv run python scripts/calibrate_reference.py" >&2
  exit 1
}

_agent_worktree_path() {
  local agent_id="$1"
  printf "%s/worktree-a%s\n" "$ROOT" "$agent_id"
}

_prepare_worktree_results_link() {
  local worktree="$1"
  local local_results="$worktree/results"
  local backup
  local expected
  local actual

  if [ "$worktree" = "$ROOT" ]; then
    echo "Refusing to use repo root as an agent worktree: $ROOT" >&2
    exit 1
  fi

  if [ ! -d "$worktree" ]; then
    echo "Missing worktree: $worktree" >&2
    exit 1
  fi

  expected=$(readlink -f "$RESULTS_DIR")

  if [ -L "$local_results" ]; then
    actual=$(readlink -f "$local_results")
    if [ "$actual" != "$expected" ]; then
      echo "$local_results points to $actual, expected $expected" >&2
      exit 1
    fi
    return
  fi

  if [ -e "$local_results" ]; then
    if [ -d "$local_results" ] && [ -z "$(find "$local_results" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
      rmdir "$local_results"
    else
      backup="$worktree/results.local.$(date -u +%Y%m%dT%H%M%SZ)"
      echo "archiving existing worktree-local results: $local_results -> $backup"
      mv "$local_results" "$backup"
    fi
  fi

  ln -s "$RESULTS_DIR" "$local_results"
}

_check_task_files() {
  local file
  local dirty

  for file in validate.py reference.py candidate/interface.py; do
    if [ ! -f "$ROOT/$file" ]; then
      echo "Missing $file. Run ./scripts/setup.sh, then fill in the task-specific implementation before launching agents." >&2
      exit 1
    fi
  done

  dirty=$(git status --porcelain -- validate.py reference.py candidate)
  if [ -z "$dirty" ]; then
    return
  fi

  echo "Uncommitted task setup changes:" >&2
  printf "%s\n" "$dirty" >&2
  echo "Commit or stash validate.py, reference.py, and candidate/ before launching agents so every worktree sees the same task setup." >&2
  exit 1
}

# Sets: _agent_pid
_launch_agent() {
  local gpu_id="$1"
  local agent_id="$2"
  local worktree="$3"
  local log="$4"
  local prompt="$5"
  local old_pwd="$PWD"
  # All launches start a fresh CLI conversation. Resume state is reconstructed
  # from the worktree, experiments TSV, logs, and notes via the prompt.
  local codex_cmd=(exec)

  case "$AGENT_CLI" in
    claude)
      AGENT_ID=$agent_id \
        WORKTREE_PATH=$worktree \
        AUTOKERNEL_ROOT=$ROOT \
        AUTOKERNEL_RESULTS_DIR=$RESULTS_DIR \
        AUTOKERNEL_EXPERIMENTS_TSV=$TSV \
        AUTOKERNEL_EXPERIMENTS_DIR=$EXPERIMENTS_DIR \
        AUTOKERNEL_REFERENCE_TIMING_PATH=$REFERENCE_TIMING_PATH \
        CUDA_VISIBLE_DEVICES=$gpu_id \
        "$CLAUDE_BIN" -p \
          --model "$CLAUDE_MODEL" \
          --effort "$CLAUDE_EFFORT" \
          --output-format stream-json \
          --verbose \
          --allowedTools "${CLAUDE_ALLOWED_TOOLS[@]}" \
          --dangerously-skip-permissions \
          "$prompt" \
          >> "$log" 2>&1 &
      _agent_pid=$!
      ;;
    codex)
      cd "$worktree"
      AGENT_ID=$agent_id \
        WORKTREE_PATH=$worktree \
        AUTOKERNEL_ROOT=$ROOT \
        AUTOKERNEL_RESULTS_DIR=$RESULTS_DIR \
        AUTOKERNEL_EXPERIMENTS_TSV=$TSV \
        AUTOKERNEL_EXPERIMENTS_DIR=$EXPERIMENTS_DIR \
        AUTOKERNEL_REFERENCE_TIMING_PATH=$REFERENCE_TIMING_PATH \
        CUDA_VISIBLE_DEVICES=$gpu_id \
        "$CODEX_BIN" "${codex_cmd[@]}" \
          "${CODEX_COMMON_ARGS[@]}" \
          - \
          >> "$log" 2>&1 <<< "$prompt" &
      _agent_pid=$!
      cd "$old_pwd"
      ;;
  esac
}

# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

cmd_stop() {
  if [ ! -f "$PID_FILE" ]; then
    echo "No .agent_pids file found."
    return
  fi
  echo "Killing agents..."
  while read -r line; do
    _parse_pid_line "$line"
    kill "$_pid" 2>/dev/null || true
  done < "$PID_FILE"
  sleep 1
  while read -r line; do
    _parse_pid_line "$line"
    if kill -0 "$_pid" 2>/dev/null; then
      echo "  a${_agent_id} (PID $_pid) still alive, sending SIGKILL..."
      kill -9 "$_pid" 2>/dev/null || true
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
  while read -r line; do
    _parse_pid_line "$line"
    if kill -0 "$_pid" 2>/dev/null; then
      state="running"
    else
      state="dead"
    fi
    printf "  a%-4s  PID %-8s  %-8s" "$_agent_id" "$_pid" "$state"
    if [ -f "$TSV" ]; then
      count=$(awk -F'\t' -v a="$_agent_id" '$3 == a' "$TSV" 2>/dev/null | wc -l)
      best=$(awk -F'\t' -v a="$_agent_id" '$3 == a && $12 == "keep" { if ($9+0 > max) max=$9+0 } END { if (max > 0) printf "%.2fx", max; else print "n/a" }' "$TSV" 2>/dev/null)
      printf "  | %3d experiments | best %s" "$count" "$best"
    fi
    echo ""
  done < "$PID_FILE"
  echo ""
}

cmd_start() {
  _require_agent_cli

  if [ -f "$PID_FILE" ]; then
    cmd_stop
  fi

  NUM_GPUS=$(_detect_num_gpus)
  echo "Detected $NUM_GPUS GPU(s)"
  echo "Agent CLI: $AGENT_CLI ($(_agent_command))"
  echo "Agent settings: $(_agent_model_summary)"
  _check_task_files
  _check_reference_calibration

  # Find next free agent prefix by checking existing a{N}/{M} branches
  MAX_PREFIX=$(git branch --list 'a*/*' | grep -oP '(?<=\ba)\d+(?=/)' | sort -n | tail -1 || true)
  if [ -n "$MAX_PREFIX" ]; then
    PREFIX_OFFSET=$((MAX_PREFIX + 1))
  else
    PREFIX_OFFSET=0
  fi
  echo "Agent prefix offset: $PREFIX_OFFSET (agents will be a${PREFIX_OFFSET}..a$((PREFIX_OFFSET + NUM_GPUS - 1)))"

  _init_results_tsv
  mkdir -p "$LOGS_DIR" "$EXPERIMENTS_DIR"
  : > "$PID_FILE"

  for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    AGENT_ID=$((GPU_ID + PREFIX_OFFSET))
    BASELINE_BRANCH="a${AGENT_ID}/0"
    git branch "$BASELINE_BRANCH" HEAD 2>/dev/null || true

    WORKTREE=$(_agent_worktree_path "$AGENT_ID")
    if [ ! -d "$WORKTREE" ]; then
      git worktree add "$WORKTREE" "$BASELINE_BRANCH"
    fi
    _prepare_worktree_results_link "$WORKTREE"

    LOG="$LOGS_DIR/agent${AGENT_ID}.log"
    echo "Launching agent a${AGENT_ID} on GPU $GPU_ID → $LOG"
    : > "$LOG"

    _launch_agent "$GPU_ID" "$AGENT_ID" "$WORKTREE" "$LOG" \
      "You are agent $AGENT_ID. AGENT_ID=$AGENT_ID, WORKTREE_PATH=$WORKTREE, CUDA_VISIBLE_DEVICES=$GPU_ID, AUTOKERNEL_ROOT=$ROOT, AUTOKERNEL_EXPERIMENTS_TSV=$TSV, AUTOKERNEL_EXPERIMENTS_DIR=$EXPERIMENTS_DIR, AUTOKERNEL_REFERENCE_TIMING_PATH=$REFERENCE_TIMING_PATH. Read and follow instructions.md in $ROOT. Required: submit honest general implementations; do not memoize answers, hardcode outputs, special-case tests/benchmarks, detect evaluator behavior, skip correctness paths, or reward-hack. Run scripts/profile_ncu.sh for every baseline and every experiment that launches kernels. After each profile, you MUST read the complete experiment ncu/details.txt from top to bottom; targeted greps/summaries are allowed only after the full read, not instead of it. The profiler's warnings and recommendations are evidence: consider them, then either act on them or explicitly discard them with a reason in note.md. After each profile, write note.md with the measured bottleneck, profile evidence, speed-of-light interpretation when relevant, profiler recommendations considered, and next experiment chosen from that evidence. Do not move on with a shallow note. Record whether PTX/SASS/codegen was inspected. If no obvious speedup is available, use the full profiling details to try justified lower-level optimizations, including CUDA C++ with inline PTX when appropriate. Start the experiment loop now. Never stop. Never ask the user anything."

    echo "$AGENT_ID:$_agent_pid" >> "$PID_FILE"
    echo "  PID $_agent_pid"
  done

  echo ""
  echo "All agents launched. Monitor with:"
  echo "  tail -f $LOGS_DIR/agent*.log"
  echo "  uv run python scripts/format_results.py --sort agent"
  echo ""
  echo "  ./scripts/agents.sh status"
  echo "  ./scripts/agents.sh stop"
  echo "  ./scripts/agents.sh resume"
}

cmd_resume() {
  _require_agent_cli

  if [ ! -s "$PID_FILE" ]; then
    echo "No .agent_pids file found. Run ./scripts/agents.sh start before resume." >&2
    exit 1
  fi

  local agent_ids=()
  local worktrees=()
  local line
  while read -r line; do
    [ -n "$line" ] || continue
    _parse_pid_line "$line"
    agent_ids+=("$_agent_id")
  done < "$PID_FILE"

  NUM_GPUS=$(_detect_num_gpus)
  echo "Detected $NUM_GPUS GPU(s)"
  echo "Agent CLI: $AGENT_CLI ($(_agent_command))"
  echo "Agent settings: $(_agent_model_summary)"
  echo "Resuming agents from previous sessions..."
  if [ "${#agent_ids[@]}" -gt "$NUM_GPUS" ]; then
    echo "Cannot resume ${#agent_ids[@]} agent(s) with only $NUM_GPUS GPU(s) detected." >&2
    exit 1
  fi
  _check_task_files
  _check_reference_calibration

  _init_results_tsv
  mkdir -p "$LOGS_DIR" "$EXPERIMENTS_DIR"

  for AGENT_ID in "${agent_ids[@]}"; do
    WORKTREE=$(_agent_worktree_path "$AGENT_ID")
    _prepare_worktree_results_link "$WORKTREE"
    worktrees+=("$WORKTREE")
  done

  echo "Stopping currently tracked agent processes before fresh resume..."
  cmd_stop
  : > "$PID_FILE"

  for GPU_ID in "${!agent_ids[@]}"; do
    AGENT_ID=${agent_ids[$GPU_ID]}
    WORKTREE=${worktrees[$GPU_ID]}

    LOG="$LOGS_DIR/agent${AGENT_ID}.log"
    echo "Starting fresh conversation for agent a${AGENT_ID} on GPU $GPU_ID → $LOG"

    _launch_agent "$GPU_ID" "$AGENT_ID" "$WORKTREE" "$LOG" \
      "You are agent $AGENT_ID in a fresh conversation. AGENT_ID=$AGENT_ID, WORKTREE_PATH=$WORKTREE, CUDA_VISIBLE_DEVICES=$GPU_ID, AUTOKERNEL_ROOT=$ROOT, AUTOKERNEL_EXPERIMENTS_TSV=$TSV, AUTOKERNEL_EXPERIMENTS_DIR=$EXPERIMENTS_DIR, AUTOKERNEL_REFERENCE_TIMING_PATH=$REFERENCE_TIMING_PATH. Do not rely on prior chat state. Treat disk artifacts as the source of truth: read and follow $ROOT/instructions.md, inspect the current git branch/status/log in $WORKTREE, read $TSV, and read relevant experiment notes under $EXPERIMENTS_DIR, especially this agent's a${AGENT_ID}_* notes and the best keep rows from the TSV. Use that provenance to identify any interrupted work, the next unused a${AGENT_ID}/N experiment id, and the strongest current parent/result to build from. Do not assume you must continue this agent's previous branch if the TSV and notes show a better global parent; choose the next experiment from the durable evidence. If you find an interrupted experiment, inspect its branch, files, run logs, NCU outputs, and note before deciding whether to finish, record, discard, or branch from the latest stronger result. Required: submit honest general implementations; do not memoize answers, hardcode outputs, special-case tests/benchmarks, detect evaluator behavior, skip correctness paths, or reward-hack. Run scripts/profile_ncu.sh for every resumed experiment that launches kernels. After each profile, you MUST read the complete experiment ncu/details.txt from top to bottom; targeted greps/summaries are allowed only after the full read, not instead of it. The profiler's warnings and recommendations are evidence: consider them, then either act on them or explicitly discard them with a reason in note.md. After each profile, write note.md with the measured bottleneck, profile evidence, speed-of-light interpretation when relevant, profiler recommendations considered, and next experiment chosen from that evidence. Do not move on with a shallow note. Record whether PTX/SASS/codegen was inspected. If no obvious speedup is available, use the full profiling details to try justified lower-level optimizations, including CUDA C++ with inline PTX when appropriate. Resume the experiment loop now. Never stop. Never ask the user anything."

    echo "$AGENT_ID:$_agent_pid" >> "$PID_FILE"
    echo "  PID $_agent_pid"
  done

  echo ""
  echo "All agents resumed with fresh conversations. Monitor with:"
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
