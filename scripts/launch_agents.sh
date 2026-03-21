#!/bin/bash
# Launch N AutoKernel agents, one per GPU.
# Each agent gets its own git worktree and a pinned CUDA device.

set -euo pipefail

NUM_GPUS=$(nvidia-smi -L | wc -l)

echo "Detected ${NUM_GPUS} GPU(s). Launching agents..."

for i in $(seq 0 $((NUM_GPUS - 1))); do
    BRANCH="a${i}/0"
    WORKTREE="worktree-a${i}"

    # Create baseline branch and worktree (ignore errors if they already exist)
    git branch "$BRANCH" HEAD 2>/dev/null || true
    git worktree add "$WORKTREE" "$BRANCH" 2>/dev/null || true

    echo "Starting agent ${i} on GPU ${i} in ${WORKTREE} (branch ${BRANCH})"

    # Launch agent with pinned GPU
    CUDA_VISIBLE_DEVICES=$i claude --agent-id $i --worktree "$WORKTREE" &
done

echo "All agents launched. Waiting for completion..."
wait
