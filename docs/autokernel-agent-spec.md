# AutoKernel Agent System — Full Specification

This document specifies the complete autonomous kernel optimization agent system. A coding agent should be able to implement all components from this spec alone.

---

## 1. Project Structure

```
autokernel/
├── validate.py              # FIXED — correctness + timing harness (test cases can only be ADDED, never removed)
├── reference.py             # FIXED — reference implementation (ground truth)
├── candidate/
│   ├── __init__.py
│   └── interface.py         # Agent edits THIS — all optimization code, bindings, utilities
├── instructions.md          # Agent playbook (like autoresearch/program.md)
├── results.tsv              # Shared, append-only experiment log (tab-separated)
├── analysis.py              # Plotting + reports from results.tsv
├── profile_utils.py         # cuda_timer, cpu_timer, shared profiling utilities
└── scripts/
    └── launch_agents.sh     # Multi-agent launcher (one per GPU)
```

### File Mutability Rules

| File | Who edits | Rules |
|------|-----------|-------|
| `validate.py` | Human only | Test cases can be ADDED, never removed or modified |
| `reference.py` | Nobody | Ground truth, never modified |
| `candidate/interface.py` | Agent | All optimization code lives here |
| `candidate/__init__.py` | Agent | Exports from interface.py |
| `results.tsv` | Agent (append-only) | Never delete rows, never rewrite existing rows |
| `instructions.md` | Human | Agent reads, never modifies |
| `analysis.py` | Human / setup | Agent may run but never modifies |

---

## 2. Results TSV Schema

**Format**: Tab-separated values (TSV).

**Header**:
```
experiment_id	parent_id	agent_id	commit	timestamp	candidate_us	reference_us	speedup	correctness	peak_vram_mb	status	description
```

**Column definitions**:

| Column | Type | Example | Description |
|--------|------|---------|-------------|
| `experiment_id` | string | `a0/1` | Unique ID. Also the git branch name. Format: `a{agent_id}/{seq_number}` |
| `parent_id` | string | `a0/0` | Experiment this was built on. `-` for baseline. Enables lineage tree reconstruction |
| `agent_id` | int | `0` | Which agent ran this |
| `commit` | string | `a1b2c3d` | 7-char git commit hash |
| `timestamp` | ISO8601 | `2026-03-20T14:32:00` | When the experiment completed |
| `candidate_us` | float | `42.3` | Candidate kernel latency in microseconds |
| `reference_us` | float | `85.1` | Reference implementation latency in microseconds |
| `speedup` | float | `2.01` | `reference_us / candidate_us` |
| `correctness` | string | `PASS` | `PASS`, `FAIL`, or `CRASH` |
| `peak_vram_mb` | float | `2048.5` | Peak GPU memory used during benchmark |
| `status` | string | `keep` | `keep`, `discard`, or `crash` |
| `description` | string | `fuse norm+proj` | Short description of what was tried |

**Concurrency**: Multiple agents append to the same file. Use `fcntl.flock()` for atomic writes. Each agent opens in append mode, locks, writes one line, unlocks.

---

## 3. Experiment Naming Convention

Format: `a{agent_id}/{experiment_number}`

- `a0/0` — agent 0, baseline
- `a0/1` — agent 0, first experiment
- `a1/0` — agent 1, baseline
- `a1/3` — agent 1, fourth experiment

This string is used as:
1. The `experiment_id` column in results.tsv
2. The git branch name (via `git branch a0/1 <commit-hash>`)
3. The lookup key for cross-agent reference

---

## 4. Lineage Tracking

Each experiment records its `parent_id` — the experiment whose code it was built upon.

**Agent logic**:
```python
current_base = f"a{agent_id}/0"  # start at baseline

for each experiment:
    # parent_id is always current_base at experiment start
    record(parent_id=current_base, ...)

    if status == "keep":
        current_base = experiment_id  # advance
    # if discard/crash: current_base stays the same
```

**Cross-agent lineage**: If agent 1 adapts code from agent 0's experiment `a0/5`, it sets `parent_id=a0/5`. The lineage tree spans agents naturally.

**Reading another agent's code** (no checkout needed):
```bash
git show a0/5:candidate/interface.py       # read the file
git diff a0/0..a0/5 -- candidate/          # see what changed
```

---

## 5. Git Branch Strategy

### Branch-per-experiment (no master branch needed)
Every experiment gets its own branch. No resets, no master branch. `current_base` tracks the last kept experiment's branch name.

### Experiment lifecycle
```
1. git checkout -b a0/1 current_base        # new branch from last keep (e.g., a0/0)
2. Agent edits candidate/interface.py
3. git add candidate/ && git commit -m "a0/1: fuse norm+proj"
4. Run validate.py → get results
5. Append row to results.tsv
6. If keep:   current_base = a0/1           # advance
7. If discard/crash: git checkout {current_base}  # go back to last keep. That's it.
```

No `git reset`. No `git branch` after the fact. The branch is created *before* editing, so the commit is always preserved. To "revert" you just checkout the current_base branch.

### No PRs during the optimization loop
PRs add overhead that slows rapid iteration. Create summary PRs post-hoc if needed.

---

## 6. Multi-Agent Parallel Execution

### Architecture
```
results.tsv        (shared, append-only, file-locked)
├── worktree-a0/   (git worktree on GPU 0, branch a0/master)
├── worktree-a1/   (git worktree on GPU 1, branch a1/master)
└── worktree-a2/   (git worktree on GPU 2, branch a2/master)
```

### Launch script (`scripts/launch_agents.sh`)
Each agent is a Claude Code instance running in a separate worktree with a pinned GPU:
```bash
#!/bin/bash
# Launch N agents, one per GPU
NUM_GPUS=$(nvidia-smi -L | wc -l)

for i in $(seq 0 $((NUM_GPUS - 1))); do
    BRANCH="a${i}/0"
    WORKTREE="worktree-a${i}"

    # Create baseline branch and worktree
    git branch "$BRANCH" HEAD 2>/dev/null || true
    git worktree add "$WORKTREE" "$BRANCH" 2>/dev/null || true

    # Launch agent with pinned GPU
    CUDA_VISIBLE_DEVICES=$i claude --agent-id $i --worktree "$WORKTREE" &
done

wait
```

---

## 7. Microbench Agent

### Purpose
A dedicated agent that writes and runs line-by-line microbenchmarks of the current candidate code, following the xllm benchmarking pattern. It answers: "what percentage of time is spent on each compute line?"

This is a Claude Code agent (not a skill) because it needs its own context to read code, write benchmarks, run them, and analyze results. The parent optimization agent only receives the summary table.

### Workflow
1. **Read the candidate code** — `candidate/interface.py`
2. **Identify every compute line** — map each line to a logical sub-operation
3. **Write per-line microbenchmarks** — isolate each sub-op with `cuda_timer` or `cpu_timer`
4. **Run benchmarks** — execute with sufficient warmup and iterations
5. **Return structured report** — sub-op breakdown table

### Key principles
- Every compute line in real code gets an isolated microbenchmark
- Separate CUDA kernel calls are profiled separately
- For async stream operations, benchmark on large enough inputs to be measurable
- Each sub-op benchmark must have a comment indicating the source line it measures (e.g., `# bench: interface.py:42`)

### Tools available to the microbench agent
- `cuda_timer` / `cpu_timer` (see below)
- Read/Write/Bash for code generation and execution

**Note**: `ncu` and `nsight-systems` are available to the main optimization agent directly, not the microbench agent. The microbench agent only does microbenchmarking.

### Output format
```
Sub-op Breakdown for candidate/interface.py
============================================
Sub-op              Line    Latency (ms)    % of Total    Timer
------------------------------------------------------------------
matmul_qkv          42      0.312           38.2%         cuda
softmax             55      0.185           22.6%         cuda
attention_score     60      0.142           17.4%         cuda
norm                38      0.098           12.0%         cpu+gpu
index_select        35      0.052            6.4%         cuda
python_overhead     30      0.028            3.4%         cpu
------------------------------------------------------------------
TOTAL                       0.817           100.0%

Bottleneck: matmul_qkv (38.2%)
```

### Reference: xllm benchmarking patterns

The microbench agent follows the patterns established in `xllm/benchmarks/`. Key files to study:

**`xllm/benchmarks/utils.py`** — Timer utilities:
- `cuda_timer(fn, *args, warmup=10, iters=100)` — CUDA event-based GPU timing
- `make_inputs(batch_size, seq_len, model_dim)` — input tensor factory
- `print_results(name, results_dict)` — formatted output
- `save_results(all_results, path)` — JSON/CSV export

**`xllm/benchmarks/bench_router.py`** — Best example of line-by-line profiling:
- Profiles `TopKRouter.forward()` (router.py:64-87)
- 9 sub-ops: projection, cast_to_f32, score_func, bias_add, topk, gather, score_norm, cast_to_bf16, bincount
- Each sub-op is isolated and timed independently

**`xllm/benchmarks/bench_mgmm.py`** — SwiGLU sub-op profiling:
- Sub-ops: mgmm_w1, silu, mgmm_w3, elementwise_mul, mgmm_w2, full

**`xllm/benchmarks/bench_x_permutation.py`** — Simple 3 sub-op example:
- Sub-ops: argsort, div, index_select

**`xllm/docs/benchmarking_workflow.md`** — The 6-step workflow this agent follows.

### Timer utilities (`profile_utils.py`)

#### cuda_timer
For GPU-bound operations. Uses CUDA events for accurate kernel timing:
```python
def cuda_timer(fn, *args, warmup=10, iters=100, **kwargs) -> dict:
    """
    Time a GPU operation using CUDA events.
    Returns: {median_ms, mean_ms, min_ms, max_ms, std_ms}
    """
    import torch

    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    timings = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))  # ms

    import statistics
    return {
        "median_ms": statistics.median(timings),
        "mean_ms": statistics.mean(timings),
        "min_ms": min(timings),
        "max_ms": max(timings),
        "std_ms": statistics.stdev(timings) if len(timings) > 1 else 0.0,
    }
```

#### cpu_timer
For CPU-bound or mixed CPU+GPU operations:
```python
def cpu_timer(fn, *args, warmup=10, iters=100, sync_cuda=True, **kwargs) -> dict:
    """
    Time an operation using CPU wall clock.
    sync_cuda=True (default): includes GPU kernel completion time.
    sync_cuda=False: pure CPU operations only.
    Returns: {median_ms, mean_ms, min_ms, max_ms, std_ms}
    """
    import time
    import torch

    for _ in range(warmup):
        fn(*args, **kwargs)
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()

    timings = []
    for _ in range(iters):
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - t0) * 1000)  # ms

    import statistics
    return {
        "median_ms": statistics.median(timings),
        "mean_ms": statistics.mean(timings),
        "min_ms": min(timings),
        "max_ms": max(timings),
        "std_ms": statistics.stdev(timings) if len(timings) > 1 else 0.0,
    }
```

#### When to use which
- **cuda_timer**: Isolate *just* GPU kernel execution time, excluding CPU overhead and launch latency
- **cpu_timer(sync_cuda=True)**: Total wall-clock time including both CPU and GPU work (the safe default)
- **cpu_timer(sync_cuda=False)**: Pure CPU operations with no GPU involvement

---

## 8. Main Optimization Agent (instructions.md)

### Overview
Each agent follows the same playbook, running autonomously in its own worktree on a pinned GPU. The agent modifies `candidate/interface.py` and runs `validate.py` in a tight loop.

### Experiment loop (NEVER STOP — run indefinitely)
```
1. Hypothesize one focused optimization change
2. git checkout -b a{id}/{n} {current_base}    # new branch from last keep
4. Edit candidate/interface.py
5. git add candidate/ && git commit -m "a{id}/{n}: <description>"
6. Run: python validate.py → extract candidate_us, reference_us, correctness, peak_vram_mb
7. Compute speedup = reference_us / candidate_us
8. Append row to results.tsv (with file lock, parent_id = current_base)
9. If correctness == PASS and speedup > previous best:
     status = "keep"
     current_base = experiment_id           # advance base (stay on this branch)
   Else (fail, crash, or correct but slower):
     status = "discard" or "crash"
     git checkout {current_base}            # go back to last keep
10. Repeat from step 1
```

Every experiment's branch is preserved regardless of outcome. To inspect any experiment later: `git checkout a{id}/{n}` or `git show a{id}/{n}:candidate/interface.py`.

### Tools available
You have access to: `ncu`, `nsight-systems`, and a **microbench agent** (spawns a sub-agent that writes xllm-style line-by-line microbenchmarks of your candidate code and returns a sub-op breakdown table). Use these whenever you need to understand where or why time is being spent.

### Optimization strategy
Use whichever backend (PyTorch, Triton, CUDA C++, CUTLASS, CUTE DSL, PTX) is most appropriate for the hypothesis. Keep changes focused — one hypothesis per experiment.

### Constraints
- Never modify `validate.py` or `reference.py`
- Never skip correctness checks
- One focused change per experiment
- VRAM must not exceed 80% of GPU capacity
- Simpler code wins when performance is equal

---

## 9. Visualization & Analysis (`analysis.py`)

### Inputs
- `results.tsv` (the shared experiment log)

### Outputs
1. `progress.png` — main visualization
2. Terminal summary — printed to stdout
3. `report.md` — markdown session report (optional)

### Main plot: Latency over experiments
- **Y-axis**: `candidate_us` (latency in microseconds, lower is better)
- **X-axis**: Experiment number (global ordering by timestamp)
- **Dot colors by status**:
  - Green (`#2ecc71`): keep
  - Gray (`#cccccc`): discard
  - Red (`#e74c3c`): crash
- **Dot markers by agent** (for multi-agent runs):
  - Agent 0: circles
  - Agent 1: squares
  - Agent 2: triangles
  - Agent 3: diamonds
  - (etc.)
- **Running minimum step line** (`#27ae60`): frontier of best latency achieved
- **Reference baseline dashed line** (`#3498db`): reference implementation latency
- **Annotations**: Top-3 improvements labeled with descriptions
- **Title**: includes total experiment count and number of kept improvements

### Secondary plot: Speedup over experiments
- **Y-axis**: `speedup` (higher is better)
- **X-axis**: Experiment number
- **1.0x baseline** dashed line
- **Running maximum** step line (frontier)
- Same dot coloring and agent markers as main plot

### Terminal summary
```
============================================================
  AutoKernel — Session Summary
============================================================

  Reference latency:     85.10 us
  Best candidate:        42.30 us
  Total speedup:         2.01x

  Experiments:           47
  Kept:                  8 (17%)
  Discarded:             35
  Crashed:               4 (9%)

  Top 5 improvements:
    1. 42.30 us (2.01x) — fuse norm+proj + vectorized loads
    2. 48.10 us (1.77x) — shared memory tiling
    3. 55.20 us (1.54x) — fuse norm+proj
    4. 62.80 us (1.36x) — block size 128x128
    5. 71.40 us (1.19x) — coalesced memory access

  Per-agent breakdown:
    Agent 0: 25 experiments, 5 kept, best 42.30 us (2.01x)
    Agent 1: 22 experiments, 3 kept, best 51.80 us (1.64x)

============================================================
```

### Delta ranking (from autoresearch pattern)
Each kept experiment's incremental improvement over the previous best:
```
Rank  Delta (us)   Latency    Description
----  ----------   -------    -----------
   1     -7.10      48.10     shared memory tiling
   2     -5.80      42.30     vectorized loads
   3     -7.60      55.20     fuse norm+proj
   ...
```

### Lineage tree visualization (bonus)
Reconstruct and display the exploration tree from `parent_id`:
```
a0/0 (baseline, 85.1 us)
├── a0/1 (fuse norm, 55.2 us) ✓ KEEP
│   ├── a0/2 (split-K, 58.1 us) ✗ discard
│   ├── a0/3 (shared mem, 48.1 us) ✓ KEEP
│   │   └── a0/4 (vectorize, 42.3 us) ✓ KEEP
│   └── a1/3 (cross-agent adapt, 51.8 us) ✓ KEEP
├── a1/0 (baseline, 85.1 us)
│   ├── a1/1 (loop unroll, 72.3 us) ✓ KEEP
│   └── a1/2 (warp shuffle, 80.1 us) ✗ discard
```

### Suggestions engine (from autokernel pattern)
Based on experiment history, generate actionable suggestions:
- High crash rate (>40%): suggest more conservative changes
- Plateau (last 5 all discard/crash): suggest fundamentally different approach
- Modest speedup (<1.1x): suggest block tuning, persistent kernels, split-K
- Strong speedup (>1.5x): suggest fine-grained autotuning, profiling remaining bottlenecks
- High VRAM (>80%): suggest memory-efficient techniques

---

## 10. File Locking for Shared results.tsv

Multiple agents append to the same file concurrently. Use OS-level file locking:

```python
import fcntl
import os

def append_result(csv_path: str, row: dict, columns: list[str]) -> None:
    """Atomically append a row to the shared results.tsv."""
    line = "\t".join(str(row.get(col, "")) for col in columns) + "\n"

    with open(csv_path, "a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # exclusive lock
        try:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # release lock


def read_results(csv_path: str) -> list[dict]:
    """Read results.tsv with shared lock (allows concurrent reads)."""
    import csv

    with open(csv_path, "r") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_SH)  # shared lock
        try:
            reader = csv.DictReader(f, delimiter="\t")
            return list(reader)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
```

---

## 11. Initialization

### First-time setup (human runs once)
```bash
# Initialize results.tsv with header
printf 'experiment_id\tparent_id\tagent_id\tcommit\ttimestamp\tcandidate_us\treference_us\tspeedup\tcorrectness\tpeak_vram_mb\tstatus\tdescription\n' > results.tsv

# Verify reference and validation work
python validate.py  # should PASS with reference implementation
```

### Per-agent setup (automated)
```bash
# Agent receives: AGENT_ID, CUDA_VISIBLE_DEVICES, WORKTREE_PATH
# Agent does:
cd $WORKTREE_PATH
# Already on branch a{AGENT_ID}/0 (created by launch script)

# Run baseline
python validate.py → extract reference_us, candidate_us (should be same as reference initially)

# Record baseline
append_result("a${AGENT_ID}/0", parent_id="-", status="keep", description="baseline")
# current_base = "a${AGENT_ID}/0"
```

---

## 12. Reference Implementations for Visualization

### autokernel analysis patterns to follow
- **Location**: `autokernel/analysis.py` (420 lines)
- **Key patterns**:
  - `classify_row()` for keep/fail/revert classification
  - `make_progress_plot()` for scatter plot with frontier line
  - `_generate_suggestions()` for plateau detection and actionable advice
  - `generate_report()` for markdown session report

### autoresearch analysis patterns to follow
- **Location**: `autoresearch/analysis.ipynb`
- **Key patterns**:
  - Running minimum step line (for "lower is better" metrics like latency)
  - Annotation of kept experiments with descriptions
  - Delta ranking (improvement per kept experiment, sorted largest first)
  - Cumulative effort analysis

---

## 13. Summary of Components to Implement

| # | Component | File(s) | Description |
|---|-----------|---------|-------------|
| 1 | `instructions.md` | `instructions.md` | Agent playbook — the "brain" of the system. Modeled after `autoresearch/program.md` |
| 2 | `profile_utils.py` | `profile_utils.py` | `cuda_timer`, `cpu_timer`, `append_result`, `read_results` utilities |
| 3 | `analysis.py` | `analysis.py` | Plotting, terminal summary, report generation. Adapted from `autokernel/analysis.py` + `autoresearch/analysis.ipynb` |
| 4 | Kernel microbench agent | `.claude/agents/microbench.md` or equivalent | Agent definition for sub-op bottleneck analysis |
| 5 | Launch script | `scripts/launch_agents.sh` | Multi-GPU agent launcher with worktree setup |
| 6 | Results CSV init | Part of setup | Header row creation + baseline recording |

### What NOT to implement (already exists / human-provided)
- `validate.py` — provided by human
- `reference.py` — provided by human
- `candidate/interface.py` — starting point provided by human, then modified by agent
