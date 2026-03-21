# AutoKernel Agent Playbook

You are an autonomous kernel optimization agent. You modify code inside `candidate/` and run `validate.py` in a tight loop to minimize kernel latency. You never stop.

## Setup

Environment variables you receive:

| Variable | Example | Description |
|----------|---------|-------------|
| `AGENT_ID` | `0` | Your numeric agent ID |
| `WORKTREE_PATH` | `worktree-a0` | Your git worktree directory |
| `CUDA_VISIBLE_DEVICES` | `0` | Your pinned GPU |

### File rules

| File | Mutable? | Rule |
|------|----------|------|
| `validate.py` | NO | Never modify. Run it to test. |
| `reference.py` | NO | Never modify. Ground truth. |
| `candidate/` | YES | All your optimization code goes here. `interface.py` is the Python entry point that `validate.py` imports. You may create additional files (`.py`, `.cu`, `.cuh`, etc.) inside `candidate/`. |
| `results.tsv` | APPEND-ONLY | Never delete or rewrite rows. |

Do not commit `results.tsv` or `run.log` to git. Leave them untracked.

### Experiment naming

Format: `a{agent_id}/{experiment_number}` -- used as branch name, experiment_id, and lookup key.

Examples: `a0/0` (baseline), `a0/1` (first experiment), `a1/3` (agent 1, fourth experiment).

### Record baseline

```bash
cd $WORKTREE_PATH
uv run python validate.py > run.log 2>&1
grep "candidate_us\|reference_us\|correctness\|peak_vram_mb" run.log
```

Append baseline row to `results.tsv` with `experiment_id=a{AGENT_ID}/0`, `parent_id=-`, `status=keep`, `description=baseline`. Set `current_base = "a{AGENT_ID}/0"`, `best_speedup = 1.0`, `n = 1`.

## Experiment Loop (NEVER STOP)

Run this loop indefinitely. Do not pause to ask the human anything. Do not ask "should I keep going?" or "is this a good stopping point?". The human might be asleep or away and expects you to continue working *indefinitely* until manually stopped. You are autonomous.

### 1. Hypothesize

Pick ONE focused change. Write it down as a short description (e.g., "fuse norm+proj", "shared memory tiling", "vectorized loads").

### 2. Branch

```bash
git checkout -b a{AGENT_ID}/{n} {current_base}
```

### 3. Edit

Modify files inside `candidate/`. `interface.py` is the entry point that `validate.py` imports — you can create additional files (`.cu`, `.py`, etc.) as needed. One hypothesis per experiment.

### 4. Commit

```bash
git add candidate/ && git commit -m "a{AGENT_ID}/{n}: {description}"  # stages all files in candidate/
```

### 5. Validate

```bash
uv run python validate.py > run.log 2>&1
grep "candidate_us\|reference_us\|correctness\|peak_vram_mb" run.log
```

If grep is empty, the run crashed. Read `tail -n 50 run.log` for the traceback. If it is a trivial fix (typo, import), fix and re-run. Otherwise log as crash and move on.

### 6. Compute speedup

```
speedup = reference_us / candidate_us
```

### 7. Log result

Append one tab-separated row to `results.tsv`. Use `profile_utils.append_result` for file-locked writes when multiple agents share the file, or just `echo -e "..." >> results.tsv` for single-agent runs.

Columns (tab-separated):

```
experiment_id  parent_id  agent_id  commit  timestamp  candidate_us  reference_us  speedup  correctness  peak_vram_mb  status  description
```

Set `parent_id = current_base`. Set `commit` = 7-char hash from `git rev-parse --short HEAD`.

### 8. Keep or discard

**If** `correctness == PASS` **and** `speedup > best_speedup`:
- `status = "keep"`, `current_base = "a{AGENT_ID}/{n}"`, `best_speedup = speedup`

**Else** (FAIL, CRASH, or not faster):
- `status = "discard"` (or `"crash"`)
- `git checkout {current_base}`

### 9. Repeat

Increment `n`. Go to step 1.

If you run out of ideas: use `ncu` or the microbench agent to find bottlenecks, try combining previous near-misses, or try a radically different backend. If you feel stuck, think harder -- re-read the kernel code, try a fundamentally different algorithm, or switch backends entirely.

## Simplicity criterion

All else being equal, simpler is better. A small speedup that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome -- that's a simplification win. When deciding whether to keep a change, weigh the complexity cost against the speedup magnitude.

## Constraints and Tools

### Tools

| Tool | Use |
|------|-----|
| `ncu` | NVIDIA Nsight Compute -- kernel-level profiling |
| `nsight-systems` | System-wide timeline profiling |
| microbench agent | Spawns sub-agent returning per-line sub-op breakdown table (see `agents/microbench.md`) |

### Constraints

- Never modify `validate.py` or `reference.py`
- Never skip correctness checks
- One focused change per experiment
- VRAM must stay below 80% of GPU capacity
- Always preserve experiment branches (no `git reset --hard`, no `git branch -D`)

### Backends

Use whichever is appropriate: PyTorch, Triton, CUDA C++, CUTLASS, CUTE DSL, PTX.
