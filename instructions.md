# AutoKernel Agent Playbook

You are an autonomous kernel optimization agent. You modify code inside `candidate/` and run `validate.py` in a tight loop to minimize kernel latency. You never stop.

## Setup

Environment variables you receive:

| Variable | Example | Description |
|----------|---------|-------------|
| `AGENT_ID` | `0` | Your numeric agent ID |
| `WORKTREE_PATH` | `worktree-a0` | Your git worktree directory |
| `CUDA_VISIBLE_DEVICES` | `0` | Your pinned GPU |
| `AUTOKERNEL_EXPERIMENTS_TSV` | `/path/to/results/experiments.tsv` | Shared experiment log. Always append here. |
| `AUTOKERNEL_NOTES_DIR` | `/path/to/results/notes` | Shared detailed experiment notes. Write one note per experiment when possible. |
| `AUTOKERNEL_REFERENCE_TIMING_PATH` | `/path/to/results/reference_timing.json` | Calibrated reference runtime used by `validate.py`. |

### File rules

| File | Mutable? | Rule |
|------|----------|------|
| `validate.py` | NO | Never modify. Run it to test. |
| `reference.py` | NO | Never modify. Ground truth. |
| `candidate/` | YES | All your optimization code goes here. `interface.py` is the Python entry point that `validate.py` imports. You may create additional files (`.py`, `.cu`, `.cuh`, etc.) inside `candidate/`. |
| `$AUTOKERNEL_EXPERIMENTS_TSV` | APPEND-ONLY | Never delete or rewrite rows. |
| `$AUTOKERNEL_NOTES_DIR` | APPEND-ONLY | Write detailed notes when possible. Never delete or rewrite existing notes. |

Do not commit `results/` or `run.log` to git. Leave them untracked.

### Experiment naming

Format: `a{agent_id}/{experiment_number}` -- used as branch name, experiment_id, and lookup key.

Examples: `a0/0` (baseline), `a0/1` (first experiment), `a1/3` (agent 1, fourth experiment).

### Record baseline

```bash
cd $WORKTREE_PATH
git branch a${AGENT_ID}/0 HEAD 2>/dev/null || true
mkdir -p "$(dirname "$AUTOKERNEL_EXPERIMENTS_TSV")"
mkdir -p "$AUTOKERNEL_NOTES_DIR"
uv run python validate.py > run.log 2>&1
grep "candidate_us\|reference_us\|correctness\|peak_vram_mb" run.log
uv run python "$AUTOKERNEL_ROOT/scripts/record_result.py" \
  --experiment-id "a${AGENT_ID}/0" \
  --parent-id "-" \
  --status keep \
  --description baseline
NOTE_PATH="$AUTOKERNEL_NOTES_DIR/a${AGENT_ID}_0.md"
# Write a detailed baseline note at $NOTE_PATH after recording the row.
```

The record script appends the baseline row to `$AUTOKERNEL_EXPERIMENTS_TSV` with file locking. The TSV row is mandatory; the note is shared learning context and should be written when possible. Set `current_base = "a{AGENT_ID}/0"`, `best_speedup = 1.0`, `n = 1`.

## Experiment Loop (NEVER STOP)

Run this loop indefinitely. Do not pause to ask the human anything. Do not ask "should I keep going?" or "is this a good stopping point?". The human might be asleep or away and expects you to continue working *indefinitely* until manually stopped. You are autonomous.

### 1. Read Shared Memory

Before choosing a hypothesis, inspect the shared experiment memory:

```bash
tail -n 40 "$AUTOKERNEL_EXPERIMENTS_TSV"
find "$AUTOKERNEL_NOTES_DIR" -maxdepth 1 -name '*.md' | sort | tail -n 12
```

Read notes for recent experiments, best kept experiments, and similar failed ideas. Avoid repeating changes that already failed unless you can explain what is different.

### 2. Hypothesize

Pick ONE focused change. Write it down as a short description (e.g., "fuse norm+proj", "shared memory tiling", "vectorized loads").

### 3. Branch

```bash
git checkout -b a{AGENT_ID}/{n} {current_base}
```

### 4. Edit

Modify files inside `candidate/`. `interface.py` is the entry point that `validate.py` imports — you can create additional files (`.cu`, `.py`, etc.) as needed. One hypothesis per experiment.

### 5. Commit

```bash
git add candidate/ && git commit -m "a{AGENT_ID}/{n}: {description}"  # stages all files in candidate/
```

### 6. Validate

```bash
uv run python validate.py > run.log 2>&1
grep "candidate_us\|reference_us\|correctness\|peak_vram_mb" run.log
```

If grep is empty, the run crashed. Read `tail -n 50 run.log` for the traceback. If it is a trivial fix (typo, import), fix and re-run. Otherwise log as crash and move on.

### 7. Compute Speedup And Status

```
speedup = reference_us / candidate_us
```

Decide `status` before recording the row:

- `keep` if `correctness == PASS` and `speedup > best_speedup`
- `discard` if correctness passed but speedup did not improve
- `crash` if validation crashed or did not print usable metrics

### 8. Log Result

Append one tab-separated row to `$AUTOKERNEL_EXPERIMENTS_TSV` using:

```bash
uv run python "$AUTOKERNEL_ROOT/scripts/record_result.py" \
  --experiment-id "a${AGENT_ID}/${n}" \
  --parent-id "${current_base}" \
  --status "${status}" \
  --description "${description}"
```

Use this script for every result, including crashes and failed experiments. It parses `run.log`, computes `speedup`, and uses `profile_utils.append_result` for file-locked writes. Do not write to a worktree-local `results/experiments.tsv`, and do not use `echo >>` for experiment rows.

### 9. Write Detailed Note

Write one Markdown note after recording the row when possible:

```bash
NOTE_PATH="$AUTOKERNEL_NOTES_DIR/a${AGENT_ID}_${n}.md"
```

The TSV row is the source of truth and must always be written. The note is shared memory for learning and should be thorough enough for other agents to learn from it. Prefer this format:

```markdown
# a{AGENT_ID}/{n}: {description}

Parent: {current_base}
Status: {keep|discard|crash}
Commit: {short_commit}

## Hypothesis
What you expected to improve and why.

## Change
What files/code paths changed. Include key parameters such as tile sizes, warps, stages, vector widths, backend, and fast-path guards.

## Result
Paste the four validate.py metrics and summarize whether latency improved versus parent/current best.

## Lessons
What this result suggests about the kernel, memory traffic, launch overhead, compiler behavior, or validation case.

## Followups
Concrete next experiments suggested by this result, or what not to try again.
```

Columns (tab-separated):

```
experiment_id  parent_id  agent_id  commit  timestamp  candidate_us  reference_us  speedup  correctness  peak_vram_mb  status  description
```

Set `parent_id = current_base`. Set `commit` = 7-char hash from `git rev-parse --short HEAD`.
`reference_us` is a calibrated constant read by `validate.py`; do not re-time the reference implementation during experiments.
The reported `candidate_us` and calibrated `reference_us` both correspond to the single stress benchmark case from `validate.make_inputs()`. Correctness-only cases are broader coverage and do not affect the reported timing case.

### 10. Keep or discard

**If** `correctness == PASS` **and** `speedup > best_speedup`:
- `status = "keep"`, `current_base = "a{AGENT_ID}/{n}"`, `best_speedup = speedup`

**Else** (FAIL, CRASH, or not faster):
- `status = "discard"` (or `"crash"`)
- `git checkout {current_base}`

### 11. Repeat

Increment `n`. Go to step 1.

If you run out of ideas: use `ncu` or the microbench agent/skill to find bottlenecks, try combining previous near-misses, or try a radically different backend. If you feel stuck, think harder -- re-read the kernel code, try a fundamentally different algorithm, or switch backends entirely.

## Simplicity criterion

All else being equal, simpler is better. A small speedup that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome -- that's a simplification win. When deciding whether to keep a change, weigh the complexity cost against the speedup magnitude.

## Constraints and Tools

### Tools

| Tool | Use |
|------|-----|
| `ncu` | NVIDIA Nsight Compute -- kernel-level profiling |
| `nsys` | NVIDIA Nsight Systems timeline profiling |
| microbench agent/skill | Returns per-line sub-op breakdown table (see `agents/microbench.md`) |

### Constraints

- Never modify `validate.py` or `reference.py`
- Never skip correctness checks
- One focused change per experiment
- VRAM must stay below 80% of GPU capacity
- Always preserve experiment branches (no `git reset --hard`, no `git branch -D`)

### Backends

Use whichever is appropriate: PyTorch, Triton, CUDA C++, CUTLASS, CUTE DSL, PTX.
