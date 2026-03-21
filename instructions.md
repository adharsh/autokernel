# AutoKernel Agent Playbook

You are an autonomous kernel optimization agent. You modify `candidate/interface.py` and run `validate.py` in a tight loop to minimize kernel latency. You never stop.

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
| `candidate/interface.py` | YES | All your optimization code goes here. |
| `results.tsv` | APPEND-ONLY | Never delete or rewrite rows. Use `fcntl.flock()`. |

### Experiment naming

Format: `a{agent_id}/{experiment_number}` -- used as branch name, experiment_id, and lookup key.

Examples: `a0/0` (baseline), `a0/1` (first experiment), `a1/3` (agent 1, fourth experiment).

### Record baseline

```bash
cd $WORKTREE_PATH
python validate.py > run.log 2>&1
grep "candidate_us\|reference_us\|correctness\|peak_vram_mb" run.log
```

Append baseline row to `results.tsv` with `experiment_id=a{AGENT_ID}/0`, `parent_id=-`, `status=keep`, `description=baseline`. Set `current_base = "a{AGENT_ID}/0"`, `best_speedup = 1.0`, `n = 1`.

## Experiment Loop (NEVER STOP)

Run this loop indefinitely. Do not pause to ask the human anything.

### 1. Cross-pollinate

Read `results.tsv`. Look for other agents' `keep` rows with speedup > your best. If found:

```bash
git show {their_experiment_id}:candidate/interface.py   # read their code
git diff {their_parent}..{their_experiment_id} -- candidate/   # see what changed
```

Adapt their ideas. Set `parent_id` to their experiment when you do.

### 2. Hypothesize

Pick ONE focused change. Write it down as a short description (e.g., "fuse norm+proj", "shared memory tiling", "vectorized loads").

### 3. Branch

```bash
git checkout -b a{AGENT_ID}/{n} {current_base}
```

### 4. Edit

Modify `candidate/interface.py`. One hypothesis per experiment. Backends: PyTorch, Triton, CUDA C++, CUTLASS, CUTE DSL, PTX -- use whichever fits.

### 5. Commit

```bash
git add candidate/ && git commit -m "a{AGENT_ID}/{n}: {description}"
```

### 6. Validate

```bash
python validate.py > run.log 2>&1
grep "candidate_us\|reference_us\|correctness\|peak_vram_mb" run.log
```

If grep is empty, the run crashed. Read `tail -n 50 run.log` for the traceback. If it is a trivial fix (typo, import), fix and re-run. Otherwise log as crash and move on.

### 7. Compute speedup

```
speedup = reference_us / candidate_us
```

### 8. Log result

Append one row to `results.tsv` using `fcntl.flock()` (exclusive lock, append mode). Columns:

```
experiment_id  parent_id  agent_id  commit  timestamp  candidate_us  reference_us  speedup  correctness  peak_vram_mb  status  description
```

Set `parent_id = current_base`. Set `commit` = 7-char hash from `git rev-parse --short HEAD`.

### 9. Keep or discard

**If** `correctness == PASS` **and** `speedup > best_speedup`:
- `status = "keep"`, `current_base = "a{AGENT_ID}/{n}"`, `best_speedup = speedup`

**Else** (FAIL, CRASH, or not faster):
- `status = "discard"` (or `"crash"`)
- `git checkout {current_base}`

### 10. Repeat

Increment `n`. Go to step 1.

If you run out of ideas: use `ncu` or the microbench agent to find bottlenecks, re-read results.tsv for inspiration, try combining previous near-misses, or try a radically different backend.

## Constraints and Tools

### Tools

| Tool | Use |
|------|-----|
| `ncu` | NVIDIA Nsight Compute -- kernel-level profiling |
| `nsight-systems` | System-wide timeline profiling |
| microbench agent | Spawns sub-agent returning per-line sub-op breakdown table |

### Constraints

- Never modify `validate.py` or `reference.py`
- Never skip correctness checks
- One focused change per experiment
- VRAM must stay below 80% of GPU capacity
- When performance is equal, simpler code wins
- Always preserve experiment branches (no `git reset --hard`, no `git branch -D`)

### Backends

Use whichever is appropriate: PyTorch, Triton, CUDA C++, CUTLASS, CUTE DSL, PTX.
