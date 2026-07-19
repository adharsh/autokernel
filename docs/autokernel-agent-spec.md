# AutoKernel Agent System — Full Specification

This document specifies the complete autonomous kernel optimization agent system. A coding agent should be able to implement all components from this spec alone.

---

## 1. Project Structure

```
autokernel/
├── validate.py              # Normally fixed correctness + calibrated reference metadata
├── reference.py             # Normally fixed reference implementation (ground truth)
├── candidate/
│   ├── __init__.py
│   └── interface.py         # Agent edits THIS — all optimization code, bindings, utilities
├── instructions.md          # Agent playbook (like autoresearch/program.md)
├── results/
│   ├── experiments.tsv      # Shared, append-only experiment log (tab-separated)
│   ├── reference_timing.json # Calibrated NCU timing for the benchmark target
│   └── experiments/         # One artifact folder per experiment
│       └── a0_1/
│           ├── note.md
│           ├── run.log
│           ├── ncu/
│           │   ├── profile.ncu-rep
│           │   ├── profile.log
│           │   └── details.txt
│           ├── nsys/
│           ├── microbench/
│           └── codegen/
├── analysis.py              # Plotting + reports from results/experiments.tsv
├── profile_utils.py         # NCU duration parser and shared TSV utilities
└── scripts/
    ├── agents.sh            # Multi-agent launcher (one per GPU)
    ├── calibrate_reference.py # One-time reference timing calibration
    ├── profile_candidate_once.py # One warmed-up candidate benchmark pass
    ├── profile_reference_once.py # One warmed-up reference benchmark pass
    ├── profile_ncu.sh        # Required per-experiment Nsight Compute profiling
    ├── record_result.py      # File-locked experiment row appender
    └── setup.sh             # Local setup helper
```

### File Mutability Rules

| File | Who edits | Rules |
|------|-----------|-------|
| `validate.py` | Human / deliberate agent reformulation | Normally fixed. May change only for a committed input/interface reformulation that preserves the same mathematical workload |
| `reference.py` | Human / deliberate agent reformulation | Normally fixed ground truth. May change only together with `validate.py` for the same input/interface reformulation |
| `candidate/interface.py` | Agent | All optimization code lives here |
| `candidate/__init__.py` | Agent | Exports from interface.py |
| `results/experiments.tsv` | Agent (append-only) | Must match the current TSV header for this run. Never delete, reorder, or alter experiment data during a run |
| `results/experiments/*` | Agent/tool output | Per-experiment artifacts. Never delete or rewrite existing experiment folders |
| `instructions.md` | Human | Agent reads, never modifies |
| `analysis.py` | Human / setup | Agent may run but never modifies |

### Validation Harness Contract

`validate.py` separates timing from correctness:

- It defines one official benchmark target in `BENCHMARK_CASES`. The target may
  be a single case or a small task-specific suite.
- `validate.make_benchmark_inputs()` and `run_benchmark_suite()` expose one
  complete timing pass. `make_stress_inputs()` may expose the primary case for
  focused microbenchmarks, but it is not the aggregate score when a suite is used.
- `scripts/calibrate_reference.py` profiles `reference.kernel_fn` on the complete benchmark target once with a lightweight NCU pass and writes `results/reference_timing.json`.
- Every `validate.py` run prints the calibrated NCU-based `reference_us`; candidate timing comes from `scripts/profile_ncu.sh`.
- `validate.py` must not compute candidate duration with CUDA events, wall-clock timers, repeated medians, or other in-harness timing. The official candidate duration is the sum of NCU `Duration us` rows from `scripts/profile_ncu.sh`.
- Correctness-only cases broaden behavioral coverage but do not change the profiled `ncu_duration_us` timing case.
- Deliberate input/API reformulations may change `validate.py` and
  `reference.py` together while preserving the same semantic task. Record the
  representation in `interface_variant` and do not add precomputed operator work
  as inputs.

---

## 2. Results TSV Schema

**Format**: Tab-separated values (TSV).

**Header**:
```
experiment_id	parent_id	agent_id	commit	timestamp	ncu_duration_us	ncu_kernel_count	reference_us	speedup	correctness	peak_vram_mb	status	interface_variant	description	experiment_elapsed_s
```

**Column definitions**:

| Column | Type | Example | Description |
|--------|------|---------|-------------|
| `experiment_id` | string | `a0/1` | Unique ID. Also the git branch name. Format: `a{agent_id}/{seq_number}` |
| `parent_id` | string | `a0/0` | Experiment this was built on. `-` for baseline. Enables lineage tree reconstruction |
| `agent_id` | int | `0` | Which agent ran this |
| `commit` | string | `a1b2c3d` | 7-char git commit hash |
| `timestamp` | ISO8601 | `2026-03-20T14:32:00Z` | When the experiment completed |
| `ncu_duration_us` | float | `42.3` | Sum of Nsight Compute kernel `Duration us` rows in microseconds |
| `ncu_kernel_count` | int | `1` | Number of NCU kernel `Duration` rows summed for the profiled invocation |
| `reference_us` | float | `85.1` | Calibrated reference latency in microseconds for one complete benchmark-target pass |
| `speedup` | float | `2.01` | `reference_us / ncu_duration_us` |
| `correctness` | string | `PASS` | `PASS`, `FAIL`, or `CRASH` |
| `peak_vram_mb` | float | `2048.5` | Peak GPU memory used during benchmark |
| `status` | string | `keep` | `keep`, `discard`, or `crash` |
| `interface_variant` | string | `default` | Input/API representation for this row, e.g. `default`, `seq_idx`, `cu_seqlens`, or `packed_layout` |
| `description` | string | `fuse norm+proj` | Short description of what was tried |
| `experiment_elapsed_s` | float | `742` | Wall-clock seconds since this agent's previous recorded row, or since session start for its first row |

`interface_variant` is provenance metadata, not an execution switch. The
experiment branch/commit is the source of truth for the actual interface and
implementation. Agents read this field from the shared TSV and notes when
deciding whether to continue from a variant branch or return to a compatible
default-interface ancestor.

**Concurrency**: Multiple agents append to the same file. Use `scripts/record_result.py`, which calls `profile_utils.append_result()` and uses `fcntl.flock()` for atomic writes. Agents must not use `echo >>` for experiment rows.

The TSV is intentionally raw tab-separated data. Values are sanitized before
append so tabs and newlines inside descriptions cannot corrupt row structure.
Use `scripts/format_results.py` for aligned human-readable output instead of
padding or manually editing the TSV.

`scripts/record_result.py` must be used for every result, including crashes. For
`status=crash`, it appends a row even if validation failed before printing all
metrics; missing timing/VRAM values are recorded as `nan`, and missing
correctness is recorded as `CRASH`.

## 2.1 Experiment Artifacts

`results/experiments.tsv` is the compact index. Detailed shared memory lives
under one folder per experiment, with deterministic folder names derived from
experiment IDs:

| Experiment | Artifact folder | Note path |
|------------|-----------------|-----------|
| `a0/1` | `results/experiments/a0_1/` | `results/experiments/a0_1/note.md` |
| `a7/23` | `results/experiments/a7_23/` | `results/experiments/a7_23/note.md` |

Canonical artifact folder:

```text
results/experiments/a{agent_id}_{n}/
├── note.md
├── run.log
├── ncu/
│   ├── profile.ncu-rep
│   ├── profile.log
│   └── details.txt
├── nsys/
├── microbench/
└── codegen/
```

Each note should include:

- `## Hypothesis`
- `## Change`
- `## Result`
- `## NCU Profile`
- `## Speed-of-Light Gap`
- `## Design Decision From Profile`
- `## Codegen/PTX/SASS`
- `## Lessons`
- `## Followups`

TSV rows are the source of truth and must be recorded for every experiment.
Notes are shared learning context and should be written when possible. This
keeps the TSV compact while giving agents context to avoid repeating failed
ideas and to build on successful ones.

`## NCU Profile` is mandatory for any experiment that launches GPU kernels. It
should name the `ncu/profile.ncu-rep` report in the experiment folder and
summarize achieved SM throughput, achieved memory throughput, occupancy, memory
traffic, instruction mix if relevant, dominant stall reasons, and notable
launch/runtime facts. If the candidate crashed before kernel launch, the note
must say NCU was not able to profile a kernel.

`## Speed-of-Light Gap` must state how far the candidate is from speed of light,
using Nsight Compute SOL/roofline percentages when available. It should identify
the current limiting factor: compute, memory bandwidth, latency/occupancy,
launch overhead, framework overhead, compiler/codegen, or another measured
limit. `## Design Decision From Profile` must connect that evidence to the next
experiment and backend choice.

`## Codegen/PTX/SASS` must always state whether generated code was inspected.
Inspection itself is profile-triggered, not mandatory for every experiment. It is
expected when NCU indicates a codegen or instruction-level issue such as
unexpected instruction mix, missing tensor cores, poor vectorization/coalescing,
register pressure, spills/local memory, excessive predication, unrolling issues,
or suspicious compiler behavior. Prefer SASS/cubin disassembly when available
because it is closer to executed machine code than PTX. Save inspected artifacts
under the experiment folder's `codegen/` subdirectory when practical.

---

## 3. Experiment Naming Convention

Format: `a{agent_id}/{experiment_number}`

- `a0/0` — agent 0, baseline
- `a0/1` — agent 0, first experiment
- `a1/0` — agent 1, baseline
- `a1/3` — agent 1, fourth experiment

This string is used as:
1. The `experiment_id` column in results/experiments.tsv
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

`keep` is decided against finalized compatible `keep` rows already present in
the shared TSV. Pending notes, partial profiles, and unrecorded results from
other agents may inform the next hypothesis, but they must not be used to record
a completed `PASS` experiment as `discard` when its speedup beats the finalized
TSV best. Such a row should be kept unless diagnostics, task hints, or fairness
constraints genuinely disqualify it.

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
5. Append row to shared root results/experiments.tsv
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
results/experiments.tsv        (shared, append-only, file-locked)
├── worktree-a0/   (GPU 0, branch a0/0)
├── worktree-a1/   (GPU 1, branch a1/0)
└── worktree-a2/   (GPU 2, branch a2/0)
```

### Launch script (`scripts/agents.sh`)
Each agent is a Codex or Claude process running with a pinned GPU:
```bash
./scripts/setup.sh
./scripts/agents.sh start

# Optional Claude backend
AGENT_CLI=claude ./scripts/agents.sh start
```

The launcher detects GPUs with `nvidia-smi`, initializes `results/experiments.tsv`, creates `a{id}/0` baseline branches, adds one worktree per agent, and writes logs under `results/logs/`.

---

## 6.1 Profiling Ground Truth

Profiling is the ground truth for optimization decisions. Every baseline and
every recordable experiment that launches GPU kernels must run the official
basic Nsight Compute timing profile before its result is used to choose the
next design:

```bash
scripts/profile_ncu.sh "a${AGENT_ID}/${n}" basic
```

The helper runs:

```bash
ncu --set basic --target-processes all --kernel-name-base demangled --profile-from-start off ...
```

and stores:

- `results/experiments/a{AGENT_ID}_{n}/ncu/profile.ncu-rep`
- `results/experiments/a{AGENT_ID}_{n}/ncu/profile.log`
- `results/experiments/a{AGENT_ID}_{n}/ncu/details.txt`

The raw `validate.py` pass remains the source of correctness, `reference_us`,
and VRAM values for `results/experiments.tsv`; NCU `Duration us` is the source
of `ncu_duration_us`. With
the default command, `scripts/profile_ncu.sh` profiles
`scripts/profile_candidate_once.py`, which calls `validate.make_benchmark_inputs()`,
warms up every benchmark case, starts CUDA profiling, runs one complete target
pass, synchronizes, and stops profiling.

Use supplemental `detailed` profiles for every non-baseline `keep`, every new
kernel family/backend/dataflow, and whenever `basic` does not explain the
bottleneck. Use `full` only when scheduler, warp-state, instruction-mix,
PM-sampling, or codegen evidence is needed. Supplemental profiles are written
under `ncu/detailed/` or `ncu/full/`; they do not replace the official basic
timing consumed by `results/experiments.tsv`. Design decisions must cite the
latest relevant NCU evidence, especially speed-of-light metrics, SM and memory
throughput, occupancy, memory traffic, instruction mix, dominant stall reasons,
and launch/runtime behavior when those metrics are available.

Before the first optimization pass for a task, agents should read NVIDIA's
official Nsight Compute documentation and Kernel Profiling Guide and use them as
the metric interpretation reference:

- https://docs.nvidia.com/nsight-compute/NsightCompute/index.html
- https://docs.nvidia.com/nsight-compute/pdf/ProfilingGuide.pdf

`nsys` and microbench profiling are optional follow-up tools. Use them when NCU
shows a timeline/launch/synchronization question or when per-line attribution is
needed. They complement the required NCU profile and do not replace it.

Backend selection is downstream of profiling and the current experiment
hypothesis. Agents should explore the core stack freely when there is a
plausible profile, codegen, or algorithmic reason: PyTorch for
reference/scaffolding, Triton, CUDA C++, CuTe/CUTLASS, and CUDA C++ with inline
PTX/SASS-guided work. The usual fit of each backend is guidance, not a
checklist; the measured limiter and the hypothesis decide.

The core stack is sufficient to reach the relevant NVIDIA hardware mechanisms:
custom memory layouts and fused kernels through Triton, explicit launch and
memory-control through CUDA C++, Tensor Core and tiled kernel construction
through CuTe/CUTLASS, and last-mile instruction/codegen work through inline PTX
or SASS-guided CUDA. Experimental DSLs may improve productivity or provide
excellent examples, but they do not unlock a separate roofline that the core
stack cannot reach in principle.

Do not allow agents to install, vendor, add to `pyproject.toml`, or use new
experimental DSLs as implementation backends by default. This includes TileLang,
Gluon/TLX, Helion, cuTile/TileIR, ThunderKittens, and similar systems. The main
risks are dependency drift across agents, CUDA/PyTorch version conflicts,
non-reproducible results, install/debug time displacing optimization time,
unreviewed third-party code in the benchmark path, and noisy experiment
provenance.

Those repositories are still valuable research inputs. Agents should read them
for algorithms, tiling, dataflow, scheduling, masking, quantization, grouped-GEMM,
MoE routing, and codegen ideas, then reimplement the idea in the allowed core
stack. Relevant examples include DeepSeek TileKernels, DeepSeek DeepGEMM,
TileLang operator libraries, Gluon/TLX kernels, Helion examples, cuTile/TileIR
examples, and ThunderKittens kernels.

An experimental DSL backend is allowed only by explicit human override. Treat
that as a scoped exception: use an isolated branch/environment, state the exact
hypothesis, avoid unrelated dependency churn, keep the experiment reproducible,
and promote the dependency only if it clearly wins and the human accepts the
maintenance cost.

PTX/SASS inspection is part of that escalation path. It is not a replacement for
NCU and should not be forced on every experiment; it is required when the NCU
evidence makes codegen the active question.

---

## 7. Microbench Agent

### Purpose
A dedicated agent that writes and runs small Nsight Compute profiling targets
for the current candidate code. It answers narrow follow-up questions such as
"which kernel launch in this path is taking time?" or "what limiter changed
after this implementation change?"

This workflow is exposed as a Claude Code agent and as a Codex skill. It needs enough context to read code, write benchmarks, run them, and analyze results. The parent optimization agent only receives the summary table.

Microbench profiling is a follow-up tool for attribution. It never replaces the
required Nsight Compute profile for an experiment.

### Workflow
1. **Read the candidate code** — `candidate/interface.py`
2. **Identify every compute line** — map each line to a logical sub-operation
3. **Write focused NCU targets** — isolate one candidate path or sub-operation
4. **Run profiles** — execute through `ncu` and import the details page
5. **Return structured report** — sub-op breakdown table

### Key principles
- Every focused question gets an isolated profiling target
- Separate CUDA kernel calls are profiled separately
- If a callable launches multiple kernels, report each kernel duration and the sum
- Each profiling target must identify the source path or line it measures

### Tools available to the microbench agent
- `ncu`, `scripts/profile_ncu.sh`, and `profile_utils.ncu_duration_rows_us`
- Read/Write/Bash for code generation and execution

**Note**: the microbench workflow complements the required NCU report. It does
not replace the official per-experiment profile.

### Output format
```
Sub-op Breakdown for candidate/interface.py
============================================
Sub-op              Line    Duration (us)   % of Total    Limiter
------------------------------------------------------------------
matmul_qkv          42      312.0           38.2%         tensor pipe
softmax             55      185.0           22.6%         memory dependency
attention_score     60      142.0           17.4%         scheduler
norm                38       98.0           12.0%         memory bandwidth
index_select        35       52.0            6.4%         uncoalesced load
------------------------------------------------------------------
TOTAL                       789.0           100.0%

Bottleneck: matmul_qkv (38.2%)
```

---

## 8. Main Optimization Agent (instructions.md)

### Overview
Each agent follows the same playbook, running autonomously in its own worktree on a pinned GPU. The agent normally modifies `candidate/interface.py` and runs `validate.py` in a tight loop. For deliberate input/interface reformulations, the agent may also update `validate.py` and `reference.py` under the constraints below.

### Experiment loop (NEVER STOP — run indefinitely)
```
1. Read shared experiments.tsv and recent/best experiment folders/notes
2. Hypothesize one focused optimization change
3. git checkout -b a{id}/{n} {current_base}    # new branch from last keep
4. Edit candidate/interface.py; for input/interface reformulations, also update validate.py and reference.py together
5. git add candidate/ && git commit -m "a{id}/{n}: <description>" (also stage validate.py/reference.py for interface reformulations)
6. Run: uv run python validate.py → write results/experiments/a{id}_{n}/run.log and extract calibrated reference_us, correctness, peak_vram_mb
7. Run required NCU profile: scripts/profile_ncu.sh "a{id}/{n}" basic, then create/update note.md from the profile evidence
8. Compute speedup = reference_us / ncu_duration_us and decide status against finalized compatible keep rows in the TSV; if this is a non-baseline keep, run detailed and update the same note.md
9. Finalize results/experiments/a{id}_{n}/note.md with interface_variant, NCU findings, speed-of-light gap, limiter, next design decision, and codegen inspected yes/no
10. Append row to the shared root results/experiments.tsv through scripts/record_result.py (with file lock, parent_id = current_base, interface_variant recorded)
11. If status == "keep":
     current_base = experiment_id           # advance base (stay on this branch)
   Else:
     git checkout {current_base}            # go back to last keep
12. Repeat from step 1
```

Every experiment's branch is preserved regardless of outcome. To inspect any experiment later: `git checkout a{id}/{n}` or `git show a{id}/{n}:candidate/interface.py`.

### Tools available
You have access to: `ncu`, `scripts/profile_ncu.sh`, `nsys`, and a **microbench agent/skill** that writes xllm-style line-by-line microbenchmarks of your candidate code and returns a sub-op breakdown table. `scripts/profile_ncu.sh EXPERIMENT_ID basic` is required for every baseline and recordable experiment that launches kernels. Use `detailed`/`full`, `nsys`, and microbench profiling for specific follow-up questions raised by NCU.

### Submission integrity
Agents optimize the implementation, not the evaluator. They must not memoize
answers, cache or replay final outputs, hardcode benchmark results, special-case
known benchmark/test inputs, detect evaluator behavior, skip correctness paths,
or use reward hacking. Legitimate compiler, extension, or autotuning artifact
caches are allowed only when they do not cache final answers or depend on
recognizing the validation case.

For convolution backward tasks, agents must not hide operator work in the input
interface. Precomputed forward outputs, activation derivatives, convolution
windows, validity matrices, partial reductions, partial gradients, transformed
inputs/weights, or other operator work are diagnostic-only unless the human
explicitly changes the benchmark contract. Compact problem metadata such as
sequence ids, sequence lengths, offsets, or declared layouts can be valid
interface variants when they preserve the same mathematical workload.

### Optimization strategy
Use the core backend stack according to the measured limiter and experiment
hypothesis: PyTorch for reference/scaffolding, Triton, CUDA C++,
CuTe/CUTLASS, and CUDA C++ with inline PTX/SASS-guided work. Keep changes
focused -- one hypothesis per experiment. Backend changes must cite the latest
NCU speed-of-light gap and limiting factor. If an obvious speedup is not
available, agents must inspect deeper profiling details and try justified
lower-level changes before moving on.

Experimental DSLs are research sources by default, not implementation
backends. Agents may read TileLang, Gluon/TLX, Helion, cuTile/TileIR,
ThunderKittens, DeepSeek TileKernels, DeepSeek DeepGEMM, and similar repos for
ideas, but must not install or depend on those toolchains unless the human has
explicitly approved a scoped exception.

### Constraints
- Never modify `validate.py` or `reference.py` except for a deliberate, committed input/interface reformulation that preserves the same mathematical workload
- Never memoize answers, hardcode outputs, special-case tests, detect evaluator behavior, or reward-hack
- Never add precomputed operator work to inputs
- Under the current contract, never promote precomputed convolution/backward
  operator work as free input state; it can become official only if the human
  explicitly changes the benchmark contract
- Never skip correctness checks
- Never skip the required NCU profile for a baseline or experiment that launches kernels
- One focused change per experiment
- VRAM must not exceed 80% of GPU capacity
- Simpler code wins when performance is equal

---

## 9. Visualization & Analysis (`analysis.py`)

### Inputs
- `results/experiments.tsv` (the shared experiment log)

### Outputs
1. `progress.html` — main visualization
2. Terminal summary — printed to stdout
3. `report.md` — markdown session report (optional)

### Main plot: Latency over experiments
- **Y-axis**: `ncu_duration_us` (NCU kernel duration in microseconds, lower is better)
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
- **Reference baseline dashed line** (`#3498db`): calibrated reference latency for one complete benchmark-target pass
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
  NCU artifacts:         47/47
  Profile note sections: 47/47

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

### Profiling coverage audit
`analysis.py` also checks that each experiment has an NCU report/log under its
`results/experiments/<experiment>/ncu/` folder and that its `note.md` includes
`## NCU Profile`, `## Speed-of-Light Gap`,
`## Design Decision From Profile`, and `## Codegen/PTX/SASS`. Missing artifacts
or sections are listed in `results/report.md`.

---

## 10. File Locking for Shared results/experiments.tsv

Multiple agents append to the same file concurrently. Use OS-level file locking:

```python
import fcntl
import os

def append_result(csv_path: str, row: dict, columns: list[str]) -> None:
    """Atomically append a row to the shared results/experiments.tsv."""
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
    """Read results/experiments.tsv with shared lock (allows concurrent reads)."""
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
# Initialize results/experiments.tsv with header and create artifact root
mkdir -p results/experiments
printf 'experiment_id\tparent_id\tagent_id\tcommit\ttimestamp\tncu_duration_us\tncu_kernel_count\treference_us\tspeedup\tcorrectness\tpeak_vram_mb\tstatus\tinterface_variant\tdescription\texperiment_elapsed_s\n' > results/experiments.tsv

# Verify reference and validation work. Calibration writes
# results/reference_timing.json and is intentionally lightweight by default.
uv run python scripts/calibrate_reference.py
AUTOKERNEL_ALLOW_REFERENCE_BASELINE=1 uv run python validate.py  # bwd2 baseline

# Make task files available to all git worktrees
git add validate.py reference.py candidate/
git commit -m "task setup"
```

### Per-agent setup (automated)
```bash
# Agent receives:
# AGENT_ID, CUDA_VISIBLE_DEVICES, WORKTREE_PATH,
# AUTOKERNEL_EXPERIMENTS_TSV, AUTOKERNEL_EXPERIMENTS_DIR,
# AUTOKERNEL_REFERENCE_TIMING_PATH
# Agent does:
cd $WORKTREE_PATH
# Baseline branch a{AGENT_ID}/0 was created by the launcher.

# Run baseline. reference_us is calibrated for the complete benchmark target.
AUTOKERNEL_ALLOW_REFERENCE_BASELINE=1 uv run python validate.py → extract reference_us, correctness, peak_vram_mb
scripts/profile_ncu.sh "a${AGENT_ID}/0" basic → write required baseline NCU report
write results/experiments/a${AGENT_ID}_0/note.md before recording

# Record baseline
append_result("a${AGENT_ID}/0", parent_id="-", status="keep", interface_variant="default", description="baseline")
# current_base = "a${AGENT_ID}/0"
```

If the initial `candidate/interface.py` is a wrapper around `reference.kernel_fn`,
use that wrapper only for the baseline. Non-baseline experiments must make a real
implementation change and must run without the baseline override; do not record
another reference wrapper as `a*/1+`.

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
| 2 | `profile_utils.py` | `profile_utils.py` | NCU duration parsing, `append_result`, `read_results` utilities |
| 3 | `analysis.py` | `analysis.py` | Plotting, terminal summary, report generation. Adapted from `autokernel/analysis.py` + `autoresearch/analysis.ipynb` |
| 4 | Required NCU profile helper | `scripts/profile_ncu.sh`, `scripts/profile_candidate_once.py`, `scripts/profile_reference_once.py` | Explicit `basic`/`detailed`/`full` Nsight Compute profiles for experiments, plus lightweight reference calibration |
| 5 | Kernel microbench workflow | `.claude/agents/microbench.md` and `~/.codex/skills/microbench/SKILL.md` | Optional sub-op bottleneck analysis that complements NCU |
| 6 | Launch script | `scripts/agents.sh` | Multi-GPU agent launcher with worktree setup |
| 7 | Results TSV init | Part of setup | Header row creation + baseline recording |

### What NOT to implement (already exists / human-provided)
- `validate.py` — provided by human
- `reference.py` — provided by human
- `candidate/interface.py` — starting point provided by human, then modified by agent
