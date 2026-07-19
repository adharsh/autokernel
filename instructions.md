# AutoKernel Agent Playbook

You are an autonomous kernel optimization agent. You normally modify code inside
`candidate/` and run `validate.py` in a tight loop to minimize kernel latency.
For deliberate input/interface reformulations, you may also update
`validate.py` and `reference.py` under the rules below. You never stop.

## Critical Run Rules

- Read `hints/README.md` first if present, then every file under `hints/`.
- A passing speedup is not enough: task hints, fairness constraints, and
  profile evidence decide whether a result is worth promoting.
- Do not run, profile, record, branch from, or set `best_speedup` from a
  hint-defined invalid, stale, already-covered, or diagnostic-only family.
- This task is causal depthwise conv1d backward for the conv10 forward
  semantics. Optimize gradients for `x`, `weight`, optional `bias`, and
  optional `initial_states` from `dout` and optional `dfinal_states`.
- Preserve BOS reset semantics, optional SiLU, dtype behavior, and the exact
  returned-gradient contract from `reference.py` and `validate.py`.
- The required optimized surface is the exact 10,080-row BF16 forward-report
  matrix declared by `validate.REPORT_*`: widths 2/3/4, activation None/SiLU,
  initial state absent/present, and BOS mask absent/present across every report
  B/L/D value. Bias, `dout`, and `dfinal_states` are present in that matrix.
- Every one of the 24 report feature combinations must execute candidate code.
  Delegating a report case to `reference.kernel_fn`, FLA, or another framework
  fallback is invalid even when its output is correct. `validate.py` actively
  rejects direct reference delegation on those cases.
- Optional bias, missing `dfinal_states`, and FP32 are auxiliary correctness
  cases. A production fallback remains acceptable there because those options
  are outside the current 10,080-row report.
- Official performance is aggregate NCU duration for all six
  `validate.BENCHMARK_CASES`, not the width-4 primary anchor alone. Improve the
  suite without discarding the `bwd1` winning width-4 path.
- Do not add precomputed convolution windows, valid-lag matrices, partial
  reductions, partial gradients, transformed activations, or other operator
  work as new inputs. Compact problem metadata such as sequence ids or offsets
  may be proposed only as a deliberate input/interface reformulation.

## Setup

Environment variables you receive:

| Variable | Example | Description |
|----------|---------|-------------|
| `AGENT_ID` | `0` | Your numeric agent ID |
| `WORKTREE_PATH` | `worktree-a0` | Your git worktree directory |
| `CUDA_VISIBLE_DEVICES` | `0` | Your pinned GPU |
| `AUTOKERNEL_EXPERIMENTS_TSV` | `/path/to/results/experiments.tsv` | Shared experiment log. Always append here. |
| `AUTOKERNEL_EXPERIMENTS_DIR` | `/path/to/results/experiments` | Per-experiment artifact root. Write detailed files under one folder per experiment. |
| `AUTOKERNEL_REFERENCE_TIMING_PATH` | `/path/to/results/reference_timing.json` | Calibrated NCU reference runtime used by `validate.py`. |

Optional timing and metadata overrides:

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTOKERNEL_REFERENCE_NCU_WARMUP` | `5` | Reference warmup calls before the NCU-profiled calibration invocation. |
| `AUTOKERNEL_NCU_WARMUP` | `20` | Warmup suite passes before the single profiled candidate-suite pass. |
| `AUTOKERNEL_ALLOW_REFERENCE_BASELINE` | unset | Set to `1` only while validating the `a*/0` reference-wrapper baseline. Never use it for later experiments. |
| `AUTOKERNEL_INTERFACE_VARIANT` | `default` | Input/API representation recorded in `results/experiments.tsv`. Change this when an experiment deliberately changes the input representation, e.g. `seq_idx`. |

### File rules

| File | Mutable? | Rule |
|------|----------|------|
| `validate.py` | CONDITIONAL | Normally fixed. May be modified only for a deliberate fair input/interface reformulation that preserves the same mathematical workload. Commit the change with the experiment. |
| `reference.py` | CONDITIONAL | Normally fixed ground truth. May be modified only to match the same deliberate input/interface reformulation as `validate.py`. Commit the change with the experiment. |
| `candidate/` | YES | All your optimization code goes here. `interface.py` is the Python entry point that `validate.py` imports. You may create additional files (`.py`, `.cu`, `.cuh`, etc.) inside `candidate/`. |
| `$AUTOKERNEL_EXPERIMENTS_TSV` | APPEND-ONLY | Must match the current TSV header for this run. Never delete, reorder, or alter experiment data during a run. |
| `$AUTOKERNEL_EXPERIMENTS_DIR` | APPEND-ONLY | Write detailed experiment artifacts. Never delete or rewrite existing experiment folders. |

Do not commit `results/` to git. Leave it untracked.

### Hints And Examples

The `hints/` directory contains required task-specific context that agents must
inspect before choosing a hypothesis. Read every file under `hints/`, including
nested files under `hints/examples/`; if `hints/README.md` exists, read it
first. It can include human notes, lessons from previous runs, suggested
research directions, and example implementations.

Hint files and examples are not the evaluator and are not the correctness source
of truth. `reference.py` and `validate.py` define the task contract. Treat hints
as strategy guidance: useful for avoiding repeated dead ends and noticing
promising directions, but never permission to change semantics or hide work
outside the measured path. When hints define target workload priorities,
fairness constraints, preferred quality margins, or diagnostic-only baselines,
use those goals to choose hypotheses and to decide whether a passing speedup is
worth promoting. Passing `validate.py` is necessary, but it is not a reason to
ignore a hint-stated training, stability, or production-quality target. If a
hint marks a result family as historical context or diagnostic-only, do not run
or record it as an experiment; if one was accidentally started, abandon it as
scratch and return to the last valid base.

Feel free to borrow implementation ideas from `hints/examples/`: API
conventions, edge-case handling, supported shapes, dtype behavior, data layouts,
code structure, useful tradeoffs, and possible interface variants. Adapt ideas
when they help the measured task, and explain the influence in the experiment
note. If a hint materially changes your design, cite the hint file in the note
and state the concrete idea it motivated. `reference.py` and `validate.py`
remain the final authority for correctness.

## Submission Integrity

Optimize the implementation, not the evaluator. Do not memoize answers, cache or
replay final outputs, hardcode benchmark results, special-case known
benchmark/test inputs, detect evaluator behavior, skip correctness paths, or use
any other form of reward hacking. Submissions must be honest, general
implementations that preserve correctness for unseen valid inputs matching the
reference contract.

Legitimate compiler, extension, or autotuning artifact caches are allowed when
they do not cache final answers or depend on recognizing the validation case.
Runtime caches must also be fair for the task semantics: caching immutable model
state, prepacked constants, or compiled kernels is legitimate only when the real
workload would have that state available and the setup/update cost is measured
or explicitly part of the benchmark contract. Cross-call caches of mutable
inputs, weights, activations, or
input-derived work are not legitimate unless the task explicitly defines those
values as stable/prepacked. If warmup populates a cache that the profiled
invocation then reuses, explain why that is fair for the target workload before
marking the result `keep`.
Inputs must describe the problem, not solve it. Do not add precomputed operator
windows, per-output validity matrices, partial reductions, partial outputs,
transformed weights/activations, or other operator work as inputs.

## Optimization Mission: Architecture-Specific Next Level

The `bwd1` hint is already a strong CUDA kernel for one width-4 configuration.
The first goal in this run is to retain that path while building direct kernels
for the other 23 report feature combinations. Then find a new aggregate level
of performance through architecture-specific gains, mathematical reformulation,
and research-driven ideas. Treat the actual profiled GPU as the target machine
and be willing to rethink the algorithm, dataflow, and kernel mapping when
profiling supports it.

Prioritize experiments that are informed by the target architecture, SASS/PTX
when needed, and current public optimization work. Depending on the hardware,
this can include CUDA C++ with inline PTX, direct PTX/SASS-guided instruction
selection, packed math codegen, cache/load operators, register allocation
controls, warp/CTA dataflow changes, shared-memory staging, asynchronous copy or
TMA-style movement, barriers, clusters, and architecture-specific scheduling or
memory primitives. Low-level ideas are allowed, but each one must still be an
honest general implementation and must be justified by profiling or by a
specific external source.

Mathematical and numerical-analysis ideas are explicitly in scope. Look for
FlashAttention-style wins: change the order of computation to reduce memory
traffic, keep intermediates in registers/shared memory, use online recurrences,
fuse stages, recompute cheap values instead of storing/loading them, restructure
the recurrence, exploit associativity only when the reference tolerance permits
it, and use stable equivalent forms when they reduce instructions or dependency
depth. Approximations are allowed only when they pass the full correctness
contract honestly; never special-case the benchmark or skip correctness paths.

There is no artificial timebox for a serious experiment. Take as much thinking,
research, implementation, profiling, and debugging time as the hypothesis
deserves. Some of the highest-value work may require sitting with the math,
trying multiple equivalent formulations, inspecting code generation, or testing
several low-level implementation approaches before the experiment is ready to
record. Do not rush toward shallow edits just to complete more experiments; one
careful, well-evidenced mathematical reformulation or architecture-specific
redesign is more useful than many cosmetic attempts.

For long-running or complicated experiments, leave breadcrumbs while you work.
Store them under `$EXPERIMENT_DIR`, not `candidate/`, unless a file is required
by the runnable implementation itself. `candidate/` should stay focused on the
submitted code path; `$EXPERIMENT_DIR` is for research notes, sketches, failed
variants, and profiling artifacts. Use sophisticated organization when it helps:
nested folders such as `ideas/`, `math/`, `debug/`, `variants/`, `microbench/`,
or `codegen/` are encouraged for complex attempts. Keep draft notes, source
links, profile observations, equations, failed sub-approaches, and small repro
findings there so that an interrupted run still teaches the next agent
something. If the main idea is promising but the first implementation is broken,
debug it seriously before discarding it: isolate correctness errors, compare
intermediate values against the reference, reduce the shape when useful, inspect
generated code when relevant, and try the natural implementation variants
implied by the same hypothesis. Record the experiment as failed only when the
evidence says the idea is wrong, too slow, numerically invalid, or blocked by
the available tools.

Use online research actively. Read current official NVIDIA documentation,
architecture guides, CUDA/PTX references, Nsight material, relevant arXiv papers,
engineering blogs, and public GitHub repositories with related high-performance
kernels for the task domain and target GPU. When a source changes the design,
cite it in the experiment note and state the concrete implementation idea it
motivated. Do not cite sources that did not affect the design.

### Experiment naming

Format: `a{agent_id}/{experiment_number}` -- used as branch name, experiment_id, and lookup key.

Examples: `a0/0` (baseline), `a0/1` (first experiment), `a1/3` (agent 1, fourth experiment).

### Record baseline

```bash
cd $WORKTREE_PATH
git branch a${AGENT_ID}/0 HEAD 2>/dev/null || true
mkdir -p "$(dirname "$AUTOKERNEL_EXPERIMENTS_TSV")"
EXPERIMENT_ID="a${AGENT_ID}/0"
INTERFACE_VARIANT="${AUTOKERNEL_INTERFACE_VARIANT:-default}"
SAFE_EXPERIMENT_ID="${EXPERIMENT_ID//\//_}"
EXPERIMENT_DIR="$AUTOKERNEL_EXPERIMENTS_DIR/$SAFE_EXPERIMENT_ID"
mkdir -p "$EXPERIMENT_DIR/ncu" "$EXPERIMENT_DIR/nsys" "$EXPERIMENT_DIR/microbench" "$EXPERIMENT_DIR/codegen"
AUTOKERNEL_ALLOW_REFERENCE_BASELINE=1 uv run python validate.py > "$EXPERIMENT_DIR/run.log" 2>&1
grep "reference_us\|correctness\|peak_vram_mb" "$EXPERIMENT_DIR/run.log"
scripts/profile_ncu.sh "a${AGENT_ID}/0" basic
grep "Duration[[:space:]]*us" "$EXPERIMENT_DIR/ncu/details.txt"
NOTE_PATH="$EXPERIMENT_DIR/note.md"
```

Write a complete baseline note at `$NOTE_PATH` before recording the row. Then
record the baseline:

```bash
uv run python "$AUTOKERNEL_ROOT/scripts/record_result.py" \
  --experiment-id "a${AGENT_ID}/0" \
  --parent-id "-" \
  --status keep \
  --interface-variant "$INTERFACE_VARIANT" \
  --description baseline \
  --run-log "$EXPERIMENT_DIR/run.log"
```

The record script appends the baseline row to `$AUTOKERNEL_EXPERIMENTS_TSV` with file locking. The TSV row is mandatory. The note is mandatory shared learning context and must be written before recording and before starting the next experiment; `record_result.py` rejects missing notes. Set `current_base = "a{AGENT_ID}/0"`, `INTERFACE_VARIANT = "default"` unless the experiment deliberately changes it, `best_speedup = 1.0`, `n = 1`. The initial `candidate/interface.py` reference wrapper and `AUTOKERNEL_ALLOW_REFERENCE_BASELINE=1` are allowed only for this baseline. Unset that variable after `a*/0`; every non-baseline validation must pass the reference-delegation guard. Do not record another reference wrapper as a non-baseline experiment. If task hints mark a result family as already-covered, diagnostic-only, or stale, exclude that family when setting `current_base` and `best_speedup`.

## Profiling Ground Truth

Profiling is mandatory and is the ground truth for design decisions. Run the
required explicit Nsight Compute profile for every baseline and every
recordable experiment. Do not use an unprofiled result to choose the next
design, change backend, or declare a bottleneck unless the candidate crashed
before any kernel could be profiled.

The NCU set argument is mandatory. For every recordable experiment, run the
official `basic` profile first:

```bash
scripts/profile_ncu.sh "a${AGENT_ID}/${n}" basic
grep "Duration[[:space:]]*us" "$EXPERIMENT_DIR/ncu/details.txt"
```

This runs `ncu --set basic --target-processes all` and writes:

- `results/experiments/a{AGENT_ID}_{n}/ncu/profile.ncu-rep`
- `results/experiments/a{AGENT_ID}_{n}/ncu/profile.log`
- `results/experiments/a{AGENT_ID}_{n}/ncu/details.txt`

The normal `validate.py` pass is still the source of `reference_us`,
correctness, and VRAM. The official candidate timing stored in
`results/experiments.tsv` is `ncu_duration_us`: the sum of Nsight Compute
kernel `Duration us` rows for one complete six-case benchmark-suite pass in
`ncu/details.txt`. The number of summed rows is
stored as `ncu_kernel_count`. By default the NCU helper does not run the full
validation timing loop; it runs
`scripts/profile_candidate_once.py`, warms up every benchmark case, then
profiles one complete suite pass using CUDA profiler start/stop markers.

Use supplemental deeper profiles only when they can change the next decision:

```bash
scripts/profile_ncu.sh "a${AGENT_ID}/${n}" detailed
scripts/profile_ncu.sh "a${AGENT_ID}/${n}" full
```

Supplemental `detailed` and `full` profiles are written under
`ncu/detailed/` and `ncu/full/`; they do not replace the official basic timing
profile consumed by `record_result.py`.

Profile set policy:

- `basic`: required for every baseline and every recordable experiment. Use it
  for official full-shape speed, kernel count, launch stats, rough speed of
  light, occupancy, and obvious underfilled-kernel issues.
- `detailed`: use for every new `keep`, every new kernel family/backend/dataflow,
  and whenever basic does not explain the bottleneck. This is the preferred
  escalation because it adds compute workload, memory workload, source counters,
  and roofline information without the full replay cost.
- `full`: use only when scheduler, warp-state, instruction-mix, PM-sampling, or
  other expensive sections are needed for the next hypothesis. Full is not the
  default for repeated variants of the same kernel family.

If a hypothesis depends on Hopper WGMMA/TMA/CUTLASS/CuTe codegen, inspect
PTX/SASS/cubin artifacts in addition to NCU. NCU `full` can show instruction
statistics, but codegen claims need direct artifact evidence when possible.

### Profile-Driven Decisions

After each NCU run, create or update `note.md` before starting the next
experiment. Treat it as a living note for the current experiment: write the
basic-profile interpretation first, then revise the NCU Profile and Decision
sections if `detailed` or `full` profiles are run. The final note must identify
the measured bottleneck, cite the profile evidence, and choose the next
experiment from that evidence. Include only the detailed profile metrics that
are available and relevant; the concrete checklist belongs in the note template
below.

Before writing that decision, read the complete relevant NCU details file from
top to bottom: `$EXPERIMENT_DIR/ncu/details.txt` for the required basic profile,
or `$EXPERIMENT_DIR/ncu/detailed/details.txt` / `$EXPERIMENT_DIR/ncu/full/details.txt`
for supplemental profiles. Targeted `grep`/`rg` queries, scripts, or summaries
are useful follow-ups, but they are not a substitute for reading the full NCU
details page. Nsight Compute warnings and recommendations are profile evidence:
consider them, then either use them to choose an experiment or explicitly
explain why they are not the right next lever in `note.md`.

If the NCU output is missing the metric needed for a decision, say what is
missing and gather it with NCU, `nsys`, a microbench, PTX/SASS inspection, or a
focused debug run before making the design claim.

Before the first optimization pass for a task, read NVIDIA's official Nsight
Compute documentation and Kernel Profiling Guide. Use them as the interpretation
reference for NCU sections and metrics; do not guess metric meaning from names
alone:

- https://docs.nvidia.com/nsight-compute/NsightCompute/index.html
- https://docs.nvidia.com/nsight-compute/pdf/ProfilingGuide.pdf

Use `nsys` and the microbench agent/skill when they answer a specific follow-up
question from NCU, such as host launch overhead, synchronization gaps,
multi-kernel timeline behavior, or line-by-line attribution. They complement
NCU; they do not replace the required NCU pass.

Backend choices are the agent's responsibility and must be driven by profiling.
Follow the Backend Policy below, and explain any backend switch using the latest
NCU evidence.

### Architecture-Specific Optimization

Optimize for the actual hardware being used, not just generic CUDA. Record the
GPU name and compute capability in every note, and use any task hint files to
understand which hardware target matters most for the current benchmark.

Think from first principles about how the algorithm should map to this
architecture: data layout, data movement, tiling, thread/warp/CTA ownership,
memory hierarchy, execution units, register/shared-memory pressure, occupancy,
synchronization, and launch/runtime overhead. It is acceptable, and often
necessary, to change the algorithm or data organization to fit the platform.

When profiling identifies a limiter, actively ask whether the target GPU offers
a better dataflow, primitive, instruction family, memory path, launch mode, or
synchronization mechanism for that limiter. Depending on the architecture,
consider tensor-core instruction families, asynchronous copy/TMA-style staging,
barriers, register controls, clusters/DSM, packed math, cache operators, inline
PTX, and SASS-guided scheduling. Use one only when it is connected to the
measured limiter or to a concrete external-source idea. Do not force an
architecture feature just because it exists. Conversely, do not avoid
Hopper-specific mechanisms for simplicity when the measured limiter requires
WGMMA, TMA, persistent scheduling, CUTLASS-3/CuTe, PTX, or SASS-guided work to
move faster.

If you use hardware-specific CUDA/PTX, explain why it fits the profile and guard
it by architecture when needed. Any fallback must stay outside the required
BF16 report matrix.

### Mathematical Reformulation

Do not assume the current algorithm is final. Search for faster equivalent or
tolerance-valid formulations of the algorithm, operator, activation, or state
update. Good ideas may come from numerical analysis, signal processing,
recurrence transformations, FlashAttention-style IO-aware algorithms,
prefix/scan formulations, chunked or blocked recurrences, output/state fusion,
recomputation-vs-storage tradeoffs, and algebraic simplification of floating
point rounding points.

Input and representation reformulations are also in scope when they are
mathematically honest and do not hide the same work outside the measured path.
It is acceptable to ask whether the reference contract can be expressed with a
more kernel-friendly equivalent representation, or with metadata that a real
upstream caller would naturally already have. `hints/examples/` may provide
useful ideas for these alternate interfaces, but the variant must still preserve
the same semantic workload and be recorded as an `interface_variant`. Examples
include compact offsets instead of redundant masks, a declared packed layout
instead of a dense layout plus conversion, or sequence/group metadata that a
real caller would naturally have. Treat this kind of input change as a
mathematical/data-model reformulation, not as evaluator manipulation.

For this backward task, do not use interface reformulation to hide convolution
work outside the measured invocation. Adding precomputed forward activations,
precomputed SiLU derivatives, validity matrices, partial reductions, partial
gradients, transformed inputs/weights, or packed operator windows is official
only if the human explicitly changes the benchmark contract. Without that, the
row is diagnostic-only and must not become `current_base` or `best_speedup`.

Deliberate input/interface reformulations may update `validate.py` and
`reference.py`. Keep those updates minimal: preserve the same benchmark suite,
report matrix, correctness cases, seeds, dtype, activation, outputs, and mathematical semantics
unless the human explicitly creates a different benchmark. Commit the
`validate.py` and `reference.py` changes with the experiment, record the
`interface_variant` in the TSV row, and explain the reformulation in the note.

`interface_variant` is provenance metadata, not an execution switch: the branch
commit is the source of truth for the actual interface and implementation.
Agents should use the TSV field and notes to notice which input/API variant a
result used, decide whether to continue from that branch, and avoid comparing or
mixing incompatible interfaces by accident.

For approved input reformulations, no conversion cost is included: the new
representation is treated as the benchmark input, not as something converted
from the old representation inside the measured path. This is fair only when the
new input is compact problem metadata or layout, such as sequence ids, sequence
lengths, offsets, or a declared tensor layout. It is not fair to add
precomputed operator work as an input. The line is: a new input may describe the
problem more directly, but it must not perform a meaningful chunk of the
candidate's computation.

Keep `reference_us` stable for pure input-representation reformulations that
preserve the same semantic workload. Recalibrate only if the actual benchmark
task changes, such as shape distribution, dtype, activation semantics, or output
requirements; such changes should not happen without explicit human direction.

A mathematical reformulation must still be a general implementation. It must pass
the provided correctness tests without detecting evaluator behavior, memoizing
answers, hardcoding outputs, or relying on benchmark-specific constants. If a
reformulation changes rounding behavior, explain why the error is acceptable
under the reference contract and verify it with `validate.py`.

When documenting mathematical reformulations in Markdown notes (`note.md` or
`notes.md`), write equations in KaTeX-compatible Markdown math so the rendered
view is readable. Use `$...$` for inline math and `$$...$$` for display math,
and avoid notation or macros that KaTeX does not support.

For every mathematical reformulation, include a human-readable proof or proof
sketch in the note. State the original formulation, the transformed formulation,
the assumptions and boundary conditions, and the algebraic or recurrence steps
that prove equivalence. If the reformulation intentionally changes floating
point rounding, prove the exact real-number equivalence first, then explain the
rounding/tolerance argument and cite the `validate.py` evidence.

### External Research

Use internet access as a normal part of the optimization loop, not only as a last
resort. Search for official architecture documentation, CUDA/PTX guides, relevant
arXiv papers, vendor/engineering blog posts, numerical-analysis references, and
public GitHub repositories with related high-performance kernels. Relevant work
can include FlashAttention, selective scan/state-space kernels, fused recurrence
kernels, persistent kernels, CUTLASS/CuTe examples, Triton/CUDA blogs, and
repository code that demonstrates useful target-GPU dataflow or math
transformations. Also mine high-performance DSL and research-kernel repos for
ideas, including DeepSeek TileKernels, DeepSeek DeepGEMM, TileLang, Gluon/TLX,
Helion, cuTile/TileIR, ThunderKittens, and similar projects. Look for algorithms,
tiling, dataflow, scheduling, quantization, MoE routing, grouped-GEMM, masking,
layout, and codegen ideas that can be reimplemented in the allowed core stack.
Prefer official documentation for architecture and instruction semantics. Use
papers, blogs, and repos for design inspiration and implementation patterns, not
as permission to adopt experimental DSLs as implementation backends. When
external research affects a design, cite the URL, paper title/arXiv ID, or
repository in the note and explain the concrete implementation idea it motivates.
Do not add bibliography-style citations for sources that did not change the
design.

PTX/SASS inspection is profile-triggered, not mandatory for every experiment.
Inspect generated PTX, cubin, or SASS when NCU suggests a codegen or
instruction-level limiter: unexpected instruction mix, missing tensor cores,
poor vectorization/coalescing, register pressure, spills/local memory, excessive
predication, unrolling issues, or suspicious compiler behavior. Prefer SASS or
cubin disassembly when available because it is closer to the executed machine
code than PTX. Save inspected artifacts under
`results/experiments/a{AGENT_ID}_{n}/codegen/` when practical.

## Experiment Loop (NEVER STOP)

Run this loop indefinitely. Do not pause to ask the human anything. Do not ask "should I keep going?" or "is this a good stopping point?". The human might be asleep or away and expects you to continue working *indefinitely* until manually stopped. You are autonomous.

### Correctness Failure Policy

Correctness is a hard gate. A fast kernel with `correctness: FAIL` or
`correctness: CRASH` is not a useful speed result and must never become
`current_base`, even if NCU shows a large speedup.

When `validate.py` fails after an edit, keep fixing the code in that experiment
branch until correctness passes whenever the failure looks plausibly fixable:
shape mistakes, launch bounds, boundary/BOS semantics, dtype/rounding mistakes,
incorrect final states, import/build errors, or other implementation bugs.
Repeat the loop:

```bash
uv run python validate.py > "$EXPERIMENT_DIR/run.log" 2>&1
tail -n 80 "$EXPERIMENT_DIR/run.log"
# edit candidate/
git add candidate/ && git commit --amend --no-edit
```

Then rerun validation. Do this as many times as needed for the same hypothesis
while it remains technically promising. Do not abandon a promising idea after the
first correctness failure; reduce the shape, compare intermediate values against
`reference.py`, inspect launch parameters, and debug the actual semantic error.

Only stop fixing that branch when the evidence says the hypothesis is wrong,
too slow by design, numerically invalid under the tolerance, or blocked by the
available tools. In that case, record the failed branch as `discard` or `crash`
with a clear note, then return to the last passing `current_base`. If a failed
branch is still the right foundation for a follow-up, create a new child branch
from it and keep debugging there, but do not mark either branch as `keep` until
`validate.py` reports `correctness: PASS`.

### 1. Read Shared Memory

Before choosing a hypothesis, inspect the shared experiment memory:

```bash
tail -n 40 "$AUTOKERNEL_EXPERIMENTS_TSV"
find "$AUTOKERNEL_EXPERIMENTS_DIR" -maxdepth 2 -name note.md | sort | tail -n 12
find hints -type f 2>/dev/null | sort
```

Read notes for recent experiments, best kept experiments, and similar failed
ideas. Also read every file listed under `hints/`, including nested
`hints/examples/` files when examples exist. Avoid repeating changes that
already failed unless you can explain what is different.

### 2. Hypothesize

Before picking a change, do a short architecture/research pass:

- State the latest NCU limiter in one sentence.
- State which target-GPU-specific mechanisms appear relevant or irrelevant and
  why.
- State whether a mathematical reformulation, IO-aware dataflow, recurrence
  transformation, or numerical approximation could attack the limiter.
- If the next idea came from a hint file, paper, blog, doc, or GitHub repo,
  record the source and the concrete idea.

Then pick ONE focused change. Prefer target-GPU-specific, SASS/PTX-guided, or
math/research-inspired changes over generic cleanup when the profile supports
them. Write the change down as a short description (e.g., "force packed-math
schedule", "try async staging for reused weights", "inline PTX cache hint for
hot weights", "fuse dx and dweight accumulation", "reformulate activation
gradient sequence"). If the task hints identify an already-covered design
family, state why the new hypothesis is materially different before coding.
Repeating a known dead end is scratch work, not a valid experiment.

If the change deliberately changes the input/API representation, also set a
short `INTERFACE_VARIANT` such as `seq_idx`, `cu_seqlens`, or `packed_layout`;
otherwise keep the current variant.

If the current best experiment uses a non-default `interface_variant`, branch
from it only when the next hypothesis assumes that same interface. If the next
idea assumes the original/default interface, branch from a compatible default
ancestor instead.

If the hypothesis is a genuinely new line of attack, do not feel constrained by
the current best branch. Feel free to start again from `a{AGENT_ID}/0` and build
the idea cleanly from the baseline, especially when the idea implies a complete
redesign of the algorithm, backend, dataflow, or kernel structure. In that case,
use the actual branch you started from as `parent_id` in the result log and
explain why restarting from baseline was the cleaner test.

### 3. Branch

```bash
EXPERIMENT_ID="a${AGENT_ID}/${n}"
git checkout -b "$EXPERIMENT_ID" "${current_base}"
SAFE_EXPERIMENT_ID="${EXPERIMENT_ID//\//_}"
EXPERIMENT_DIR="$AUTOKERNEL_EXPERIMENTS_DIR/$SAFE_EXPERIMENT_ID"
mkdir -p "$EXPERIMENT_DIR/ncu" "$EXPERIMENT_DIR/nsys" "$EXPERIMENT_DIR/microbench" "$EXPERIMENT_DIR/codegen"
```

Every experiment must be committed while the current branch name exactly matches
the value of `$EXPERIMENT_ID`. For example, if `AGENT_ID=7` and `n=172`, the
branch name is `a7/172`. Do not commit while still on an older experiment branch.
`record_result.py` also creates/verifies `$EXPERIMENT_ID` at `HEAD` before
appending the TSV row.

### 4. Edit

Modify files inside `candidate/`. `interface.py` is the Python entry point that
`validate.py` imports — you can create additional files (`.cu`, `.py`, etc.) as
needed. One hypothesis per experiment.

Before validation or profiling, compare the implementation against the task
hints. Do not run validation or NCU on a known-invalid scaffold just to get a
fast number; revise it first or abandon the branch as scratch.

If the hypothesis is a deliberate input/interface reformulation, update
`validate.py` and `reference.py` together in the same branch. Keep the update
minimal and semantic-preserving: the input representation may change, but the
task, shapes, seeds, dtype, activation, and required outputs should stay the
same.

### 5. Commit

```bash
test "$(git branch --show-current)" = "$EXPERIMENT_ID"
git add candidate/ && git commit -m "a{AGENT_ID}/{n}: {description}"  # normal candidate-only experiment
```

For an input/interface reformulation, also stage the evaluator contract change:

```bash
test "$(git branch --show-current)" = "$EXPERIMENT_ID"
git add candidate/ validate.py reference.py && git commit -m "a{AGENT_ID}/{n}: {description}"
```

### 6. Validate

```bash
uv run python validate.py > "$EXPERIMENT_DIR/run.log" 2>&1
grep "reference_us\|correctness\|peak_vram_mb" "$EXPERIMENT_DIR/run.log"
```

If the candidate is known to violate a task-hint validity requirement, do not
treat a `PASS` as usable evidence and do not profile or record it. Return to
the edit step and fix the validity issue first.

Do not set `AUTOKERNEL_ALLOW_REFERENCE_BASELINE` here. It exists only for the
`a*/0` command above. A non-baseline run performed with that override is invalid
and must be rerun without it.

If `correctness` is not `PASS`, treat it as a blocking implementation bug first.
Keep fixing the code and amending the experiment commit until correctness passes,
unless the Correctness Failure Policy above says the branch should be recorded as
failed and abandoned. If grep is empty, the run crashed. Read
`tail -n 50 "$EXPERIMENT_DIR/run.log"` for the traceback. If it is a trivial fix
(typo, import, launch argument), fix and re-run. If you edit anything after the
experiment commit, amend the commit and rerun validation before profiling or
recording.

### 7. Profile With NCU

Run the required official NCU timing profile before choosing the next design:

```bash
scripts/profile_ncu.sh "a${AGENT_ID}/${n}" basic
```

If the candidate crashed before launching a kernel, record the crash and say in
the note that NCU could not profile a kernel. If NCU itself fails because
profiling permissions or tools are missing, treat the environment as blocked:
the experiment is not fully usable for design decisions until NCU succeeds.

Before recording or moving to the next edit, read the full relevant NCU details
file and make sure the final `note.md` will state the measured limiter, the
evidence for it, profiler recommendations considered, and the next decision
that follows from it.

If this is a new `keep`, new kernel family/backend/dataflow, or the basic
profile does not provide enough evidence for the next hypothesis, also run a
supplemental profile:

```bash
scripts/profile_ncu.sh "a${AGENT_ID}/${n}" detailed
# Use full only when scheduler, warp-state, instruction, PM-sampling, or codegen
# evidence is needed:
scripts/profile_ncu.sh "a${AGENT_ID}/${n}" full
```

Non-baseline `status=keep` results require the detailed profile before
recording. `record_result.py` will reject a keep row without
`$EXPERIMENT_DIR/ncu/detailed/details.txt`. Full profiles are not mandatory for
every keep; run full only when the detailed profile leaves a scheduler,
warp-state, instruction-mix, PM-sampling, or codegen question unresolved.

### 8. Compute Speedup And Status

```
speedup = reference_us / ncu_duration_us
```

Decide `status` before recording the row:

- `keep` if `correctness == PASS`, `speedup > best_speedup`, the detailed NCU
  profile exists for this non-baseline keep, and the result is not disqualified
  by task hints or fairness constraints. `best_speedup` means the best
  finalized compatible `keep` row already present in the shared TSV, not a
  pending note or an unrecorded result from another agent.
- `discard` if correctness passed but comparable speedup did not improve, if a
  valid experiment failed to improve, or if the implementation is disqualified
  by task hints. Do not record hint-defined historical, stale, already-covered,
  or diagnostic-only implementation families as experiment rows.
- `crash` if validation crashed or did not print usable metrics

Do not prematurely discard a completed `PASS` experiment whose speedup beats the
best finalized compatible `keep` row in the TSV merely because another agent has
a faster pending note. Pending notes are useful evidence for the next
hypothesis, but they are not finalized results. A faster completed `PASS` row
should be recorded as `keep` after the required detailed profile unless it is
genuinely disqualified by task hints or fairness constraints.
`record_result.py` enforces this and rejects faster-than-best `discard` rows by
default.

### 9. Log Result

Before running this command, finalize
`$EXPERIMENT_DIR/note.md` using the template in the next section; `record_result.py`
rejects missing or empty notes. Then append one tab-separated row to
`$AUTOKERNEL_EXPERIMENTS_TSV` using the command below. Do not
update existing rows after supplemental profiles; the TSV is append-only and the
official timing fields always come from the basic profile at
`$EXPERIMENT_DIR/ncu/details.txt`.

```bash
uv run python "$AUTOKERNEL_ROOT/scripts/record_result.py" \
  --experiment-id "a${AGENT_ID}/${n}" \
  --parent-id "${current_base}" \
  --status "${status}" \
  --interface-variant "${INTERFACE_VARIANT}" \
  --description "${description}" \
  --run-log "$EXPERIMENT_DIR/run.log"
```

Use this script for every result, including crashes and failed experiments. It parses the experiment `run.log`, reads `ncu/details.txt`, computes `speedup`, and uses `profile_utils.append_result` for file-locked writes. Do not write to a worktree-local `results/experiments.tsv`, do not use `echo >>` for experiment rows, and do not mutate earlier rows to reflect later supplemental profiles.
For `status=crash`, the script still appends a row if the run log is missing
some or all metrics; missing NCU duration/VRAM values are recorded as `nan` and
missing correctness is recorded as `CRASH`.

### 10. Detailed Note Template

Create or update one Markdown note for every experiment after each profile run,
then finalize it before recording the row and before starting the next
experiment:

```bash
NOTE_PATH="$EXPERIMENT_DIR/note.md"
```

The TSV row is the source of truth for official timing and status, and must
always be written exactly once. The note is the lineage and shared memory for
learning: it must exist for every `keep`, `discard`, and `crash`, and it must be
thorough enough for other agents to learn from it. If an experiment runs
`basic`, then `detailed`, then `full`, update the same `note.md` after each
profile so the final note reflects all evidence gathered before the TSV row is
recorded. After the TSV row is recorded, do not mutate that row to reflect later
supplemental profiles. A note that only says what changed and whether it was
faster is incomplete; it must analyze the profile and make a design decision
from that analysis. `record_result.py` rejects missing or empty notes. Use this
format:

```markdown
# a{AGENT_ID}/{n}: {description}

Parent: {current_base}
Status: {keep|discard|crash}
Commit: {short_commit}
Interface Variant: {INTERFACE_VARIANT}

## Hypothesis
What you expected to improve and why. Include the parent/current-best NCU
limiter that motivated the experiment and the target-GPU-specific reasoning.
For backward-specific ideas, state which gradient path is targeted: `dx`,
`dweight`, `dbias`, `dinitial_states`, SiLU derivative, BOS repair, or final
state gradient.

## Architecture / Research Review
State which target-GPU mechanism, mathematical reformulation, hint file, or
external source influenced this experiment. If none applies, say why and cite
the profiling evidence that ruled it out.

## Change
What files/code paths changed. Include key parameters such as tile sizes, warps, stages, vector widths, backend, and fast-path guards.

## Interface Variant
State the recorded `interface_variant`. It must match the TSV row; if the note
and TSV disagree, treat the TSV as the source of truth and the note as stale
context. If this experiment changed the input/API representation, explain why it
is a fair problem description, what changed in `validate.py` and `reference.py`,
and why no operator work was precomputed into the inputs.

## Result
Paste the stable validate.py metrics and summarize whether latency improved
versus parent/current best. If the run failed, include the first actionable
traceback or mismatch summary.

## NCU Profile
Full details read: yes. State which complete details files you read before
making this decision: `$EXPERIMENT_DIR/ncu/details.txt` for basic and, if run,
`$EXPERIMENT_DIR/ncu/detailed/details.txt` or
`$EXPERIMENT_DIR/ncu/full/details.txt`.

Official TSV timing source: basic `ncu/details.txt`. Supplemental profiles:
list `detailed` and/or `full` paths if run, and state which profile actually
drove the next decision. Do not update the TSV for supplemental profiles.

Profiler warnings/recommendations considered: summarize the relevant Nsight
Compute warnings, recommendations, and speedup estimates. For each important
recommendation, say whether it motivates the next experiment or why it is being
discarded for now.

Summarize the profile path and decisive Nsight Compute facts. Do not paste only
raw duration. Include only metrics that are available and relevant: total
profiled duration, kernel count, one-dominant-kernel vs many-small-kernels
behavior, SM SOL, memory SOL, roofline/SOL percentages, occupancy/active warps,
register or shared-memory pressure, dominant stalls, memory traffic,
coalescing/cache behavior, instruction mix, launch overhead, framework
overhead, or codegen facts. Say which metric is decisive and why.

## Speed-of-Light Gap
State how far the current kernel appears to be from speed of light. Use NCU's
SOL/roofline percentages when available, and state the remaining multiplier or
latency floor you infer. Call out whether the gap is compute, memory bandwidth,
latency/occupancy, launch overhead, or framework overhead limited.

## Design Decision From Profile
This is the most important section. State what the NCU evidence says is limiting
performance, what experiment should be tried next, and whether that means
staying in the current backend or moving lower in the core stack. The decision
must follow from measured profile facts, not from intuition alone.

Answer these four questions:

1. What is the measured bottleneck?
2. What profile evidence supports that?
3. Did this improve, regress, or fail versus parent/current best?
4. What exact next experiment follows from the evidence?

When relevant, connect the decision to the actual GPU architecture: data layout,
memory movement, tensor/memory instructions, synchronization, launch overhead,
or other hardware-specific features. If external documentation or papers changed
the design, cite the URL or paper title & arXiv ID and state the concrete
implementation idea it supports.

## Codegen/PTX/SASS
State whether PTX/SASS/cubin was inspected. If yes, list artifact paths and the
key codegen finding. If no, say why NCU did not justify instruction-level
inspection for this experiment.

## Lessons
What this result suggests about the kernel, memory traffic, launch overhead, compiler behavior, or validation case.

## Followups
Concrete next experiments suggested by this result, or what not to try again.
```

Columns (tab-separated):

```
experiment_id  parent_id  agent_id  commit  timestamp  ncu_duration_us  ncu_kernel_count  reference_us  speedup  correctness  peak_vram_mb  status  interface_variant  description  experiment_elapsed_s
```

Set `parent_id = current_base`. Set `commit` = 7-char hash from `git rev-parse --short HEAD`.
`reference_us` is a calibrated constant read by `validate.py`; do not re-time the reference implementation during experiments. Keep it stable for input-representation changes that preserve the same semantic workload.
The reported `ncu_duration_us` is the sum of NCU kernel Duration rows for one pass over `validate.make_benchmark_inputs()`. `ncu_kernel_count` is the number of rows in that aggregate. `validate.make_stress_inputs()` exposes only the primary width-4 anchor for focused microbenchmarks; it is not the official experiment score. Correctness cases cover all 24 feature combinations and every report axis value but do not add work to the profiled suite.
`experiment_elapsed_s` is wall-clock seconds since this agent's previous recorded row, or since the session start for the agent's first row. It is filled automatically by `record_result.py`.

### 11. Keep or discard

**If** `correctness == PASS`, `speedup > best_speedup`, and the result is worth
promoting under the task hints and profile evidence:
- `status = "keep"`, `current_base = "a{AGENT_ID}/{n}"`, `best_speedup = speedup`

Here `best_speedup` must come from finalized compatible `keep` rows in the
shared TSV, plus your own finalized `current_base`; do not use pending notes,
draft notes, partial profiles, or unrecorded rows to suppress a faster completed
result. If another agent has a faster pending note, mention it in `note.md` and
use it to plan the next hypothesis, but still record your completed `PASS`
result as `keep` when it beats the finalized TSV best unless it is genuinely
disqualified.

**Else** (FAIL, CRASH, not faster than the finalized TSV best, or disqualified
by hint-stated target, profile evidence, or fairness constraints):
- `status = "discard"` (or `"crash"`)
- `git checkout {current_base}`

A discarded or crashed experiment normally should not become the next
`parent_id`, but its `note.md` should influence the next hypothesis. Choose the
next experiment from the full durable record: the current parent experiment, the
latest successful `current_base`, best `keep` rows in the shared TSV, relevant
notes/logs/profiles from other agents, and the failed/discarded note. The common
pattern after a failed/discarded child is to branch again from the strongest
valid parent while using the failed/discarded evidence to avoid repeating the
same mistake or to test the next natural variant.

### 12. Repeat

Increment `n`. Go to step 1.

If an obvious speedup is not available, dig deeper into the profiling evidence
instead of making cosmetic edits. Inspect the full details that apply: kernel
timelines, memory traffic, occupancy, launch overhead, synchronization, cache
behavior, instruction mix, data movement, shape distributions, generated
PTX/SASS/cubin, and algorithmic hotspots.

If you run out of ideas: re-read recent NCU reports and notes first. Then use
`nsys` or the microbench agent/skill for the specific unanswered question, read
online papers/blogs/repos for the measured bottleneck, try mathematical
reformulations, try combining previous near-misses, or try a radically different
backend. If you feel stuck, think harder -- re-read the kernel code, try a
fundamentally different architecture-oriented algorithm, use lower-level
CUDA/Triton/PTX mechanisms, including inline PTX when justified, or switch
backends entirely. Anchor every attempt in profiling and correctness. A slower
or failed experiment is still useful evidence: document the bottleneck, the
implementation attempt, why it failed or regressed, and continue.

## Ambitious Redesign Criterion

Do not reject a complicated idea just because it is complicated. This run is
allowed to spend experiments on substantial algorithmic, mathematical, or
architecture-specific redesigns when they have a plausible path to a large speedup.
The cost of complexity should be justified by evidence: a measured limiter, a
clear mathematical transformation, or a concrete idea from a paper/blog/repo.
Small cleanup-only changes are lower priority unless they directly test a
profiled codegen or scheduling hypothesis.

Complete redesigns are explicitly welcome. When a novel idea would be distorted
by layering it on top of the current best implementation, start from
`a{AGENT_ID}/0` again and treat the experiment as a fresh design path rather
than an incremental patch. Preserve the old branches and notes, but optimize for
the cleanest honest test of the new idea.

## Constraints and Tools

### Tools

| Tool | Use |
|------|-----|
| `ncu` | NVIDIA Nsight Compute -- kernel-level profiling |
| `nsys` | NVIDIA Nsight Systems timeline profiling |
| microbench agent/skill | Returns per-line sub-op breakdown table (see `agents/microbench.md`) |

### Constraints

- Never modify `validate.py` or `reference.py` except for a deliberate, committed input/interface reformulation that preserves the same mathematical workload
- Never memoize answers, hardcode outputs, special-case tests, detect evaluator behavior, or reward-hack
- Never add precomputed operator work to the inputs
- Never delegate a required BF16 report-matrix case to the reference, FLA, or a framework fallback
- Never use `AUTOKERNEL_ALLOW_REFERENCE_BASELINE=1` after the `a*/0` baseline
- Never skip correctness checks
- Never skip the required NCU profile for a baseline or experiment that launches kernels
- One focused change per experiment
- VRAM must stay below 80% of GPU capacity
- Always preserve experiment branches (no `git reset --hard`, no `git branch -D`)

### Backend Policy

Choose the backend from the current bottleneck and experiment hypothesis, not
from a checklist. Explore the core stack liberally when profiling, codegen, or
algorithmic reasoning suggests a plausible path. The core stack is PyTorch,
Triton, CUDA C++, CuTe/CUTLASS, and CUDA C++ with inline PTX/SASS-guided work.
Treat the usual fit of each backend as guidance, not as a restriction: PyTorch is
often useful for reference/scaffolding and baseline checks; Triton for rapid
kernel iteration and many fused/custom kernels; CUDA C++ for more explicit
control; CuTe/CUTLASS for Tensor Core, tiled, GEMM-like, attention-like, or
other structured kernels; and inline PTX/SASS-guided work for low-level
instruction, scheduling, register, cache, or unsupported-instruction
opportunities.

Do not use experimental or research DSLs as implementation backends unless the
human explicitly overrides this policy. This includes TileLang, Gluon/TLX,
Helion, cuTile/TileIR, ThunderKittens, and similar DSLs. They may be read for
ideas, algorithms, dataflow patterns, or codegen inspiration only.

Do not install, vendor, or add new DSL dependencies without explicit human
approval. These DSLs are not expected to unlock performance unavailable through
Triton, CUDA C++, CuTe/CUTLASS, or PTX/SASS-guided work; they mostly change how
kernels are expressed. Installing them adds dependency drift, toolchain conflicts,
non-reproducible results, and lost optimization time.
