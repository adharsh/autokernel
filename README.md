# AutoKernel

AutoKernel is a scaffold for running autonomous GPU kernel optimization agents
against a fixed validation harness.

This README is ordered for setup. Follow it from top to bottom when creating a
new task. Do not launch agents until the task files have been populated,
validated, profiled with the smoke check, and committed.

## AI Handoff Boundary

An AI coding tool can complete sections 1 through 6 in order: repo setup, task
definition, reference calibration, validation, profiling smoke check, and the
setup commit. The AI setup tool must stop after reporting the commands run,
metrics, commit hash, and blockers.

Sections 7 and 8 are human-only. Launching and operating long-running agents is
a human action, so an AI setup tool must not run `./scripts/agents.sh start`.

To hand off a new task, tell the LLM to read this README and provide the
reference implementation. Also provide expected shapes, dtypes, edge cases, and
desired output layout if they are not obvious from the reference implementation.

## 1. Prerequisites

- NVIDIA GPU host with CUDA Toolkit 12.x
- `nvidia-smi`, `ncu`, and `nsys` on `PATH`
- git
- uv

Install uv if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2. Set Up The Repo

From a fresh checkout:

```bash
git clone <repo-url> autokernel
cd autokernel
./scripts/setup.sh --sync --verify
```

For an existing checkout, still run the setup command from the repo root before
editing task files:

```bash
./scripts/setup.sh --sync --verify
```

`setup.sh` installs/syncs the uv environment, links the microbench workflow for
Claude Code and Codex, creates `reference.py` and `validate.py` from templates
when missing, and initializes `results/experiments.tsv`.

Useful setup options:

```bash
./scripts/setup.sh --dry-run
./scripts/setup.sh --force
```

If setup fails because dependencies need to be downloaded or the environment
blocks access to a cache directory, stop and report that blocker. Retry only with
an explicit writable cache or approved network/filesystem access.

## 3. Define The Task

Populate these files only after `./scripts/setup.sh --sync --verify` completes:

| File | Purpose |
|------|---------|
| `reference.py` | Trusted ground-truth implementation exposed as `kernel_fn`. |
| `validate.py` | Correctness, calibrated reference timing metadata, and input test cases. Normally fixed after task setup. |
| `candidate/interface.py` | Starting optimization entry point exposed as `kernel_fn`. |

Root `reference.py` and `validate.py` are task-specific. Reusable harness policy
belongs in `docs/templates/` so future tasks inherit it.

Validation requirements:

- Put the trusted implementation in `reference.kernel_fn`.
- Define exactly one stress/benchmark case in `validate.py`; this same case is
  used for calibrated reference timing and NCU candidate profiling.
- Expose that timing case through `validate.make_stress_inputs()`.
- Do not time candidate implementations inside `validate.py`. Candidate duration
  must come from the NCU `Duration us` rows produced by `scripts/profile_ncu.sh`.
- Build separate correctness-only cases for representative shapes and edge
  cases.
- Compare all returned outputs, including state tensors.
- Keep the printed labels stable: `reference_us`, `correctness`, `peak_vram_mb`.
- Keep timing and correctness roles separate. Correctness-only cases broaden
  coverage but must not change the profiled timing case.
- Deliberate input/API reformulations may update `reference.py` and
  `validate.py` together when they preserve the same mathematical workload.
  Record the input representation through `interface_variant`; do not put
  precomputed operator work into inputs.

## 4. Calibrate And Validate

After task files are populated, run:

```bash
uv run python scripts/calibrate_reference.py
uv run python validate.py
```

`scripts/calibrate_reference.py` profiles `reference.kernel_fn` once with
Nsight Compute on the single stress benchmark case and stores
`results/reference_timing.json`. Calibration is intentionally lightweight by
default: it uses the `basic` NCU section set and only enough warmup to keep lazy
initialization out of the profiled invocation. Its purpose is to produce the
reference Duration total, not a full performance analysis.

`validate.py` still runs the reference implementation for correctness, but uses
the calibrated `results/reference_timing.json` value for the printed
`reference_us` metric. This keeps speedup reporting stable across agent
experiments.

Candidate profiling uses `scripts/profile_ncu.sh`, not `validate.py` timing.
Generated task harnesses must not add CUDA-event timers, wall-clock timers, or
median timing tables for candidate speed inside `validate.py`.
Reference calibration profiles one reference invocation after warmup; override
its warmup with `AUTOKERNEL_REFERENCE_NCU_WARMUP` or
`scripts/calibrate_reference.py --warmup`. Override the NCU section set with
`AUTOKERNEL_REFERENCE_NCU_SET` or `scripts/calibrate_reference.py --ncu-set`
only when you specifically need deeper reference profiling.

## 5. Check Profiling

Confirm Nsight Compute can run before launching agents:

```bash
scripts/profile_ncu.sh smoke-test basic
```

If this fails because hardware counters are restricted, see
[Profiling Permissions](#profiling-permissions).

## 6. Commit The Task Setup

Commit the fixed task setup before launching agents:

```bash
git add validate.py reference.py candidate/
git commit -m "task setup"
```

Git worktrees only contain committed files, so uncommitted task setup changes
will not be visible to agents.

Restart Claude Code and/or Codex after setup so the microbench agent/skill is
discovered.

## 7. Launch Agents

Human-only step. Launch agents only after setup, validation, profiling smoke
check, and commit are complete.

Codex is the default backend. It launches with `gpt-5.5` at `xhigh` effort:

```bash
./scripts/agents.sh start
```

Use Claude instead:

```bash
AGENT_CLI=claude ./scripts/agents.sh start
```

Claude launches with `claude-opus-4-7` at `high` effort.

The launcher detects GPUs with `nvidia-smi` and starts one agent per GPU. Every
agent runs in its own git worktree, including agent 0 at `worktree-a0`, so the
repo root remains the human/control checkout.

## 8. Operate

Human-only step.

```bash
./scripts/agents.sh status
watch -n 5 ./scripts/agents.sh status # poll status
./scripts/agents.sh watch 5 # poll status and resume dead agents every 5 seconds
./scripts/agents.sh stop
./scripts/agents.sh cleanup # remove old agent worktrees, branches, results, and caches
./scripts/agents.sh resume
./scripts/agents.sh resume a3 a7
tail -F results/logs/agent0.log
uv run python scripts/format_results.py --sort agent
uv run python analysis.py
```

Analysis outputs are written under `results/`.
`resume` keeps tracked agents whose PIDs are still alive on the current node and
restarts only dead tracked agents. Pass agent IDs to limit resume to a known
subset.
`watch` runs in the foreground, clears the screen between refreshes, shows
session age, experiment counts, per-agent experiment rate, each agent's last
recorded experiment time, and exits on Ctrl+C without stopping agents.
`cleanup` is destructive and prompts before removing generated agent state. Use
`./scripts/agents.sh cleanup --yes` for non-interactive cleanup. It preserves
`.venv/`.
Agents append through `scripts/record_result.py`, which uses file locking and
the shared root `results/experiments.tsv`. Worktree `results/` paths are
symlinked to the root results directory when agents launch. Crash rows are still
recorded even when validation fails before printing metrics; missing NCU
duration and VRAM fields are written as `nan`, and missing correctness is
written as `CRASH`.

`results/experiments.tsv` is the compact machine-readable index. Detailed
experiment memory lives under one folder per experiment, for example
`results/experiments/a0_1/` for `a0/1`. The TSV row is the source of truth and
should be written once for every experiment after `note.md` has been finalized;
`note.md`, `run.log`, NCU reports, and optional artifacts live under the
per-experiment folder. `analysis.py` audits this coverage and flags missing
basic NCU artifacts, missing detailed profiles for non-baseline keeps, and
notes that are missing the required NCU, speed-of-light, design-decision, or
codegen/PTX/SASS sections.

Rows include `interface_variant` so input/API reformulations stay visible in the
shared table. Use `default` for the task's original interface and a short name
such as `seq_idx` or `packed_layout` for deliberate variants.
Rows also include `experiment_elapsed_s`, the wall-clock seconds since that
agent's previous recorded row, or since the session start for its first row.
This is filled automatically by `scripts/record_result.py` and surfaced by
`./scripts/agents.sh status` and `./scripts/agents.sh watch`.

`interface_variant` is provenance metadata, not an execution switch; the
experiment branch/commit remains the source of truth for the actual interface.

Keep `results/experiments.tsv` as raw tab-separated data. Use
`scripts/format_results.py` for aligned human-readable output; do not pad or
manually edit the TSV.

## Profiling Policy

Profiling is mandatory. Every baseline and every recordable experiment that
launches GPU kernels must run the official basic Nsight Compute timing pass:

```bash
scripts/profile_ncu.sh "a${AGENT_ID}/${n}" basic
```

This stores `profile.ncu-rep`, `profile.log`, and `details.txt` under
`results/experiments/a{AGENT_ID}_{n}/ncu/`. The `ncu/details.txt` kernel
`Duration us` rows are the source of `ncu_duration_us` in
`results/experiments.tsv`, and their row count is stored as `ncu_kernel_count`;
`validate.py` remains the source of correctness, `reference_us`, and VRAM.

Use `scripts/profile_ncu.sh "a${AGENT_ID}/${n}" detailed` for every
non-baseline `keep`, new kernel family/backend/dataflow, or unclear basic
profile. Use `full` only when scheduler, warp-state, instruction-mix,
PM-sampling, or codegen evidence is needed. Supplemental profiles live under
`ncu/detailed/` or `ncu/full/` and do not replace the official basic timing in
`experiments.tsv`.

By default, `scripts/profile_ncu.sh` runs
`uv run python scripts/profile_candidate_once.py` under Nsight Compute with
profiling disabled at process start. That target calls
`validate.make_stress_inputs()`, warms up the candidate, starts CUDA profiling,
runs one candidate invocation, synchronizes, and stops profiling. Set
`AUTOKERNEL_NCU_WARMUP` to override the warmup count for this profile-only pass.

Before the first optimization pass for a task, agents should read NVIDIA's
official Nsight Compute documentation and Kernel Profiling Guide and use them as
the metric interpretation reference:

- https://docs.nvidia.com/nsight-compute/NsightCompute/index.html
- https://docs.nvidia.com/nsight-compute/pdf/ProfilingGuide.pdf

Experiment notes must summarize the NCU findings, how far the candidate is from
speed of light, the current limiting factor, and what the profile says to try
next. `nsys` and the microbench workflow are optional follow-up tools for
timeline, launch overhead, synchronization, or per-line attribution questions;
they do not replace the required NCU profile.

Agents must optimize the implementation, not the evaluator. Do not memoize
answers, cache or replay final outputs, hardcode benchmark results, special-case
known benchmark/test inputs, detect evaluator behavior, skip correctness paths,
or use reward hacking. Legitimate compiler, extension, or autotuning artifact
caches are fine only when they do not cache final answers or depend on
recognizing the validation case.

Backend choices are explicitly profile-driven. PyTorch, Triton, CUDA C++,
CUDA C++ with inline PTX, CUTLASS, CUTE DSL, and PTX are all allowed, but moving
lower level should be justified by Nsight Compute evidence such as codegen,
occupancy, memory coalescing, instruction mix, scheduling, or other kernel-level
limits.

If an obvious speedup is not available, inspect deeper profiling details before
moving on: kernel timelines, memory traffic, occupancy, launch overhead,
synchronization, cache behavior, instruction mix, data movement, generated
PTX/SASS/cubin, and algorithmic hotspots.

PTX/SASS inspection is optional by default and required when NCU points at a
codegen or instruction-level limiter. Prefer SASS/cubin disassembly when
available because it is closer to executed machine code than PTX. When
inspected, save artifacts under `results/experiments/<experiment>/codegen/` and
record the finding in the experiment note.

## Profiling Permissions

`nsys` usually works without extra permissions. `ncu` hardware counters may
require:

```bash
sudo tee /etc/modprobe.d/nvidia-profiling.conf <<'EOF'
options nvidia NVreg_RestrictProfilingToAdminUsers=0
EOF
sudo update-initramfs -u -k all
```

Reboot, then check:

```bash
cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly
```

Then run a smoke profile from the repo root:

```bash
scripts/profile_ncu.sh smoke-test basic
```

## Agent Playbook

The runtime instructions passed to launched agents live in `instructions.md`.
