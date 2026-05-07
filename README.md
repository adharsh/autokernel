# AutoKernel

AutoKernel is a scaffold for running autonomous GPU kernel optimization agents against a fixed validation harness.

The main workflow is:

1. Set up the repo.
2. Fill in the task-specific reference and validation files.
3. Smoke test the task.
4. Calibrate the reference timing.
5. Commit the task setup so worktrees can see it.
6. Launch one optimization agent per GPU.

## Prerequisites

- NVIDIA GPU host with CUDA Toolkit 12.x
- `nvidia-smi`, `ncu`, and `nsys` on `PATH`
- git
- uv

Install uv if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Quick Start

```bash
git clone <repo-url> autokernel
cd autokernel
./scripts/setup.sh --sync --verify
```

`setup.sh` installs/syncs the uv environment, links the microbench workflow for Claude Code and Codex, creates `reference.py` and `validate.py` from templates when missing, and initializes `results/experiments.tsv`.

Useful setup options:

```bash
./scripts/setup.sh --dry-run
./scripts/setup.sh --force
```

## Task Setup

Before launching agents, populate these files:

| File | Purpose |
|------|---------|
| `reference.py` | Trusted ground-truth implementation exposed as `kernel_fn`. |
| `validate.py` | Fixed correctness, timing, and input test cases. |
| `candidate/interface.py` | Starting optimization entry point exposed as `kernel_fn`. |

Root `reference.py` and `validate.py` are task-specific. Reusable harness
policy belongs in `docs/templates/` so future tasks inherit it.

This is a good step to do with an AI coding tool before starting the autonomous agents. Give the tool the target function, expected shapes, dtypes, edge cases, and desired output layout, then ask it to fill `reference.py` and `validate.py`.

For example, ask it to:

- put the trusted implementation in `reference.kernel_fn`
- define exactly one stress/benchmark case in `validate.py`; this same case is used for calibrated reference timing and candidate timing
- build separate correctness-only cases for representative shapes and edge cases
- compare both returned outputs and final state tensors
- keep the printed labels stable: `candidate_us`, `reference_us`, `correctness`, `peak_vram_mb`

Keep the timing and correctness roles separate:

- The stress/benchmark case should be the performance target agents optimize. Calibrate `reference_us` on this case once, then time every candidate on this same case.
- Correctness-only cases should broaden coverage without changing the reported timing. Use them for edge cases like optional arguments, width variants, reset-mask behavior, activations, and small shapes.

Run the smoke test:

```bash
uv run python scripts/calibrate_reference.py
uv run python validate.py
```

`validate.py` still runs the reference implementation for correctness, but uses
the calibrated `results/reference_timing.json` value for the printed `reference_us`
metric. This keeps speedup reporting stable across agent experiments.

Commit the task setup before running multiple agents:

```bash
git add validate.py reference.py candidate/
git commit -m "task setup"
```

Git worktrees only contain tracked files, so untracked `validate.py` or `reference.py` files will not be visible to secondary agents.

Restart Claude Code and/or Codex after setup so the microbench agent/skill is discovered.

## Launch Agents

Codex is the default backend. It launches with `gpt-5.5` at `xhigh` effort:

```bash
./scripts/agents.sh start
```

Use Claude instead:

```bash
AGENT_CLI=claude ./scripts/agents.sh start
```

Claude launches with `claude-opus-4-7` at `high` effort.

The launcher detects GPUs with `nvidia-smi` and starts one agent per GPU. Agent 0 runs in the repo root; agents 1+ run in separate git worktrees.

## Operate

```bash
./scripts/agents.sh status
./scripts/agents.sh stop
./scripts/agents.sh resume
tail -F results/logs/agent0.log
uv run python analysis.py
```

Analysis outputs are written under `results/`.
Agents append through `scripts/record_result.py`, which uses file locking and
the shared root `results/experiments.tsv`. Worktree `results/` paths are
symlinked to the root results directory when agents launch.

## Profiling Permissions

`nsys` usually works without extra permissions. `ncu` hardware counters may require:

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

## Agent Playbook

The runtime instructions passed to launched agents live in `instructions.md`.
