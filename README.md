# AutoKernel

AutoKernel is a scaffold for running autonomous GPU kernel optimization agents against a fixed validation harness.

The main workflow is:

1. Set up the repo.
2. Fill in the task-specific reference and validation files.
3. Smoke test the task.
4. Commit the task setup so worktrees can see it.
5. Launch one optimization agent per GPU.

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

This is a good step to do with an AI coding tool before starting the autonomous agents. Give the tool the target function, expected shapes, dtypes, edge cases, and desired output layout, then ask it to fill `reference.py` and `validate.py`.

For example, ask it to:

- put the trusted implementation in `reference.kernel_fn`
- build `validate.py` test cases for representative shapes and edge cases
- compare both returned outputs and final state tensors
- keep the printed labels stable: `candidate_us`, `reference_us`, `correctness`, `peak_vram_mb`

Run the smoke test:

```bash
uv run python validate.py
```

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
