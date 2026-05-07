# Installation Guide

## Prerequisites

- NVIDIA GPU host with CUDA Toolkit 12.x
- `nvidia-smi`, `ncu`, and `nsys` on `PATH`
- git
- uv

Install uv if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup

```bash
git clone <repo-url> autokernel
cd autokernel
./scripts/setup.sh --sync --verify
```

`setup.sh` handles the repo-local setup:

- installs/syncs the uv environment when `--sync` is passed
- links the microbench workflow for Claude Code and Codex
- copies `docs/templates/validate.py` and `docs/templates/reference.py` to the repo root if missing
- initializes `results/experiments.tsv`
- runs lightweight dependency checks when `--verify` is passed

Useful setup options:

```bash
./scripts/setup.sh --dry-run
./scripts/setup.sh --force
```

## Fill In The Task

Edit the generated root files:

- `reference.py`: trusted ground-truth implementation
- `validate.py`: task inputs, tolerances, timing, and correctness checks
- `candidate/interface.py`: starting candidate entry point

Smoke test before launching agents:

```bash
uv run python validate.py
```

Commit the task files before running multiple agents. Git worktrees only contain tracked files.

```bash
git add validate.py reference.py candidate/
git commit -m "task setup"
```

Restart Claude Code and/or Codex after setup so the microbench agent/skill is discovered.

## Launch

Codex is the default backend. It launches with `gpt-5.5` at `xhigh` effort.

```bash
./scripts/agents.sh start
```

Use Claude instead:

```bash
AGENT_CLI=claude ./scripts/agents.sh start
```

Claude launches with `claude-opus-4-7` at `high` effort.

## Operate

```bash
./scripts/agents.sh status
./scripts/agents.sh stop
./scripts/agents.sh resume
```

Logs:

```bash
tail -F results/logs/agent0.log
```

Analysis:

```bash
uv run python analysis.py
```

Outputs are written under `results/`, including `progress.html`, `speedup.html`, and `report.md`.

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
