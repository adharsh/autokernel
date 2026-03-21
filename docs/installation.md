# Installation Guide (Internal)

## Prerequisites

- NVIDIA GPU with compute capability >= 8.0 (tested on RTX A5500, sm_86)
- NVIDIA driver >= 570
- CUDA Toolkit 12.x installed at `/usr/local/cuda`
- `CUDA_HOME=/usr/local/cuda` and `/usr/local/cuda/lib64` in `LD_LIBRARY_PATH`
- ncu (Nsight Compute) and nsys (Nsight Systems) on PATH (bundled with CUDA Toolkit)
- git

## Step 1: Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Step 2: Clone and sync

```bash
git clone <repo-url> autokernel
cd autokernel
uv python install 3.10      # uv-managed Python (includes headers for Triton/pybind11)
uv sync                     # installs everything from uv.lock
```

That's it. `uv sync` installs all dependencies from the lock file:

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.9.1+cu128 | PyTorch with CUDA 12.8 |
| triton | 3.5.1 | Triton GPU compiler |
| cuda-python | 13.2.0 | CUDA driver/runtime Python bindings (PTX, NVRTC) |
| nvidia-cutlass | 4.2.0 | CUTLASS/CUTE DSL for tensor core kernels |
| pybind11 | 3.0.2 | C++/CUDA extension bindings |
| ninja | 1.13.0 | Fast CUDA C++ compilation |
| setuptools | 82.0.1 | Required by `torch.utils.cpp_extension` |
| numpy | 2.2.6 | Numeric operations |
| pandas | 2.3.3 | Results TSV handling |
| matplotlib | 3.10.8 | Plotting (progress.png) |
| tabulate | 0.10.0 | Table formatting |

## Step 3: Set up microbench agent symlink

```bash
mkdir -p .claude/agents
ln -sf ../../agents/microbench.md .claude/agents/microbench.md
```

This makes the microbench agent discoverable by Claude Code. The source file (`agents/microbench.md`) is tracked in git; the symlink in `.claude/` is gitignored.

## Step 4: Clean up AI tooling artifacts

```bash
rm CLAUDE.md
```

## Step 5: Verify

Each backend the agent can use must be verified independently.

```bash
# 1. PyTorch + CUDA
uv run python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')"

# 2. Triton
uv run python -c "import triton; print(f'Triton {triton.__version__}')"

# 3. CUDA C++ (torch.utils.cpp_extension — compiles via ninja)
uv run python -c "
from torch.utils.cpp_extension import load_inline
import torch
mod = load_inline('test_ext', cpp_sources=['torch::Tensor test_fn(torch::Tensor x) { return x + 1; }'],
    cuda_sources=[], functions=['test_fn'], verbose=False)
t = mod.test_fn(torch.tensor([1.0, 2.0], device='cuda'))
print(f'CUDA C++ extension: OK (result={t.tolist()})')
"

# 4. CUTLASS Python (code generation)
uv run python -c "import cutlass_library; import cutlass_cppgen; print(f'CUTLASS Python {cutlass_library.__version__}: OK')"

# 5. CUTE DSL C++ headers (verify headers exist for #include)
uv run python -c "
import cutlass_library, os
inc = os.path.join(os.path.dirname(cutlass_library.__file__), 'source', 'include')
assert os.path.isfile(os.path.join(inc, 'cute', 'tensor.hpp')), 'CuTe headers missing'
assert os.path.isfile(os.path.join(inc, 'cutlass', 'cutlass.h')), 'CUTLASS headers missing'
print(f'CUTE DSL headers: OK ({inc})')
"

# 6. PTX (compile via NVRTC)
uv run python -c "
from cuda.bindings import nvrtc
src = b'extern \"C\" __global__ void k(float* o) { o[threadIdx.x] = threadIdx.x; }'
err, prog = nvrtc.nvrtcCreateProgram(src, b'test.cu', 0, [], [])
nvrtc.nvrtcCompileProgram(prog, 1, [b'--gpu-architecture=compute_86'])
err, size = nvrtc.nvrtcGetPTXSize(prog)
ptx = b' ' * size
nvrtc.nvrtcGetPTX(prog, ptx)
print(f'PTX compilation: OK ({size} bytes)')
nvrtc.nvrtcDestroyProgram(prog)
"

# 7. nsys profiling (should work without sudo)
nsys profile -o /tmp/nsys_test --force-overwrite true -- uv run python -c "import torch; torch.randn(100,100,device='cuda') @ torch.randn(100,100,device='cuda')"
echo "nsys: OK"
```

## Step 6. Fill out code

Create validate.py and reference.py based on the provided spec under docs.

### Step 6.1 Smoke test

Replace interface.py by calling reference.py to check if validate.py passes. Then revert changes. This is just to test everything is running smoothly.

## Step 7. Restart Claude Code to access microbench agent. (Manual)

Restart Claude Code to access microbench agent. This is a manual step.

## Step 8. Launch agents

Auto-detects GPUs and spawns one agent per GPU. Agent 0 uses the repo directly; agents 1+ get their own git worktree.

```bash
./scripts/launch.sh
```

Logs go to `agent{N}.log`. PIDs are saved to `.agent_pids`.

```bash
# Monitor
tail -f agent*.log

# Stop all
kill $(cat .agent_pids)

# Resume from where agents left off
./scripts/launch.sh --resume
```

Resume uses `claude --continue` to pick up the previous session. The agent reads `results.tsv` and the current git branch to figure out where it was.

## Step 9. Monitor from Claude Code interactive session (optional)

Open a separate `claude` interactive session in the project root. Use `/loop` to auto-check progress:

```
/loop 1hr Read results.tsv, show the latest rows, plot speedup over time to progress.png using @analysis.py, and summarize what's working.
```

This re-runs every hour. You can also ask one-off questions or spawn the microbench agent anytime.

Also view logs for a given agent via:

```bash
tail -F agent0.log | jq -r --stream 'select(length==2 and .[0][-1]=="content") | .[1]'
```

## Profiling permissions (ncu)

`nsys` works out of the box. `ncu` hardware counters require one of:

**One-time fix** (requires root, persists across reboots):
```bash
sudo tee /etc/modprobe.d/nvidia-profiling.conf <<'EOF'
options nvidia NVreg_RestrictProfilingToAdminUsers=0
EOF
sudo update-initramfs -u -k all
# Reboot
```

To check current status:
```bash
cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly
# 1 = restricted (need sudo for ncu), 0 = open
```

## Notes

- `uv python install 3.10` is critical — it provides Python headers needed by Triton's JIT compiler. System Python 3.10 without `-dev` package will fail with `Python.h: No such file or directory`.
- Triton `@triton.jit` kernels cannot be defined inline via `python -c "..."` — they must be in a `.py` file (Triton needs `inspect.getsource()`).
- CUTLASS v4 (nvidia-cutlass >= 4.0) changed import names: `import cutlass_library` and `import cutlass_cppgen` instead of `import cutlass`.
- cuda-python v13 changed import: `from cuda.bindings import driver, runtime` instead of `import cuda.cuda`.
- PyTorch is pinned to cu128 wheels via `[tool.uv.sources]` in pyproject.toml.
