"""Run one warmed-up candidate benchmark suite for Nsight Compute.

This script is the default target for scripts/profile_ncu.sh. It imports the
task's validate.py to build the official parity suite, warms up the candidate,
then profiles exactly one pass over that suite between CUDA profiler start/stop
markers.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import validate  # noqa: E402
from candidate.interface import kernel_fn as candidate_kernel_fn  # noqa: E402


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Nsight Compute profiling.")

    warmup = int(os.environ.get("AUTOKERNEL_NCU_WARMUP", "20"))

    torch.cuda.reset_peak_memory_stats()

    benchmark_inputs = validate.make_benchmark_inputs()

    with torch.enable_grad():
        for _ in range(warmup):
            validate.run_benchmark_suite(candidate_kernel_fn, benchmark_inputs)
        torch.cuda.synchronize()

        torch.cuda.profiler.start()
        try:
            validate.run_benchmark_suite(candidate_kernel_fn, benchmark_inputs)
            torch.cuda.synchronize()
        finally:
            torch.cuda.profiler.stop()

    print(f"ncu_warmup: {warmup}")
    print(f"profiled_candidate_cases: {len(benchmark_inputs)}")
    print("profiled_candidate_suite_invocations: 1")
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2
    print(f"peak_vram_mb: {peak_vram_mb:.1f}")


if __name__ == "__main__":
    main()
