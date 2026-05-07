"""Run one warmed-up candidate invocation for Nsight Compute.

This script is the default target for scripts/profile_ncu.sh. It imports the
task's validate.py only to call validate.make_stress_inputs(), warms up the
candidate, then profiles exactly one candidate call between CUDA profiler
start/stop markers.
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

    warmup = int(
        os.environ.get(
            "AUTOKERNEL_NCU_WARMUP",
            int(getattr(validate, "CANDIDATE_WARMUP", 20)),
        )
    )

    torch.cuda.reset_peak_memory_stats()

    bench_args = validate.make_stress_inputs()

    with torch.no_grad():
        for _ in range(warmup):
            candidate_kernel_fn(*bench_args)
        torch.cuda.synchronize()

        torch.cuda.profiler.start()
        try:
            candidate_kernel_fn(*bench_args)
            torch.cuda.synchronize()
        finally:
            torch.cuda.profiler.stop()

    print(f"ncu_warmup: {warmup}")
    print("profiled_candidate_invocations: 1")
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2
    print(f"peak_vram_mb: {peak_vram_mb:.1f}")


if __name__ == "__main__":
    main()
