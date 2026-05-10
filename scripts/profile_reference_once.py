"""Run one warmed-up reference invocation for Nsight Compute calibration."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import validate  # noqa: E402
from reference import kernel_fn as reference_kernel_fn  # noqa: E402


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Nsight Compute profiling.")

    warmup = int(
        os.environ.get(
            "AUTOKERNEL_REFERENCE_NCU_WARMUP",
            os.environ.get("AUTOKERNEL_NCU_WARMUP", "20"),
        )
    )

    bench_args = validate.make_stress_inputs()

    with torch.no_grad():
        for _ in range(warmup):
            reference_kernel_fn(*bench_args)
        torch.cuda.synchronize()

        torch.cuda.profiler.start()
        try:
            reference_kernel_fn(*bench_args)
            torch.cuda.synchronize()
        finally:
            torch.cuda.profiler.stop()

    print(f"ncu_reference_warmup: {warmup}")
    print("profiled_reference_invocations: 1")


if __name__ == "__main__":
    main()
