"""Run one warmed-up reference invocation for Nsight Compute calibration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WARMUP = 5
sys.path.insert(0, str(ROOT))

import validate  # noqa: E402
from reference import kernel_fn as reference_kernel_fn  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help="Reference warmup calls before the single profiled invocation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Nsight Compute profiling.")

    bench_args = validate.make_stress_inputs()

    with torch.enable_grad():
        for _ in range(args.warmup):
            reference_kernel_fn(*bench_args)
        torch.cuda.synchronize()

        torch.cuda.profiler.start()
        try:
            reference_kernel_fn(*bench_args)
            torch.cuda.synchronize()
        finally:
            torch.cuda.profiler.stop()

    print(f"ncu_reference_warmup: {args.warmup}")
    print("profiled_reference_invocations: 1")


if __name__ == "__main__":
    main()
