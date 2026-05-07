"""Task-specific validation and timing harness template.

Copy this file to the repository root as `validate.py`, then fill in the TODOs.
Agents parse the printed keys below, so keep these labels stable:

candidate_us
reference_us
correctness
peak_vram_mb
"""

from __future__ import annotations

import math
import traceback

import torch

from candidate.interface import kernel_fn as candidate_kernel_fn
from profile_utils import cuda_timer
from reference import kernel_fn as reference_kernel_fn


RTOL = 1e-2
ATOL = 1e-2
WARMUP = 10
ITERS = 100


def make_inputs():
    """Create one representative input case.

    Replace this with the exact shapes, dtypes, device placement, and value
    ranges for the benchmark task. Return a tuple of positional args.
    """
    raise NotImplementedError("Fill in task-specific benchmark inputs")


def clone_inputs(args):
    """Clone tensor inputs so candidate/reference runs do not share mutation."""
    cloned = []
    for arg in args:
        if torch.is_tensor(arg):
            cloned.append(arg.clone())
        else:
            cloned.append(arg)
    return tuple(cloned)


def assert_close(candidate, reference):
    """Task-specific correctness check.

    Extend this if the task returns nested structures or needs custom tolerances.
    """
    torch.testing.assert_close(candidate, reference, rtol=RTOL, atol=ATOL)


def peak_vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**2


def time_us(fn) -> float:
    """Return median runtime in microseconds, or NaN if timing fails."""
    try:
        return cuda_timer(fn, warmup=WARMUP, iters=ITERS)["median_ms"] * 1000.0
    except Exception:
        traceback.print_exc()
        return math.nan


def main() -> None:
    args = make_inputs()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    try:
        with torch.no_grad():
            reference = reference_kernel_fn(*clone_inputs(args))
            candidate = candidate_kernel_fn(*clone_inputs(args))
            assert_close(candidate, reference)
            correctness = "PASS"
    except AssertionError:
        correctness = "FAIL"
        traceback.print_exc()
    except Exception:
        correctness = "CRASH"
        traceback.print_exc()

    reference_us = time_us(lambda: reference_kernel_fn(*clone_inputs(args)))
    candidate_us = time_us(lambda: candidate_kernel_fn(*clone_inputs(args)))
    if correctness == "PASS" and (math.isnan(reference_us) or math.isnan(candidate_us)):
        correctness = "CRASH"

    print(f"candidate_us: {candidate_us:.3f}")
    print(f"reference_us: {reference_us:.3f}")
    print(f"correctness: {correctness}")
    print(f"peak_vram_mb: {peak_vram_mb():.1f}")


if __name__ == "__main__":
    main()
