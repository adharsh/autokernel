"""Task-specific validation and timing harness template.

Copy this file to the repository root as `validate.py`, then fill in the TODOs.
Agents parse the printed keys below, so keep these labels stable:

candidate_us
reference_us
correctness
peak_vram_mb

Design contract:
- Define exactly one stress benchmark case. It is the only case used for
  calibrated reference timing and candidate timing.
- Expose that stress case through `make_stress_inputs()`.
- Define separate correctness-only cases for edge coverage. These cases compare
  candidate outputs to reference outputs but do not affect reported timing.
- `reference_us` is loaded from `results/reference_timing.json`, produced by
  `uv run python scripts/calibrate_reference.py`.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import traceback
from dataclasses import dataclass
from typing import Any

import torch

from candidate.interface import kernel_fn as candidate_kernel_fn
from profile_utils import cpu_timer, cuda_timer
from reference import kernel_fn as reference_kernel_fn


RTOL = 1e-2
ATOL = 1e-2
DEFAULT_CANDIDATE_WARMUP = 20
DEFAULT_CANDIDATE_ITERS = 200

CANDIDATE_WARMUP = int(
    os.environ.get("AUTOKERNEL_CANDIDATE_WARMUP", DEFAULT_CANDIDATE_WARMUP)
)
CANDIDATE_ITERS = int(
    os.environ.get("AUTOKERNEL_CANDIDATE_ITERS", DEFAULT_CANDIDATE_ITERS)
)

ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCE_TIMING_PATH = ROOT / "results" / "reference_timing.json"


@dataclass(frozen=True)
class CaseSpec:
    """Task-specific case description.

    Replace these placeholder fields with the shape/options needed by the task.
    """

    name: str
    seed: int


STRESS_BENCHMARK_CASE = CaseSpec(
    name="stress_main",
    seed=1001,
)

CORRECTNESS_CASES = (
    STRESS_BENCHMARK_CASE,
    # Add correctness-only edge cases here. Examples:
    # CaseSpec("small_shape", 2001),
    # CaseSpec("no_optional_args", 3001),
)


def benchmark_dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_case(spec: CaseSpec) -> tuple[Any, ...]:
    """Create positional args for one case.

    The stress case and correctness-only cases should all be deterministic.
    Place tensors directly on `device()` and use `benchmark_dtype()` for the
    stress case unless the task requires a different dtype.
    """
    raise NotImplementedError("Fill in task-specific input construction")


def make_stress_inputs() -> tuple[Any, ...]:
    """Create the single stress case used for candidate and reference timing."""
    return make_case(STRESS_BENCHMARK_CASE)


def clone_inputs(args: tuple[Any, ...]) -> tuple[Any, ...]:
    """Clone tensor inputs so candidate/reference correctness runs are isolated."""
    cloned = []
    for arg in args:
        if torch.is_tensor(arg):
            cloned.append(arg.clone())
        else:
            cloned.append(arg)
    return tuple(cloned)


def assert_close(candidate: Any, reference: Any) -> None:
    """Task-specific correctness check.

    Extend this if the task returns tuples, nested structures, mutated inputs, or
    needs per-output tolerances.
    """
    torch.testing.assert_close(candidate, reference, rtol=RTOL, atol=ATOL)


def peak_vram_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**2


def time_us(
    fn,
    *,
    warmup: int = CANDIDATE_WARMUP,
    iters: int = CANDIDATE_ITERS,
) -> float:
    """Return median runtime in microseconds, or NaN if timing fails."""
    try:
        timer = cuda_timer if torch.cuda.is_available() else cpu_timer
        return timer(fn, warmup=warmup, iters=iters)["median_ms"] * 1000.0
    except Exception:
        traceback.print_exc()
        return math.nan


def reference_timing_path() -> Path:
    override = os.environ.get("AUTOKERNEL_REFERENCE_TIMING_PATH")
    if override:
        return Path(override)
    return DEFAULT_REFERENCE_TIMING_PATH


def calibrated_reference_us() -> float:
    override = os.environ.get("AUTOKERNEL_REFERENCE_US")
    if override:
        return float(override)

    path = reference_timing_path()
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Missing calibrated reference timing. Run "
            "`uv run python scripts/calibrate_reference.py` before validation, or set "
            "AUTOKERNEL_REFERENCE_US."
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid reference timing file: {path}") from exc

    expected_device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = payload.get("device_type")
    if device_type != expected_device_type:
        raise RuntimeError(
            f"Reference timing was calibrated for {device_type!r}, but validation is "
            f"running on {expected_device_type!r}. Re-run "
            "`uv run python scripts/calibrate_reference.py` in the target environment."
        )

    if payload.get("case") != STRESS_BENCHMARK_CASE.name:
        raise RuntimeError(
            f"Reference timing case {payload.get('case')!r} does not match "
            f"stress case {STRESS_BENCHMARK_CASE.name!r}."
        )

    return float(payload["reference_us"])


def check_correctness() -> str:
    try:
        with torch.no_grad():
            for spec in CORRECTNESS_CASES:
                args = make_case(spec)
                reference = reference_kernel_fn(*clone_inputs(args))
                candidate = candidate_kernel_fn(*clone_inputs(args))
                try:
                    assert_close(candidate, reference)
                except AssertionError as exc:
                    raise AssertionError(f"{spec.name}: {exc}") from exc
        return "PASS"
    except AssertionError:
        traceback.print_exc()
        return "FAIL"
    except Exception:
        traceback.print_exc()
        return "CRASH"


def main() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    correctness = check_correctness()
    bench_args = make_stress_inputs()
    with torch.no_grad():
        reference_us = calibrated_reference_us()
        candidate_us = time_us(lambda: candidate_kernel_fn(*bench_args))

    if correctness == "PASS" and (math.isnan(reference_us) or math.isnan(candidate_us)):
        correctness = "CRASH"

    print(f"candidate_us: {candidate_us:.3f}")
    print(f"reference_us: {reference_us:.3f}")
    print(f"correctness: {correctness}")
    print(f"peak_vram_mb: {peak_vram_mb():.1f}")


if __name__ == "__main__":
    main()
