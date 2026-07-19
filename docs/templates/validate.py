"""Task-specific validation harness template.

Copy this file to the repository root as `validate.py`, then fill in the TODOs.
Agents parse the printed keys below, so keep these labels stable:

reference_us
correctness
peak_vram_mb

Design contract:
- Define one benchmark target as `BENCHMARK_CASES`. It may contain one case or
  a small task-specific suite; calibration and candidate profiling use the same
  complete target.
- Expose the target through `make_benchmark_inputs()` and
  `run_benchmark_suite()`. `make_stress_inputs()` may remain as a focused
  compatibility helper for the primary case.
- Define separate correctness-only cases for edge coverage. These cases compare
  candidate outputs to reference outputs but do not affect NCU profiling.
- `reference_us` is loaded from `results/reference_timing.json`, produced by
  `uv run python scripts/calibrate_reference.py`.
- Do not time candidate implementations in this file. Candidate speed comes from
  Nsight Compute `Duration us` rows produced by `scripts/profile_ncu.sh`, whose
  default target is `scripts/profile_candidate_once.py`.
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
from reference import kernel_fn as reference_kernel_fn


RTOL = 1e-2
ATOL = 1e-2
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
BENCHMARK_SUITE_NAME = "stress_main_v1"
BENCHMARK_CASES = (STRESS_BENCHMARK_CASE,)

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

    Benchmark and correctness-only cases should all be deterministic. Place
    tensors directly on `device()` and use `benchmark_dtype()` for benchmark
    cases unless the task requires a different dtype.
    """
    raise NotImplementedError("Fill in task-specific input construction")


def make_stress_inputs() -> tuple[Any, ...]:
    """Create the primary benchmark case for focused profiling."""
    return make_case(STRESS_BENCHMARK_CASE)


def make_benchmark_inputs() -> tuple[tuple[Any, ...], ...]:
    """Create all inputs used for official calibration and profiling."""
    return tuple(make_case(spec) for spec in BENCHMARK_CASES)


def run_benchmark_suite(kernel_fn: Any, cases: tuple[tuple[Any, ...], ...]) -> None:
    """Run one complete benchmark-suite pass without retaining outputs."""
    for args in cases:
        kernel_fn(*args)


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

    if payload.get("case") != BENCHMARK_SUITE_NAME:
        raise RuntimeError(
            f"Reference timing case {payload.get('case')!r} does not match "
            f"benchmark suite {BENCHMARK_SUITE_NAME!r}."
        )

    if payload.get("cases") != [spec.name for spec in BENCHMARK_CASES]:
        raise RuntimeError(
            "Reference timing cases do not match BENCHMARK_CASES. Re-run "
            "`uv run python scripts/calibrate_reference.py`."
        )

    timing_source = payload.get("timing_source")
    if timing_source != "ncu_duration_us":
        raise RuntimeError(
            f"Reference timing source was {timing_source!r}, expected "
            "'ncu_duration_us'. Re-run `uv run python scripts/calibrate_reference.py`."
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
    with torch.no_grad():
        reference_us = calibrated_reference_us()

    if correctness == "PASS" and math.isnan(reference_us):
        correctness = "CRASH"

    print(f"reference_us: {reference_us:.3f}")
    print(f"correctness: {correctness}")
    print(f"peak_vram_mb: {peak_vram_mb():.1f}")


if __name__ == "__main__":
    main()
