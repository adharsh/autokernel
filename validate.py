"""Validation and benchmark harness for causal conv1d backward.

Stable labels parsed by AutoKernel:

reference_us
correctness
peak_vram_mb
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from itertools import product
import json
import math
import os
from pathlib import Path
import traceback
from typing import Any

import torch

from candidate.interface import kernel_fn as candidate_kernel_fn
from reference import (
    CandidateReferenceDelegationError,
    forbid_candidate_reference_delegation,
    kernel_fn as reference_kernel_fn,
)


DEFAULT_GRAD_RTOL = 1e-3
DEFAULT_GRAD_ATOL = 1e-3
LARGE_BF16_GRAD_RTOL = 2e-2
LARGE_BF16_GRAD_ATOL = 2e-2
FP32_GRAD_RTOL = 1e-4
FP32_GRAD_ATOL = 1e-4
ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCE_TIMING_PATH = ROOT / "results" / "reference_timing.json"
ALLOW_REFERENCE_BASELINE_ENV = "AUTOKERNEL_ALLOW_REFERENCE_BASELINE"

# This is the exact BF16 matrix used by the xllm forward and backward reports.
REPORT_BATCHES = (1, 2, 3, 4, 5, 8)
REPORT_SEQLENS = (
    3,
    4,
    11,
    32,
    128,
    255,
    512,
    733,
    1024,
    4096,
    8192,
    16384,
    32768,
    65536,
)
REPORT_DIMS = (8, 64, 123, 256, 1024)
REPORT_WIDTHS = (2, 3, 4)
REPORT_ACTIVATIONS = (None, "silu")
REPORT_INITIAL_STATE_OPTIONS = (False, True)
REPORT_BOS_OPTIONS = (False, True)
REPORT_CASE_COUNT = (
    len(REPORT_BATCHES)
    * len(REPORT_SEQLENS)
    * len(REPORT_DIMS)
    * len(REPORT_WIDTHS)
    * len(REPORT_ACTIVATIONS)
    * len(REPORT_INITIAL_STATE_OPTIONS)
    * len(REPORT_BOS_OPTIONS)
)

BENCHMARK_SUITE_NAME = "backward_report_parity_v1"


@dataclass(frozen=True, kw_only=True)
class CaseSpec:
    name: str
    seed: int
    batch: int
    seqlen: int
    dim: int
    width: int = 4
    use_bias: bool = True
    use_initial_states: bool = True
    use_bos_mask: bool = True
    use_dfinal_states: bool = True
    activation: str | None = "silu"
    bos_prob: float = 0.05
    force_bos_first: bool = False
    dtype: torch.dtype | None = None
    requires_optimized_path: bool = True
    grad_rtol: float
    grad_atol: float


BENCHMARK_CASES = (
    # Preserve the bwd1 winning workload family, but use D=1024 so the primary
    # anchor is also an exact member of the 10,080-case report matrix.
    CaseSpec(
        name="anchor_w4_stateful_bos_silu",
        seed=1001,
        batch=8,
        seqlen=65536,
        dim=1024,
        width=4,
        use_initial_states=True,
        use_bos_mask=True,
        activation="silu",
        bos_prob=0.01,
        grad_rtol=LARGE_BF16_GRAD_RTOL,
        grad_atol=LARGE_BF16_GRAD_ATOL,
    ),
    CaseSpec(
        name="anchor_w4_stateless_dense_linear",
        seed=1002,
        batch=4,
        seqlen=16384,
        dim=1024,
        width=4,
        use_initial_states=False,
        use_bos_mask=False,
        activation=None,
        grad_rtol=LARGE_BF16_GRAD_RTOL,
        grad_atol=LARGE_BF16_GRAD_ATOL,
    ),
    CaseSpec(
        name="anchor_w3_stateful_dense_silu",
        seed=1003,
        batch=4,
        seqlen=16384,
        dim=1024,
        width=3,
        use_initial_states=True,
        use_bos_mask=False,
        activation="silu",
        grad_rtol=LARGE_BF16_GRAD_RTOL,
        grad_atol=LARGE_BF16_GRAD_ATOL,
    ),
    CaseSpec(
        name="anchor_w3_stateless_bos_linear",
        seed=1004,
        batch=4,
        seqlen=16384,
        dim=1024,
        width=3,
        use_initial_states=False,
        use_bos_mask=True,
        activation=None,
        bos_prob=0.01,
        force_bos_first=True,
        grad_rtol=LARGE_BF16_GRAD_RTOL,
        grad_atol=LARGE_BF16_GRAD_ATOL,
    ),
    CaseSpec(
        name="anchor_w2_stateful_bos_linear",
        seed=1005,
        batch=4,
        seqlen=16384,
        dim=1024,
        width=2,
        use_initial_states=True,
        use_bos_mask=True,
        activation=None,
        bos_prob=0.01,
        force_bos_first=True,
        grad_rtol=LARGE_BF16_GRAD_RTOL,
        grad_atol=LARGE_BF16_GRAD_ATOL,
    ),
    CaseSpec(
        name="anchor_w2_stateless_dense_silu",
        seed=1006,
        batch=4,
        seqlen=16384,
        dim=1024,
        width=2,
        use_initial_states=False,
        use_bos_mask=False,
        activation="silu",
        grad_rtol=LARGE_BF16_GRAD_RTOL,
        grad_atol=LARGE_BF16_GRAD_ATOL,
    ),
)

# Kept as a compatibility alias for focused microbench scripts. Official
# calibration and experiment profiling use all of BENCHMARK_CASES.
STRESS_BENCHMARK_CASE = BENCHMARK_CASES[0]


def _make_report_feature_cases() -> tuple[CaseSpec, ...]:
    cases = []
    combinations = product(
        REPORT_WIDTHS,
        REPORT_ACTIVATIONS,
        REPORT_INITIAL_STATE_OPTIONS,
        REPORT_BOS_OPTIONS,
    )
    for index, (width, activation, use_initial_states, use_bos_mask) in enumerate(
        combinations
    ):
        activation_name = activation or "linear"
        state_name = "stateful" if use_initial_states else "stateless"
        bos_name = "bos" if use_bos_mask else "dense"
        cases.append(
            CaseSpec(
                name=f"report_w{width}_{state_name}_{bos_name}_{activation_name}",
                seed=2000 + index,
                batch=2,
                seqlen=11,
                dim=123,
                width=width,
                use_initial_states=use_initial_states,
                use_bos_mask=use_bos_mask,
                activation=activation,
                bos_prob=0.2,
                force_bos_first=use_bos_mask,
                grad_rtol=LARGE_BF16_GRAD_RTOL,
                grad_atol=LARGE_BF16_GRAD_ATOL,
            )
        )
    return tuple(cases)


def _make_report_shape_cases() -> tuple[CaseSpec, ...]:
    """Exercise every B, L, and D value without materializing all 10,080 cases."""
    cases = []
    for index, seqlen in enumerate(REPORT_SEQLENS):
        use_initial_states = bool(index & 1)
        use_bos_mask = bool(index & 2)
        activation = "silu" if index & 4 else None
        cases.append(
            CaseSpec(
                name=f"report_shape_axis_{index:02d}",
                seed=3000 + index,
                batch=REPORT_BATCHES[index % len(REPORT_BATCHES)],
                seqlen=seqlen,
                dim=REPORT_DIMS[index % len(REPORT_DIMS)],
                width=REPORT_WIDTHS[index % len(REPORT_WIDTHS)],
                use_initial_states=use_initial_states,
                use_bos_mask=use_bos_mask,
                activation=activation,
                bos_prob=0.1,
                force_bos_first=use_bos_mask and index % 2 == 0,
                grad_rtol=LARGE_BF16_GRAD_RTOL,
                grad_atol=LARGE_BF16_GRAD_ATOL,
            )
        )
    return tuple(cases)


REPORT_FEATURE_CASES = _make_report_feature_cases()
REPORT_SHAPE_CASES = _make_report_shape_cases()

# These remain part of the public backward contract, but are not columns in the
# current 10,080-row BF16 report. A production fallback is allowed for them.
AUXILIARY_CORRECTNESS_CASES = (
    CaseSpec(
        name="auxiliary_no_bias",
        seed=4001,
        batch=2,
        seqlen=17,
        dim=37,
        width=4,
        use_bias=False,
        use_initial_states=True,
        use_bos_mask=True,
        activation="silu",
        bos_prob=0.2,
        force_bos_first=True,
        grad_rtol=DEFAULT_GRAD_RTOL,
        grad_atol=DEFAULT_GRAD_ATOL,
        requires_optimized_path=False,
    ),
    CaseSpec(
        name="auxiliary_no_final_state_grad",
        seed=4002,
        batch=3,
        seqlen=29,
        dim=65,
        width=2,
        use_dfinal_states=False,
        activation="silu",
        grad_rtol=DEFAULT_GRAD_RTOL,
        grad_atol=DEFAULT_GRAD_ATOL,
        requires_optimized_path=False,
    ),
    CaseSpec(
        name="auxiliary_fp32",
        seed=4003,
        batch=2,
        seqlen=17,
        dim=37,
        width=3,
        use_bos_mask=False,
        activation=None,
        dtype=torch.float32,
        grad_rtol=FP32_GRAD_RTOL,
        grad_atol=FP32_GRAD_ATOL,
        requires_optimized_path=False,
    ),
)

CORRECTNESS_CASES = (
    REPORT_FEATURE_CASES + REPORT_SHAPE_CASES + AUXILIARY_CORRECTNESS_CASES
)

assert len(REPORT_FEATURE_CASES) == 24
assert REPORT_CASE_COUNT == 10_080


def benchmark_dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _randn(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.randn(shape, device=device(), dtype=dtype, generator=generator)


def make_case(spec: CaseSpec) -> tuple[Any, ...]:
    """Create positional args for one deterministic backward case."""
    dtype = spec.dtype or benchmark_dtype()
    dev = device()
    generator = torch.Generator(device=dev)
    generator.manual_seed(spec.seed)

    x = _randn(
        (spec.batch, spec.seqlen, spec.dim),
        generator=generator,
        dtype=dtype,
    )
    weight = _randn(
        (spec.dim, spec.width),
        generator=generator,
        dtype=dtype,
    )
    bias = (
        _randn((spec.dim,), generator=generator, dtype=dtype)
        if spec.use_bias
        else None
    )
    initial_states = (
        _randn(
            (spec.batch, spec.dim, spec.width - 1),
            generator=generator,
            dtype=dtype,
        )
        if spec.use_initial_states
        else None
    )

    bos_mask = None
    if spec.use_bos_mask:
        bos_mask = torch.rand(
            (spec.batch, spec.seqlen),
            device=dev,
            generator=generator,
        ) < spec.bos_prob
        if spec.force_bos_first:
            bos_mask[:, 0] = True
        elif spec.seqlen > 0:
            bos_mask[:, 0] = False

    dout = _randn(
        (spec.batch, spec.seqlen, spec.dim),
        generator=generator,
        dtype=dtype,
    )
    dfinal_states = (
        _randn(
            (spec.batch, spec.dim, spec.width - 1),
            generator=generator,
            dtype=dtype,
        )
        if spec.use_dfinal_states
        else None
    )

    return (
        x,
        weight,
        bias,
        initial_states,
        bos_mask,
        spec.activation,
        dout,
        dfinal_states,
    )


def make_stress_inputs() -> tuple[Any, ...]:
    """Create the primary width-4 anchor for focused debugging/microbenchmarks."""
    return make_case(STRESS_BENCHMARK_CASE)


def make_benchmark_inputs() -> tuple[tuple[Any, ...], ...]:
    """Create all cases in the official aggregate performance suite."""
    return tuple(make_case(spec) for spec in BENCHMARK_CASES)


def run_benchmark_suite(kernel_fn: Any, cases: tuple[tuple[Any, ...], ...]) -> None:
    """Run every official performance case once without retaining outputs."""
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


def _assert_tensor_close(
    label: str,
    candidate: torch.Tensor,
    reference: torch.Tensor,
    spec: CaseSpec,
) -> None:
    try:
        torch.testing.assert_close(
            candidate,
            reference,
            rtol=spec.grad_rtol,
            atol=spec.grad_atol,
        )
    except AssertionError as exc:
        raise AssertionError(
            f"{label} mismatch "
            f"(rtol={spec.grad_rtol:g}, atol={spec.grad_atol:g}): {exc}"
        ) from exc


def assert_close(candidate: Any, reference: Any, spec: CaseSpec) -> None:
    """Compare returned gradient tensors."""
    if not isinstance(candidate, tuple) or not isinstance(reference, tuple):
        raise AssertionError(
            "kernel_fn must return (dx, dweight, dbias, dinitial_states)"
        )
    if len(candidate) != 4 or len(reference) != 4:
        raise AssertionError("kernel_fn must return exactly four outputs")

    labels = ("dx", "dweight", "dbias", "dinitial_states")
    for label, cand, ref in zip(labels, candidate, reference, strict=True):
        if ref is None:
            if cand is not None:
                raise AssertionError(f"{label} must be None")
            continue
        if cand is None:
            raise AssertionError(f"{label} is None, expected a tensor")
        _assert_tensor_close(label, cand, ref, spec)


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

    expected_cases = [spec.name for spec in BENCHMARK_CASES]
    if payload.get("cases") != expected_cases:
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
        allow_reference_baseline = os.environ.get(ALLOW_REFERENCE_BASELINE_ENV) == "1"
        for spec in CORRECTNESS_CASES:
            args = make_case(spec)
            reference = reference_kernel_fn(*clone_inputs(args))
            reject_delegation = (
                spec.requires_optimized_path and not allow_reference_baseline
            )
            guard = (
                forbid_candidate_reference_delegation()
                if reject_delegation
                else nullcontext()
            )
            try:
                with guard:
                    candidate = candidate_kernel_fn(*clone_inputs(args))
            except CandidateReferenceDelegationError as exc:
                raise AssertionError(f"{spec.name}: {exc}") from exc
            try:
                assert_close(candidate, reference, spec)
            except AssertionError as exc:
                raise AssertionError(f"{spec.name}: {exc}") from exc
            del reference, candidate, args
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
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
    reference_us = calibrated_reference_us()

    if correctness == "PASS" and math.isnan(reference_us):
        correctness = "CRASH"

    print(f"reference_us: {reference_us:.3f}")
    print(f"correctness: {correctness}")
    print(f"peak_vram_mb: {peak_vram_mb():.1f}")


if __name__ == "__main__":
    main()
