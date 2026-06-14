"""Validation and benchmark harness for causal conv1d backward.

Stable labels parsed by AutoKernel:

reference_us
correctness
peak_vram_mb
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import traceback
from typing import Any

import torch

from candidate.interface import kernel_fn as candidate_kernel_fn
from reference import kernel_fn as reference_kernel_fn


DEFAULT_GRAD_RTOL = 1e-3
DEFAULT_GRAD_ATOL = 1e-3
LARGE_BF16_GRAD_RTOL = 2e-2
LARGE_BF16_GRAD_ATOL = 2e-2
FP32_GRAD_RTOL = 1e-4
FP32_GRAD_ATOL = 1e-4
ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCE_TIMING_PATH = ROOT / "results" / "reference_timing.json"


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
    grad_rtol: float
    grad_atol: float


STRESS_BENCHMARK_CASE = CaseSpec(
    name="backward_stress_b8_l65536_d4096_w4_bos_silu",
    seed=1001,
    batch=8,
    seqlen=65536,
    dim=4096,
    width=4,
    use_bias=True,
    use_initial_states=True,
    use_bos_mask=True,
    use_dfinal_states=True,
    activation="silu",
    bos_prob=0.01,
    grad_rtol=LARGE_BF16_GRAD_RTOL,
    grad_atol=LARGE_BF16_GRAD_ATOL,
)

CORRECTNESS_CASES = (
    # The full stress shape is intentionally reserved for reference calibration
    # and NCU profiling through make_stress_inputs(). Running the analytical
    # reference plus candidate on that shape for every validation pass makes the
    # optimization loop impractical.
    CaseSpec(
        name="primary_b8_l4096_d4096_w4_bos_silu",
        seed=1234,
        batch=8,
        seqlen=4096,
        dim=4096,
        width=4,
        use_bias=True,
        use_initial_states=True,
        use_bos_mask=True,
        use_dfinal_states=True,
        activation="silu",
        bos_prob=0.01,
        force_bos_first=True,
        grad_rtol=LARGE_BF16_GRAD_RTOL,
        grad_atol=LARGE_BF16_GRAD_ATOL,
    ),
    CaseSpec(
        name="latency_small_b2_l128_d256_w4_bos_silu",
        seed=1235,
        batch=2,
        seqlen=128,
        dim=256,
        width=4,
        use_bias=True,
        use_initial_states=True,
        use_bos_mask=True,
        use_dfinal_states=True,
        activation="silu",
        bos_prob=0.01,
        force_bos_first=True,
        grad_rtol=DEFAULT_GRAD_RTOL,
        grad_atol=DEFAULT_GRAD_ATOL,
    ),
    CaseSpec(
        name="small_with_initial_no_bos_no_activation",
        seed=2001,
        batch=2,
        seqlen=17,
        dim=37,
        width=4,
        use_bias=True,
        use_initial_states=True,
        use_bos_mask=False,
        use_dfinal_states=True,
        activation=None,
        dtype=torch.float32,
        grad_rtol=FP32_GRAD_RTOL,
        grad_atol=FP32_GRAD_ATOL,
    ),
    CaseSpec(
        name="bos_at_first_without_initial_or_bias",
        seed=3001,
        batch=3,
        seqlen=23,
        dim=64,
        width=4,
        use_bias=False,
        use_initial_states=False,
        use_bos_mask=True,
        use_dfinal_states=True,
        force_bos_first=True,
        grad_rtol=DEFAULT_GRAD_RTOL,
        grad_atol=DEFAULT_GRAD_ATOL,
    ),
    CaseSpec(
        name="short_sequence_dense_bos",
        seed=4001,
        batch=4,
        seqlen=3,
        dim=31,
        width=4,
        use_bias=True,
        use_initial_states=True,
        use_bos_mask=True,
        use_dfinal_states=True,
        bos_prob=0.45,
        force_bos_first=True,
        grad_rtol=DEFAULT_GRAD_RTOL,
        grad_atol=DEFAULT_GRAD_ATOL,
    ),
    CaseSpec(
        name="width2_small_bos_silu_no_final_grad",
        seed=5001,
        batch=3,
        seqlen=29,
        dim=65,
        width=2,
        use_bias=True,
        use_initial_states=True,
        use_bos_mask=True,
        use_dfinal_states=False,
        activation="silu",
        bos_prob=0.17,
        force_bos_first=True,
        grad_rtol=DEFAULT_GRAD_RTOL,
        grad_atol=DEFAULT_GRAD_ATOL,
    ),
    CaseSpec(
        name="width3_small_bos_silu",
        seed=5002,
        batch=3,
        seqlen=31,
        dim=65,
        width=3,
        use_bias=True,
        use_initial_states=True,
        use_bos_mask=True,
        use_dfinal_states=True,
        activation="silu",
        bos_prob=0.17,
        force_bos_first=True,
        grad_rtol=DEFAULT_GRAD_RTOL,
        grad_atol=DEFAULT_GRAD_ATOL,
    ),
)


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

    if payload.get("case") != STRESS_BENCHMARK_CASE.name:
        raise RuntimeError(
            f"Reference timing case {payload.get('case')!r} does not match "
            f"stress case {STRESS_BENCHMARK_CASE.name!r}."
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
        for spec in CORRECTNESS_CASES:
            args = make_case(spec)
            reference = reference_kernel_fn(*clone_inputs(args))
            candidate = candidate_kernel_fn(*clone_inputs(args))
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
