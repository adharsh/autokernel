"""Validation and benchmark harness for FP8 xllm Expert forward.

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

import candidate.interface as candidate_interface
from candidate.interface import kernel_fn as candidate_kernel_fn
from reference import kernel_fn as reference_kernel_fn


ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCE_TIMING_PATH = ROOT / "results" / "reference_timing.json"

# These are hard pass/fail gates for exploration, not the final quality target.
# Ranking/hints should still prefer relative MAE <= 0.10 and cosine >= 0.995.
MAX_RELATIVE_MAE = float(os.environ.get("AUTOKERNEL_MAX_RELATIVE_MAE", "0.15"))
MIN_COSINE = float(os.environ.get("AUTOKERNEL_MIN_COSINE", "0.99"))
MAX_P99_RELATIVE_ERROR = float(os.environ.get("AUTOKERNEL_MAX_P99_RELATIVE_ERROR", "12.0"))
P99_SAMPLE_ELEMS = int(os.environ.get("AUTOKERNEL_P99_SAMPLE_ELEMS", str(4 * 1024 * 1024)))
METRIC_CHUNK_ELEMS = int(os.environ.get("AUTOKERNEL_METRIC_CHUNK_ELEMS", str(16 * 1024 * 1024)))
DIAGNOSTIC_RELATIVE_REF_ABS_THRESHOLD = float(
    os.environ.get("AUTOKERNEL_DIAGNOSTIC_RELATIVE_REF_ABS_THRESHOLD", "1e-3")
)


@dataclass(frozen=True)
class CaseSpec:
    name: str
    batch: int
    seqlen: int
    model_dim: int
    expert_inter_dim: int
    n_local_experts: int
    seed: int
    group_sizes: tuple[int, ...] | None = None

    @property
    def tokens(self) -> int:
        return self.batch * self.seqlen

    @property
    def shape_label(self) -> str:
        return (
            f"B={self.batch},S={self.seqlen},T={self.tokens},"
            f"D={self.model_dim},I={self.expert_inter_dim},E_local={self.n_local_experts}"
        )


@dataclass(frozen=True)
class DiagnosticCase:
    name: str
    spec: CaseSpec
    x_scale: float = 1.0
    token_outlier_stride: int = 0
    token_outlier_scale: float = 1.0
    weight_global_scale: float = 1.0
    weight_expert_stride: int = 0
    weight_expert_scale: float = 1.0
    weight_channel_stride: int = 0
    weight_channel_scale: float = 1.0


STRESS_BENCHMARK_CASE = CaseSpec(
    name="stress_h200_1t_local_8x65536_d8192_i2048_e32",
    batch=8,
    seqlen=65536,
    model_dim=8192,
    expert_inter_dim=2048,
    n_local_experts=32,
    seed=1001,
)

CORRECTNESS_CASES = (
    CaseSpec(
        name="small_balanced",
        batch=1,
        seqlen=16,
        model_dim=64,
        expert_inter_dim=128,
        n_local_experts=4,
        seed=2001,
    ),
    CaseSpec(
        name="zero_token_experts",
        batch=1,
        seqlen=32,
        model_dim=128,
        expert_inter_dim=256,
        n_local_experts=8,
        seed=2002,
        group_sizes=(0, 5, 0, 10, 1, 0, 8, 8),
    ),
    CaseSpec(
        name="skewed_groups",
        batch=2,
        seqlen=64,
        model_dim=256,
        expert_inter_dim=1024,
        n_local_experts=8,
        seed=2003,
        group_sizes=(1, 2, 4, 8, 16, 32, 64, 1),
    ),
    CaseSpec(
        name="medium_balanced",
        batch=4,
        seqlen=512,
        model_dim=512,
        expert_inter_dim=2048,
        n_local_experts=16,
        seed=2004,
    ),
)

# Diagnostic cases are part of the training-forward correctness contract. They
# are not timed, but they must pass the same aggregate quality gates so static
# scale shortcuts do not win only on the default random stress distribution.
DIAGNOSTIC_CASES = (
    DiagnosticCase(
        name="stress_shape_x_scale_4",
        spec=CaseSpec(
            name="diag_stress_shape_x_scale_4",
            batch=1,
            seqlen=1024,
            model_dim=8192,
            expert_inter_dim=2048,
            n_local_experts=32,
            seed=3001,
        ),
        x_scale=4.0,
    ),
    DiagnosticCase(
        name="small_x_scale_0p125",
        spec=CaseSpec(
            name="diag_small_x_scale_0p125",
            batch=1,
            seqlen=1024,
            model_dim=512,
            expert_inter_dim=1024,
            n_local_experts=8,
            seed=3002,
        ),
        x_scale=0.125,
    ),
    DiagnosticCase(
        name="small_x_scale_16",
        spec=CaseSpec(
            name="diag_small_x_scale_16",
            batch=1,
            seqlen=1024,
            model_dim=512,
            expert_inter_dim=1024,
            n_local_experts=8,
            seed=3006,
        ),
        x_scale=16.0,
    ),
    DiagnosticCase(
        name="small_weight_global_scale_8",
        spec=CaseSpec(
            name="diag_small_weight_global_scale_8",
            batch=1,
            seqlen=1024,
            model_dim=512,
            expert_inter_dim=1024,
            n_local_experts=8,
            seed=3007,
        ),
        weight_global_scale=8.0,
    ),
    DiagnosticCase(
        name="small_weight_global_scale_0p125",
        spec=CaseSpec(
            name="diag_small_weight_global_scale_0p125",
            batch=1,
            seqlen=1024,
            model_dim=512,
            expert_inter_dim=1024,
            n_local_experts=8,
            seed=3008,
        ),
        weight_global_scale=0.125,
    ),
    DiagnosticCase(
        name="small_weight_expert_outliers",
        spec=CaseSpec(
            name="diag_small_weight_expert_outliers",
            batch=1,
            seqlen=1024,
            model_dim=512,
            expert_inter_dim=1024,
            n_local_experts=8,
            seed=3009,
        ),
        weight_expert_stride=2,
        weight_expert_scale=8.0,
    ),
    DiagnosticCase(
        name="small_token_outliers",
        spec=CaseSpec(
            name="diag_small_token_outliers",
            batch=1,
            seqlen=1024,
            model_dim=512,
            expert_inter_dim=1024,
            n_local_experts=8,
            seed=3003,
        ),
        token_outlier_stride=257,
        token_outlier_scale=8.0,
    ),
    DiagnosticCase(
        name="small_weight_channel_outliers",
        spec=CaseSpec(
            name="diag_small_weight_channel_outliers",
            batch=1,
            seqlen=1024,
            model_dim=512,
            expert_inter_dim=1024,
            n_local_experts=8,
            seed=3004,
        ),
        weight_channel_stride=127,
        weight_channel_scale=6.0,
    ),
    DiagnosticCase(
        name="small_skewed_groups_m16",
        spec=CaseSpec(
            name="diag_small_skewed_groups_m16",
            batch=1,
            seqlen=1024,
            model_dim=512,
            expert_inter_dim=1024,
            n_local_experts=8,
            seed=3005,
            group_sizes=(16, 16, 32, 64, 128, 192, 256, 320),
        ),
    ),
)


def benchmark_dtype() -> torch.dtype:
    return torch.bfloat16


def device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Expert FP8 benchmark task.")
    return torch.device("cuda")


def _balanced_group_sizes(total: int, n_groups: int) -> tuple[int, ...]:
    base, rem = divmod(total, n_groups)
    return tuple(base + (1 if i < rem else 0) for i in range(n_groups))


def _group_sizes_for_case(spec: CaseSpec) -> tuple[int, ...]:
    if spec.group_sizes is not None:
        group_sizes = spec.group_sizes
    else:
        group_sizes = _balanced_group_sizes(spec.tokens, spec.n_local_experts)
    if len(group_sizes) != spec.n_local_experts:
        raise ValueError(f"{spec.name}: expected {spec.n_local_experts} group sizes")
    if any(size < 0 for size in group_sizes):
        raise ValueError(f"{spec.name}: group sizes must be non-negative")
    if sum(group_sizes) != spec.tokens:
        raise ValueError(f"{spec.name}: group sizes sum to {sum(group_sizes)}, expected {spec.tokens}")
    return tuple(int(size) for size in group_sizes)


def _normal(shape: tuple[int, ...], std: float, generator: torch.Generator) -> torch.Tensor:
    out = torch.empty(shape, dtype=benchmark_dtype(), device=device())
    return out.normal_(mean=0.0, std=std, generator=generator)


def make_case(spec: CaseSpec) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...]]:
    gen = torch.Generator(device=device())
    gen.manual_seed(spec.seed)
    group_sizes = _group_sizes_for_case(spec)

    x = _normal((spec.tokens, spec.model_dim), 1.0, gen)
    w1 = _normal((spec.n_local_experts, spec.expert_inter_dim, spec.model_dim), 1.0 / math.sqrt(spec.model_dim), gen)
    w3 = _normal((spec.n_local_experts, spec.expert_inter_dim, spec.model_dim), 1.0 / math.sqrt(spec.model_dim), gen)
    w2 = _normal((spec.n_local_experts, spec.model_dim, spec.expert_inter_dim), 1.0 / math.sqrt(spec.expert_inter_dim), gen)
    return x, w1, w2, w3, group_sizes


def make_diagnostic_case(
    case: DiagnosticCase,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...]]:
    x, w1, w2, w3, group_sizes = make_case(case.spec)

    if case.x_scale != 1.0:
        x.mul_(case.x_scale)
    if case.token_outlier_stride > 0:
        x[:: case.token_outlier_stride].mul_(case.token_outlier_scale)
    if case.weight_global_scale != 1.0:
        w1.mul_(case.weight_global_scale)
        w3.mul_(case.weight_global_scale)
        w2.mul_(case.weight_global_scale)
    if case.weight_expert_stride > 0:
        w1[:: case.weight_expert_stride].mul_(case.weight_expert_scale)
        w3[:: case.weight_expert_stride].mul_(case.weight_expert_scale)
        w2[:: case.weight_expert_stride].mul_(case.weight_expert_scale)
    if case.weight_channel_stride > 0:
        w1[:, :: case.weight_channel_stride, :].mul_(case.weight_channel_scale)
        w3[:, :: case.weight_channel_stride, :].mul_(case.weight_channel_scale)
        w2[:, :, :: case.weight_channel_stride].mul_(case.weight_channel_scale)

    return x, w1, w2, w3, group_sizes


def make_stress_inputs() -> tuple[Any, ...]:
    return make_case(STRESS_BENCHMARK_CASE)


def clone_inputs(args: tuple[Any, ...]) -> tuple[Any, ...]:
    cloned: list[Any] = []
    for arg in args:
        if torch.is_tensor(arg):
            cloned.append(arg.clone())
        elif isinstance(arg, tuple):
            cloned.append(tuple(arg))
        elif isinstance(arg, list):
            cloned.append(list(arg))
        else:
            cloned.append(arg)
    return tuple(cloned)


def _sample_pair(candidate: torch.Tensor, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    cand = candidate.reshape(-1)
    ref = reference.reshape(-1)
    n = cand.numel()
    if n == 0:
        empty = torch.empty(0, dtype=torch.float32, device=candidate.device)
        return empty, empty
    stride = max(1, math.ceil(n / P99_SAMPLE_ELEMS))
    cand_sample = cand[::stride][:P99_SAMPLE_ELEMS].float()
    ref_sample = ref[::stride][:P99_SAMPLE_ELEMS].float()
    return cand_sample, ref_sample


def output_metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    cand = candidate.reshape(-1)
    ref = reference.reshape(-1)
    n = cand.numel()
    if n != ref.numel():
        raise AssertionError(f"numel mismatch: candidate={n}, reference={ref.numel()}")

    sum_abs_err = 0.0
    sum_err = 0.0
    sum_abs_cand = 0.0
    sum_abs_ref = 0.0
    dot = 0.0
    cand_norm_sq = 0.0
    ref_norm_sq = 0.0
    max_abs_err = 0.0

    for start in range(0, n, METRIC_CHUNK_ELEMS):
        end = min(start + METRIC_CHUNK_ELEMS, n)
        c = cand[start:end].float()
        r = ref[start:end].float()
        diff = c - r
        abs_diff = diff.abs()
        sum_abs_err += float(abs_diff.sum().item())
        sum_err += float(diff.sum().item())
        sum_abs_cand += float(c.abs().sum().item())
        sum_abs_ref += float(r.abs().sum().item())
        dot += float((c * r).sum().item())
        cand_norm_sq += float((c * c).sum().item())
        ref_norm_sq += float((r * r).sum().item())
        max_abs_err = max(max_abs_err, float(abs_diff.max().item()) if abs_diff.numel() else 0.0)

    cand_sample, ref_sample = _sample_pair(candidate, reference)
    abs_err_sample = (cand_sample - ref_sample).abs()
    ref_abs_sample = ref_sample.abs()
    rel = abs_err_sample / ref_abs_sample.clamp_min(1e-6)
    p99 = float(torch.quantile(rel, 0.99).item()) if rel.numel() else 0.0
    p99_abs = float(torch.quantile(abs_err_sample, 0.99).item()) if abs_err_sample.numel() else 0.0
    ref_mask = ref_abs_sample > DIAGNOSTIC_RELATIVE_REF_ABS_THRESHOLD
    if ref_mask.numel():
        near_zero_fraction = float((~ref_mask).float().mean().item())
        masked_rel = rel[ref_mask]
        masked_p99 = float(torch.quantile(masked_rel, 0.99).item()) if masked_rel.numel() else float("nan")
    else:
        near_zero_fraction = 0.0
        masked_p99 = 0.0

    cand_norm = math.sqrt(cand_norm_sq)
    ref_norm = math.sqrt(ref_norm_sq)
    denom = max(sum_abs_ref / max(n, 1), 1e-12)
    cosine = dot / max(cand_norm * ref_norm, 1e-30)

    return {
        "mean_abs_error": sum_abs_err / max(n, 1),
        "mean_signed_error": sum_err / max(n, 1),
        "candidate_abs_mean": sum_abs_cand / max(n, 1),
        "reference_abs_mean": sum_abs_ref / max(n, 1),
        "relative_mae": (sum_abs_err / max(n, 1)) / denom,
        "p99_relative_error": p99,
        "p99_relative_error_ref_gt_threshold": masked_p99,
        "near_zero_reference_fraction": near_zero_fraction,
        "p99_abs_error": p99_abs,
        "max_abs_error": max_abs_err,
        "cosine_similarity": cosine,
        "norm_ratio": cand_norm / max(ref_norm, 1e-30),
    }


def assert_output_contract(candidate: torch.Tensor, reference: torch.Tensor) -> None:
    if candidate.shape != reference.shape:
        raise AssertionError(f"shape mismatch: candidate={tuple(candidate.shape)}, reference={tuple(reference.shape)}")
    if candidate.dtype != reference.dtype:
        raise AssertionError(f"dtype mismatch: candidate={candidate.dtype}, reference={reference.dtype}")
    if not torch.isfinite(candidate).all():
        raise AssertionError("candidate output contains non-finite values")


def quality_failures(metrics: dict[str, float]) -> list[str]:
    failures = []
    if metrics["relative_mae"] > MAX_RELATIVE_MAE:
        failures.append(f"relative_MAE={metrics['relative_mae']:.6g} > {MAX_RELATIVE_MAE}")
    if metrics["p99_relative_error"] > MAX_P99_RELATIVE_ERROR:
        failures.append(f"p99_relative_error={metrics['p99_relative_error']:.6g} > {MAX_P99_RELATIVE_ERROR}")
    if metrics["cosine_similarity"] < MIN_COSINE:
        failures.append(f"cosine_similarity={metrics['cosine_similarity']:.8f} < {MIN_COSINE}")
    return failures


def validate_quality(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    assert_output_contract(candidate, reference)
    metrics = output_metrics(candidate, reference)
    failures = quality_failures(metrics)
    if failures:
        raise AssertionError(", ".join(failures))
    return metrics


def assert_quality(candidate: torch.Tensor, reference: torch.Tensor) -> None:
    validate_quality(candidate, reference)


def worst_expert_quality(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    group_sizes: tuple[int, ...],
) -> dict[str, float]:
    start = 0
    worst_relative_mae = -1.0
    worst_relative_mae_expert = -1
    worst_cosine = float("inf")
    worst_cosine_expert = -1

    for expert_idx, size in enumerate(group_sizes):
        end = start + int(size)
        if end > start:
            metrics = output_metrics(candidate[start:end], reference[start:end])
            if metrics["relative_mae"] > worst_relative_mae:
                worst_relative_mae = metrics["relative_mae"]
                worst_relative_mae_expert = expert_idx
            if metrics["cosine_similarity"] < worst_cosine:
                worst_cosine = metrics["cosine_similarity"]
                worst_cosine_expert = expert_idx
        start = end

    return {
        "worst_expert_relative_mae": worst_relative_mae,
        "worst_expert_relative_mae_id": float(worst_relative_mae_expert),
        "worst_expert_cosine": worst_cosine,
        "worst_expert_cosine_id": float(worst_cosine_expert),
    }


def peak_vram_mb() -> float:
    return torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0


def reference_timing_path() -> Path:
    override = os.environ.get("AUTOKERNEL_REFERENCE_TIMING_PATH")
    return Path(override) if override else DEFAULT_REFERENCE_TIMING_PATH


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
    if payload.get("device_type") != expected_device_type:
        raise RuntimeError(
            f"Reference timing was calibrated for {payload.get('device_type')!r}, "
            f"but validation is running on {expected_device_type!r}."
        )
    if payload.get("case") != STRESS_BENCHMARK_CASE.name:
        raise RuntimeError(
            f"Reference timing case {payload.get('case')!r} does not match "
            f"stress case {STRESS_BENCHMARK_CASE.name!r}."
        )
    if payload.get("timing_source") != "ncu_duration_us":
        raise RuntimeError("Reference timing must come from ncu_duration_us.")
    return float(payload["reference_us"])


def check_correctness() -> str:
    try:
        with torch.no_grad():
            for spec in CORRECTNESS_CASES:
                args = make_case(spec)
                reference = reference_kernel_fn(*clone_inputs(args))
                candidate = candidate_kernel_fn(*clone_inputs(args))
                try:
                    assert_quality(candidate, reference)
                except AssertionError as exc:
                    raise AssertionError(f"{spec.name}: {exc}") from exc
                del args, reference, candidate
                torch.cuda.empty_cache()
        return "PASS"
    except AssertionError:
        traceback.print_exc()
        return "FAIL"
    except Exception:
        traceback.print_exc()
        return "CRASH"


def check_stress_quality() -> str:
    try:
        with torch.no_grad():
            args = make_stress_inputs()
            reference = reference_kernel_fn(*clone_inputs(args))
            candidate = candidate_kernel_fn(*clone_inputs(args))
            metrics = validate_quality(candidate, reference)
            diagnostics = collect_candidate_diagnostics()
        print_stress_quality(metrics)
        print_candidate_diagnostics("candidate_diagnostics_stress", [(STRESS_BENCHMARK_CASE.name, diagnostics)])
        del args, reference, candidate
        torch.cuda.empty_cache()
        return "PASS"
    except AssertionError:
        traceback.print_exc()
        return "FAIL"
    except Exception:
        traceback.print_exc()
        return "CRASH"


def check_diagnostic_quality() -> str:
    rows = []
    diagnostic_rows = []
    try:
        with torch.no_grad():
            for case in DIAGNOSTIC_CASES:
                args = make_diagnostic_case(case)
                reference = reference_kernel_fn(*clone_inputs(args))
                candidate = candidate_kernel_fn(*clone_inputs(args))
                assert_output_contract(candidate, reference)
                metrics = output_metrics(candidate, reference)
                metrics.update(worst_expert_quality(candidate, reference, args[-1]))
                diagnostics = collect_candidate_diagnostics()
                rows.append((case.name, case.spec.shape_label, metrics))
                diagnostic_rows.append((case.name, diagnostics))
                failures = quality_failures(metrics)
                del args, reference, candidate
                torch.cuda.empty_cache()
                if failures:
                    raise AssertionError(f"{case.name}: {', '.join(failures)}")
        print_quality_table("diagnostic_quality", rows, include_worst_expert=True)
        print_candidate_diagnostics("candidate_diagnostics_diagnostic", diagnostic_rows)
        print("diagnostic_quality_status: PASS")
        return "PASS"
    except AssertionError:
        traceback.print_exc()
        if rows:
            print_quality_table("diagnostic_quality", rows, include_worst_expert=True)
            print_candidate_diagnostics("candidate_diagnostics_diagnostic", diagnostic_rows)
        print("diagnostic_quality_status: FAIL")
        return "FAIL"
    except Exception:
        traceback.print_exc()
        if rows:
            print_quality_table("diagnostic_quality", rows, include_worst_expert=True)
            print_candidate_diagnostics("candidate_diagnostics_diagnostic", diagnostic_rows)
        print("diagnostic_quality_status: CRASH")
        return "CRASH"


def print_stress_quality(metrics: dict[str, float]) -> None:
    print_quality_table(
        "stress_quality",
        [(STRESS_BENCHMARK_CASE.name, STRESS_BENCHMARK_CASE.shape_label, metrics)],
        include_worst_expert=False,
    )
    if STRESS_BENCHMARK_CASE.tokens * STRESS_BENCHMARK_CASE.model_dim > P99_SAMPLE_ELEMS:
        print(f"p99 relative error uses a deterministic sample of up to {P99_SAMPLE_ELEMS} elements.")


def print_quality_table(
    title: str,
    rows: list[tuple[str, str, dict[str, float]]],
    *,
    include_worst_expert: bool,
) -> None:
    masked_label = f"p99 rel err ref>{DIAGNOSTIC_RELATIVE_REF_ABS_THRESHOLD:g}"
    expert_columns = " worst expert rel MAE | worst expert id | worst expert cosine | worst cosine id |"
    print(f"{title}:")
    print(
        "| case | shape | "
        "mean abs error | relative MAE | mean signed error | p99 relative error | "
        f"{masked_label} | p99 abs error | max abs error | cosine similarity | "
        "norm ratio | cand/ref abs mean | near-zero ref % |"
        + (expert_columns if include_worst_expert else "")
    )
    print(
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        + ("---:|---:|---:|---:|" if include_worst_expert else "")
    )
    for name, shape_label, metrics in rows:
        cand_ref_abs_mean = (
            f"{metrics['candidate_abs_mean']:.6g}/{metrics['reference_abs_mean']:.6g}"
        )
        line = (
            f"| {name} | {shape_label} | "
            f"{metrics['mean_abs_error']:.6g} | "
            f"{metrics['relative_mae']:.6g} | "
            f"{metrics['mean_signed_error']:.6g} | "
            f"{metrics['p99_relative_error']:.6g} | "
            f"{metrics['p99_relative_error_ref_gt_threshold']:.6g} | "
            f"{metrics['p99_abs_error']:.6g} | "
            f"{metrics['max_abs_error']:.6g} | "
            f"{metrics['cosine_similarity']:.8f} | "
            f"{metrics['norm_ratio']:.6g} | "
            f"{cand_ref_abs_mean} | "
            f"{100.0 * metrics['near_zero_reference_fraction']:.4g}% |"
        )
        if include_worst_expert:
            line += (
                f" {metrics['worst_expert_relative_mae']:.6g} | "
                f"{int(metrics['worst_expert_relative_mae_id'])} | "
                f"{metrics['worst_expert_cosine']:.8f} | "
                f"{int(metrics['worst_expert_cosine_id'])} |"
            )
        print(line)


def collect_candidate_diagnostics() -> dict[str, str] | None:
    hook = getattr(candidate_interface, "diagnostics", None)
    if hook is None:
        return None
    try:
        payload = hook()
    except Exception:
        traceback.print_exc()
        return {"status": "CRASH"}
    if not payload:
        return None

    rendered: dict[str, str] = {}
    for key in sorted(payload):
        value = payload[key]
        if torch.is_tensor(value):
            value = value.detach()
            if value.numel() == 1:
                value = value.item()
            else:
                value = value.float().mean().item()
        if isinstance(value, float):
            rendered_value = f"{value:.6g}"
        else:
            rendered_value = str(value)
        rendered[str(key)] = rendered_value
    return rendered


def print_candidate_diagnostics(title: str, rows: list[tuple[str, dict[str, str] | None]]) -> None:
    rows = [(name, payload) for name, payload in rows if payload]
    if not rows:
        return

    print(f"{title}:")
    print("| case | key | value |")
    print("|---|---|---:|")
    for case_name, payload in rows:
        assert payload is not None
        for key in sorted(payload):
            print(f"| {case_name} | {key} | {payload[key]} |")


def main() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    correctness = check_correctness()
    reference_us = calibrated_reference_us()

    if correctness == "PASS":
        correctness = check_stress_quality()

    if correctness == "PASS":
        diagnostic_correctness = check_diagnostic_quality()
        if diagnostic_correctness != "PASS":
            correctness = diagnostic_correctness

    if correctness == "PASS" and math.isnan(reference_us):
        correctness = "CRASH"

    print(f"reference_us: {reference_us:.3f}")
    print(f"correctness: {correctness}")
    print(f"peak_vram_mb: {peak_vram_mb():.1f}")


if __name__ == "__main__":
    main()
