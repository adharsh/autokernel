"""Historical unit-scale FP8 dataflow reference, not an experiment.

Do not run, submit, profile, or record this file as an experiment in this
session. It uses unit FP8 scales and is not a valid scale-aware FP8 training
implementation. Use it only to understand the prior fast dataflow shape.

Any real FP8 experiment must add an explicit scale policy before validation or
NCU profiling. That policy must cover `x`, packed/up weights, down weights, and
hidden activation quantization. Under the current harness, official
training-forward results must compute scales and FP8 weight state inside the
measured forward. Prebuilt state is diagnostic-only under the current contract.
It can become official only if the human explicitly changes the benchmark
contract and a measured training-state update is added.

Historical dataflow properties:
- all FP8 conversions happen inside the measured invocation;
- no cross-call weight, activation, or output caches are used;
- `weight1` and `weight3` are packed into one wider up projection;
- the up projection writes FP8, the hidden activation writes FP8, and the final
  down projection writes BF16;
- unsupported shapes fall back to the BF16 reference.

Known limits:
- unit scales are used throughout, so this is not scale-aware training FP8;
- it still writes the widened up result (`h1/h3`) to global memory as FP8 before
  the hidden activation;
- the 32 up GEMMs and 32 down GEMMs dominate runtime;
- quality is close to the preferred target margin but does not make this a valid
  experiment for this session.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import triton
import triton.language as tl

from reference import kernel_fn as _bf16_reference_kernel


_FP8_DTYPE = torch.float8_e4m3fn


@triton.jit
def _cast_fp8_kernel(src_ptr, out_ptr, n_elements: tl.constexpr, block_size: tl.constexpr) -> None:
    offsets = tl.program_id(0).to(tl.int64) * block_size + tl.arange(0, block_size).to(tl.int64)
    mask = offsets < n_elements
    value = tl.load(src_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, value, mask=mask)


@triton.jit
def _pack_w13_fp8_kernel(
    w1_ptr,
    w3_ptr,
    out_ptr,
    total: tl.constexpr,
    inter_dim: tl.constexpr,
    model_dim: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    offsets = tl.program_id(0).to(tl.int64) * block_size + tl.arange(0, block_size).to(tl.int64)
    mask = offsets < total
    d = offsets % model_dim
    j2 = (offsets // model_dim) % (inter_dim * 2)
    expert = offsets // (model_dim * inter_dim * 2)
    from_w3 = j2 >= inter_dim
    j = tl.where(from_w3, j2 - inter_dim, j2)
    src_offset = (expert * inter_dim + j) * model_dim + d
    value = tl.load(tl.where(from_w3, w3_ptr + src_offset, w1_ptr + src_offset), mask=mask, other=0.0)
    tl.store(out_ptr + offsets, value, mask=mask)


@triton.jit
def _silu_mul_fp8_kernel(
    up_ptr,
    out_ptr,
    total: tl.constexpr,
    inter_dim: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < total
    row = offsets // inter_dim
    col = offsets - row * inter_dim
    base = row * (inter_dim * 2) + col
    h1 = tl.load(up_ptr + base, mask=mask, other=0.0).to(tl.float32)
    h3 = tl.load(up_ptr + base + inter_dim, mask=mask, other=0.0).to(tl.float32)
    sigmoid = 1.0 / (1.0 + tl.exp(-h1))
    hidden = h1 * sigmoid * h3
    tl.store(out_ptr + offsets, hidden, mask=mask)


def _cast_fp8(src: torch.Tensor) -> torch.Tensor:
    out = torch.empty(src.shape, dtype=_FP8_DTYPE, device=src.device)
    n_elements = src.numel()
    if n_elements:
        block_size = 1024
        grid = (triton.cdiv(n_elements, block_size),)
        _cast_fp8_kernel[grid](src, out, n_elements, block_size, num_warps=4)
    return out


def _pack_w13_fp8(w1: torch.Tensor, w3: torch.Tensor) -> torch.Tensor:
    n_experts, inter_dim, model_dim = w1.shape
    out = torch.empty((n_experts, inter_dim * 2, model_dim), dtype=_FP8_DTYPE, device=w1.device)
    total = out.numel()
    if total:
        block_size = 1024
        grid = (triton.cdiv(total, block_size),)
        _pack_w13_fp8_kernel[grid](w1, w3, out, total, inter_dim, model_dim, block_size, num_warps=4)
    return out


def _silu_mul_fp8(up: torch.Tensor, out: torch.Tensor, rows: int, inter_dim: int) -> None:
    total = rows * inter_dim
    if total == 0:
        return
    block_size = 1024
    grid = (triton.cdiv(total, block_size),)
    _silu_mul_fp8_kernel[grid](up, out, total, inter_dim, block_size, num_warps=4)


def _can_use_fp8_path(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    group_sizes: Sequence[int],
) -> bool:
    if not x.is_cuda or x.dtype != torch.bfloat16:
        return False
    if w1.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16 or w3.dtype != torch.bfloat16:
        return False
    if x.ndim != 2 or w1.ndim != 3 or w2.ndim != 3 or w3.ndim != 3:
        return False
    tokens, model_dim = x.shape
    n_experts, inter_dim, w_model_dim = w1.shape
    if w3.shape != w1.shape or w2.shape != (n_experts, model_dim, inter_dim):
        return False
    if w_model_dim != model_dim or len(group_sizes) != n_experts:
        return False
    if sum(int(size) for size in group_sizes) != tokens:
        return False
    if model_dim % 16 or inter_dim % 16:
        return False
    return all(int(size) >= 0 and int(size) % 16 == 0 for size in group_sizes)


def kernel_fn(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    group_sizes: Sequence[int],
) -> torch.Tensor:
    if not _can_use_fp8_path(x, w1, w2, w3, group_sizes):
        return _bf16_reference_kernel(x, w1, w2, w3, group_sizes)

    n_experts, inter_dim, _ = w1.shape
    tokens, model_dim = x.shape

    # Unit scales are the reason this file is invalid for current experiments.
    # Keep this code as a dataflow reference only; real candidates need a scale
    # policy for x, weights, and hidden activation before validation/profiling.
    scale = torch.ones((), dtype=torch.float32, device=x.device)
    x_fp8 = _cast_fp8(x)
    w13_fp8 = _pack_w13_fp8(w1, w3)
    w2_fp8 = _cast_fp8(w2)

    y = torch.empty((tokens, model_dim), dtype=torch.bfloat16, device=x.device)
    max_group = max((int(size) for size in group_sizes), default=0)
    up_buf = torch.empty((max_group, inter_dim * 2), dtype=_FP8_DTYPE, device=x.device)
    hidden_fp8_buf = torch.empty((max_group, inter_dim), dtype=_FP8_DTYPE, device=x.device)

    start = 0
    for expert_idx, raw_size in enumerate(group_sizes):
        size = int(raw_size)
        end = start + size
        if size:
            x_e = x_fp8[start:end]
            up = up_buf[:size]
            torch._scaled_mm(
                x_e,
                w13_fp8[expert_idx].t(),
                scale_a=scale,
                scale_b=scale,
                out_dtype=_FP8_DTYPE,
                use_fast_accum=False,
                out=up,
            )
            hidden_fp8 = hidden_fp8_buf[:size]
            _silu_mul_fp8(up, hidden_fp8, size, inter_dim)
            torch._scaled_mm(
                hidden_fp8,
                w2_fp8[expert_idx].t(),
                scale_a=scale,
                scale_b=scale,
                out_dtype=torch.bfloat16,
                use_fast_accum=True,
                out=y[start:end],
            )
        start = end

    return y
