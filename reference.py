"""BF16 reference for the xllm MoE Expert forward pass.

The workload mirrors xllm.modules.moe.expert.Expert.forward with
hidden_dropout fixed to zero:

    h1 = silu(mgmm(x, w1, group_sizes, transpose=True))
    h3 = mgmm(x, w3, group_sizes, transpose=True)
    y = mgmm(h1 * h3, w2, group_sizes, transpose=True)

The grouped matmul uses PyTorch's grouped_mm, matching xllm's torch mgmm
backend semantics.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def _group_offsets(group_sizes: Sequence[int], device: torch.device) -> torch.Tensor:
    sizes = torch.tensor(group_sizes, dtype=torch.int32, device=device)
    return torch.cumsum(sizes, dim=0, dtype=torch.int32)


def _mgmm_transpose(
    x: torch.Tensor,
    weight: torch.Tensor,
    group_sizes: Sequence[int],
) -> torch.Tensor:
    """Grouped `x_i @ weight_i.T` with concatenated group outputs."""
    offsets = _group_offsets(group_sizes, x.device)
    return F.grouped_mm(x, weight.transpose(1, 2), offs=offsets)


def kernel_fn(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    group_sizes: Sequence[int],
) -> torch.Tensor:
    """Return the BF16 Expert forward output."""
    h1 = F.silu(_mgmm_transpose(x, w1, group_sizes))
    h3 = _mgmm_transpose(x, w3, group_sizes)
    hidden = h1 * h3
    return _mgmm_transpose(hidden, w2, group_sizes)
