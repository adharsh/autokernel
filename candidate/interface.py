"""Starting candidate implementation for the FP8 Expert forward task.

Agents should replace this BF16 baseline with an FP8 implementation that keeps
the same public `kernel_fn` signature.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from reference import kernel_fn as _bf16_reference_kernel


def kernel_fn(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    group_sizes: Sequence[int],
) -> torch.Tensor:
    return _bf16_reference_kernel(x, w1, w2, w3, group_sizes)
