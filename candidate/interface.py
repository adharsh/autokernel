"""Baseline candidate for causal depthwise conv1d backward.

Agents should replace this reference wrapper with an optimized implementation
that keeps the same public ``kernel_fn`` signature and returns
``(dx, dweight, dbias, dinitial_states)``.
"""

from __future__ import annotations

import torch

from reference import kernel_fn as reference_kernel_fn


def kernel_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
    bos_mask: torch.Tensor | None = None,
    activation: str | None = None,
    dout: torch.Tensor | None = None,
    dfinal_states: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    return reference_kernel_fn(
        x,
        weight,
        bias,
        initial_states,
        bos_mask,
        activation,
        dout,
        dfinal_states,
    )
