"""Trusted backward reference for causal depthwise conv1d with BOS resets.

The backward contract is the gradient of the conv10 forward task:

    out, final_states = forward(x, weight, bias, initial_states, bos_mask, activation)

Given upstream gradients for ``out`` and optionally ``final_states``, return
``(dx, dweight, dbias, dinitial_states)``. The forward path accumulates in fp32
and casts back to the input dtype; this file implements the corresponding
backward equations directly so validation does not need to build huge autograd
graphs for the benchmark suite.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch


class CandidateReferenceDelegationError(RuntimeError):
    """Raised when a report-parity candidate delegates to the reference."""


_CANDIDATE_REFERENCE_DELEGATION_FORBIDDEN = False


@contextmanager
def forbid_candidate_reference_delegation():
    """Make reference delegation fail while validating an optimized path."""
    global _CANDIDATE_REFERENCE_DELEGATION_FORBIDDEN
    previous = _CANDIDATE_REFERENCE_DELEGATION_FORBIDDEN
    _CANDIDATE_REFERENCE_DELEGATION_FORBIDDEN = True
    try:
        yield
    finally:
        _CANDIDATE_REFERENCE_DELEGATION_FORBIDDEN = previous


def _bos_prefix(bos_mask: torch.Tensor | None) -> torch.Tensor | None:
    if bos_mask is None:
        return None
    return bos_mask.to(torch.int32).cumsum(dim=1)


def _lag_valid(prefix: torch.Tensor, lag: int) -> torch.Tensor:
    batch, seqlen = prefix.shape
    valid = torch.zeros(batch, seqlen, device=prefix.device, dtype=torch.bool)
    if lag < seqlen:
        valid[:, lag:] = (prefix[:, lag:] - prefix[:, :-lag]) == 0
    return valid


def _preactivation(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
    bos_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the fp32 forward preactivation before optional SiLU."""
    batch, seqlen, dim = x.shape
    width = weight.shape[1]
    prefix = _bos_prefix(bos_mask)

    x_f = x.float()
    weight_f = weight.float()
    initial_f = initial_states.float() if initial_states is not None else None

    out = x_f * weight_f[:, width - 1].view(1, 1, dim)
    for lag in range(1, width):
        shifted = torch.zeros_like(x_f)
        if lag < seqlen:
            shifted[:, lag:, :] = x_f[:, : seqlen - lag, :]
            if prefix is not None:
                shifted = shifted * _lag_valid(prefix, lag).unsqueeze(2)
        out = out + shifted * weight_f[:, width - 1 - lag].view(1, 1, dim)

    if initial_f is not None:
        init_steps = min(seqlen, width - 1)
        for t in range(init_steps):
            init_contrib = torch.sum(
                initial_f[:, :, t:]
                * weight_f[:, : width - 1 - t].unsqueeze(0),
                dim=2,
            )
            if prefix is not None:
                init_contrib = init_contrib * (prefix[:, t] == 0).view(batch, 1)
            out[:, t, :] = out[:, t, :] + init_contrib

    if bias is not None:
        out = out + bias.float().view(1, 1, dim)
    return out


def _silu_grad(x: torch.Tensor) -> torch.Tensor:
    sigmoid = torch.sigmoid(x)
    return sigmoid * (1.0 + x * (1.0 - sigmoid))


def _final_state_backward(
    dx: torch.Tensor,
    dinitial: torch.Tensor | None,
    initial_states: torch.Tensor | None,
    bos_mask: torch.Tensor | None,
    dfinal_states: torch.Tensor | None,
) -> None:
    if dfinal_states is None:
        return

    batch, seqlen, _dim = dx.shape
    width_minus_one = dfinal_states.shape[2]
    prefix = _bos_prefix(bos_mask)
    dfinal_f = dfinal_states.float()

    for state_idx in range(width_minus_one):
        source_t = seqlen - (width_minus_one - state_idx)
        grad = dfinal_f[:, :, state_idx]
        if source_t >= 0:
            if prefix is not None and source_t + 1 < seqlen:
                grad = grad * (
                    (prefix[:, seqlen - 1] - prefix[:, source_t]) == 0
                ).view(batch, 1)
            dx[:, source_t, :] = dx[:, source_t, :] + grad
        elif initial_states is not None and dinitial is not None:
            init_idx = state_idx + seqlen
            grad = dfinal_f[:, :, state_idx]
            if prefix is not None and seqlen > 0:
                grad = grad * (prefix[:, seqlen - 1] == 0).view(batch, 1)
            dinitial[:, :, init_idx] = dinitial[:, :, init_idx] + grad


def _backward_from_g(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    initial_states: torch.Tensor | None,
    bos_mask: torch.Tensor | None,
    g: torch.Tensor,
    dfinal_states: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    batch, seqlen, dim = x.shape
    width = weight.shape[1]
    prefix = _bos_prefix(bos_mask)

    x_f = x.float()
    weight_f = weight.float()
    initial_f = initial_states.float() if initial_states is not None else None

    dx = torch.zeros_like(x_f)
    dweight = torch.zeros_like(weight_f)
    dbias = g.sum(dim=(0, 1)) if bias is not None else None
    dinitial = torch.zeros_like(initial_f) if initial_f is not None else None

    for lag in range(width):
        weight_idx = width - 1 - lag
        if lag == 0:
            g_lag = g
            x_lag = x_f
            dx[:, :, :] = dx + g_lag * weight_f[:, weight_idx].view(1, 1, dim)
        elif lag < seqlen:
            valid = (
                torch.ones(batch, seqlen - lag, device=x.device, dtype=torch.bool)
                if prefix is None
                else _lag_valid(prefix, lag)[:, lag:]
            )
            valid_f = valid.unsqueeze(2).to(torch.float32)
            g_lag = g[:, lag:, :] * valid_f
            x_lag = x_f[:, : seqlen - lag, :]
            dweight[:, weight_idx] = dweight[:, weight_idx] + (
                g_lag * x_lag
            ).sum(dim=(0, 1))
            dx[:, : seqlen - lag, :] = dx[:, : seqlen - lag, :] + (
                g_lag * weight_f[:, weight_idx].view(1, 1, dim)
            )
            continue
        else:
            continue

        dweight[:, weight_idx] = dweight[:, weight_idx] + (
            g_lag * x_lag
        ).sum(dim=(0, 1))

    if initial_f is not None and dinitial is not None:
        init_steps = min(seqlen, width - 1)
        for t in range(init_steps):
            valid = (
                torch.ones(batch, device=x.device, dtype=torch.bool)
                if prefix is None
                else prefix[:, t] == 0
            )
            g_t = g[:, t, :] * valid.view(batch, 1).to(torch.float32)
            for state_idx in range(t, width - 1):
                weight_idx = state_idx - t
                dweight[:, weight_idx] = dweight[:, weight_idx] + (
                    g_t * initial_f[:, :, state_idx]
                ).sum(dim=0)
                dinitial[:, :, state_idx] = dinitial[:, :, state_idx] + (
                    g_t * weight_f[:, weight_idx].view(1, dim)
                )

    _final_state_backward(dx, dinitial, initial_states, bos_mask, dfinal_states)

    return (
        dx.to(dtype=x.dtype),
        dweight.to(dtype=weight.dtype),
        dbias.to(dtype=bias.dtype) if dbias is not None and bias is not None else None,
        dinitial.to(dtype=initial_states.dtype)
        if dinitial is not None and initial_states is not None
        else None,
    )


def _final_states_forward(
    x: torch.Tensor,
    initial_states: torch.Tensor | None,
    bos_mask: torch.Tensor | None,
    width: int,
) -> torch.Tensor:
    """Kept for small autograd parity tests and documentation."""
    batch, seqlen, dim = x.shape
    prefix = _bos_prefix(bos_mask)
    final_states = torch.zeros(batch, dim, width - 1, device=x.device, dtype=x.dtype)
    for state_idx in range(width - 1):
        source_t = seqlen - (width - 1 - state_idx)
        if source_t >= 0:
            value = x[:, source_t, :]
            if prefix is not None and source_t + 1 < seqlen:
                value = value * (
                    (prefix[:, seqlen - 1] - prefix[:, source_t]) == 0
                ).view(batch, 1)
        elif initial_states is not None:
            value = initial_states[:, :, state_idx + seqlen]
            if prefix is not None and seqlen > 0:
                value = value * (prefix[:, seqlen - 1] == 0).view(batch, 1)
        else:
            continue
        final_states[:, :, state_idx] = value

    return final_states


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
    """Return gradients ``(dx, dweight, dbias, dinitial_states)``.

    ``dout`` is required and has the same shape as the forward output.
    ``dfinal_states`` may be ``None`` to indicate that the final-state output is
    not used by the loss.
    """
    if _CANDIDATE_REFERENCE_DELEGATION_FORBIDDEN:
        raise CandidateReferenceDelegationError(
            "report-matrix cases must execute candidate kernels, not "
            "reference.kernel_fn"
        )

    if dout is None:
        raise TypeError("dout is required for the backward benchmark")

    if activation not in (None, "silu"):
        raise NotImplementedError("activation must be None or silu")

    z = _preactivation(x, weight, bias, initial_states, bos_mask)
    if activation == "silu":
        g = dout.float() * _silu_grad(z)
    else:
        g = dout.float()

    return _backward_from_g(
        x,
        weight,
        bias,
        initial_states,
        bos_mask,
        g,
        dfinal_states,
    )
