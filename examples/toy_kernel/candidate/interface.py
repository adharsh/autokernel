"""Candidate kernel implementation.

The agent edits THIS file -- all optimization code lives here.
Starting point: identical to reference (plain matmul).
"""
import torch


def kernel_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Candidate matrix multiplication -- optimize this."""
    return a @ b
