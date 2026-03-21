"""Reference implementation -- ground truth matmul.

This file is FIXED and must never be modified.
"""
import torch


def reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Reference matrix multiplication using PyTorch's built-in operator."""
    return a @ b
