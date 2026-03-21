"""Validation harness for the toy matmul kernel.

Times reference vs. candidate, checks correctness, and prints results
in the key: value format expected by agent_loop.py.

Usage:  cd examples/toy_kernel && uv run python validate.py
"""
import torch

from reference import reference
from candidate.interface import kernel_fn


def main() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda")

    N = 1024
    a = torch.randn(N, N, device=device, dtype=torch.float32)
    b = torch.randn(N, N, device=device, dtype=torch.float32)

    # -- correctness check --
    ref_out = reference(a, b)
    cand_out = kernel_fn(a, b)
    correct = torch.allclose(ref_out, cand_out, atol=1e-3, rtol=1e-3)

    # -- timing setup --
    warmup = 10
    iters = 100

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # -- time reference --
    for _ in range(warmup):
        reference(a, b)
    torch.cuda.synchronize()

    start_event.record()
    for _ in range(iters):
        reference(a, b)
    end_event.record()
    torch.cuda.synchronize()
    ref_us = start_event.elapsed_time(end_event) * 1000.0 / iters  # ms -> us

    # -- time candidate --
    for _ in range(warmup):
        kernel_fn(a, b)
    torch.cuda.synchronize()

    start_event.record()
    for _ in range(iters):
        kernel_fn(a, b)
    end_event.record()
    torch.cuda.synchronize()
    cand_us = start_event.elapsed_time(end_event) * 1000.0 / iters  # ms -> us

    # -- peak VRAM --
    torch.cuda.reset_peak_memory_stats()
    kernel_fn(a, b)
    torch.cuda.synchronize()
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # -- print results in expected format --
    print(f"candidate_us: {cand_us:.1f}")
    print(f"reference_us: {ref_us:.1f}")
    print(f"correctness: {'PASS' if correct else 'FAIL'}")
    print(f"peak_vram_mb: {peak_vram_mb:.1f}")


if __name__ == "__main__":
    main()
