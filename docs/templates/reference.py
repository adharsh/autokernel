"""Task-specific reference implementation template.

Copy this file to the repository root as `reference.py`, then replace
`kernel_fn` with the trusted ground-truth implementation for the task.

Optimization agents must not edit the root `reference.py`.
`validate.py` will call this implementation for every correctness case, while
`scripts/calibrate_reference.py` profiles it once with Nsight Compute on the
complete benchmark target declared by `validate.BENCHMARK_CASES` and stores
that calibrated timing under
`results/reference_timing.json`.
"""


def kernel_fn(*args, **kwargs):
    """Return the ground-truth output for the task-specific inputs."""
    raise NotImplementedError("Fill in the task-specific reference implementation")
