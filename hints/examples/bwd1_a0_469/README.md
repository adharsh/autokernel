# bwd1 Winner Snapshot

This directory is an exact source snapshot of `candidate/` from branch
`a0/469`, commit `f301731`, in the `bwd1` run:

- `interface.py`
- `final_reduce_cuda.py`

The session report recorded `3267.36 us` and `90.82x` speedup for the old
width-4 BF16 stateful BOS+SiLU stress case.

This is hint code, not an importable candidate package. Copy and adapt the
relevant pieces under `candidate/`. The snapshot's dispatcher delegates all
unsupported configurations to `reference.kernel_fn`; that behavior is invalid
for the 24 `bwd2` report feature combinations and is intentionally rejected by
the evaluator.

The interface file also contains fail-closed SM90 cubin rewrites keyed to exact
code hashes. Keep those guards intact if reusing them, and do not assume the
rewrites apply after changing kernel code or specialization parameters.
