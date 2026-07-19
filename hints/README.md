# Hints Start Here

Read this file first, then every other file under `hints/`, including all files
under `hints/examples/`.

## Required Outcome

This run must produce an optimized backward implementation for every row in the
same BF16 feature matrix used by the xllm forward report:

- `B`: `1, 2, 3, 4, 5, 8`
- `L`: `3, 4, 11, 32, 128, 255, 512, 733, 1024, 4096, 8192, 16384, 32768, 65536`
- `D`: `8, 64, 123, 256, 1024`
- width: `2, 3, 4`
- activation: `None`, `silu`
- initial state: absent, present
- BOS mask: absent, present
- bias: present
- `dout`: present
- final-state gradient: present
- dtype: BF16

That is `6 * 14 * 5 * 3 * 2 * 2 * 2 = 10,080` cases. The 24 combinations
of width, activation, initial-state presence, and BOS presence are the feature
parity contract. A report-matrix case may not delegate to `reference.kernel_fn`,
FLA, or another framework fallback. `validate.py` rejects direct delegation to
`reference.kernel_fn` for these cases.

Optional bias, a missing final-state gradient, and FP32 remain valid auxiliary
API cases. They are correctness-tested, but this run does not require an
optimized path for them because they are not rows in the current report.

## Performance Target

Official NCU timing is the total duration of one pass over six benchmark
anchors. For each width, the suite includes two opposing feature configurations
so that both values of activation, initial-state presence, and BOS presence
affect the score. The primary anchor is the largest report shape,
`B=8, L=65536, D=1024, W=4`, with initial state, BOS, and SiLU.

`validate.make_stress_inputs()` remains available for focused work on that
primary anchor. Official calibration and experiment profiles use
`validate.make_benchmark_inputs()` and all six cases.

## Previous Winner

The best `bwd1` result was `a0/469` (`f301731`) at `3267.36 us` on the old
`B=8, L=65536, D=4096, W=4` stateful BOS+SiLU stress case, a `90.82x`
speedup over its calibrated reference. Its exact two-file candidate snapshot is
under `hints/examples/bwd1_a0_469/`.

Treat that snapshot as the performance seed for the width-4 stateful BOS+SiLU
path, not as a complete solution. Its dispatcher falls back to
`reference.kernel_fn` outside a narrow width-4 contract, and much of its final
speed comes from SM90 cubin peepholes tied to exact generated code. Preserve or
adapt the useful fast path, then add honest candidate kernels for the other 23
feature combinations. Do not claim parity by weakening the support check.

The copied forward winner in `hints/examples/conv10_best_forward_interface.py`
is additional design evidence. It is not a backward candidate.

## Mathematical Contract

For preactivation `z[t,d]`, compute
`g[t,d] = dout[t,d] * silu'(z[t,d])` for SiLU and `g = dout` otherwise. Then:

- `dbias[d]` reduces `g` over batch and time.
- `dweight[d,k]` reduces `g[t,d] * source_x[t,k,d]` with the same causal and
  BOS-reset validity as forward.
- `dx` is the reverse causal stencil with the same reset boundaries.
- `dinitial_states` collects valid prefix contributions.
- `dfinal_states` contributes to tail `dx` and, for short sequences, possibly
  `dinitial_states`.

The forward path accumulates in FP32 and casts outputs back to input dtype.
Preserve that behavior and the exact return tuple
`(dx, dweight, dbias, dinitial_states)`.

## Fairness

- Do not add precomputed forward outputs, preactivations, SiLU derivatives,
  convolution windows, validity matrices, partial reductions, partial
  gradients, or transformed inputs/weights as inputs.
- Compact sequence metadata such as BOS offsets may be proposed only as a
  deliberate interface reformulation.
- Do not special-case the known validation inputs or benchmark shapes. Kernels
  must be shape-generic across the declared report axes.
- The baseline reference wrapper is allowed only for `a*/0`, using
  `AUTOKERNEL_ALLOW_REFERENCE_BASELINE=1`. Every later experiment must validate
  without that override.

## Suggested Decomposition

- Keep width-specialized kernels where that produces better code, but share
  dispatch and common algebra when it stays readable.
- Separate dense and BOS-aware paths when reset repair would otherwise burden
  every row.
- Specialize activation at compile time so the linear path does not recompute
  preactivation or SiLU derivatives.
- Treat absent initial state as a real fast path, not a synthetic zero tensor
  that adds avoidable traffic.
- Make width-dependent reduction layouts explicit: the old winner's five-plane
  layout is `dbias + 4 * dweight` and cannot simply be relabeled for widths 2/3.
- Retain a production fallback only outside the report-parity contract.

Use the aggregate profile and focused per-anchor microbenchmarks to prevent a
large gain in one path from hiding a severe regression in another.
