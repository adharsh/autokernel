# Hints Start Here

Read this file first, then read every other file under `hints/`, including
`hints/examples/`.

Critical rules for this causal conv1d backward run:

- The task is the backward pass for the conv10 forward semantics, not another
  forward-only pass. `kernel_fn` must return `(dx, dweight, dbias,
  dinitial_states)` for the forward inputs plus `dout` and optional
  `dfinal_states`.
- Preserve the conv10 behavior exactly: causal depthwise width-2/3/4 conv,
  optional bias, optional `initial_states`, optional `bos_mask` resets, optional
  SiLU, fp32 forward accumulation, output cast behavior, and final-state
  gradient contribution.
- The stress case is the same shape family as conv10 forward:
  `batch=8`, `seqlen=65536`, `dim=4096`, `width=4`, BF16, bias, initial
  states, BOS mask, SiLU, `dout`, and `dfinal_states`. This intentionally
  matches the primary `../conv10` forward stress case. The baseline backward
  reference uses about 132 GB under NCU on H200, so keep extra workspaces and
  temporary tensors tight.
- Do not add precomputed forward outputs, preactivation tensors, SiLU
  derivatives, convolution windows, valid-lag matrices, partial reductions,
  partial gradients, transformed inputs/weights, or packed operator work as
  inputs. Compact metadata such as BOS offsets or sequence ids is allowed only
  as a deliberate interface reformulation.
- Treat the copied conv10 forward implementation as strategy evidence, not as a
  candidate. It optimized the forward main path to about `2013.55 us`, but
  backward has different bottlenecks: `dx` needs a reversed causal stencil,
  `dweight` reduces over batch/time, `dbias` reduces post-activation gradients,
  and `dfinal_states` adds tail/prefix gradient paths.
- If a forward trick is reused, explain the backward analog in the note. Good
  candidates include width-specialized kernels, separating dense no-BOS regions
  from BOS repair, packed BOS metadata, dirty row masks, vectorized contiguous-D
  loads/stores, explicit SiLU derivative math, cache modifiers, and careful
  tail/final-state handling.
- Do not spend early experiments only on rewriting the baseline autograd call.
  The first serious goal should be a direct fused backward implementation for
  the width-4 BF16 stress path, with fallbacks for small width/no-bias/no-BOS
  correctness cases.

Useful first directions:

- Start from the algebra. For preactivation `z[t,d]`, compute
  `g[t,d] = dout[t,d] * silu'(z[t,d])` when activation is SiLU, otherwise
  `g = dout`. Then `dbias[d] = sum_t g[t,d]`, `dweight[d,k]` is a masked
  reduction of `g[t,d] * source_x[t,k,d]`, and `dx` is the reverse causal
  convolution of `g` with the same BOS/reset rules plus `dfinal_states`.
- Split the width-4 stress path from general fallbacks. A fast path can assume
  BF16, contiguous `(batch, seqlen, dim)` inputs, width 4, bias present,
  initial states present, BOS mask present, SiLU, and `dfinal_states` present;
  keep the reference wrapper or a simple generic implementation for other
  correctness cases until they are worth optimizing.
- Profile whether the first fused candidate is dominated by recomputing
  preactivation/SiLU derivative, by `dweight` reductions, by `dx` stores, or by
  many tiny BOS repair/final-state kernels. Let that decide whether to fuse or
  split kernels.
- The conv10 forward frontier suggests dense no-BOS regions are worth treating
  differently from BOS boundary rows. For backward, the same idea may apply to
  `g`, `dx`, and `dweight`: do the dense body with simple vectorized kernels and
  repair rows near BOS boundaries separately.
