# Conv10 Forward Lessons For Backward

The best recorded conv10 forward result was `a4/725`, commit
`291eb552b72c0074bc011f698c11e53d5c10575f`, with an official NCU duration of
`2013.55 us` and about `59.90x` speedup over the PyTorch reference. The copied
source snapshot is in `hints/examples/conv10_best_forward_interface.py`.

What carried the forward result:

- Width-4 specialization for the dominant BF16 stress path.
- A dense main kernel over eight time rows with contiguous-D vectorization.
- Separate handling for prefix/tail/final-state work and BOS repair.
- Packed BOS metadata and dirty-row masks to keep the dense main path simple.
- Explicit fast SiLU and FFMA spelling in Triton inline asm.
- Cache/store policy tuning such as `.cg` stores and targeted prefetching.
- Many small changes after the main dataflow was saturated; the last few
  microseconds were mostly secondary-kernel scheduling/store details.

Backward implications:

- Reuse the idea of dense no-BOS paths plus targeted repair, but do not copy the
  forward dataflow blindly. Backward has three large jobs: recompute or recover
  `z` for SiLU derivative, accumulate `dweight/dbias`, and write `dx`.
- `dweight` is a reduction over `(batch, time)` for each `(dim, width)`. It may
  need a different tiling and reduction strategy than the forward contiguous-D
  map. Its partial/final reduction layout must cover widths 2, 3, and 4.
- `dx` is a reversed masked width-2/3/4 stencil. Dense regions can use a simple
  width-specialized reverse conv, while rows adjacent to BOS boundaries need
  repair.
- `dfinal_states` contributes directly to tail `dx` and, for short sequences,
  to `dinitial_states`. Do not optimize only the `dout` path.
- Forward's final frontier was L2/DRAM heavy. If backward profiles similarly,
  prioritize reducing global traffic, recomputing cheap preactivation values
  only when it saves stores/loads, and fusing reductions only when it does not
  create excessive register pressure or atomics.

Known traps:

- A forward-only candidate or a wrapper around the conv10 forward code is not a
  valid experiment for this repo.
- The `bwd1` winner is not feature-complete. Reusing its fast path is encouraged,
  but retaining its reference fallback for any report-matrix case is invalid.
- Precomputing forward outputs, preactivations, SiLU derivatives, valid masks,
  or partial reductions as inputs changes the benchmark and is not allowed.
- A fast `dx`-only path is incomplete. The task returns all four gradients and
  correctness compares all present tensors.
- A single width-4 BOS+SiLU fast path is also incomplete. `validate.py`
  exhaustively checks all 24 report feature combinations, and official timing
  includes representative paths for every width and feature value.
