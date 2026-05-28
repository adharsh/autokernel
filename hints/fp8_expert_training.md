# FP8 Expert Training Hints

Read this before choosing an FP8 Expert-forward hypothesis. These are distilled
lessons from earlier FP8 experiments; `reference.py` and `validate.py` remain
the task contract.

## Critical Takeaways

- Objective: make a scale-aware FP8 training path as fast as possible, ideally
  faster than the simple `1.70x` unit-scale reference. Do not trade away
  scale-aware FP8, fair weight handling, or quality margin for a faster
  validator-only result.
- Training is the default target for this hint file, not inference with
  immutable prepacked weights. Cross-call FP8 weight caches are not
  training-fair unless the task explicitly models maintained FP8 weight state
  and accounts for update/conversion cost.
- Scale-aware FP8 is required for every FP8 training experiment in this session.
  Unit-scale FP8 is historical context only: do not implement, profile, record,
  mark `keep`, make `current_base`, or use it to set `best_speedup`, even if it
  is faster.
- A simple unit-scale FP8 implementation reached about `1.70x` speedup and
  `46.8 ms` NCU kernel-sum time in a previous run while doing all FP8
  conversions inside the measured call. See
  `hints/examples/training_fair_fp8_reference.py` only to understand the
  historical dataflow. Do not rerun it as an experiment; the real goal is to
  beat it with a scale-aware implementation.
- A small follow-up on the same family reached about `46.7 ms`, but most of the
  real savings are already present in the simpler reference example.
- The largest real opportunity is a fused up-projection/activation path that
  avoids writing `h1/h3` to global memory and writes only `hidden_fp8`.
- After obvious FP8 conversion and hidden-fusion wins, runtime was dominated by FP8
  GEMMs. Small Python/API tweaks often gave tiny, noisy improvements.
- A prior non-training-fair run also reached about `1.71x` by reusing
  FP8-converted weights across calls. Do not reproduce that cache for a
  training-fair result; use it only as evidence about dataflow and bottlenecks.

## Experiment Validity

For this run, an FP8 candidate is a real experiment only if it uses an explicit
scale policy for the tensors that enter FP8 GEMMs: `x`, `weight1`, `weight3`,
`weight2`, and the hidden activation. The policy can compute scales inside the
measured call, or use maintained/accounted scale state that is credible for
training.

A candidate that casts to FP8 and then calls `_scaled_mm` with `scale_a=1` and
`scale_b=1` for all operands is unit-scale FP8. Do not implement, profile, or
record this as an experiment in this session. The existing example is enough
historical evidence for the unit-scale path.

If an existing `a*/1` branch is the simple unit-scale reference, treat it as an
aborted scratch attempt. Do not record it in the TSV, do not use its speed as
`best_speedup`, and do not branch future work from it. Return to the last valid
base and make the next numbered experiment scale-aware. Mention the aborted
unit-scale branch only in the next valid scale-aware note if useful.

Before running validation or NCU for any FP8 candidate, check the source for the
scale policy. The experiment is not ready to run unless the implementation and
hypothesis note identify:

- the scale granularity for `x`, packed/up weights, down weights, and hidden;
- whether scales are computed in the measured call or maintained/accounted
  training state;
- how the scales are passed into FP8 GEMMs and hidden quantization;
- why the policy is credible for training rather than only for this validator.

Do not spend `a*/1` on a trivial fixed/static-scale port of the historical FP8
dataflow. Earlier rows already cover the simple static-scale family well enough
to show that it can be fast but hard to beat. If all the candidate adds is
hardcoded non-unit constants around the reference FP8 structure, abandon it as
scratch and choose a more informative scale-aware hypothesis.

A static, calibrated, or power-of-two scale policy is acceptable only when it is
training-realistic and materially different from the known simple static-scale
baseline: for example a different granularity, a maintained/delayed state model,
hidden-activation scale handling, or a dataflow/codegen change that preserves
fast GEMM dispatch. A single scalar `torch.ones` scale reused for every FP8
operand is not a scale policy.

## Cache Fairness

Do not cache outputs, activations, input-derived intermediates, or anything that
recognizes benchmark tensor reuse.

For training-fair forward results, either:

- include BF16-to-FP8 weight conversion inside the measured path,
- explicitly model FP8 weight copies as maintained training state and account
  for their update cost elsewhere, or
- avoid cross-call FP8 weight caches.

Warmup-populated FP8 weight caches may be profiled only as clearly labeled
non-training-fair diagnostics. Do not let a diagnostic cache become the selected
training result.

## What Worked

Useful ideas from the strongest previous implementations:

- Convert `x` to FP8 once inside the measured path and reuse it.
- Use FP8 operands and tensor-core GEMMs. For weights, include conversion in the
  measured path or use explicitly accounted FP8 training state.
- Combine `weight1` and `weight3` into one wider per-expert up projection.
- Fuse `silu(h1) * h3` and cast the hidden activation to FP8.
- Replace native BF16-to-FP8 activation copy kernels with a Triton cast.

The simple unit-scale dataflow reference profile roughly broke down as:

```text
x/w2 FP8 casts:       3.3 ms
w1/w3 FP8 pack:       0.8 ms
32 widened up GEMMs: 26.5 ms
32 hidden kernels:    1.2 ms
32 down GEMMs:       15.0 ms
total:               46.8 ms
```

The 64 per-expert FP8 GEMMs dominated runtime. The up projection was the largest
piece, so avoiding BF16 `h1/h3` traffic or changing GEMM dataflow matters more
than more standalone pointwise tuning.

## Reference Example

`hints/examples/training_fair_fp8_reference.py` is the preferred historical
dataflow reference from the prior run because it is simple, self-contained, and
contains the main savings:

- Triton BF16-to-FP8 casts for `x` and `w2`.
- Triton pack of `weight1` and `weight3` into one FP8 `w13` tensor.
- One wider FP8-output up `_scaled_mm` per expert.
- One Triton hidden activation kernel per expert that reads FP8 `h1/h3` and
  writes FP8 `hidden`.
- One FP8-input/FP8-weight down `_scaled_mm` per expert that writes BF16 output.

Use it to understand the known good dataflow. If copying from it, still explain
what new change is being tested and why the profile supports it.

Important: the example uses simple unit-scale FP8 conversion. It is historical
context only, not a candidate to rerun. Valid FP8 training experiments need
explicit scale-aware quantization while preserving as much of this structure and
speed as possible.

## Scale-Aware FP8 Target

FP8 is a recipe, not just a dtype. The next search should keep the known fast
dataflow while adding the cheapest scale policy that is credible for training.

Start with this subset:

- Use E4M3 for forward activations and weights unless quality shows clear
  clipping that justifies E5M2 or a mixed format.
- Try per-tensor or per-expert scales before fine-grained per-token/per-channel
  scales. Coarser scales are easier to keep on fast GEMM paths.
- Prefer calibrated, delayed, or maintained scale state when it is
  training-realistic and its update cost is accounted for. Static or
  power-of-two scales need a concrete reason beyond "easy first experiment."
- Avoid naive amax reductions over huge tensors inside every forward unless the
  profile shows the cost is acceptable. Scale computation can erase FP8 gains.
- Track speed/quality Pareto results. A slightly slower scale-aware result can
  be more valuable than a faster unit-scale result if it is more credible for
  stable training.

Concrete target:

```text
x_bf16 -> x_fp8 plus x_scale
w1/w3/w2 bf16 -> fp8 plus weight scales, inside measured path or accounted state
packed w1/w3 scale-aware widened up projection
hidden activation -> hidden_fp8 plus hidden_scale
scale-aware down projection -> y_bf16
```

The main question is how to add scales without leaving the fast nvjet/cuBLASLt
kernel family. If a scale choice changes GEMM dispatch and slows the hot GEMMs,
record that tradeoff clearly instead of judging only by total speed. Do not
settle for a scale-aware path that is obviously capped below the `1.70x`
reference unless it proves an important quality/fairness point and suggests a
follow-up that can recover speed.

For Hopper/H200, use Hopper-specific mechanisms when they are the plausible way
to beat the reference: native FP8 tensor cores/WGMMA, CUTLASS-3 or CuTe SM90
schedules, persistent or grouped GEMM structure, TMA/asynchronous movement when
useful, FP8 epilogues, and SASS/PTX inspection for codegen blockers. Avoid
falling back to generic Triton tile mutations if profiling says the missing win
is nvjet-level Hopper scheduling or an FP8 epilogue that the high-level path
cannot express.

## What To Be Careful About

- `use_fast_accum=True` was not uniformly better. It improved down projection
  in one profile but hurt the widened up projection. Test accumulation choices
  per projection instead of applying one global setting.
- `_scaled_mm.out` with preallocated outputs was mixed. It helped the
  down/output path in one profile but hurt the widened up path. Use direct
  output writes only where profiling shows a benefit.
- Lower-kernel grouped FP8 paths were interesting, but they were still dominated
  by grouped/library GEMM time and carried the same cached-weight caveat when
  they reused FP8 weights across calls.
- Merely using FP8 operands is not enough. If the up projection writes `h1/h3`
  as global intermediates before activation, the implementation has not removed
  the key intermediate-traffic problem. BF16 `h1/h3` is especially expensive;
  FP8 `h1/h3` is better but still not the ideal fused dataflow.
- The Triton BF16-to-FP8 cast was a real win over native copy kernels. Later
  block-size tuning was marginal/noisy, so further cast gains likely need
  codegen inspection, vectorized/custom CUDA/PTX, fusion, or avoiding the cast.
- Be skeptical of very small wins. If a change is under roughly 0.5%, treat it
  as weak evidence unless repeated profiles or kernel breakdowns explain it.
- Passing the validator is a floor, not the quality target. Prefer results with
  relative MAE near or below `0.10` and cosine near or above `0.995` when
  comparing otherwise similar speedups.
- Treat `diagnostic_quality` rows from `validate.py` as required
  training-stability evidence. They are not official speed cases, but worse
  near-zero-relative behavior, norm drift, saturation-like outliers, or
  worst-expert quality should affect ranking, `keep` decisions, and the next
  experiment choice. Do not promote a faster result whose diagnostics make it
  less credible for training.
- If the candidate can cheaply expose a `diagnostics()` hook, report scale and
  quantization health there: `x`/weight/hidden scale ranges, saturation rates,
  zero rates, and hidden/output norm drift. These are especially useful when a
  scale-aware FP8 path passes the aggregate hard gates.
- The simple reference example is close to the preferred quality target but not
  comfortably past it. Prior measured quality was roughly relative MAE `0.101`,
  p99 relative error `6.44`, and cosine `0.9949`. A fused version that avoided
  intermediate up-output rounding improved quality to roughly relative MAE
  `0.093` and cosine `0.9957`, but did not yet win on speed.
- Do not optimize only for one-forward pass/fail gates. Stable training needs a
  credible FP8 scale policy, including how scales are computed, stored, updated,
  and accounted for in the measured workload.

## Overnight Dead Ends

These ideas were tried enough that repeating them needs a materially different
hypothesis:

- Standalone hidden-kernel rewrites, row-indexing changes, in-place hidden
  storage, and hidden approximations gave local wins at best. Hidden is only
  about `1.1 ms`, and GEMM timing variation usually erased those gains.
- Static FP8 scaling improved quality but slowed the library GEMMs enough to
  lose on speed. Use scaling only if the experiment is explicitly about quality
  margin or if a new profile explains why GEMM slowdown will not happen.
- Exposed `_scaled_mm` knobs were mostly exhausted: up projection wants no fast
  accumulation, down projection wants fast accumulation, direct `out=` writes
  matter by projection, and newer wrapper variants did not reveal a hidden
  faster backend.
- E5M2/mixed FP8 dtype variants did not beat E4M3 for this path and can fail the
  cosine gate.
- Vectorized/custom CUDA casts and standalone pack/cast rewrites did not create
  a large win. Cast/pack kernels are memory-bound and small relative to GEMMs.
- Lower-launch grouped FP8 paths did not beat the strong per-expert nvjet GEMM
  path. Fewer launches alone are not enough if the GEMM kernel family is slower.

## Fusion Lessons

Fusing up projection with `silu/mul` is still the right conceptual target, but
the prior direct fusion attempts did not beat the simple reference.

What happened:

- A straightforward Triton fused up-hidden kernel wrote only `hidden_fp8` and
  improved numerical quality, but its matmul schedule was much slower than the
  PyTorch/nvjet FP8 up GEMMs.
- The best simple Triton fused tile became close: fused up-hidden took about
  `28.0 ms` versus about `27.5 ms` for separate up GEMMs plus hidden kernels.
  It reached Hopper tensor-core instructions and high tensor utilization, but
  still lost to scheduler/barrier/resource pressure.
- Larger K/N tiles and narrower tile variants regressed. The common failure mode
  was low eligible warps, CTA barrier pressure, high shared-memory use, or lower
  occupancy.
- A CUTLASS dual-GEMM fused attempt was much slower because that path did not
  generate the desired native Hopper FP8 WGMMA-style code; it unpacked FP8 and
  used older HMMA-like instructions.
- Standalone CUTLASS replacements for the existing up/down GEMMs were slower
  than the nvjet kernels.

The lesson: do not repeat naive fused Triton tile mutations or one-for-one
CUTLASS GEMM replacements. A winning fused path probably needs a different
schedule class, such as a persistent/CuTe/CUTLASS-3 SM90 FP8 WGMMA design with a
real activation/FP8-output epilogue, while preserving nvjet-like tensor-core
efficiency.

## Better Target Dataflow

Aim for this training-fair dataflow:

```text
x_bf16 -> x_fp8 plus x_scale
w1/w3/w2 bf16 -> fp8 plus scales inside measured path,
                  or maintained/accounted FP8 state
up projection: FP8 operands, high-precision accumulation
compute silu(up1) * up3 before storing h1/h3 as any global intermediate
store only hidden_fp8 plus hidden_scale
down projection: FP8 operands, high-precision accumulation
write final y_bf16
```

Pseudocode:

```python
x_fp8, x_scale = quantize_fp8_with_scale(x_bf16)
w1_fp8, w1_scale = quantize_or_load_accounted_fp8_with_scale(w1_bf16)
w3_fp8, w3_scale = quantize_or_load_accounted_fp8_with_scale(w3_bf16)
w2_fp8, w2_scale = quantize_or_load_accounted_fp8_with_scale(w2_bf16)

for expert in experts:
    # Ideal custom kernel/CUTLASS epilogue keeps intermediates local.
    acc1 = fp8_gemm_accum_fp32(
        x_fp8[expert], w1_fp8[expert], x_scale, w1_scale[expert]
    )
    acc3 = fp8_gemm_accum_fp32(
        x_fp8[expert], w3_fp8[expert], x_scale, w3_scale[expert]
    )
    hidden_fp8[expert], hidden_scale[expert] = quantize_fp8_with_scale(
        silu(acc1) * acc3
    )

for expert in experts:
    y_bf16[expert] = fp8_gemm_out_bf16(
        hidden_fp8[expert],
        w2_fp8[expert],
        hidden_scale[expert],
        w2_scale[expert],
    )
```

The missing piece is a custom fused up-projection/activation path or
CUTLASS/CuTe-style epilogue that avoids global `h1/h3` traffic.

## Best Next Experiments

- Rebuild the same promising dataflow without cross-call weight caches: include
  weight quantization in the measured path, or explicitly model accounted FP8
  weight state.
- Use the simple reference example only as a scaffold for code structure. Add
  explicit scale-aware FP8 quantization before running validation or NCU for a
  real experiment.
- Build a scale-aware FP8 MGMM path first, then re-apply the known structure:
  packed `weight1/weight3`, FP8 up output, FP8 hidden, BF16 final output.
- Test the fastest credible scale granularities: per-tensor, per-expert, and
  maintained/delayed or calibrated power-of-two scales. Do not repeat the simple
  static-scale baseline as the first experiment. Move to finer
  per-block/per-channel scaling only if quality requires it and the GEMM API
  supports it efficiently.
- Compare scale-aware variants on both speed and quality. Prefer relative MAE
  near or below `0.10` and cosine near or above `0.995`. The target is a
  training-credible scale-aware result that beats the `1.70x` diagnostic
  reference, not a unit-scale result that merely passes validation.
- Prototype a fused `weight1/weight3` up projection that applies SiLU/multiply
  before global `h1/h3` stores and writes only `hidden_fp8`, but only with a
  schedule intended to preserve nvjet-level tensor-core efficiency.
- Investigate CUTLASS-3/CuTe SM90 grouped or persistent GEMM epilogues for
  activation plus FP8 output conversion. Avoid older dual-GEMM examples that do
  not emit native Hopper FP8 WGMMA-style instructions.
- Compose known mixed wins: no-fast accumulation for widened up, fast
  accumulation for down; direct-output writes only for down if the profile still
  supports it.
- If library GEMMs remain dominant and near speed-of-light, move to dataflow,
  launch-count, or custom GEMM/epilogue changes instead of more pointwise tuning.
- Keep the official stress shape for recorded speedups. Smaller dev cases are
  acceptable for debugging or compile checks, but do not record a result as
  `keep` based only on a smaller case because the bottleneck mix changes.
