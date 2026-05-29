# FP8 Expert Task Notes

Read this as task context. `reference.py` and `validate.py` remain the
correctness contract.

## Task Boundary

Optimize the full forward pass of `xllm.modules.moe.expert.Expert`, not a
standalone grouped-matmul microbenchmark:

```text
h1 = silu(mgmm(x, weight1, group_sizes, transpose=True))
h3 = mgmm(x, weight3, group_sizes, transpose=True)
hidden = h1 * h3
y = mgmm(hidden, weight2, group_sizes, transpose=True)
```

Keep the public task scoped to Expert forward. Internal FP8 grouped/per-expert
GEMM helpers are fine, but the measured implementation must compute the complete
Expert output.

This is still a training-forward task. It is acceptable to optimize only the
forward pass for now, but the assumptions must match training, not inference:
weights are mutable across optimizer steps, activation/weight scales must be
computed or refreshed by a real training policy, and prepacked immutable FP8
weights are not free inputs to the measured forward.

Do not turn the forward harness into an inference-prepack benchmark. In the
current harness, a result that consumes prebuilt FP8 weights, pre-expanded scale
metadata, or fixed activation scales is useful only as diagnostic evidence. It
can become an official training-forward `keep` only if the human explicitly
changes the benchmark contract and the refresh/update path and cost are
measured.

The current benchmark contract does not include a separate refresh/update hook.
Agents should not add FP8-packed weights, precomputed scales, expanded scale
metadata, or activation scale state to `validate.py`/`reference.py` as official
inputs. For now, official training-forward candidates compute that work inside
the measured `candidate.kernel_fn` invocation.

Why this matters:

- Expert output is the accuracy boundary; FP8 error compounds through two up
  projections, SiLU, multiply, and down projection.
- A standalone `mgmm` task would miss the main fusion opportunity:
  `silu(h1) * h3` plus quantizing `hidden` before `weight2`.
- `weight1` and `weight3` share `x` and `group_sizes`, so full Expert scope
  allows combined or co-scheduled up projections.
- Generic `mgmm` is used elsewhere in xllm; this task should not accidentally
  optimize the wrong contract.

## Stress Case

Default H200 stress case:

```text
batch = 8
seqlen = 65536
T = batch * seqlen = 524288 local routed tokens
D = model_dim = 8192
I = expert_inter_dim = 2048
E = n_local_experts = 32
```

This mirrors one local Expert.forward invocation for the xllm `k2moe-1T-a48B`
MoE shape after routing/all-to-all. The global config has `num_experts=256`,
`num_activated_experts=8`, and 8-way expert/model parallelism, so local routed
tokens are effectively `original_tokens * topk / world_size`.

`batch=16, seqlen=32768` is equivalent for Expert forward because the module
only sees the flattened local routed token count.

## Memory Scale

Weights are:

```text
weight1: [E, I, D]
weight3: [E, I, D]
weight2: [E, D, I]
```

At BF16, the three weights are about 3 GiB. For the default stress case, BF16
activations `x`, `h1`, `h3`, `hidden`, and `y` add about 22 GiB, for a lower
bound near 25 GiB before allocator overhead, workspaces, FP8 buffers, and
validation outputs.

This is why avoiding BF16 `h1/h3` global materialization is a meaningful target.

## Measurement

Official speed uses Nsight Compute kernel-sum duration:

```text
speedup = reference_us / ncu_duration_us
```

`validate.py` reports correctness, `reference_us`, peak VRAM, a stress-shape
quality comparison, and `diagnostic_quality` rows for training-forward scale
robustness. Smaller correctness cases cover balanced groups, skewed groups,
zero-token experts, and medium production-like shapes. Diagnostic quality cases
are not the official speed benchmark, but they are correctness evidence: use
them to catch fragile FP8 scaling, activation/weight magnitude shifts, outlier
behavior, near-zero relative-error noise, and per-expert regressions.
Current diagnostic cases include small/large activation scale shifts, global
small/large weight scale shifts, per-expert weight outliers, token outliers,
per-channel weight outliers, and skewed groups. These are meant to make static
scale shortcuts fail early.

## Implementation Guidance

Start from FP8 grouped or per-expert GEMM pieces, then profile the full Expert
forward. The first serious optimization targets are quantization, layout,
combining or co-scheduling `weight1/weight3`, fusing SiLU/multiply, reducing
launches, and avoiding unnecessary BF16 intermediate traffic.

For this training task, FP8 means scale-aware FP8. Unit-scale FP8 casts are
historical context only and are not valid experiments; see
`hints/fp8_expert_training.md` before coding, validating, profiling, or
recording any FP8 candidate.

Any custom grouped/per-expert path must preserve `group_sizes` semantics exactly,
including skewed groups and zero-token experts.

Do not start with one giant hand-written CUDA kernel for all three matmuls
unless profiling has already shown why that risk is justified.
