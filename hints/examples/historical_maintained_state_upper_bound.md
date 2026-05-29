# Historical Maintained-State Upper Bound

This is a historical artifact from the previous run, not a valid official base
for this training-forward run.

The previous run found a maintained-state FP8 forward path that reached about
`1.81x` speedup, but it did so with state that was not refreshed inside the
measured forward:

- `x_scale = 1/16` was treated as maintained activation scale state.
- FP8-packed `w1/w3` and FP8 `w2` were accepted as input state.
- Expanded `w2` column scale metadata was accepted as input state.
- The measured profile did not include BF16-to-FP8 weight refresh, scale update,
  or metadata refresh cost.

That makes this useful as an upper-bound/dataflow clue, not as production
training-forward evidence. Do not submit this structure as `keep` unless the
human explicitly changes the benchmark contract and the missing training-state
refresh/update work is measured.

## Useful Structure

The useful ideas to preserve are:

- use scale-aware FP8 `_scaled_mm` for the up and down projections;
- pack `w1` and `w3` into one widened up-projection weight layout;
- store the widened up projection as FP8 rather than BF16;
- run one full-token hidden pass that computes `silu(h1) * h3`;
- quantize hidden rows with exact current-call row scales;
- use an inline approximate reciprocal for `hidden / hidden_scale`;
- return BF16 output.

## Rough Kernel Call Sequence

Each line below is one separate CUDA kernel or library GEMM launch in the
historical shape:

```text
x_bf16 -> x_fp8 with fixed/maintained x_scale=1/16

for each expert:
    scaled_mm(x_fp8[expert], w13_fp8_state[expert].T) -> up_fp8[h1,h3]

silu_mul_rowwise_quantize(up_fp8) -> hidden_fp8 plus hidden_row_scale

for each expert:
    scaled_mm(hidden_fp8[expert], w2_fp8_state[expert].T) -> y_bf16
```

The official timing was fast because the profile did not include:

```text
w1/w3/w2 BF16 -> FP8 refresh after weight updates
weight scale refresh
x activation scale update
w2 column scale metadata expansion
```

## How To Use This Artifact

Good next experiments may reuse the dataflow idea, but must repair the training
fairness gap:

- include weight quantization/packing and scale computation inside the measured
  forward; or
- include a measured refresh/update path for maintained FP8 weight state after
  the human explicitly changes the benchmark contract; or
- use the result only as diagnostic evidence and do not make it `keep`,
  `current_base`, or `best_speedup`.

Do not use this artifact to justify hardcoded activation scales or free FP8
weight caches.
