# Hints Start Here

Read this file first, then read every other file under `hints/`, including
`hints/examples/`.

Critical rules for this FP8 Expert training run:

- The goal is a fast, training-credible, scale-aware FP8 Expert forward path.
  This is forward-only benchmarking for training, not inference prepacking.
- Treat weights as mutable training weights. In the current harness, prebuilt
  FP8 weight tensors, pre-expanded scale metadata, or warmup-populated weight
  caches are diagnostic-only. They can become official only if the human
  explicitly changes the benchmark contract and the refresh/update path is
  measured.
- The current harness has no separate training-state refresh hook. Do not add
  FP8-packed weights, precomputed scales, expanded scale metadata, or activation
  scale state as new inputs in `validate.py`/`reference.py` unless the human
  explicitly changes the benchmark contract. For official `keep` rows, compute
  that work inside the measured candidate invocation.
- Do not promote a candidate that hardcodes an activation scale such as
  `x_scale = 1/16`. Delayed/maintained activation scales need an explicit
  training update policy and must pass the scale-shift diagnostic cases.
- Unit-scale FP8 is historical context only. Do not run, profile, record, or
  promote it.
- Do not spend `a*/1` on a trivial static-scale port of the known FP8 dataflow.
  Existing static-scale rows are comparison points, not strong bases to branch
  from or speed thresholds that should block more credible experiments.
- A real FP8 hypothesis must state scale granularity, how scales are computed
  inside the measured forward, how scales enter GEMMs and hidden quantization,
  and why the policy is credible for training. Mention a separate measured
  training-state update only if the human has explicitly changed the benchmark
  contract to include one.
- `diagnostic_quality` rows are required evidence for ranking. They are not the
  official speed benchmark, but they are hard validation evidence. Bad norm
  drift, near-zero-relative behavior, outliers, scale-shift failures, or
  worst-expert quality should prevent promotion.
- Do not skip stress or diagnostic quality for official `keep` rows.
- The simple example under `hints/examples/` is a dataflow reference only. Do
  not submit it or lightly wrap it as a new experiment.
- The historical maintained-state example is an upper-bound artifact, not a
  production training-forward result: it uses maintained FP8 weight state and a
  fixed activation scale without measuring the state refresh/update cost.

Most useful first directions:

- Maintain the known fast dataflow while adding a nontrivial scale policy:
  dynamic per-tensor/per-expert scales first; delayed or maintained scales only
  with a real measured refresh/update policy.
- Preserve fast Hopper FP8 GEMM dispatch. If a scale policy changes dispatch and
  slows hot GEMMs, document that tradeoff and pivot.
- Look for ways to fuse or avoid global `h1/h3` traffic without losing native
  Hopper FP8/WGMMA-class efficiency.
