# Hints Start Here

Read this file first, then read every other file under `hints/`, including
`hints/examples/`.

Critical rules for this FP8 Expert training run:

- The goal is a fast, training-credible, scale-aware FP8 Expert forward path.
- Unit-scale FP8 is historical context only. Do not run, profile, record, or
  promote it.
- Do not spend `a*/1` on a trivial static-scale port of the known FP8 dataflow.
  Existing static-scale rows are comparison points, not strong bases to branch
  from or speed thresholds that should block more credible experiments.
- A real FP8 hypothesis must state scale granularity, how scales are computed or
  maintained/accounted as training state, how scales enter GEMMs and hidden
  quantization, and why the policy is credible for training.
- `diagnostic_quality` rows are required evidence for ranking. They are not the
  official speed benchmark, but bad norm drift, near-zero-relative behavior,
  outliers, or worst-expert quality should prevent promotion.
- The simple example under `hints/examples/` is a dataflow reference only. Do
  not submit it or lightly wrap it as a new experiment.

Most useful first directions:

- Maintain the known fast dataflow while adding a nontrivial scale policy:
  calibrated, delayed, maintained, per-expert, or hidden-aware scales.
- Preserve fast Hopper FP8 GEMM dispatch. If a scale policy changes dispatch and
  slows hot GEMMs, document that tradeoff and pivot.
- Look for ways to fuse or avoid global `h1/h3` traffic without losing native
  Hopper FP8/WGMMA-class efficiency.
