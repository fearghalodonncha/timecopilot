# GIFT-Eval Validation Report

## Scope

This note summarizes validation comparisons between local TimeCopilot runs and public GIFT-Eval benchmark results for:

- `patchtst-fm-all` vs public PatchTST benchmarks
- `flowstate-all` vs `benchmark/FlowState-r1.1`

The goal is not strict reproduction, but to check whether local runs are directionally aligned with the public benchmark and to identify where the largest mismatches occur.

## Local Comparison Context

The local multi-run comparison across 97 datasets currently shows:

- `ibm-r3-all` is the best overall run on mean `eval_metrics/MASE[0.5]`
- `ibm-best-all` is competitive and stronger on some absolute-error metrics
- `ttm-r3-all` gets the most outright dataset wins in some views, but is much less consistent overall
- `flowstate-all` and `patchtst-fm-all` trail the IBM ensembles on the current aggregate summary

## PatchTST-FM Validation

### Compared Against

- `benchmark/PatchTST-FM-r1`
- `benchmark/Granite-PatchTST-FM-r1`
- `benchmark/PatchTST`

### Overall Summary

From the focused leaderboard comparison:

- `benchmark/PatchTST-FM-r1`
  - `MASE[0.5] = 1.447`
  - `mean_weighted_sum_quantile_loss = 0.205`
- `benchmark/Granite-PatchTST-FM-r1`
  - `MASE[0.5] = 1.463`
  - `mean_weighted_sum_quantile_loss = 0.206`
- `patchtst-fm-all`
  - `MASE[0.5] = 1.673`
  - `mean_weighted_sum_quantile_loss = 0.228`
- `benchmark/PatchTST`
  - `MASE[0.5] = 1.703`
  - `mean_weighted_sum_quantile_loss = 0.232`

Interpretation:

- The local `patchtst-fm-all` run is clearly behind the public PatchTST-FM benchmarks.
- It still outperforms the classic `PatchTST` benchmark overall.
- This suggests the implementation is broadly credible but not yet benchmark-matching.

### Delta Analysis vs `benchmark/PatchTST-FM-r1`

- Datasets compared: `97`
- Candidate wins: `23`
- Benchmark wins: `74`
- Mean delta (`candidate - benchmark`) on `MASE[0.5]`: `+0.2258`
- Median delta: `+0.0354`

By term:

- `short`: 55 datasets, 19 wins, mean delta `+0.0871`
- `medium`: 21 datasets, 1 win, mean delta `+0.4575`
- `long`: 21 datasets, 3 wins, mean delta `+0.3575`

Interpretation:

- The local run is relatively close on `short` tasks.
- The main gap is on `medium` and `long` tasks.
- This looks more like a configuration or wrapper mismatch than random variation.

Most favorable datasets for the local run included:

- `electricity/W/short`
- `m4_daily/D/short`
- `electricity/H/short`
- `ett1/H/long`
- `kdd_cup_2018/D/short`
- `solar/H/long`

Largest regressions were concentrated in:

- `bizitobs_application/10S/*`
- `bizitobs_service/10S/*`
- `bizitobs_l2c/*`
- `solar/10T/*`
- `bitbrains_rnd/5T/*`

### Delta Analysis vs `benchmark/Granite-PatchTST-FM-r1`

- Datasets compared: `97`
- Candidate wins: `24`
- Benchmark wins: `73`
- Mean delta (`candidate - benchmark`) on `MASE[0.5]`: `+0.2105`
- Median delta: `+0.0328`

By term:

- `short`: 55 datasets, 20 wins, mean delta `+0.0645`
- `medium`: 21 datasets, 1 win, mean delta `+0.4475`
- `long`: 21 datasets, 3 wins, mean delta `+0.3559`

Interpretation:

- The story is almost identical to the comparison against `PatchTST-FM-r1`.
- The local run is directionally aligned, but still below benchmark parity.
- The gap remains structured around the same dataset families rather than being spread uniformly across the benchmark.

### PatchTST-FM Conclusion

The local `patchtst-fm-all` run is:

- plausible and directionally aligned with public PatchTST-FM behavior
- stronger than classic `PatchTST` overall
- still materially below the public PatchTST-FM benchmark level

The mismatch is concentrated in specific dataset families and in medium/long horizons, which strongly suggests a systematic configuration or wrapper difference rather than a complete implementation failure.

## FlowState Validation

### Compared Against

- `benchmark/FlowState-r1.1`

### Overall Summary

From the leaderboard comparison:

- `benchmark/FlowState-r1.1`
  - `MASE[0.5] = 1.455`
  - `mean_weighted_sum_quantile_loss = 0.200`
- `flowstate-all`
  - `MASE[0.5] = 1.641`
  - `mean_weighted_sum_quantile_loss = 0.232`

Interpretation:

- The local FlowState run is clearly behind the benchmark overall.
- The gap is noticeable on both deterministic and probabilistic summary metrics.

### Delta Analysis vs `benchmark/FlowState-r1.1`

- Datasets compared: `97`
- Candidate wins: `13`
- Benchmark wins: `84`
- Mean delta (`candidate - benchmark`) on `MASE[0.5]`: `+0.1860`
- Median delta: `+0.0666`

By term:

- `short`: 55 datasets, 8 wins, mean delta `+0.1610`
- `medium`: 21 datasets, 1 win, mean delta `+0.2815`
- `long`: 21 datasets, 4 wins, mean delta `+0.1557`

Interpretation:

- The local FlowState run is consistently below benchmark across all horizon groups.
- `medium` remains the weakest region, but the gap is not as horizon-specific as PatchTST-FM.
- This suggests a more general configuration mismatch, not just a long-horizon issue.

Most favorable datasets for the local run included:

- `bizitobs_application/10S/long`
- `solar/W/short`
- `m4_daily/D/short`
- `ett2/D/short`
- `ett2/H/long`
- `sz_taxi/H/short`

Largest regressions were concentrated in:

- `covid_deaths/D/short`
- `bizitobs_application/10S/medium`
- `bizitobs_service/10S/medium`
- `bizitobs_l2c/*`
- `loop_seattle/5T/*`
- `solar/10T/long`

### FlowState Conclusion

The local `flowstate-all` run is:

- directionally aligned with the public benchmark
- consistently below the benchmark across most of the suite
- particularly weak on a small number of concentrated dataset families

Compared with PatchTST-FM validation, FlowState appears less competitive in terms of win count, but its underperformance is also somewhat more consistent rather than being driven only by a few severe outliers.

## Cross-Model Validation Takeaways

Across both validations:

- The local wrappers are producing plausible, benchmark-like behavior.
- Neither local PatchTST-FM nor local FlowState is yet matching the public benchmark level.
- The errors are structured by dataset family rather than looking random.
- The most recurring problem area is the `bizitobs_*` family.
- PatchTST-FM looks closer on short horizons and farther on medium/long horizons.
- FlowState looks more uniformly below benchmark, with medium horizons still the weakest area.

## Likely Next Debugging Directions

The validation results suggest investigating:

- context length choices
- horizon handling
- preprocessing and normalization differences
- frequency handling / scale factor mapping
- quantile handling and forecast post-processing
- any benchmark-specific model or wrapper configuration not mirrored locally

## Supporting Files

Key outputs used in this report:

- `results/timecopilot/hpc/patchtst_validation/leaderboard_summary.csv`
- `results/timecopilot/hpc/patchtst_validation/validation_delta_summary.csv`
- `results/timecopilot/hpc/patchtst_validation/validation_delta_by_term.csv`
- `results/timecopilot/hpc/leaderboard/patchtst/validation_delta_summary.csv`
- `results/timecopilot/hpc/leaderboard/patchtst/validation_delta_by_term.csv`
- `results/timecopilot/hpc/leaderboard/flowstate/leaderboard_summary.csv`
- `results/timecopilot/hpc/leaderboard/flowstate/validation_delta_summary.csv`
- `results/timecopilot/hpc/leaderboard/flowstate/validation_delta_by_term.csv`
