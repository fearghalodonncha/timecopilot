# TimeCopilot Ensemble Comparison

## Executive Summary

- Best overall mean `eval_metrics/MASE[0.5]`: `ibm-r3-all` (1.5500).
- Best overall mean `eval_metrics/mean_weighted_sum_quantile_loss`: `ibm-r3-all` (0.2144).
- Most dataset wins on `eval_metrics/MASE[0.5]`: `ttm-r3-all` (37 wins).
- Most consistent average rank: `ibm-r3-all` (2.000).
- The IBM ensembles lead overall, while single-model runs are more specialized and less consistent.

## Runs Compared

- `ibm-best-all`
- `ibm-r3-all`
- `patchtst-fm-all`
- `flowstate-all`
- `ttm-r3-all`

## Overall Summary

| run_name        | model           |   eval_metrics/MASE[0.5] |   eval_metrics/mean_weighted_sum_quantile_loss |   eval_metrics/MAE[0.5] |   eval_metrics/RMSE[mean] |   n_datasets |
|:----------------|:----------------|-------------------------:|-----------------------------------------------:|------------------------:|--------------------------:|-------------:|
| ibm-r3-all      | TimeCopilot-IBM |                  1.54998 |                                       0.214378 |                 549.015 |                   2974.68 |           97 |
| ibm-best-all    | TimeCopilot-IBM |                  1.60677 |                                       0.223634 |                 506.304 |                   2766.33 |           97 |
| flowstate-all   | FlowState       |                  1.64117 |                                       0.232145 |                 516.287 |                   2756.45 |           97 |
| patchtst-fm-all | PatchTST-FM     |                  1.67304 |                                       0.228007 |                 548.509 |                   3135.61 |           97 |
| ttm-r3-all      | TTM-R3          |                  1.98854 |                                       0.303424 |                1229.12  |                   8586.04 |           95 |

## Fairness-Aware Ranking

| run_name        | model           |   n_datasets |   wins |   top2 |   mean_rank |   median_rank |
|:----------------|:----------------|-------------:|-------:|-------:|------------:|--------------:|
| ibm-r3-all      | TimeCopilot-IBM |           97 |     26 |     74 |     2       |             2 |
| ibm-best-all    | TimeCopilot-IBM |           97 |     20 |     45 |     2.53608 |             3 |
| ttm-r3-all      | TTM-R3          |           95 |     37 |     45 |     3.0101  |             4 |
| patchtst-fm-all | PatchTST-FM     |           97 |     10 |     20 |     3.56701 |             4 |
| flowstate-all   | FlowState       |           97 |      8 |     16 |     3.72165 |             4 |

## Performance by Horizon

| run_name        | model           |   short |   medium |    long |   overall_mean_rank |
|:----------------|:----------------|--------:|---------:|--------:|--------------------:|
| ibm-r3-all      | TimeCopilot-IBM | 2.21818 |  1.7619  | 1.66667 |             2       |
| ibm-best-all    | TimeCopilot-IBM | 2.14545 |  3.2381  | 2.85714 |             2.53608 |
| ttm-r3-all      | TTM-R3          | 3.89286 |  1.69565 | 2.05    |             3.0101  |
| patchtst-fm-all | PatchTST-FM     | 3.25455 |  4.14286 | 3.80952 |             3.56701 |
| flowstate-all   | FlowState       | 3.43636 |  4.09524 | 4.09524 |             3.72165 |

## Visual Summary

### Dataset Wins

![Dataset wins](ensemble_wins.png)

How to read:
- Taller bars mean the run is the best on more datasets when ranked by `MASE[0.5]`.
- The annotation above each bar shows outright wins and top-2 finishes.
- This plot is good for spotting specialist runs that win often but may still be unstable overall.

### Average Rank by Term

![Average rank by term](ensemble_rank_heatmap.png)

How to read:
- Lower values are better.
- Each cell shows the average per-dataset rank for that run within a horizon group.
- Darker cells indicate stronger relative performance.
- This plot is useful for checking whether a run is strongest on `short`, `medium`, or `long` tasks.

### Primary Metric by Term

![MASE by term](ensemble_primary_metric_by_term.png)

How to read:
- Each group compares runs within `short`, `medium`, and `long` datasets using the actual mean `MASE[0.5]` value.
- Lower bars are better.
- This plot complements the rank heatmap by showing effect size, not just ordering.

## Key Observations

- `ibm-r3-all` is the strongest overall run on the primary benchmark metrics and also the most consistent by average rank.
- `ibm-best-all` is competitive, especially on some absolute-error metrics, but is less balanced across horizon groups.
- `ttm-r3-all` wins many individual datasets, but its average performance is much less stable than the IBM ensembles.
- `ttm-r3-all` is a good example of why the bar plot helps: it looks weak overall, but its medium and long `MASE[0.5]` values are very strong.
- `patchtst-fm-all` and `flowstate-all` provide useful single-model baselines but trail the ensembles on most aggregate views.

## Suggested Discussion Points

- Whether to favor the most consistent run (`ibm-r3-all`) or a more specialized run with stronger niche performance.
- Whether additional ensemble tuning should focus on medium/long horizons.
- Which single-model baselines are still worth retaining for future comparison and ablation studies.

## Appendix

- Detailed per-dataset comparisons are available in `comparison_appendix.csv`.
- Win counts are available in `comparison_win_counts.csv`.
- Rank-by-term details are available in `comparison_avg_rank_by_term.csv`.
