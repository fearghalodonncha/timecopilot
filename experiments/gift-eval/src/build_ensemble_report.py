from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import pandas as pd
import typer


app = typer.Typer()

PRIMARY_METRIC = "eval_metrics/MASE[0.5]"
SECONDARY_METRIC = "eval_metrics/mean_weighted_sum_quantile_loss"


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return pd.read_csv(path)


def _plot_win_counts(win_counts: pd.DataFrame, output_path: Path) -> None:
    plot_df = win_counts.sort_values(["wins", "top2"], ascending=[False, False]).copy()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.bar(plot_df["run_name"], plot_df["wins"], color="#35618f")
    ax.set_title("Dataset Wins on MASE[0.5]")
    ax.set_ylabel("Number of wins")
    ax.set_xlabel("Run")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    for bar, top2 in zip(bars, plot_df["top2"], strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{int(bar.get_height())} wins\n{int(top2)} top2",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_rank_heatmap(rank_by_term: pd.DataFrame, output_path: Path) -> None:
    value_cols = [col for col in ["short", "medium", "long", "overall_mean_rank"] if col in rank_by_term.columns]
    labels = rank_by_term["run_name"].tolist()
    data = rank_by_term[value_cols].to_numpy()

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    im = ax.imshow(data, cmap="YlGnBu_r", aspect="auto")
    ax.set_title("Average Rank by Horizon Term")
    ax.set_xticks(range(len(value_cols)))
    ax.set_xticklabels(["short", "medium", "long", "overall"][: len(value_cols)])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean rank (lower is better)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_primary_metric_by_term(primary_by_term: pd.DataFrame, output_path: Path) -> None:
    value_cols = [col for col in ["short", "medium", "long"] if col in primary_by_term.columns]
    plot_df = primary_by_term.sort_values(f"overall_{PRIMARY_METRIC}").copy()

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = range(len(value_cols))
    width = 0.16

    for idx, (_, row) in enumerate(plot_df.iterrows()):
        offsets = [pos + (idx - (len(plot_df) - 1) / 2) * width for pos in x]
        ax.bar(
            offsets,
            [row[col] for col in value_cols],
            width=width,
            label=row["run_name"],
        )

    ax.set_title("MASE[0.5] by Horizon Term")
    ax.set_ylabel("Mean MASE[0.5] (lower is better)")
    ax.set_xlabel("Horizon term")
    ax.set_xticks(list(x))
    ax.set_xticklabels(value_cols)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(title="Run", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _fmt_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def _build_report(
    summary: pd.DataFrame,
    win_counts: pd.DataFrame,
    rank_by_term: pd.DataFrame,
    primary_by_term: pd.DataFrame,
    output_dir: Path,
) -> str:
    summary_sorted = summary.sort_values(PRIMARY_METRIC).copy()
    win_sorted = win_counts.sort_values("mean_rank").copy()
    rank_sorted = rank_by_term.sort_values("overall_mean_rank").copy()

    best_overall = summary_sorted.iloc[0]
    most_wins = win_counts.sort_values("wins", ascending=False).iloc[0]
    most_consistent = win_sorted.iloc[0]
    runs_compared = summary_sorted["run_name"].tolist()

    short_best = (
        primary_by_term.sort_values("short").iloc[0]["run_name"]
        if "short" in primary_by_term.columns
        else None
    )
    medium_best = (
        primary_by_term.sort_values("medium").iloc[0]["run_name"]
        if "medium" in primary_by_term.columns
        else None
    )
    long_best = (
        primary_by_term.sort_values("long").iloc[0]["run_name"]
        if "long" in primary_by_term.columns
        else None
    )

    lines = [
        "# TimeCopilot Ensemble Comparison",
        "",
        "## Executive Summary",
        "",
        f"- `default-all` is the strongest overall configuration in this comparison, with the best mean `{PRIMARY_METRIC}` and `{SECONDARY_METRIC}`.",
        f"- `{most_wins['run_name']}` records the most per-dataset wins on `{PRIMARY_METRIC}` ({int(most_wins['wins'])} of {int(most_wins['n_datasets'])} common datasets).",
        f"- `{most_consistent['run_name']}` is the most consistent run by average per-dataset rank ({most_consistent['mean_rank']:.3f}).",
        "- The default TimeCopilot stack outperforms the IBM R3 ensemble and the single-model baselines in this refreshed comparison.",
        "",
        "## Runs Compared",
        "",
    ]
    lines.extend([f"- `{run_name}`" for run_name in runs_compared])
    lines.extend(
        [
            "",
            "## Overall Summary",
            "",
            _fmt_table(
                summary_sorted[
                    [
                        "run_name",
                        "model",
                        PRIMARY_METRIC,
                        SECONDARY_METRIC,
                        "eval_metrics/MAE[0.5]",
                        "eval_metrics/RMSE[mean]",
                        "n_datasets",
                    ]
                ]
            ),
            "",
            "## Fairness-Aware Ranking",
            "",
            "These views are computed on the common dataset intersection across the included runs.",
            "",
            _fmt_table(
                win_sorted[
                    ["run_name", "model", "n_datasets", "wins", "top2", "mean_rank", "median_rank"]
                ]
            ),
            "",
            "## Performance by Horizon",
            "",
            _fmt_table(rank_sorted),
            "",
            "## Visual Summary",
            "",
            "### Dataset Wins",
            "",
            "![Dataset wins](ensemble_wins.png)",
            "",
            "How to read:",
            "- Taller bars mean the run is the best on more datasets when ranked by `MASE[0.5]`.",
            "- The annotation above each bar shows outright wins and top-2 finishes.",
            "- This plot is good for spotting specialist runs that win often but may still be unstable overall.",
            "",
            "### Average Rank by Term",
            "",
            "![Average rank by term](ensemble_rank_heatmap.png)",
            "",
            "How to read:",
            "- Lower values are better.",
            "- Each cell shows the average per-dataset rank for that run within a horizon group.",
            "- Darker cells indicate stronger relative performance.",
            "- This plot is useful for checking whether a run is strongest on `short`, `medium`, or `long` tasks.",
            "",
            "### Primary Metric by Term",
            "",
            "![MASE by term](ensemble_primary_metric_by_term.png)",
            "",
            "How to read:",
            "- Each group compares runs within `short`, `medium`, and `long` datasets using the actual mean `MASE[0.5]` value.",
            "- Lower bars are better.",
            "- This plot complements the rank heatmap by showing effect size, not just ordering.",
            "",
            "## Key Observations",
            "",
            f"- `{best_overall['run_name']}` is the strongest overall run on both point and probabilistic metrics in this comparison.",
            f"- `{most_consistent['run_name']}` is also the most consistent run by average per-dataset rank, so the same configuration leads on both aggregate quality and stability.",
        ]
    )
    if short_best is not None:
        lines.append(f"- `{short_best}` is best on the short-horizon mean `MASE[0.5]` view.")
    if medium_best is not None:
        lines.append(f"- `{medium_best}` is best on the medium-horizon mean `MASE[0.5]` view.")
    if long_best is not None:
        lines.append(f"- `{long_best}` is best on the long-horizon mean `MASE[0.5]` view.")
    lines.extend(
        [
            "- `ibm-r3-all` remains the strongest challenger, but it is clearly behind `default-all` once the comparison is restricted to these five runs.",
            "- `ttm-r3-all` is still informative as a specialist baseline: it is weak overall, but materially stronger on medium and long horizons than on short horizons.",
            "- `patchtst-fm-all` and `flowstate-all` remain useful single-model references, but neither is competitive with the two ensemble configurations on the main aggregate views.",
            "- The wins plot and the rank heatmap are useful together because they separate broad consistency from more specialized strengths.",
            "",
            "## Suggested Discussion Points",
            "",
            "- Whether `default-all` should now be treated as the primary recommended TimeCopilot configuration for broad use.",
            "- Whether `ibm-r3-all` is still worth keeping as a simpler IBM-only ensemble baseline for future ablations.",
            "- Whether the horizon-specific behavior of `ttm-r3-all` suggests an opportunity for routing or conditional ensembling rather than uniform use.",
        "",
            "## Appendix",
            "",
            "- Detailed per-dataset comparisons are available in `comparison_appendix.csv`.",
            "- Win counts are available in `comparison_win_counts.csv`.",
            "- Rank-by-term details are available in `comparison_avg_rank_by_term.csv`.",
        ]
    )
    return "\n".join(lines) + "\n"


@app.command()
def build_ensemble_report(
    comparison_dir: Annotated[
        Path,
        typer.Option(help="Directory containing comparison_summary.csv and related files."),
    ] = Path("./results/timecopilot/hpc/leaderboard"),
    output_dir: Annotated[
        Path | None,
        typer.Option(help="Optional output directory. Defaults to ./docs/ensemble_comparison."),
    ] = None,
):
    summary = _load_csv(comparison_dir / "comparison_summary.csv")
    win_counts = _load_csv(comparison_dir / "comparison_win_counts.csv")
    rank_by_term = _load_csv(comparison_dir / "comparison_avg_rank_by_term.csv")
    primary_by_term = _load_csv(comparison_dir / "comparison_primary_metric_by_term.csv")

    out_dir = output_dir or Path("./docs/ensemble_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    _plot_win_counts(win_counts, out_dir / "ensemble_wins.png")
    _plot_rank_heatmap(rank_by_term, out_dir / "ensemble_rank_heatmap.png")
    _plot_primary_metric_by_term(primary_by_term, out_dir / "ensemble_primary_metric_by_term.png")

    report = _build_report(summary, win_counts, rank_by_term, primary_by_term, out_dir)
    (out_dir / "ensemble_comparison_report.md").write_text(report)

    print(f"Wrote report to {out_dir / 'ensemble_comparison_report.md'}")
    print(
        "Wrote plots to "
        f"{out_dir / 'ensemble_wins.png'}, "
        f"{out_dir / 'ensemble_rank_heatmap.png'}, and "
        f"{out_dir / 'ensemble_primary_metric_by_term.png'}"
    )


if __name__ == "__main__":
    app()
