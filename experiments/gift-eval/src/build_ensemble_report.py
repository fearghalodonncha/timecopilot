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

    lines = [
        "# TimeCopilot Ensemble Comparison",
        "",
        "## Executive Summary",
        "",
        f"- Best overall mean `{PRIMARY_METRIC}`: `{best_overall['run_name']}` ({best_overall[PRIMARY_METRIC]:.4f}).",
        f"- Best overall mean `{SECONDARY_METRIC}`: `{summary.sort_values(SECONDARY_METRIC).iloc[0]['run_name']}` ({summary.sort_values(SECONDARY_METRIC).iloc[0][SECONDARY_METRIC]:.4f}).",
        f"- Most dataset wins on `{PRIMARY_METRIC}`: `{most_wins['run_name']}` ({int(most_wins['wins'])} wins).",
        f"- Most consistent average rank: `{most_consistent['run_name']}` ({most_consistent['mean_rank']:.3f}).",
        "- The IBM ensembles lead overall, while single-model runs are more specialized and less consistent.",
        "",
        "## Runs Compared",
        "",
        "- `ibm-best-all`",
        "- `ibm-r3-all`",
        "- `patchtst-fm-all`",
        "- `flowstate-all`",
        "- `ttm-r3-all`",
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
        "- `ibm-r3-all` is the strongest overall run on the primary benchmark metrics and also the most consistent by average rank.",
        "- `ibm-best-all` is competitive, especially on some absolute-error metrics, but is less balanced across horizon groups.",
        "- `ttm-r3-all` wins many individual datasets, but its average performance is much less stable than the IBM ensembles.",
        "- `ttm-r3-all` is a good example of why the bar plot helps: it looks weak overall, but its medium and long `MASE[0.5]` values are very strong.",
        "- `patchtst-fm-all` and `flowstate-all` provide useful single-model baselines but trail the ensembles on most aggregate views.",
        "",
        "## Suggested Discussion Points",
        "",
        "- Whether to favor the most consistent run (`ibm-r3-all`) or a more specialized run with stronger niche performance.",
        "- Whether additional ensemble tuning should focus on medium/long horizons.",
        "- Which single-model baselines are still worth retaining for future comparison and ablation studies.",
        "",
        "## Appendix",
        "",
        "- Detailed per-dataset comparisons are available in `comparison_appendix.csv`.",
        "- Win counts are available in `comparison_win_counts.csv`.",
        "- Rank-by-term details are available in `comparison_avg_rank_by_term.csv`.",
    ]
    return "\n".join(lines) + "\n"


@app.command()
def build_ensemble_report(
    comparison_dir: Annotated[
        Path,
        typer.Option(help="Directory containing comparison_summary.csv and related files."),
    ] = Path("./results/timecopilot/hpc/leaderboard"),
    output_dir: Annotated[
        Path | None,
        typer.Option(help="Optional output directory. Defaults to comparison_dir."),
    ] = None,
):
    summary = _load_csv(comparison_dir / "comparison_summary.csv")
    win_counts = _load_csv(comparison_dir / "comparison_win_counts.csv")
    rank_by_term = _load_csv(comparison_dir / "comparison_avg_rank_by_term.csv")
    primary_by_term = _load_csv(comparison_dir / "comparison_primary_metric_by_term.csv")

    out_dir = output_dir or comparison_dir
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
