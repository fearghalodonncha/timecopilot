from pathlib import Path
from typing import Annotated

import pandas as pd
import typer


app = typer.Typer()

DEFAULT_PRIMARY_METRIC = "eval_metrics/MASE[0.5]"
DEFAULT_SECONDARY_METRIC = "eval_metrics/mean_weighted_sum_quantile_loss"


def _extract_term(dataset_config: str) -> str:
    return dataset_config.rsplit("/", maxsplit=1)[-1]


def _deduplicate_per_dataset(per_dataset: pd.DataFrame) -> pd.DataFrame:
    return per_dataset.drop_duplicates(subset=["dataset", "run_name"], keep="first").copy()


def _filter_to_common_datasets(per_dataset: pd.DataFrame) -> pd.DataFrame:
    common_datasets = None
    for run_name, group in per_dataset.groupby("run_name"):
        datasets = set(group["dataset"].unique().tolist())
        common_datasets = datasets if common_datasets is None else common_datasets.intersection(datasets)
    if common_datasets is None:
        return per_dataset.copy()
    return per_dataset[per_dataset["dataset"].isin(common_datasets)].copy()


def _build_rank_table(
    per_dataset: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    ranked = per_dataset.copy()
    ranked["term"] = ranked["dataset"].map(_extract_term)
    ranked[f"{metric}_rank"] = ranked.groupby("dataset")[metric].rank(method="dense")
    return ranked


def _build_win_counts(
    ranked: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    rank_col = f"{metric}_rank"
    grouped = ranked.groupby("run_name", as_index=False)
    summary = grouped.agg(
        model=("model", "first"),
        n_datasets=("dataset", "nunique"),
        wins=(rank_col, lambda s: int((s == 1).sum())),
        top2=(rank_col, lambda s: int((s <= 2).sum())),
        mean_rank=(rank_col, "mean"),
        median_rank=(rank_col, "median"),
    )
    return summary.sort_values(["mean_rank", "wins", "top2"], ascending=[True, False, False])


def _build_rank_by_term(
    ranked: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    rank_col = f"{metric}_rank"
    grouped = (
        ranked.groupby(["run_name", "model", "term"], as_index=False)[rank_col]
        .mean()
        .rename(columns={rank_col: "mean_rank"})
    )
    pivoted = grouped.pivot(
        index=["run_name", "model"],
        columns="term",
        values="mean_rank",
    ).reset_index()
    overall = (
        ranked.groupby(["run_name", "model"], as_index=False)[rank_col]
        .mean()
        .rename(columns={rank_col: "overall_mean_rank"})
    )
    result = pivoted.merge(overall, on=["run_name", "model"], how="left")
    ordered_cols = ["run_name", "model", "short", "medium", "long", "overall_mean_rank"]
    existing_cols = [col for col in ordered_cols if col in result.columns]
    return result[existing_cols].sort_values("overall_mean_rank")


def _build_metric_by_term(
    per_dataset: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    metric_by_term = per_dataset.copy()
    metric_by_term["term"] = metric_by_term["dataset"].map(_extract_term)
    grouped = (
        metric_by_term.groupby(["run_name", "model", "term"], as_index=False)[metric]
        .mean()
    )
    pivoted = grouped.pivot(
        index=["run_name", "model"],
        columns="term",
        values=metric,
    ).reset_index()
    overall = (
        metric_by_term.groupby(["run_name", "model"], as_index=False)[metric]
        .mean()
        .rename(columns={metric: f"overall_{metric}"})
    )
    result = pivoted.merge(overall, on=["run_name", "model"], how="left")
    overall_col = f"overall_{metric}"
    ordered_cols = ["run_name", "model", "short", "medium", "long", overall_col]
    existing_cols = [col for col in ordered_cols if col in result.columns]
    return result[existing_cols].sort_values(overall_col)


def _build_appendix(
    ranked_primary: pd.DataFrame,
    ranked_secondary: pd.DataFrame,
    primary_metric: str,
    secondary_metric: str,
) -> pd.DataFrame:
    appendix = ranked_primary[
        ["dataset", "run_name", "model", primary_metric, f"{primary_metric}_rank"]
    ].merge(
        ranked_secondary[
            ["dataset", "run_name", secondary_metric, f"{secondary_metric}_rank"]
        ],
        on=["dataset", "run_name"],
        how="left",
    )
    return appendix.sort_values(
        ["dataset", f"{primary_metric}_rank", f"{secondary_metric}_rank", "run_name"]
    ).reset_index(drop=True)


def _build_markdown_summary(
    leaderboard_summary: pd.DataFrame,
    win_counts: pd.DataFrame,
    primary_metric: str,
) -> str:
    best_mean = leaderboard_summary.sort_values(primary_metric).iloc[0]
    best_rank = win_counts.sort_values("mean_rank").iloc[0]
    most_wins = win_counts.sort_values("wins", ascending=False).iloc[0]

    lines = [
        "# Comparison Summary",
        "",
        f"- Best mean `{primary_metric}`: `{best_mean['run_name']}` ({best_mean[primary_metric]:.4f})",
        f"- Best average dataset rank on `{primary_metric}`: `{best_rank['run_name']}` ({best_rank['mean_rank']:.3f})",
        f"- Most dataset wins on `{primary_metric}`: `{most_wins['run_name']}` ({int(most_wins['wins'])} wins)",
        "",
        "## Included Runs",
    ]
    for _, row in leaderboard_summary.sort_values(primary_metric).iterrows():
        lines.append(
            f"- `{row['run_name']}` / `{row['model']}`: "
            f"{primary_metric}={row[primary_metric]:.4f}, "
            f"datasets={int(row['n_datasets'])}"
        )
    return "\n".join(lines) + "\n"


@app.command()
def build_comparison_views(
    leaderboard_dir: Annotated[
        Path,
        typer.Option(help="Directory containing leaderboard_summary.csv and leaderboard_per_dataset.csv."),
    ] = Path("./results/timecopilot/hpc/leaderboard"),
    output_dir: Annotated[
        Path | None,
        typer.Option(help="Optional output directory. Defaults to the leaderboard directory."),
    ] = None,
    run_name: Annotated[
        list[str] | None,
        typer.Option(help="Optional repeated run-name filter for a clean comparison set."),
    ] = None,
    primary_metric: Annotated[
        str,
        typer.Option(help="Primary metric for fair ranking views."),
    ] = DEFAULT_PRIMARY_METRIC,
    secondary_metric: Annotated[
        str,
        typer.Option(help="Secondary metric for appendix/ranking support."),
    ] = DEFAULT_SECONDARY_METRIC,
):
    summary_path = leaderboard_dir / "leaderboard_summary.csv"
    per_dataset_path = leaderboard_dir / "leaderboard_per_dataset.csv"
    leaderboard_summary = pd.read_csv(summary_path)
    per_dataset = pd.read_csv(per_dataset_path)
    if run_name:
        leaderboard_summary = leaderboard_summary[leaderboard_summary["run_name"].isin(run_name)].copy()
        per_dataset = per_dataset[per_dataset["run_name"].isin(run_name)].copy()
    per_dataset = _deduplicate_per_dataset(per_dataset)
    fair_per_dataset = _filter_to_common_datasets(per_dataset)

    out_dir = output_dir or leaderboard_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ranked_primary = _build_rank_table(fair_per_dataset, primary_metric)
    ranked_secondary = _build_rank_table(fair_per_dataset, secondary_metric)

    win_counts = _build_win_counts(ranked_primary, primary_metric)
    rank_by_term = _build_rank_by_term(ranked_primary, primary_metric)
    primary_metric_by_term = _build_metric_by_term(fair_per_dataset, primary_metric)
    appendix = _build_appendix(
        ranked_primary=ranked_primary,
        ranked_secondary=ranked_secondary,
        primary_metric=primary_metric,
        secondary_metric=secondary_metric,
    )

    leaderboard_summary.to_csv(out_dir / "comparison_summary.csv", index=False)
    win_counts.to_csv(out_dir / "comparison_win_counts.csv", index=False)
    rank_by_term.to_csv(out_dir / "comparison_avg_rank_by_term.csv", index=False)
    primary_metric_by_term.to_csv(out_dir / "comparison_primary_metric_by_term.csv", index=False)
    appendix.to_csv(out_dir / "comparison_appendix.csv", index=False)
    (out_dir / "comparison_summary.md").write_text(
        _build_markdown_summary(
            leaderboard_summary=leaderboard_summary,
            win_counts=win_counts,
            primary_metric=primary_metric,
        )
    )

    with pd.option_context("display.max_columns", None, "display.width", 200):
        print("\nSummary leaderboard")
        print(leaderboard_summary.to_string(index=False))
        print("\nWin counts")
        print(win_counts.to_string(index=False))
        print("\nAverage rank by term")
        print(rank_by_term.to_string(index=False))


if __name__ == "__main__":
    app()
