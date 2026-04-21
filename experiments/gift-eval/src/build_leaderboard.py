from pathlib import Path
from typing import Annotated

import pandas as pd
import typer


app = typer.Typer()

DEFAULT_METRICS = [
    "eval_metrics/MASE[0.5]",
    "eval_metrics/mean_weighted_sum_quantile_loss",
    "eval_metrics/MAE[0.5]",
    "eval_metrics/RMSE[mean]",
]


def _read_results(csv_path: Path, run_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()
    df["run_name"] = run_name
    df = df.drop_duplicates(subset=["dataset", "model"], keep="last")
    return df


def _normalize_local_run_name(rel_parent: Path) -> str:
    if str(rel_parent) == ".":
        return "root"
    parts = rel_parent.parts
    # Collapse nested multi-dataset outputs like
    # `ibm-r3-all/m4_weekly/short/all_results.csv` under a single run name.
    if parts and parts[0].endswith("-all"):
        return parts[0]
    return rel_parent.as_posix()


def _discover_local_results(
    results_root: Path,
    run_names: list[str] | None = None,
) -> pd.DataFrame:
    csv_paths = sorted(results_root.glob("**/all_results.csv"))
    if not csv_paths:
        raise ValueError(f"No all_results.csv files found under {results_root}")
    allowed_runs = set(run_names) if run_names is not None else None
    dfs = []
    for csv_path in csv_paths:
        rel_parent = csv_path.parent.relative_to(results_root)
        run_name = _normalize_local_run_name(rel_parent)
        if allowed_runs is not None and run_name not in allowed_runs:
            continue
        dfs.append(_read_results(csv_path, run_name=run_name))
    return pd.concat(dfs, ignore_index=True)


def _load_benchmark_subset(benchmark_root: Path, benchmark_models: list[str]) -> pd.DataFrame:
    dfs = []
    missing = []
    for model in benchmark_models:
        csv_path = benchmark_root / model / "all_results.csv"
        if not csv_path.exists():
            missing.append(model)
            continue
        dfs.append(_read_results(csv_path, run_name=f"benchmark/{model}"))
    if missing:
        raise ValueError(
            f"Could not find benchmark results for: {missing} under {benchmark_root}"
        )
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _prepare_leaderboard(
    local_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    metrics: list[str],
    datasets: list[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat(
        [df for df in [local_df, benchmark_df] if not df.empty],
        ignore_index=True,
    )
    if datasets is None:
        local_datasets = set(local_df["dataset"].unique().tolist())
        benchmark_datasets = (
            set(benchmark_df["dataset"].unique().tolist())
            if not benchmark_df.empty and "dataset" in benchmark_df.columns
            else set()
        )
        datasets = sorted(
            local_datasets.intersection(benchmark_datasets)
            if benchmark_datasets
            else local_datasets
        )
    combined = combined[combined["dataset"].isin(datasets)].copy()
    if combined.empty:
        raise ValueError("No rows matched the selected dataset scope.")

    agg_spec = {metric: "mean" for metric in metrics}
    leaderboard = (
        combined.groupby(["run_name", "model"], as_index=False)
        .agg(agg_spec)
        .sort_values(by=metrics)
        .reset_index(drop=True)
    )
    dataset_counts = (
        combined.groupby(["run_name", "model"], as_index=False)["dataset"]
        .nunique()
        .rename(columns={"dataset": "n_datasets"})
    )
    leaderboard = leaderboard.merge(dataset_counts, on=["run_name", "model"], how="left")

    for metric in metrics:
        rank_col = f"{metric}_rank"
        leaderboard[rank_col] = leaderboard[metric].rank(method="dense")

    per_dataset = (
        combined[["dataset", "run_name", "model", *metrics]]
        .sort_values(["dataset", *metrics])
        .reset_index(drop=True)
    )
    return leaderboard, per_dataset


@app.command()
def build_leaderboard(
    local_results_root: Annotated[
        Path,
        typer.Option(help="Root directory containing local TimeCopilot result folders."),
    ] = Path("./results/timecopilot"),
    local_run: Annotated[
        list[str] | None,
        typer.Option(
            "--local-run",
            help="Local run folder(s) to include. Repeat the flag for multiple runs.",
        ),
    ] = None,
    benchmark_results_root: Annotated[
        Path,
        typer.Option(help="Cloned GIFT-Eval results directory."),
    ] = Path(
        "/Users/fearghalodonncha/Work/DigitalTwin/TSFM_Agent/Development/Benchmarks/gift-eval/results"
    ),
    benchmark_model: Annotated[
        list[str] | None,
        typer.Option(
            "--benchmark-model",
            help="Benchmark model folder(s) to include. Repeat the flag for multiple models.",
        ),
    ] = None,
    dataset: Annotated[
        list[str] | None,
        typer.Option(
            "--dataset",
            help="Limit the leaderboard to specific dataset config(s), e.g. m4_weekly/W/short.",
        ),
    ] = None,
    metric: Annotated[
        list[str] | None,
        typer.Option(
            "--metric",
            help="Metric columns to rank by. Defaults to a compact benchmark set.",
        ),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(help="Optional directory to save leaderboard CSVs."),
    ] = None,
):
    metrics = metric or DEFAULT_METRICS
    local_df = _discover_local_results(local_results_root, run_names=local_run)
    benchmark_df = (
        _load_benchmark_subset(benchmark_results_root, benchmark_model)
        if benchmark_model
        else pd.DataFrame()
    )
    leaderboard, per_dataset = _prepare_leaderboard(
        local_df=local_df,
        benchmark_df=benchmark_df,
        metrics=metrics,
        datasets=dataset,
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        leaderboard.to_csv(output_dir / "leaderboard_summary.csv", index=False)
        per_dataset.to_csv(output_dir / "leaderboard_per_dataset.csv", index=False)

    with pd.option_context("display.max_columns", None, "display.width", 200):
        selected_datasets = dataset or sorted(
            set(local_df["dataset"].unique()).intersection(
                set(benchmark_df["dataset"].unique())
            )
            if not benchmark_df.empty
            else local_df["dataset"].unique().tolist()
        )
        print("\nDatasets used")
        print(selected_datasets)
        print("\nSummary leaderboard")
        print(leaderboard.to_string(index=False))
        print("\nPer-dataset rows")
        print(per_dataset.to_string(index=False))


if __name__ == "__main__":
    app()
