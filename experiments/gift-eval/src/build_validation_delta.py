from pathlib import Path
from typing import Annotated

import pandas as pd
import typer


app = typer.Typer()

DEFAULT_METRIC = "eval_metrics/MASE[0.5]"


def _extract_term(dataset_config: str) -> str:
    return dataset_config.rsplit("/", maxsplit=1)[-1]


def _build_delta_table(
    per_dataset: pd.DataFrame,
    candidate_run: str,
    benchmark_run: str,
    metric: str,
) -> pd.DataFrame:
    cols = ["dataset", "run_name", "model", metric]
    subset = per_dataset[cols].copy()

    candidate = (
        subset[subset["run_name"] == candidate_run]
        .rename(
            columns={
                "model": "candidate_model",
                metric: "candidate_metric",
            }
        )
        .drop(columns=["run_name"])
    )
    benchmark = (
        subset[subset["run_name"] == benchmark_run]
        .rename(
            columns={
                "model": "benchmark_model",
                metric: "benchmark_metric",
            }
        )
        .drop(columns=["run_name"])
    )

    merged = candidate.merge(benchmark, on="dataset", how="inner")
    merged["term"] = merged["dataset"].map(_extract_term)
    merged["delta"] = merged["candidate_metric"] - merged["benchmark_metric"]
    merged["abs_delta"] = merged["delta"].abs()
    merged["candidate_better"] = merged["delta"] < 0
    return merged.sort_values("delta")


def _build_delta_summary(
    delta_df: pd.DataFrame,
) -> pd.DataFrame:
    summary = {
        "n_datasets_compared": int(len(delta_df)),
        "candidate_wins": int(delta_df["candidate_better"].sum()),
        "benchmark_wins": int((~delta_df["candidate_better"]).sum()),
        "mean_delta": float(delta_df["delta"].mean()),
        "median_delta": float(delta_df["delta"].median()),
        "mean_abs_delta": float(delta_df["abs_delta"].mean()),
        "max_improvement": float(delta_df["delta"].min()),
        "max_regression": float(delta_df["delta"].max()),
    }
    return pd.DataFrame([summary])


def _build_delta_by_term(
    delta_df: pd.DataFrame,
) -> pd.DataFrame:
    grouped = (
        delta_df.groupby("term", as_index=False)
        .agg(
            n_datasets=("dataset", "count"),
            candidate_wins=("candidate_better", "sum"),
            mean_delta=("delta", "mean"),
            median_delta=("delta", "median"),
            mean_abs_delta=("abs_delta", "mean"),
        )
        .sort_values("mean_delta")
    )
    grouped["benchmark_wins"] = grouped["n_datasets"] - grouped["candidate_wins"]
    return grouped


def _build_markdown_summary(
    summary: pd.DataFrame,
    by_term: pd.DataFrame,
    candidate_run: str,
    benchmark_run: str,
    metric: str,
) -> str:
    row = summary.iloc[0]
    lines = [
        "# Validation Delta Summary",
        "",
        f"Comparing `{candidate_run}` against `{benchmark_run}` on `{metric}`.",
        "",
        f"- Datasets compared: {int(row['n_datasets_compared'])}",
        f"- Candidate wins: {int(row['candidate_wins'])}",
        f"- Benchmark wins: {int(row['benchmark_wins'])}",
        f"- Mean delta (`candidate - benchmark`): {row['mean_delta']:.6f}",
        f"- Median delta: {row['median_delta']:.6f}",
        f"- Mean absolute delta: {row['mean_abs_delta']:.6f}",
        f"- Best candidate improvement: {row['max_improvement']:.6f}",
        f"- Worst candidate regression: {row['max_regression']:.6f}",
        "",
        "## By Term",
    ]
    for _, term_row in by_term.iterrows():
        lines.append(
            f"- `{term_row['term']}`: "
            f"n={int(term_row['n_datasets'])}, "
            f"candidate_wins={int(term_row['candidate_wins'])}, "
            f"benchmark_wins={int(term_row['benchmark_wins'])}, "
            f"mean_delta={term_row['mean_delta']:.6f}"
        )
    return "\n".join(lines) + "\n"


@app.command()
def build_validation_delta(
    validation_dir: Annotated[
        Path,
        typer.Option(help="Directory containing leaderboard_per_dataset.csv from a validation run."),
    ] = Path("./results/timecopilot/hpc/patchtst_validation"),
    candidate_run: Annotated[
        str,
        typer.Option(help="Run name to evaluate, e.g. patchtst-fm-all."),
    ] = "patchtst-fm-all",
    benchmark_run: Annotated[
        str,
        typer.Option(help="Benchmark run name to compare against."),
    ] = "benchmark/PatchTST-FM-r1",
    metric: Annotated[
        str,
        typer.Option(help="Metric column to compare."),
    ] = DEFAULT_METRIC,
    output_dir: Annotated[
        Path | None,
        typer.Option(help="Optional output directory. Defaults to validation_dir."),
    ] = None,
):
    per_dataset_path = validation_dir / "leaderboard_per_dataset.csv"
    per_dataset = pd.read_csv(per_dataset_path)
    delta_df = _build_delta_table(
        per_dataset=per_dataset,
        candidate_run=candidate_run,
        benchmark_run=benchmark_run,
        metric=metric,
    )
    summary = _build_delta_summary(delta_df)
    by_term = _build_delta_by_term(delta_df)
    best = delta_df.nsmallest(15, "delta")
    worst = delta_df.nlargest(15, "delta")

    out_dir = output_dir or validation_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    delta_df.to_csv(out_dir / "validation_delta_full.csv", index=False)
    summary.to_csv(out_dir / "validation_delta_summary.csv", index=False)
    by_term.to_csv(out_dir / "validation_delta_by_term.csv", index=False)
    best.to_csv(out_dir / "validation_delta_best_cases.csv", index=False)
    worst.to_csv(out_dir / "validation_delta_worst_cases.csv", index=False)
    (out_dir / "validation_delta_summary.md").write_text(
        _build_markdown_summary(
            summary=summary,
            by_term=by_term,
            candidate_run=candidate_run,
            benchmark_run=benchmark_run,
            metric=metric,
        )
    )

    with pd.option_context("display.max_columns", None, "display.width", 200):
        print("\nDelta summary")
        print(summary.to_string(index=False))
        print("\nDelta by term")
        print(by_term.to_string(index=False))
        print("\nBest candidate cases")
        print(best[["dataset", "candidate_metric", "benchmark_metric", "delta"]].to_string(index=False))
        print("\nWorst candidate cases")
        print(worst[["dataset", "candidate_metric", "benchmark_metric", "delta"]].to_string(index=False))


if __name__ == "__main__":
    app()
