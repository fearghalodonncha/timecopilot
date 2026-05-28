"""Run Planning Analytics forecasts and write comparison artifacts."""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from planning_analytics_loader import load_all_long, mark_leading_zeros_as_missing


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = EXPERIMENT_DIR / "PA Anonymized Dataset"
DEFAULT_DATA_PATH = EXPERIMENT_DIR / "data" / "planning_analytics_long.parquet"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "results" / "timecopilot"
TTM_R3_36_12_PATH = DATASET_DIR / "36-12-combined-v3-minus-m4"
PATCHTST_CONTEXT_LENGTH = 48
DEFAULT_FLOWSTATE_SCALE_FACTORS = [2.0]

MANUAL_MODELS = {"seasonal-naive", "last-value", "mean-12"}
STATS_MODELS = {"auto-ets", "historic-average", "stats-seasonal-naive"}
FORECASTER_MODELS = {
    "chronos",
    "flowstate",
    "ibm-ensemble",
    "ibm-granite-tsfm",
    "ibm-research-tsfm",
    "patchtst-fm",
    "timesfm",
    "ttm-r3",
    "timecopilot",
    "timecopilot+ibm",
    "timecopilot-default",
}
MODEL_CHOICES = sorted(MANUAL_MODELS | STATS_MODELS | FORECASTER_MODELS)
TRANSFORM_CHOICES = ["none", "abs-scale", "neg-abs-scale", "pos-neg-components"]
DEFAULT_MODELS = ["auto-ets", "timecopilot-default"]


@dataclass(frozen=True)
class RunConfig:
    data_path: str
    output_dir: str
    country: str | None
    granularity: str | None
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    target_column: str
    transform: str
    models: list[str]
    flowstate_scale_factors: list[float]
    min_history: int
    series_selection: str | None
    cohort_label: str | None
    continue_on_error: bool


def _scale_factor_slug(scale_factor: float) -> str:
    return f"{scale_factor:g}".replace("-", "m").replace(".", "p")


def _load_model_table(data_path: Path, target_column: str) -> pd.DataFrame:
    if data_path.exists():
        logging.info("Loading exported model table from %s", data_path)
        frame = pd.read_parquet(data_path)
    else:
        logging.info("No parquet found at %s; loading workbooks from %s", data_path, DATASET_DIR)
        frame = mark_leading_zeros_as_missing(load_all_long(DATASET_DIR))
        frame = frame.rename(columns={"value_leading_zero_na": "target"})
        data_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(data_path, index=False)
        logging.info("Wrote exported model table to %s", data_path)

    if target_column not in frame.columns:
        if target_column == "target" and "value_leading_zero_na" in frame.columns:
            frame = frame.rename(columns={"value_leading_zero_na": "target"})
        elif target_column == "target" and "value" in frame.columns:
            frame = mark_leading_zeros_as_missing(frame).rename(
                columns={"value_leading_zero_na": "target"}
            )
        else:
            raise ValueError(f"Target column {target_column!r} not found.")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame[target_column] = pd.to_numeric(frame[target_column], errors="coerce")
    return frame


def _country_slug(country: str | None) -> str | None:
    return None if country is None else country.replace(" ", "_")


def _read_series_selection(path: Path, *, country: str | None, granularity: str | None) -> pd.DataFrame:
    if path.is_dir():
        if country is None or granularity is None:
            raise ValueError("Directory --series-selection requires --country and --granularity.")
        candidates = sorted(path.glob(f"{_country_slug(country)}_{granularity}*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No selection CSV in {path} for {country=} {granularity=}")
        frames = [pd.read_csv(candidate) for candidate in candidates]
    else:
        frames = [pd.read_csv(path)]
    selection = pd.concat(frames, ignore_index=True).rename(
        columns={
            "BUSINESS_UNIT": "business_unit",
            "BUDGETARY_NATURES": "budgetary_nature",
            "PL_COST": "pl_cost",
            "country_name": "country",
            "item_id": "series_id",
        }
    )
    if "country" in selection.columns:
        selection["country"] = selection["country"].astype(str).str.replace("_", " ", regex=False)
    return selection


def _apply_series_selection(
    frame: pd.DataFrame,
    *,
    selection_path: str | None,
    country: str | None,
    granularity: str | None,
) -> pd.DataFrame:
    if selection_path is None:
        return frame
    selection = _read_series_selection(Path(selection_path), country=country, granularity=granularity)
    join_columns = [
        column
        for column in ["country", "granularity", "business_unit", "budgetary_nature", "pl_cost"]
        if column in selection.columns and column in frame.columns
    ]
    if "series_id" in selection.columns:
        candidate_ids = set(selection["series_id"].dropna().astype(str))
        frame_ids = set(frame["series_id"].dropna().astype(str))
        if candidate_ids & frame_ids:
            join_columns = ["series_id"]
    if join_columns != ["series_id"] and not all(
        column in join_columns for column in ["business_unit", "budgetary_nature", "pl_cost"]
    ):
        raise ValueError("Selection must contain series_id or business_unit/budgetary_nature/pl_cost.")
    result = frame.merge(selection[join_columns].drop_duplicates(), on=join_columns, how="inner")
    logging.info(
        "Applied series selection %s on %s: %s series retained",
        selection_path,
        join_columns,
        result["series_id"].nunique(),
    )
    if result.empty:
        raise ValueError(f"No rows matched --series-selection {selection_path}")
    return result


def _filter_panel(
    frame: pd.DataFrame,
    *,
    country: str | None,
    granularity: str | None,
    target_column: str,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    min_history: int,
) -> pd.DataFrame:
    filtered = frame.copy()
    if country is not None:
        filtered = filtered[filtered["country"] == country]
    if granularity is not None:
        filtered = filtered[filtered["granularity"] == granularity]
    window = filtered[(filtered["date"] >= train_start) & (filtered["date"] <= test_end)]
    train = window[(window["date"] >= train_start) & (window["date"] <= train_end)]
    test = window[(window["date"] >= test_start) & (window["date"] <= test_end)]
    train_counts = train.groupby("series_id")[target_column].apply(lambda s: int(s.notna().sum()))
    has_history = set(train_counts[train_counts >= min_history].index)
    has_test = set(test.loc[test[target_column].notna(), "series_id"].unique())
    eligible = has_history & has_test
    result = window[window["series_id"].isin(eligible)].copy()
    if result.empty:
        raise ValueError("No eligible series after filtering.")
    logging.info(
        "Filtered panel: %s rows, %s series, countries=%s, granularities=%s",
        len(result),
        result["series_id"].nunique(),
        sorted(result["country"].unique()),
        sorted(result["granularity"].unique()),
    )
    return result


def _to_forecaster_train(
    panel: pd.DataFrame,
    *,
    target_column: str,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
) -> pd.DataFrame:
    train = panel[(panel["date"] >= train_start) & (panel["date"] <= train_end)].copy()
    return train.rename(columns={"series_id": "unique_id", "date": "ds", target_column: "y"})[
        ["unique_id", "ds", "y"]
    ].dropna(subset=["y"])


def _actuals(
    panel: pd.DataFrame,
    *,
    target_column: str,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> pd.DataFrame:
    actual = panel[(panel["date"] >= test_start) & (panel["date"] <= test_end)].copy()
    return actual.rename(columns={target_column: "y"})[
        ["series_id", "country", "granularity", "business_unit", "budgetary_nature", "pl_cost", "date", "y"]
    ].dropna(subset=["y"])


def _manual_predictions(train_df: pd.DataFrame, *, model_names: list[str], test_dates: pd.DatetimeIndex) -> pd.DataFrame:
    requested = [name for name in model_names if name in MANUAL_MODELS]
    if not requested:
        return pd.DataFrame()
    records = []
    train = train_df.rename(columns={"unique_id": "series_id", "ds": "date"}).sort_values(["series_id", "date"])
    for series_id, group in train.groupby("series_id", sort=False):
        history = group.set_index("date")["y"].dropna()
        if history.empty:
            continue
        last_value = float(history.iloc[-1])
        mean_12 = float(history.tail(12).mean())
        for ds in test_dates:
            if "seasonal-naive" in requested:
                previous_year = ds - pd.DateOffset(years=1)
                if previous_year in history.index:
                    records.append({"series_id": series_id, "date": ds, "model": "seasonal-naive", "yhat": float(history.loc[previous_year])})
            if "last-value" in requested:
                records.append({"series_id": series_id, "date": ds, "model": "last-value", "yhat": last_value})
            if "mean-12" in requested:
                records.append({"series_id": series_id, "date": ds, "model": "mean-12", "yhat": mean_12})
    return pd.DataFrame.from_records(records)


def _statsforecast_predictions(train_df: pd.DataFrame, *, model_names: list[str], h: int, freq: str) -> pd.DataFrame:
    requested = [name for name in model_names if name in STATS_MODELS]
    if not requested:
        return pd.DataFrame()
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS, HistoricAverage, SeasonalNaive

    registry = {
        "auto-ets": AutoETS(season_length=12, alias="auto-ets"),
        "historic-average": HistoricAverage(alias="historic-average"),
        "stats-seasonal-naive": SeasonalNaive(season_length=12, alias="stats-seasonal-naive"),
    }
    start = time.monotonic()
    forecast = StatsForecast(models=[registry[name] for name in requested], freq=freq, n_jobs=1).forecast(df=train_df, h=h)
    elapsed = time.monotonic() - start
    frames = []
    for name in requested:
        frame = forecast[["unique_id", "ds", name]].rename(columns={"unique_id": "series_id", "ds": "date", name: "yhat"})
        frame["model"] = name
        frame["inference_time_s"] = elapsed
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _build_forecaster_model(name: str, *, flowstate_scale_factor: float = 2.0):
    def _chronos(alias: str = "chronos"):
        from timecopilot.models.foundation.chronos import Chronos
        return Chronos(repo_id="amazon/chronos-2", batch_size=256, alias=alias)

    def _timesfm(alias: str = "timesfm"):
        from timecopilot.models.foundation.timesfm import TimesFM
        return TimesFM(repo_id="google/timesfm-2.5-200m-pytorch", batch_size=256, alias=alias)

    def _patch(repo_id: str, alias: str):
        from timecopilot.models.foundation.patchtst_fm import PatchTSTFM
        return PatchTSTFM(repo_id=repo_id, context_length=PATCHTST_CONTEXT_LENGTH, batch_size=16, gift_eval_compat=True, alias=alias)

    def _flow(repo_id: str, alias: str):
        from timecopilot.models.foundation.flowstate import FlowState
        return FlowState(repo_id=repo_id, revision="r1.1", scale_factor=flowstate_scale_factor, context_length=None, batch_size=16, domain="finance", no_daily=True, gift_eval_compat=True, alias=alias)

    def _ttm(alias: str = "ttm-r3-36-12"):
        from timecopilot.models.foundation.ttm_r3 import TTMR3
        return TTMR3(
            repo_id=str(TTM_R3_36_12_PATH),
            context_length=36,
            batch_size=256,
            alias=alias,
            model_revision="",
            use_lite=True,
            rolling_norm=True,
        )

    if name == "chronos":
        return _chronos()
    if name == "timesfm":
        return _timesfm()
    if name == "patchtst-fm":
        return _patch("ibm-research/patchtst-fm-r1", "patchtst-fm")
    if name == "flowstate":
        return _flow("ibm-research/flowstate", f"flowstate-sf-{_scale_factor_slug(flowstate_scale_factor)}")
    if name == "ttm-r3":
        return _ttm("ttm-r3")
    if name == "timecopilot":
        name = "timecopilot-default"

    from timecopilot.models.ensembles.median import MedianEnsemble
    if name == "timecopilot-default":
        return MedianEnsemble(models=[_chronos(), _timesfm()], alias="timecopilot-default")
    if name == "timecopilot+ibm":
        return MedianEnsemble(
            models=[_chronos(), _timesfm(), _patch("ibm-research/patchtst-fm-r1", "research-patchtst"), _flow("ibm-research/flowstate", "research-flowstate"), _ttm()],
            alias="timecopilot+ibm",
        )
    if name in {"ibm-research-tsfm", "ibm-ensemble"}:
        return MedianEnsemble(
            models=[_patch("ibm-research/patchtst-fm-r1", "research-patchtst"), _flow("ibm-research/flowstate", "research-flowstate"), _ttm()],
            alias=name,
        )
    if name == "ibm-granite-tsfm":
        return MedianEnsemble(
            models=[_patch("ibm-granite/granite-timeseries-patchtst-fm-r1", "granite-patchtst"), _flow("ibm-granite/granite-timeseries-flowstate-r1", "granite-flowstate"), _ttm()],
            alias="ibm-granite-tsfm",
        )
    raise ValueError(f"Unsupported forecaster model: {name}")


def _forecaster_predictions(train_df: pd.DataFrame, *, model_names: list[str], flowstate_scale_factors: list[float], h: int, freq: str, continue_on_error: bool) -> pd.DataFrame:
    requested = [name for name in model_names if name in FORECASTER_MODELS]
    frames = []
    for model_name in requested:
        scale_values = flowstate_scale_factors if model_name == "flowstate" else [flowstate_scale_factors[0]]
        for scale_factor in scale_values:
            logging.info("Running forecaster model: %s", model_name)
            start = time.monotonic()
            try:
                model = _build_forecaster_model(model_name, flowstate_scale_factor=scale_factor)
                forecast = model.forecast(df=train_df, h=h, freq=freq)
            except Exception:
                if not continue_on_error:
                    raise
                logging.exception("Model %s failed; continuing.", model_name)
                continue
            elapsed = time.monotonic() - start
            if model.alias not in forecast.columns:
                raise ValueError(f"Model {model_name} returned {list(forecast.columns)}; expected {model.alias}")
            frame = forecast[["unique_id", "ds", model.alias]].rename(columns={"unique_id": "series_id", "ds": "date", model.alias: "yhat"})
            frame["model"] = model.alias
            frame["inference_time_s"] = elapsed
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _smape_components(actual: pd.Series, forecast: pd.Series) -> np.ndarray:
    denominator = (actual.abs() + forecast.abs()) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denominator == 0, 0.0, (forecast - actual).abs() / denominator)


def _attach_actuals(predictions: pd.DataFrame, actuals: pd.DataFrame) -> pd.DataFrame:
    merged = predictions.merge(actuals, on=["series_id", "date"], how="inner")
    merged["error"] = merged["y"] - merged["yhat"]
    merged["abs_error"] = merged["error"].abs()
    merged["squared_error"] = merged["error"] ** 2
    merged["abs_y"] = merged["y"].abs()
    merged["smape_component"] = _smape_components(merged["y"], merged["yhat"])
    return merged


def _summarize_metrics(scored: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    summary = scored.groupby(group_columns, as_index=False).agg(
        evaluated_points=("abs_error", "size"),
        evaluated_series=("series_id", "nunique"),
        mae=("abs_error", "mean"),
        mse=("squared_error", "mean"),
        smape=("smape_component", "mean"),
        total_abs_error=("abs_error", "sum"),
        total_abs_y=("abs_y", "sum"),
        bias=("error", "mean"),
    )
    summary["rmse"] = np.sqrt(summary["mse"])
    summary["accuracy"] = 1.0 - summary["smape"]
    summary["accuracy_pct"] = 100.0 * summary["accuracy"]
    summary["wape"] = summary["total_abs_error"] / summary["total_abs_y"].replace(0, np.nan)
    return summary


def _aggregate_scored(scored: pd.DataFrame) -> pd.DataFrame:
    aggregate = scored.groupby(["country", "granularity", "date", "model"], as_index=False).agg(
        y=("y", "sum"), yhat=("yhat", "sum"), series=("series_id", "nunique")
    )
    aggregate["error"] = aggregate["y"] - aggregate["yhat"]
    aggregate["abs_error"] = aggregate["error"].abs()
    aggregate["squared_error"] = aggregate["error"] ** 2
    aggregate["abs_y"] = aggregate["y"].abs()
    aggregate["smape_component"] = _smape_components(aggregate["y"], aggregate["yhat"])
    aggregate["series_id"] = aggregate["country"] + "|" + aggregate["granularity"] + "|aggregate"
    return aggregate


def run_experiment(
    *,
    data_path: Path = DEFAULT_DATA_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    country: str | None = "Country 1",
    granularity: str | None = "BN_Leaf_PL_Leaf",
    model: list[str] | None = None,
    target_column: str = "target",
    transform: str = "none",
    flowstate_scale_factor: list[float] | None = None,
    train_start: str = "2021-01-01",
    train_end: str = "2024-12-01",
    test_start: str = "2025-01-01",
    test_end: str = "2025-12-01",
    min_history: int = 24,
    series_selection: str | None = None,
    cohort_label: str | None = None,
    continue_on_error: bool = False,
) -> None:
    model_names = model or DEFAULT_MODELS
    unknown = sorted(set(model_names) - set(MODEL_CHOICES))
    if unknown:
        raise ValueError(f"Unknown models {unknown}. Valid choices: {MODEL_CHOICES}")
    train_start_ts, train_end_ts = pd.Timestamp(train_start), pd.Timestamp(train_end)
    test_start_ts, test_end_ts = pd.Timestamp(test_start), pd.Timestamp(test_end)
    horizon = len(pd.date_range(test_start_ts, test_end_ts, freq="MS"))
    flowstate_scale_factors = flowstate_scale_factor or DEFAULT_FLOWSTATE_SCALE_FACTORS
    config = RunConfig(str(data_path), str(output_dir), country, granularity, train_start, train_end, test_start, test_end, target_column, transform, model_names, flowstate_scale_factors, min_history, series_selection, cohort_label, continue_on_error)

    model_table = _apply_series_selection(_load_model_table(data_path, target_column), selection_path=series_selection, country=country, granularity=granularity)
    panel = _filter_panel(model_table, country=country, granularity=granularity, target_column=target_column, train_start=train_start_ts, train_end=train_end_ts, test_start=test_start_ts, test_end=test_end_ts, min_history=min_history)
    train_df = _to_forecaster_train(panel, target_column=target_column, train_start=train_start_ts, train_end=train_end_ts)
    actual = _actuals(panel, target_column=target_column, test_start=test_start_ts, test_end=test_end_ts)
    test_dates = pd.date_range(test_start_ts, test_end_ts, freq="MS")

    frames = [
        _manual_predictions(train_df, model_names=model_names, test_dates=test_dates),
        _statsforecast_predictions(train_df, model_names=model_names, h=horizon, freq="MS"),
        _forecaster_predictions(train_df, model_names=model_names, flowstate_scale_factors=flowstate_scale_factors, h=horizon, freq="MS", continue_on_error=continue_on_error),
    ]
    predictions = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    if predictions.empty:
        raise ValueError("No predictions were produced.")
    scored = _attach_actuals(predictions, actual)
    series_metrics = _summarize_metrics(scored, ["country", "granularity", "model"])
    per_series_metrics = _summarize_metrics(scored, ["country", "granularity", "model", "series_id"])
    aggregate_predictions = _aggregate_scored(scored)
    aggregate_metrics = _summarize_metrics(aggregate_predictions, ["country", "granularity", "model"])

    model_slug = "_".join(model_names).replace("/", "-")
    cohort_slug = f"__{cohort_label}" if cohort_label else ""
    run_dir = output_dir / f"{country or 'all_countries'}__{granularity or 'all_granularities'}__{model_slug}{cohort_slug}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(json.dumps(asdict(config), indent=2))
    scored.to_csv(run_dir / "predictions_scored.csv", index=False)
    series_metrics.to_csv(run_dir / "series_metrics.csv", index=False)
    per_series_metrics.to_csv(run_dir / "per_series_metrics.csv", index=False)
    aggregate_predictions.to_csv(run_dir / "aggregate_predictions.csv", index=False)
    aggregate_metrics.to_csv(run_dir / "aggregate_metrics.csv", index=False)
    print(f"\nPer-series metrics written to {run_dir / 'per_series_metrics.csv'}")
    print("\nAggregated country-month metrics")
    print(aggregate_metrics.sort_values("wape").to_string(index=False))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Planning Analytics forecasts.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--country", default="Country 1")
    parser.add_argument("--granularity", default="BN_Leaf_PL_Leaf")
    parser.add_argument("--model", action="append", choices=MODEL_CHOICES)
    parser.add_argument("--target-column", default="target")
    parser.add_argument("--transform", choices=TRANSFORM_CHOICES, default="none")
    parser.add_argument("--flowstate-scale-factor", action="append", type=float, default=None)
    parser.add_argument("--train-start", default="2021-01-01")
    parser.add_argument("--train-end", default="2024-12-01")
    parser.add_argument("--test-start", default="2025-01-01")
    parser.add_argument("--test-end", default="2025-12-01")
    parser.add_argument("--min-history", type=int, default=24)
    parser.add_argument("--series-selection", default=None)
    parser.add_argument("--cohort-label", default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_experiment(
        data_path=args.data_path,
        output_dir=args.output_dir,
        country=None if args.country == "all" else args.country,
        granularity=None if args.granularity == "all" else args.granularity,
        model=args.model,
        target_column=args.target_column,
        transform=args.transform,
        flowstate_scale_factor=args.flowstate_scale_factor,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        min_history=args.min_history,
        series_selection=args.series_selection,
        cohort_label=args.cohort_label,
        continue_on_error=args.continue_on_error,
    )


if __name__ == "__main__":
    main()
