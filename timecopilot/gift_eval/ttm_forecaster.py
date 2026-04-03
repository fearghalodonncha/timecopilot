import logging
from collections.abc import Iterable

import numpy as np
import pandas as pd
from gluonts.dataset.split import TrainingDataset
from scipy.stats import norm

from ..models.foundation.ttm import TTM
from ..models.utils.forecaster import Forecaster, QuantileConverter

logger = logging.getLogger(__name__)


class TTMGiftEvalForecaster(Forecaster):
    """
    Gift-eval specific adapter that turns TTM point forecasts into quantile forecasts
    using per-series in-sample absolute errors, following the same high-level idea as
    IBM's gift leaderboard predictor.
    """

    def __init__(
        self,
        forecaster: TTM,
        calibration_dataset: TrainingDataset,
        prediction_length: int,
        freq: str,
    ):
        self.forecaster = forecaster
        self.prediction_length = prediction_length
        self.freq = freq
        self.alias = forecaster.alias
        self.insample_errors = self._compute_insample_errors(calibration_dataset)

    def _training_dataset_to_df(self, dataset: TrainingDataset) -> pd.DataFrame:
        dfs: list[pd.DataFrame] = []
        for entry in dataset:
            target = np.asarray(entry["target"], dtype=np.float32)
            if target.ndim > 1:
                raise ValueError("TTMGiftEvalForecaster only supports univariate data.")
            ds = pd.date_range(
                start=entry["start"].to_timestamp(),
                freq=entry["start"].freq,
                periods=len(target),
            )
            dfs.append(
                pd.DataFrame(
                    {
                        "unique_id": entry["item_id"],
                        "ds": ds,
                        "y": target,
                    }
                )
            )
        return pd.concat(dfs, ignore_index=True)

    def _normalize_error_vector(self, errors: Iterable[float]) -> np.ndarray:
        arr = np.asarray(list(errors), dtype=np.float32).reshape(-1)
        if len(arr) == 0:
            arr = np.ones(self.prediction_length, dtype=np.float32)
        if len(arr) < self.prediction_length:
            pad = np.repeat(arr[-1], self.prediction_length - len(arr))
            arr = np.concatenate([arr, pad])
        arr = arr[: self.prediction_length]
        arr = np.nan_to_num(arr, nan=float(np.nanmean(arr)) if not np.isnan(arr).all() else 1.0)
        arr[arr <= 0] = 1e-5
        return arr

    def _compute_insample_errors(
        self,
        calibration_dataset: TrainingDataset,
    ) -> pd.Series:
        calibration_df = self._training_dataset_to_df(calibration_dataset)
        cv_df = self.forecaster.cross_validation(
            calibration_df,
            h=self.prediction_length,
            freq=self.freq,
            n_windows=1,
        )
        cv_df["horizon_idx"] = cv_df.groupby(["unique_id", "cutoff"]).cumcount()
        cv_df["abs_error"] = (cv_df["y"] - cv_df[self.forecaster.alias]).abs()
        errors = (
            cv_df.groupby(["unique_id", "horizon_idx"])["abs_error"]
            .mean()
            .unstack(level="horizon_idx")
            .sort_index(axis=1)
        )
        if errors.empty:
            logger.warning("TTM calibration produced no in-sample errors; using unit scale.")
            return pd.Series(dtype=object)
        return errors.apply(self._normalize_error_vector, axis=1)

    def _lookup_error_scale(self, source_item_id: str, horizon_idx: int) -> float:
        if source_item_id in self.insample_errors.index:
            return float(self.insample_errors.loc[source_item_id][horizon_idx])
        if len(self.insample_errors) > 0:
            mean_error = np.mean(np.stack(self.insample_errors.values), axis=0)
            return float(mean_error[horizon_idx])
        return 1.0

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        freq = self._maybe_infer_freq(df, freq)
        if h != self.prediction_length:
            raise ValueError(
                f"TTM gift-eval calibration expects horizon {self.prediction_length}, got {h}."
            )
        qc = QuantileConverter(level=level, quantiles=quantiles)
        point_df = self.forecaster.forecast(df=df, h=h, freq=freq)
        if qc.quantiles is None:
            return point_df

        if "source_item_id" in df.columns:
            source_item_ids = df.groupby("unique_id")["source_item_id"].first()
        else:
            source_item_ids = pd.Series(df["unique_id"].unique(), index=df["unique_id"].unique())
        point_df = point_df.copy()
        point_df["_source_item_id"] = point_df["unique_id"].map(source_item_ids).fillna(
            point_df["unique_id"]
        )
        point_df["_horizon_idx"] = point_df.groupby("unique_id").cumcount()
        scales = point_df.apply(
            lambda row: self._lookup_error_scale(
                str(row["_source_item_id"]),
                int(row["_horizon_idx"]),
            ),
            axis=1,
        ).to_numpy(dtype=np.float32)
        loc = point_df[self.alias].to_numpy(dtype=np.float32)
        for q in qc.quantiles:
            col_name = f"{self.alias}-q-{int(q * 100)}"
            point_df[col_name] = norm.ppf(q, loc=loc, scale=scales)
        point_df = point_df.drop(columns=["_source_item_id", "_horizon_idx"])
        if 0.5 in qc.quantiles:
            point_df[self.alias] = point_df[f"{self.alias}-q-50"].to_numpy()
        return qc.maybe_convert_quantiles_to_level(point_df, models=[self.alias])
