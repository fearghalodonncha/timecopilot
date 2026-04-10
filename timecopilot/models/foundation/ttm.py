from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction

from ..utils.forecaster import Forecaster, _DataProcessor
from .utils import TimeSeriesDataset, flatten_forecast_values


class TTM(Forecaster, _DataProcessor):
    """
    TinyTimeMixer (TTM) is an IBM time series foundation model designed to provide
    compact and fast zero-shot forecasting.
    """

    def __init__(
        self,
        repo_id: str = "ibm-granite/granite-timeseries-ttm-r2",
        context_length: int = 512,
        batch_size: int = 1_024,
        alias: str = "TTM",
    ):
        self.repo_id = repo_id
        self.context_length = context_length
        self.batch_size = batch_size
        self.alias = alias
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32

    @contextmanager
    def _get_model(self) -> TinyTimeMixerForPrediction:
        model = TinyTimeMixerForPrediction.from_pretrained(self.repo_id).to(self.device)
        try:
            model.eval()
            yield model
        finally:
            del model
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
            elif self.device.startswith("mps"):
                torch.mps.empty_cache()

    def _predict_batch(
        self,
        model: TinyTimeMixerForPrediction,
        batch: list[torch.Tensor] | torch.Tensor,
        h: int,
    ) -> np.ndarray:
        context = self._prepare_and_validate_context(batch)
        if context.shape[1] > self.context_length:
            context = context[..., -self.context_length :]
        context = self._maybe_impute_missing(context)
        context = context.unsqueeze(-1).to(self.device)
        fcst = model(context, return_loss=False).prediction_outputs
        fcst = fcst[..., 0]
        if fcst.shape[1] < h:
            raise ValueError(
                f"{self.alias} returned horizon {fcst.shape[1]}, expected at least {h}. "
                "Choose a model checkpoint whose native prediction length covers "
                "the requested evaluation horizon."
            )
        fcst = fcst[:, :h]
        return fcst.detach().cpu().numpy()

    def _predict(
        self,
        model: TinyTimeMixerForPrediction,
        dataset: TimeSeriesDataset,
        h: int,
    ) -> np.ndarray:
        fcsts = [self._predict_batch(model, batch, h) for batch in tqdm(dataset)]
        return np.concatenate(fcsts)

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        freq = self._maybe_infer_freq(df, freq)
        if level is not None or quantiles is not None:
            raise ValueError("TTM does not support level or quantile forecasts yet.")
        dataset = TimeSeriesDataset.from_df(
            df,
            batch_size=self.batch_size,
            dtype=self.dtype,
        )
        fcst_df = dataset.make_future_dataframe(h=h, freq=freq)
        with self._get_model() as model:
            fcsts_mean_np = self._predict(model, dataset, h)
        fcst_df[self.alias] = flatten_forecast_values(
            fcsts_mean_np,
            expected_rows=len(fcst_df),
            model_alias=self.alias,
            column_name=self.alias,
        )
        return fcst_df
