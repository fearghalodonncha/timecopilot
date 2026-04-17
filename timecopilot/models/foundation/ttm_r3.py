from contextlib import contextmanager
from importlib import import_module
import logging

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..utils.forecaster import Forecaster, QuantileConverter, _DataProcessor
from .utils import TimeSeriesDataset, flatten_forecast_values

LOGGER = logging.getLogger(__name__)


def _load_ttm_r3_helpers():
    try:
        toolkit_module = import_module("tsfm_public.toolkit.get_model")
        return getattr(toolkit_module, "get_model")
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "TTMR3 requires an R3-capable `tsfm_public` installation. "
            "Install the Granite TSFM `ttm-r3-release-mq2` branch into the "
            "environment used by TimeCopilot."
        ) from exc


def _fallback_ttm_r3_revision_for_base_revision(
    base_revision: str,
    prediction_length: int,
    use_lite: bool,
) -> str | None:
    def _maybe_lite(revision: str) -> str:
        return revision.replace("-r3", "-lite-r3") if use_lite else revision

    if base_revision.endswith("-96-r2") or base_revision.endswith("-96-ft-r2"):
        return _maybe_lite("2048-96-r3")
    if base_revision.endswith("-720-r2") or base_revision.endswith("-720-ft-r2"):
        return _maybe_lite("2048-720-r3")
    if prediction_length <= 96:
        return _maybe_lite("2048-96-r3")
    if prediction_length <= 720:
        return _maybe_lite("2048-720-r3")
    return None


class TTMR3(Forecaster, _DataProcessor):
    """
    TinyTimeMixer R3 wrapper with native quantile output support.
    """

    def __init__(
        self,
        repo_id: str = "ibm-research/ttm-r3",
        context_length: int = 512,
        batch_size: int = 1_024,
        alias: str = "TTM-R3",
        quantile_list: list[float] | None = None,
        model_revision: str | None = None,
        use_lite: bool = True,
    ):
        self.repo_id = repo_id
        self.context_length = context_length
        self.batch_size = batch_size
        self.alias = alias
        self.quantile_list = quantile_list or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.model_revision = model_revision
        self.use_lite = use_lite
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32

    def _get_effective_context_length(self, model) -> int:
        context_lengths = [self.context_length]
        for config_owner in (model, getattr(model, "trend_forecaster", None)):
            config = getattr(config_owner, "config", None)
            context_length = getattr(config, "context_length", None)
            if isinstance(context_length, int) and context_length > 0:
                context_lengths.append(context_length)
        return max(context_lengths)

    def _load_model(self, get_model, h: int, freq: str | None):
        try:
            return get_model(
                model_path=self.repo_id,
                model_name="ttm",
                context_length=self.context_length,
                prediction_length=h,
                freq=freq,
                use_lite=self.use_lite,
                model_revision=self.model_revision,
            )
        except ValueError as exc:
            if "prediction_filter_length should be positive" in str(exc):
                revision_msg = (
                    f"Requested TTMR3 revision `{self.model_revision}`"
                    if self.model_revision is not None
                    else "The selected TTMR3 revision"
                )
                raise ValueError(
                    f"{revision_msg} is incompatible with forecast horizon h={h}. "
                    "This usually means the revision's native prediction length is "
                    "shorter than the requested horizon. Leave `model_revision=None` "
                    "to let `tsfm_public` auto-select a compatible R3 revision, or "
                    "choose a revision whose prediction length is at least the "
                    "requested horizon."
                ) from exc
            error_prefix = "Invalid base revision for r3 mapping: "
            if self.model_revision is None and error_prefix in str(exc):
                base_revision = str(exc).split(error_prefix, maxsplit=1)[1].strip()
                fallback_revision = _fallback_ttm_r3_revision_for_base_revision(
                    base_revision=base_revision,
                    prediction_length=h,
                    use_lite=self.use_lite,
                )
                if fallback_revision is not None:
                    LOGGER.warning(
                        "%s retrying with explicit fallback revision=%s for unsupported base_revision=%s",
                        self.alias,
                        fallback_revision,
                        base_revision,
                    )
                    return get_model(
                        model_path=self.repo_id,
                        model_name="ttm",
                        context_length=self.context_length,
                        prediction_length=h,
                        freq=freq,
                        use_lite=self.use_lite,
                        model_revision=fallback_revision,
                    )
            raise

    @contextmanager
    def _get_model(self, h: int, freq: str | None):
        get_model = _load_ttm_r3_helpers()
        LOGGER.info(
            "%s loading repo_id=%s revision=%s context_length=%s prediction_length=%s use_lite=%s",
            self.alias,
            self.repo_id,
            self.model_revision or "<auto>",
            self.context_length,
            h,
            self.use_lite,
        )
        model = self._load_model(get_model, h=h, freq=freq).to(self.device)
        if hasattr(model.config, "multi_quantile_head"):
            model.config.multi_quantile_head = True
        if hasattr(model.config, "quantile_list"):
            model.config.quantile_list = self.quantile_list
        LOGGER.info(
            "%s loaded class=%s revision=%s model_context_length=%s model_prediction_length=%s quantiles=%s",
            self.alias,
            model.__class__.__name__,
            getattr(model, "name_or_path", self.repo_id),
            getattr(model.config, "context_length", None),
            getattr(model.config, "prediction_length", None),
            getattr(model.config, "quantile_list", None),
        )
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
        model,
        batch: list[torch.Tensor] | torch.Tensor,
        h: int,
        quantiles: list[float] | None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        context = self._prepare_and_validate_context(batch)
        effective_context_length = self._get_effective_context_length(model)
        observed_mask = torch.ones_like(context, dtype=torch.bool)
        if context.shape[1] > effective_context_length:
            context = context[..., -effective_context_length:]
            observed_mask = observed_mask[..., -effective_context_length:]
        elif context.shape[1] < effective_context_length:
            pad = effective_context_length - context.shape[1]
            context = torch.nn.functional.pad(context, (pad, 0), value=0.0)
            observed_mask = torch.nn.functional.pad(observed_mask, (pad, 0), value=False)
            LOGGER.info(
                "%s zero-padded context from %s to %s",
                self.alias,
                context.shape[1] - pad,
                effective_context_length,
            )
        context = self._maybe_impute_missing(context)
        context = context.unsqueeze(-1).to(self.device)
        observed_mask = observed_mask.unsqueeze(-1).to(self.device)
        outputs = model(
            past_values=context,
            past_observed_mask=observed_mask,
            return_loss=False,
        )
        point_fcst = outputs.prediction_outputs[..., 0]
        if point_fcst.shape[1] < h:
            raise ValueError(
                f"{self.alias} returned horizon {point_fcst.shape[1]}, expected at least {h}."
            )
        point_fcst = point_fcst[:, :h]
        LOGGER.info(
            "%s point output shape=%s requested_h=%s",
            self.alias,
            tuple(point_fcst.shape),
            h,
        )

        quantile_fcst = None
        if quantiles is not None:
            if not hasattr(outputs, "quantile_outputs"):
                raise ValueError(f"{self.alias} did not return native quantile outputs.")
            quantile_fcst = outputs.quantile_outputs
            if quantile_fcst is None:
                raise ValueError(f"{self.alias} returned no native quantile outputs.")
            if quantile_fcst.ndim != 4:
                raise ValueError(
                    f"{self.alias} quantile output must have 4 dims, got shape {tuple(quantile_fcst.shape)}."
                )
            quantile_fcst = quantile_fcst[:, :, :h, 0]
            if quantile_fcst.shape[1] != len(self.quantile_list):
                raise ValueError(
                    f"{self.alias} returned {quantile_fcst.shape[1]} quantiles, "
                    f"expected {len(self.quantile_list)} from config.quantile_list."
                )
            index_map = {round(q, 6): idx for idx, q in enumerate(self.quantile_list)}
            try:
                selected_idx = [index_map[round(q, 6)] for q in quantiles]
            except KeyError as exc:
                raise ValueError(
                    f"{self.alias} does not support requested quantiles {quantiles}. "
                    f"Available quantiles are {self.quantile_list}."
                ) from exc
            quantile_fcst = quantile_fcst[:, selected_idx, :]
            LOGGER.info(
                "%s quantile output shape=%s selected_quantiles=%s",
                self.alias,
                tuple(quantile_fcst.shape),
                quantiles,
            )
        return point_fcst.detach().cpu().numpy(), (
            quantile_fcst.detach().cpu().numpy() if quantile_fcst is not None else None
        )

    def _predict(
        self,
        model,
        dataset: TimeSeriesDataset,
        h: int,
        quantiles: list[float] | None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        point_fcsts = []
        quantile_fcsts = []
        for batch in tqdm(dataset):
            point_batch, quantile_batch = self._predict_batch(model, batch, h, quantiles)
            point_fcsts.append(point_batch)
            if quantile_batch is not None:
                quantile_fcsts.append(quantile_batch)
        point_np = np.concatenate(point_fcsts)
        quantile_np = np.concatenate(quantile_fcsts) if quantile_fcsts else None
        return point_np, quantile_np

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        freq = self._maybe_infer_freq(df, freq)
        qc = QuantileConverter(level=level, quantiles=quantiles)
        dataset = TimeSeriesDataset.from_df(
            df,
            batch_size=self.batch_size,
            dtype=self.dtype,
        )
        fcst_df = dataset.make_future_dataframe(h=h, freq=freq)
        with self._get_model(h=h, freq=freq) as model:
            point_fcst_np, quantile_fcst_np = self._predict(
                model,
                dataset,
                h,
                quantiles=qc.quantiles,
            )
        fcst_df[self.alias] = flatten_forecast_values(
            point_fcst_np,
            expected_rows=len(fcst_df),
            model_alias=self.alias,
            column_name=self.alias,
        )
        if qc.quantiles is not None and quantile_fcst_np is not None:
            for i, q in enumerate(qc.quantiles):
                col_name = f"{self.alias}-q-{int(q * 100)}"
                fcst_df[col_name] = flatten_forecast_values(
                    quantile_fcst_np[:, i, :],
                    expected_rows=len(fcst_df),
                    model_alias=self.alias,
                    column_name=col_name,
                )
            if 0.5 in qc.quantiles:
                fcst_df[self.alias] = fcst_df[f"{self.alias}-q-50"].to_numpy()
            fcst_df = qc.maybe_convert_quantiles_to_level(
                fcst_df,
                models=[self.alias],
            )
        return fcst_df
