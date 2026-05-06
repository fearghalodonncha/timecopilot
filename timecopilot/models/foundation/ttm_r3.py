from contextlib import contextmanager
from importlib import import_module
import logging
from typing import Any

from gluonts.dataset.util import forecast_start
from gluonts.model.forecast import QuantileForecast
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..utils.forecaster import Forecaster, QuantileConverter, _DataProcessor
from .utils import TimeSeriesDataset, flatten_forecast_values

LOGGER = logging.getLogger(__name__)
TTM_MAX_FORECAST_HORIZON = 720
TTM_LOW_RESOLUTION_MODELS_MAX_CONTEXT = 512

RESOLUTION_MAP = {
    "oov": "oov",
    "OOV": "oov",
    "min": "min",
    "1min": "min",
    "T": "min",
    "1T": "min",
    "2min": "2min",
    "2T": "2min",
    "5min": "5min",
    "5T": "5min",
    "10min": "10min",
    "10T": "10min",
    "15min": "15min",
    "15T": "15min",
    "30min": "30min",
    "30T": "30min",
    "h": "h",
    "1h": "h",
    "H": "h",
    "1H": "h",
    "d": "d",
    "1d": "d",
    "D": "d",
    "1D": "d",
    "w": "W",
    "1w": "W",
    "W": "W",
    "1W": "W",
    "W-FRI": "W",
    "W-TUE": "W",
    "W-MON": "W",
    "W-WED": "W",
    "W-THU": "W",
    "W-SAT": "W",
    "W-SUN": "W",
    "M": "oov",
    "1M": "oov",
    "Q-DEC": "oov",
    "A-DEC": "oov",
    "A": "oov",
    "10S": "oov",
}


def _load_ttm_r3_helpers():
    try:
        toolkit_module = import_module("tsfm_public.toolkit.get_model")
        return (
            getattr(toolkit_module, "get_model"),
            getattr(
                toolkit_module,
                "TTM_LOW_RESOLUTION_MODELS_MAX_CONTEXT",
                TTM_LOW_RESOLUTION_MODELS_MAX_CONTEXT,
            ),
        )
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
        context_length: int | None = 512,
        batch_size: int = 1_024,
        alias: str = "TTM-R3",
        quantile_list: list[float] | None = None,
        model_revision: str | None = None,
        use_lite: bool = True,
        gift_eval_compat: bool = False,
        term: str | None = None,
        use_mask: bool = True,
        rolling_norm: bool = False,
    ):
        self.repo_id = repo_id
        self.context_length = context_length
        self.batch_size = batch_size
        self.alias = alias
        self.quantile_list = quantile_list or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.model_revision = model_revision
        self.use_lite = use_lite
        self.gift_eval_compat = gift_eval_compat
        self.term = term
        self.use_mask = use_mask
        self.rolling_norm = rolling_norm
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float32

    def _get_effective_context_length(self, model) -> int:
        context_lengths = [self.context_length] if self.context_length is not None else []
        for config_owner in (model, getattr(model, "trend_forecaster", None)):
            config = getattr(config_owner, "config", None)
            context_length = getattr(config, "context_length", None)
            if isinstance(context_length, int) and context_length > 0:
                context_lengths.append(context_length)
        return max(context_lengths)

    def _public_gift_model_kwargs(
        self,
        h: int,
        freq: str | None,
        low_resolution_max_context: int,
    ) -> dict[str, Any]:
        prefer_l1_loss = False
        prefer_longer_context = True
        freq_prefix_tuning = False
        force_return = "zeropad"
        term = self.term or ""
        freq_str = str(freq or "")
        context_length = self.context_length or 1

        if term == "short" and (
            freq_str.startswith("W")
            or freq_str.startswith("M")
            or freq_str.startswith("Q")
            or freq_str.startswith("A")
        ):
            prefer_l1_loss = True
            prefer_longer_context = False
            freq_prefix_tuning = True

        if term == "short" and freq_str.startswith("D"):
            prefer_l1_loss = True
            freq_prefix_tuning = True
            prefer_longer_context = context_length >= 2 * low_resolution_max_context

        if term == "short" and freq_str.startswith("A"):
            force_return = "random_init_small"

        if h > TTM_MAX_FORECAST_HORIZON:
            force_return = "rolling"

        return {
            "freq_prefix_tuning": freq_prefix_tuning,
            "freq": RESOLUTION_MAP.get(freq_str, "oov"),
            "prefer_l1_loss": prefer_l1_loss,
            "prefer_longer_context": prefer_longer_context,
            "force_return": force_return,
        }

    def _load_model(
        self,
        get_model,
        h: int,
        freq: str | None,
        low_resolution_max_context: int,
    ):
        prediction_length = min(h, TTM_MAX_FORECAST_HORIZON)
        model_kwargs = {}
        if self.gift_eval_compat:
            model_kwargs.update(
                self._public_gift_model_kwargs(
                    h=h,
                    freq=freq,
                    low_resolution_max_context=low_resolution_max_context,
                )
            )
        else:
            model_kwargs["freq"] = freq
        try:
            return get_model(
                model_path=self.repo_id,
                model_name="ttm",
                context_length=self.context_length,
                prediction_length=prediction_length,
                use_lite=self.use_lite,
                model_revision=self.model_revision,
                **model_kwargs,
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
                        prediction_length=prediction_length,
                        use_lite=self.use_lite,
                        model_revision=fallback_revision,
                        **model_kwargs,
                    )
            raise

    @contextmanager
    def _get_model(self, h: int, freq: str | None):
        get_model, low_resolution_max_context = _load_ttm_r3_helpers()
        LOGGER.info(
            "%s loading repo_id=%s revision=%s context_length=%s prediction_length=%s use_lite=%s gift_eval_compat=%s",
            self.alias,
            self.repo_id,
            self.model_revision or "<auto>",
            self.context_length,
            h,
            self.use_lite,
            self.gift_eval_compat,
        )
        model = self._load_model(
            get_model,
            h=h,
            freq=freq,
            low_resolution_max_context=low_resolution_max_context,
        ).to(self.device)
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

    def _prepare_gift_entry(
        self,
        entry: Any,
        model_context_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray | None, np.ndarray | None]:
        target = np.asarray(entry["target"], dtype=np.float32)
        if target.ndim == 1:
            target = target.reshape(1, -1)
        elif target.ndim != 2:
            raise ValueError(f"{self.alias} only supports 1D or 2D GIFT targets.")
        mean = None
        std = None
        if self.rolling_norm:
            target = self._impute_gift_target_np(target)
            mean = np.mean(target, axis=1).reshape(1, -1)
            std = np.std(target, axis=1).reshape(1, -1)
            std[std == 0] = 1.0
            target = ((target.T - mean) / std).T
        target_tensor = torch.tensor(target, dtype=self.dtype)
        if target_tensor.shape[1] > model_context_length:
            target_tensor = target_tensor[:, -model_context_length:]
            mask = torch.ones_like(target_tensor, dtype=torch.bool)
        elif target_tensor.shape[1] < model_context_length:
            pad_len = model_context_length - target_tensor.shape[1]
            padding = torch.zeros((target_tensor.shape[0], pad_len), dtype=self.dtype)
            target_tensor = torch.cat((padding, target_tensor), dim=1)
            mask = torch.ones_like(target_tensor, dtype=torch.bool)
            mask[:, :pad_len] = False
        else:
            mask = torch.ones_like(target_tensor, dtype=torch.bool)
        target_tensor = self._maybe_impute_missing(target_tensor)
        return target_tensor.T, mask.T, mean, std

    def _impute_gift_target_np(self, target: np.ndarray) -> np.ndarray:
        target = np.asarray(target, dtype=np.float32).copy()
        for idx in range(target.shape[0]):
            row = target[idx]
            if not np.isnan(row).any():
                continue
            valid = np.flatnonzero(~np.isnan(row))
            if len(valid) == 0:
                target[idx] = np.zeros_like(row)
                continue
            first = valid[0]
            row[:first] = row[first]
            for pos in range(first + 1, len(row)):
                if np.isnan(row[pos]):
                    row[pos] = row[pos - 1]
            target[idx] = row
        return target

    def _inverse_rolling_norm(
        self,
        quantile_outputs: torch.Tensor,
        means: list[np.ndarray | None],
        stds: list[np.ndarray | None],
    ) -> torch.Tensor:
        if not self.rolling_norm:
            return quantile_outputs
        mean_values = []
        std_values = []
        for mean, std in zip(means, stds, strict=True):
            if mean is None or std is None:
                raise ValueError("Rolling normalization statistics were not computed.")
            mean_values.append(mean.reshape(-1))
            std_values.append(std.reshape(-1))
        mean_tensor = torch.tensor(
            np.stack(mean_values),
            dtype=quantile_outputs.dtype,
            device=quantile_outputs.device,
        )[:, None, None, :]
        std_tensor = torch.tensor(
            np.stack(std_values),
            dtype=quantile_outputs.dtype,
            device=quantile_outputs.device,
        )[:, None, None, :]
        return quantile_outputs * std_tensor + mean_tensor

    def _normalize_quantile_outputs(
        self,
        outputs,
        h: int,
        num_channels: int,
    ) -> torch.Tensor:
        quantile_outputs = outputs.quantile_outputs
        if quantile_outputs is None:
            raise ValueError(f"{self.alias} did not return quantile outputs.")
        if quantile_outputs.ndim != 4:
            raise ValueError(
                f"{self.alias} quantile output must have 4 dims, got {tuple(quantile_outputs.shape)}."
            )
        quantile_outputs = quantile_outputs[:, :, :h, :num_channels]
        return quantile_outputs

    def _predict_gift_batch_tensor(
        self,
        model,
        past_values: torch.Tensor,
        past_observed_mask: torch.Tensor,
        h: int,
        num_channels: int,
    ) -> torch.Tensor:
        model_prediction_length = int(getattr(model.config, "prediction_length", h))
        if h <= TTM_MAX_FORECAST_HORIZON:
            batch_ttm = {"past_values": past_values}
            if self.use_mask:
                batch_ttm["past_observed_mask"] = past_observed_mask
            outputs = model(**batch_ttm)
            return self._normalize_quantile_outputs(outputs, h=h, num_channels=num_channels)

        remaining = h
        quantile_outputs = []
        batch_ttm = {
            "past_values": past_values,
            "return_loss": False,
        }
        if self.use_mask:
            batch_ttm["past_observed_mask"] = past_observed_mask
        with torch.no_grad():
            while remaining > 0:
                outputs = model(**batch_ttm)
                step_outputs = self._normalize_quantile_outputs(
                    outputs,
                    h=min(model_prediction_length, remaining),
                    num_channels=num_channels,
                )
                quantile_outputs.append(step_outputs)
                median = outputs.quantile_outputs[:, 4, ...]
                batch_ttm["past_values"] = torch.cat(
                    [batch_ttm["past_values"], median],
                    dim=1,
                )[:, -model.config.context_length :, :]
                if self.use_mask:
                    observed = torch.ones_like(median, dtype=torch.bool)
                    batch_ttm["past_observed_mask"] = torch.cat(
                        [batch_ttm["past_observed_mask"], observed],
                        dim=1,
                    )[:, -model.config.context_length :, :]
                remaining -= step_outputs.shape[2]
        return torch.cat(quantile_outputs, dim=2)[:, :, :h, :]

    def predict_gluonts_batch(
        self,
        batch: list[Any],
        h: int,
        quantiles: list[float] | None,
    ) -> list[QuantileForecast]:
        requested_quantiles = self.quantile_list if quantiles is None else quantiles
        with self._get_model(h=h, freq=batch[0].get("freq")) as model:
            model_context_length = int(model.config.context_length)
            prepared = [
                self._prepare_gift_entry(entry, model_context_length=model_context_length)
                for entry in batch
            ]
            past_values = torch.stack([item[0] for item in prepared]).to(self.device)
            past_observed_mask = torch.stack([item[1] for item in prepared]).to(self.device)
            means = [item[2] for item in prepared]
            stds = [item[3] for item in prepared]
            num_channels = past_values.shape[-1]
            with torch.no_grad():
                quantile_outputs = self._predict_gift_batch_tensor(
                    model=model,
                    past_values=past_values,
                    past_observed_mask=past_observed_mask,
                    h=h,
                    num_channels=num_channels,
                )
                quantile_outputs = self._inverse_rolling_norm(
                    quantile_outputs,
                    means=means,
                    stds=stds,
                )
        index_map = {round(q, 6): idx for idx, q in enumerate(self.quantile_list)}
        selected_idx = [index_map[round(q, 6)] for q in requested_quantiles]
        selected = quantile_outputs[:, selected_idx, :, :]
        mean_like = selected[:, requested_quantiles.index(0.5), :, :] if 0.5 in requested_quantiles else selected[:, 0]
        forecast_arrays = torch.cat([selected, mean_like.unsqueeze(1)], dim=1)
        forecast_keys = [str(q) for q in requested_quantiles] + ["mean"]
        if forecast_arrays.shape[-1] == 1:
            forecast_arrays = forecast_arrays.squeeze(-1)
        return [
            QuantileForecast(
                forecast_arrays=forecast_array.detach().cpu().numpy(),
                forecast_keys=forecast_keys,
                item_id=entry.get("item_id"),
                start_date=forecast_start(entry),
            )
            for forecast_array, entry in zip(forecast_arrays, batch, strict=True)
        ]

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
