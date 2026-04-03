import numpy as np
import pandas as pd
import pytest

from timecopilot.models.foundation.ttm import TTM


def test_ttm_uses_configured_dtype(mocker):
    mock_from_df = mocker.patch("timecopilot.models.foundation.ttm.TimeSeriesDataset.from_df")

    model = TTM(batch_size=4)

    df = pd.DataFrame(
        {
            "unique_id": ["A"] * 10,
            "ds": pd.date_range("2020-01-01", periods=10),
            "y": range(10),
        }
    )

    def _from_df_side_effect(*args, **kwargs):
        assert kwargs.get("dtype") == model.dtype
        raise RuntimeError("stop after dtype check")

    mock_from_df.side_effect = _from_df_side_effect

    with pytest.raises(RuntimeError, match="stop after dtype check"):
        model.forecast(df=df, h=2)


def test_ttm_rejects_quantiles():
    model = TTM()
    df = pd.DataFrame(
        {
            "unique_id": ["A"] * 10,
            "ds": pd.date_range("2020-01-01", periods=10),
            "y": range(10),
        }
    )

    with pytest.raises(ValueError, match="does not support level or quantile"):
        model.forecast(df=df, h=2, quantiles=[0.1, 0.5, 0.9])


def test_ttm_h1_single_uid(mocker):
    mock_model = mocker.Mock()
    mock_model.to.return_value = mock_model
    mock_model.eval.return_value = None
    mock_model.return_value.prediction_outputs = mocker.Mock(
        detach=mocker.Mock(
            return_value=mocker.Mock(
                cpu=mocker.Mock(
                    return_value=mocker.Mock(
                        numpy=mocker.Mock(return_value=np.array([[[1.23]]], dtype=np.float32))
                    )
                )
            )
        )
    )
    mocker.patch(
        "timecopilot.models.foundation.ttm.TinyTimeMixerForPrediction.from_pretrained",
        return_value=mock_model,
    )

    ds = pd.date_range("2024-01-01", periods=20, freq="W")
    df = pd.DataFrame({"unique_id": "u1", "ds": ds, "y": np.arange(20)})

    fcst = TTM().forecast(df=df, h=1, freq="W")

    assert isinstance(fcst, pd.DataFrame)
    assert len(fcst) == 1
    assert "unique_id" in fcst.columns
    assert "ds" in fcst.columns
    assert "TTM" in fcst.columns
