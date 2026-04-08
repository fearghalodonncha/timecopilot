import numpy as np
import pandas as pd
import pytest

from timecopilot.models.foundation.ttm_r3 import TTMR3, _load_ttm_r3_class


def test_ttm_r3_import_error_is_helpful(mocker):
    mocker.patch(
        "timecopilot.models.foundation.ttm_r3.import_module",
        side_effect=ImportError("missing"),
    )
    with pytest.raises(ImportError, match="R3-capable `tsfm_public` installation"):
        _load_ttm_r3_class()


def test_ttm_r3_h1_single_uid(mocker):
    mock_model = mocker.Mock()
    mock_model.to.return_value = mock_model
    mock_model.eval.return_value = None
    mock_model.config.multi_quantile_head = False
    mock_model.config.quantile_list = [0.1, 0.5, 0.9]
    outputs = mocker.Mock()
    outputs.prediction_outputs = np_to_torch(np.array([[[1.23], [2.34]]], dtype=np.float32))
    outputs.quantile_outputs = np_to_torch(
        np.array([[[[1.0], [2.0]], [[1.23], [2.34]], [[1.5], [2.7]]]], dtype=np.float32)
    )
    mock_model.return_value = outputs
    mock_loader = mocker.patch(
        "timecopilot.models.foundation.ttm_r3._load_ttm_r3_class",
        return_value=mocker.Mock(from_pretrained=mocker.Mock(return_value=mock_model)),
    )

    ds = pd.date_range("2024-01-01", periods=20, freq="W")
    df = pd.DataFrame({"unique_id": "u1", "ds": ds, "y": np.arange(20)})

    fcst = TTMR3().forecast(df=df, h=1, freq="W", quantiles=[0.1, 0.5, 0.9])

    assert mock_loader.called
    assert isinstance(fcst, pd.DataFrame)
    assert len(fcst) == 1
    assert "TTM-R3" in fcst.columns
    assert "TTM-R3-q-10" in fcst.columns
    assert "TTM-R3-q-50" in fcst.columns
    assert "TTM-R3-q-90" in fcst.columns


def np_to_torch(arr):
    import torch

    return torch.tensor(arr)
