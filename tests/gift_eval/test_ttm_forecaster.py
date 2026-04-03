import numpy as np
import pandas as pd

from timecopilot.gift_eval.ttm_forecaster import TTMGiftEvalForecaster


class DummyTTMForecaster:
    alias = "TTM"

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level=None,
        quantiles=None,
    ) -> pd.DataFrame:
        unique_ids = df["unique_id"].unique()
        rows = []
        for uid in unique_ids:
            base_time = df.loc[df["unique_id"] == uid, "ds"].max()
            for step in range(h):
                rows.append(
                    {
                        "unique_id": uid,
                        "ds": base_time + pd.Timedelta(days=step + 1),
                        self.alias: 10.0 + step,
                    }
                )
        return pd.DataFrame(rows)

    def cross_validation(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        n_windows: int = 1,
        level=None,
        quantiles=None,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "unique_id": ["series_a", "series_a"],
                "cutoff": [pd.Timestamp("2020-01-08"), pd.Timestamp("2020-01-08")],
                "ds": [pd.Timestamp("2020-01-09"), pd.Timestamp("2020-01-10")],
                "y": [12.0, 14.0],
                self.alias: [10.0, 11.0],
            }
        )


def test_ttm_gift_eval_forecaster_builds_quantiles():
    calibration_dataset = [
        {
            "item_id": "series_a",
            "start": pd.Period("2020-01-01", freq="D"),
            "target": np.arange(10, dtype=np.float32),
        }
    ]
    adapter = TTMGiftEvalForecaster(
        forecaster=DummyTTMForecaster(),
        calibration_dataset=calibration_dataset,
        prediction_length=2,
        freq="D",
    )
    input_df = pd.DataFrame(
        {
            "unique_id": ["series_a-window"] * 6,
            "source_item_id": ["series_a"] * 6,
            "ds": pd.date_range("2020-01-01", periods=6, freq="D"),
            "y": np.arange(6, dtype=np.float32),
        }
    )

    fcst_df = adapter.forecast(
        df=input_df,
        h=2,
        freq="D",
        quantiles=[0.1, 0.5, 0.9],
    )

    assert "TTM" in fcst_df.columns
    assert "TTM-q-10" in fcst_df.columns
    assert "TTM-q-50" in fcst_df.columns
    assert "TTM-q-90" in fcst_df.columns
    np.testing.assert_allclose(fcst_df["TTM"].to_numpy(), fcst_df["TTM-q-50"].to_numpy())
    assert (fcst_df["TTM-q-10"] < fcst_df["TTM"]).all()
    assert (fcst_df["TTM"] < fcst_df["TTM-q-90"]).all()
