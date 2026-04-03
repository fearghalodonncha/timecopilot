import logging
from collections.abc import Callable
from typing import Annotated

import typer

from timecopilot.gift_eval.eval import GIFTEval
from timecopilot.gift_eval.gluonts_predictor import GluonTSPredictor
from timecopilot.gift_eval.ttm_forecaster import TTMGiftEvalForecaster
from timecopilot.models.ensembles.median import MedianEnsemble
from timecopilot.models.foundation.chronos import Chronos
from timecopilot.models.foundation.flowstate import FlowState
from timecopilot.models.foundation.patchtst_fm import PatchTSTFM
from timecopilot.models.foundation.ttm import TTM
from timecopilot.models.foundation.timesfm import TimesFM
from timecopilot.models.foundation.tirex import TiRex
from timecopilot.models.utils.forecaster import Forecaster

logging.basicConfig(level=logging.INFO)


app = typer.Typer()

MODEL_PRESETS = {
    "default": ["chronos", "timesfm", "tirex"],
    "ibm": ["ttm", "flowstate", "patchtst-fm"]
}


def _build_models(
    model_names: list[str],
    batch_size: int,
    gift_eval: GIFTEval,
) -> list[Forecaster]:
    registry: dict[str, Callable[[], Forecaster]] = {
        "chronos": lambda: Chronos(
            repo_id="amazon/chronos-2",
            batch_size=batch_size,
        ),
        "timesfm": lambda: TimesFM(
            repo_id="google/timesfm-2.5-200m-pytorch",
            batch_size=batch_size,
        ),
        "tirex": lambda: TiRex(
            batch_size=batch_size,
        ),
        "ttm": lambda: TTMGiftEvalForecaster(
            forecaster=TTM(
                batch_size=batch_size,
            ),
            calibration_dataset=gift_eval.dataset.validation_dataset,
            prediction_length=gift_eval.dataset.prediction_length,
            freq=gift_eval.dataset.freq,
        ),
        "flowstate": lambda: FlowState(
            context_length=512,
            batch_size=32,
        ),
        "patchtst-fm": lambda: PatchTSTFM(
            context_length=512,
            batch_size=16,
        ),
    }
    unknown = sorted(set(model_names) - set(registry))
    if unknown:
        raise ValueError(
            f"Unknown models: {unknown}. Valid choices are {sorted(registry)}."
        )
    return [registry[name]() for name in model_names]


@app.command()
def run_timecopilot(
    dataset_name: Annotated[
        str,
        typer.Option(help="The name of the dataset to evaluate"),
    ],
    term: Annotated[
        str,
        typer.Option(help="The term to evaluate"),
    ],
    output_path: Annotated[
        str,
        typer.Option(help="The directory to save the results"),
    ],
    storage_path: Annotated[
        str,
        typer.Option(help="The directory were the GIFT data is stored"),
    ],
    model_preset: Annotated[
        str,
        typer.Option(
            help="Named model bundle to run. Use `default` for the current ensemble or `ibm` for IBM-only models.",
        ),
    ] = "default",
    model: Annotated[
        list[str] | None,
        typer.Option(
            "--model",
            help="Explicit model selection. Repeat the flag, e.g. `--model ttm --model flowstate`.",
        ),
    ] = None,
):
    logging.info(f"Running {dataset_name} {term} {output_path}")
    # Keep the model batch size larger than the predictor batch here so the
    # foundation-model backends see the full request in one pass.
    batch_size = 512
    gifteval = GIFTEval(
        dataset_name=dataset_name,
        term=term,
        output_path=output_path,
        storage_path=storage_path,
    )
    model_names = model or MODEL_PRESETS.get(model_preset)
    if model_names is None:
        raise ValueError(
            f"Unknown model preset `{model_preset}`. "
            f"Valid presets are {sorted(MODEL_PRESETS)}."
        )
    models = _build_models(model_names, batch_size=batch_size, gift_eval=gifteval)
    forecaster = (
        models[0]
        if len(models) == 1
        else MedianEnsemble(
            models=models,
            alias="TimeCopilot" if model_preset == "default" else "TimeCopilot-IBM",
        )
    )
    predictor = GluonTSPredictor(
        forecaster=forecaster,
        max_length=4_096,
        # data batch size
        batch_size=1_024,
    )
    gifteval.evaluate_predictor(predictor, batch_size=512)


if __name__ == "__main__":
    app()
