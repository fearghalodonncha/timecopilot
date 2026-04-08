import logging
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer

from timecopilot.gift_eval.eval import GIFTEval
from timecopilot.gift_eval.gluonts_predictor import GluonTSPredictor
from timecopilot.gift_eval.ttm_forecaster import TTMGiftEvalForecaster
from timecopilot.gift_eval.utils import DATASETS_WITH_TERMS
from timecopilot.models.ensembles.median import MedianEnsemble
from timecopilot.models.foundation.chronos import Chronos
from timecopilot.models.foundation.flowstate import FlowState
from timecopilot.models.foundation.patchtst_fm import PatchTSTFM
from timecopilot.models.foundation.ttm import TTM
from timecopilot.models.foundation.ttm_r3 import TTMR3
from timecopilot.models.foundation.timesfm import TimesFM
from timecopilot.models.foundation.tirex import TiRex
from timecopilot.models.utils.forecaster import Forecaster

logging.basicConfig(level=logging.INFO)


app = typer.Typer()

MODEL_PRESETS = {
    "default": ["chronos", "timesfm", "tirex"],
    "ibm": ["ttm", "flowstate", "patchtst-fm"],
    "ibm-r3": ["ttm-r3", "flowstate", "patchtst-fm"],
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
        "ttm-r3": lambda: TTMR3(
            batch_size=batch_size,
            model_revision="512-30-dec-90-lite-r3",
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


def _run_single_dataset(
    dataset_name: str,
    term: str,
    output_path: str,
    storage_path: str,
    model_preset: str,
    model: list[str] | None,
) -> None:
    logging.info("Running dataset=%s term=%s output=%s", dataset_name, term, output_path)
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
    logging.info("Using models: %s", model_names)
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
        batch_size=1_024,
    )
    gifteval.evaluate_predictor(predictor, batch_size=512)


@app.command()
def run_timecopilot(
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
    dataset_name: Annotated[
        str | None,
        typer.Option(help="The name of the dataset to evaluate"),
    ] = None,
    term: Annotated[
        str | None,
        typer.Option(help="The term to evaluate"),
    ] = None,
    all_datasets: Annotated[
        bool,
        typer.Option(
            "--all-datasets",
            help="Run across all GIFT-Eval dataset/term combinations.",
        ),
    ] = False,
    limit: Annotated[
        int | None,
        typer.Option(
            help="Optional limit on the number of dataset/term combinations to run.",
        ),
    ] = None,
):
    if all_datasets:
        runs = DATASETS_WITH_TERMS[:limit] if limit is not None else DATASETS_WITH_TERMS
        logging.info("Running %s dataset/term combinations", len(runs))
        for dataset_name_i, term_i in runs:
            output_dir = str(Path(output_path) / dataset_name_i / term_i)
            _run_single_dataset(
                dataset_name=dataset_name_i,
                term=term_i,
                output_path=output_dir,
                storage_path=storage_path,
                model_preset=model_preset,
                model=model,
            )
        return

    if dataset_name is None or term is None:
        raise ValueError(
            "Provide both `--dataset-name` and `--term`, or use `--all-datasets`."
        )

    _run_single_dataset(
        dataset_name=dataset_name,
        term=term,
        output_path=output_path,
        storage_path=storage_path,
        model_preset=model_preset,
        model=model,
    )


if __name__ == "__main__":
    app()
