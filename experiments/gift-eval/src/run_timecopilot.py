import logging
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import pandas as pd
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
    "ibm-best": ["flowstate", "patchtst-fm"],
}
TTM_MAX_PREDICTION_LENGTH = 720
TTM_FAMILY_MODELS = {"ttm", "ttm-r3"}


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
        ),
        "flowstate": lambda: FlowState(
            context_length=512,
            batch_size=32,
        ),
        "patchtst-fm": lambda: PatchTSTFM(
            context_length=batch_size,
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
    skip_completed: bool,
) -> None:
    logging.info("Running dataset=%s term=%s output=%s", dataset_name, term, output_path)
    output_csv = Path(output_path) / "all_results.csv"
    if skip_completed and output_csv.exists():
        logging.info(
            "Skipping completed dataset=%s term=%s because %s already exists",
            dataset_name,
            term,
            output_csv,
        )
        return
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
    logging.info("Prediction length; %s", gifteval.dataset.prediction_length)
    ttm_family_models = [name for name in model_names if name in TTM_FAMILY_MODELS]
    if ttm_family_models and gifteval.dataset.prediction_length > TTM_MAX_PREDICTION_LENGTH:
        if model is not None and set(model).issubset(TTM_FAMILY_MODELS):
            raise ValueError(
                "TTM/TTM-R3 does not support this GIFT-Eval horizon in the current Granite "
                "model-selection path. "
                f"Requested prediction_length={gifteval.dataset.prediction_length}, "
                f"but the largest available Granite TTM checkpoint supports "
                f"prediction_length<={TTM_MAX_PREDICTION_LENGTH}. "
                "Use another model, or implement rolling/recursive forecasting outside the model."
            )
        model_names = [name for name in model_names if name not in TTM_FAMILY_MODELS]
        logging.warning(
            "Skipping %s for dataset=%s term=%s because prediction_length=%s exceeds "
            "the supported maximum of %s for the current Granite TTM selection path.",
            ttm_family_models,
            dataset_name,
            term,
            gifteval.dataset.prediction_length,
            TTM_MAX_PREDICTION_LENGTH,
        )
        if not model_names:
            raise ValueError(
                "After removing unsupported TTM, no models remain to run for this dataset."
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
    skip_completed: Annotated[
        bool,
        typer.Option(
            "--skip-completed",
            help="Skip dataset/term outputs whose all_results.csv already exists.",
        ),
    ] = False,
):
    if all_datasets:
        runs = DATASETS_WITH_TERMS[:limit] if limit is not None else DATASETS_WITH_TERMS
        logging.info("Running %s dataset/term combinations", len(runs))
        failures: list[dict[str, str]] = []
        for dataset_name_i, term_i in runs:
            output_dir = str(Path(output_path) / dataset_name_i / term_i)
            try:
                _run_single_dataset(
                    dataset_name=dataset_name_i,
                    term=term_i,
                    output_path=output_dir,
                    storage_path=storage_path,
                    model_preset=model_preset,
                    model=model,
                    skip_completed=skip_completed,
                )
            except Exception as exc:
                logging.exception(
                    "Failed dataset=%s term=%s output=%s",
                    dataset_name_i,
                    term_i,
                    output_dir,
                )
                failures.append(
                    {
                        "dataset_name": dataset_name_i,
                        "term": term_i,
                        "output_path": output_dir,
                        "model_preset": model_preset,
                        "models": ",".join(model) if model is not None else ",".join(
                            MODEL_PRESETS.get(model_preset, [])
                        ),
                        "error_type": exc.__class__.__name__,
                        "error_message": str(exc),
                    }
                )
        if failures:
            failures_path = Path(output_path) / "failures.csv"
            failures_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(failures).to_csv(failures_path, index=False)
            logging.warning(
                "Completed with %s failed dataset/term combinations. Failure summary written to %s",
                len(failures),
                failures_path,
            )
        else:
            logging.info("Completed all dataset/term combinations successfully.")
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
        skip_completed=skip_completed,
    )


if __name__ == "__main__":
    app()
