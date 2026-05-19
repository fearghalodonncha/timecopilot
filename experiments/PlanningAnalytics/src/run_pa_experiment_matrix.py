"""Run the Planning Analytics selected-series experiment matrix."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


DATASET_DIR = Path("experiments/PlanningAnalytics/PA Anonymized Dataset")
SELECTED_SERIES_DIR = DATASET_DIR / "selected_series"
ACTIVE_SERIES_DIR = DATASET_DIR / "Evaluation Series" / "PA_ACTIVE_EVALUATION_SERIES"
DEFAULT_MODELS = [
    "auto-ets",
    "timecopilot-default",
    "timecopilot+ibm",
    "ibm-granite-tsfm",
    "ibm-research-tsfm",
    "patchtst-fm",
]
DEFAULT_COUNTRIES = ["Country 1", "Country 2", "Country 3"]
DEFAULT_GRANULARITIES = [
    "BN_Leaf_PL_Leaf",
    "BN_Leaf_PL_Total",
    "BN_Lvl1_PL_Total",
]


@dataclass(frozen=True)
class Experiment:
    granularity: str
    label: str
    selection_path: Path


def _experiments() -> list[Experiment]:
    return [
        Experiment("BN_Leaf_PL_Leaf", "most-active", SELECTED_SERIES_DIR / "clean_series_leaf_48_12.csv"),
        Experiment("BN_Leaf_PL_Leaf", "active", ACTIVE_SERIES_DIR),
        Experiment("BN_Leaf_PL_Total", "most-active", SELECTED_SERIES_DIR / "clean_series_leaf_total_48_12.csv"),
        Experiment("BN_Leaf_PL_Total", "active", ACTIVE_SERIES_DIR),
        Experiment("BN_Lvl1_PL_Total", "most-active", SELECTED_SERIES_DIR / "clean_series_lvl1_total_48_12.csv"),
        Experiment("BN_Lvl1_PL_Total", "active", ACTIVE_SERIES_DIR),
    ]


def _build_command(
    *,
    python: str,
    runner: Path,
    country: str,
    experiment: Experiment,
    models: list[str],
    transform: str,
    continue_on_error: bool,
    extra_args: list[str],
) -> list[str]:
    command = [
        python,
        str(runner),
        "--country",
        country,
        "--granularity",
        experiment.granularity,
        "--transform",
        transform,
        "--cohort-label",
        experiment.label,
        "--series-selection",
        str(experiment.selection_path),
    ]
    for model in models:
        command.extend(["--model", model])
    if continue_on_error:
        command.append("--continue-on-error")
    command.extend(extra_args)
    return command


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the selected-series Planning Analytics experiments.",
    )
    parser.add_argument("--python", default="experiments/gift-eval/.venv/bin/python3")
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path("experiments/PlanningAnalytics/src/run_timecopilot.py"),
    )
    parser.add_argument("--country", action="append", choices=DEFAULT_COUNTRIES)
    parser.add_argument("--granularity", action="append", choices=DEFAULT_GRANULARITIES)
    parser.add_argument("--model", action="append", default=None)
    parser.add_argument(
        "--transform",
        default="none",
        choices=["none", "abs-scale", "neg-abs-scale", "pos-neg-components"],
    )
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    countries = args.country or DEFAULT_COUNTRIES
    granularities = set(args.granularity or DEFAULT_GRANULARITIES)
    models = args.model or DEFAULT_MODELS
    extra_args = args.extra_args
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    experiments = [
        experiment
        for experiment in _experiments()
        if experiment.granularity in granularities
    ]
    commands = [
        _build_command(
            python=args.python,
            runner=args.runner,
            country=country,
            experiment=experiment,
            models=models,
            transform=args.transform,
            continue_on_error=args.continue_on_error,
            extra_args=extra_args,
        )
        for country in countries
        for experiment in experiments
    ]

    for index, command in enumerate(commands, start=1):
        print(f"\n[{index}/{len(commands)}] {shlex.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
