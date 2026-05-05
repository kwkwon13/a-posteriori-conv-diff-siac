#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
RUN_EXPERIMENT_SCRIPT = SCRIPT_DIR / "run_experiment.py"

DEFAULT_MODELS = ("linear", "burgers", "p-system")
DEFAULT_POLYNOMIAL_DEGREES = (1, 2)
DEFAULT_EPSILONS = (0.0, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1)
DEFAULT_MESH_COUNTS = (16, 32, 64, 128)
DEFAULT_CFL_ADV = 0.1
DEFAULT_FINAL_TIMES = {
    "linear": 1.0,
    "burgers": 0.1,
    "p-system": 0.05,
}
MODEL_EXPERIMENT_NAMES = {
    "linear": "linear_advection_diffusion",
    "burgers": "viscous_burgers",
    "p-system": "diffusive_p_system",
}
MODEL_LAMBDA_MAX = {
    "linear": 1.0,
    "burgers": 2.2,
    "p-system": math.sqrt(2.0 / 0.9**3),
}


@dataclass(frozen=True, slots=True)
class PaperExperimentCase:
    model: str
    experiment_name: str
    polynomial_degree_q: int
    epsilon: float
    mesh_counts: tuple[int, ...]
    final_time: float
    cfl_adv: float
    lambda_max: float
    cfl_number: float


def main() -> None:
    args = _parse_args()
    results_directory = args.results_directory
    json_directory = results_directory / "json"
    csv_directory = results_directory / "csv"
    checkpoint_directory = results_directory / "checkpoints"
    cases = _build_cases(
        models=args.models,
        polynomial_degrees=args.polynomial_degrees,
        epsilons=args.epsilons,
        mesh_counts=args.mesh_counts,
        final_times={
            "linear": args.linear_final_time,
            "burgers": args.burgers_final_time,
            "p-system": args.p_system_final_time,
        },
        cfl_adv=args.cfl_adv,
    )
    experiment_set = {
        "models": list(args.models),
        "polynomial_degrees": list(args.polynomial_degrees),
        "epsilons": list(args.epsilons),
        "mesh_counts": list(args.mesh_counts),
        "final_times": {
            "linear": args.linear_final_time,
            "burgers": args.burgers_final_time,
            "p-system": args.p_system_final_time,
        },
        "cfl_adv": args.cfl_adv,
        "lambda_max": MODEL_LAMBDA_MAX,
        "case_count": len(cases),
        "cases": [asdict(case) for case in cases],
    }
    _write_json(results_directory / "experiment_set.json", experiment_set)

    completed_cases = []
    if not args.preview:
        for case in cases:
            completed_cases.append(
                _run_case(
                    case,
                    json_directory=json_directory,
                    csv_directory=csv_directory,
                    checkpoint_directory=checkpoint_directory,
                ),
            )

    summary = {
        "preview": bool(args.preview),
        "case_count": len(cases),
        "completed_case_count": len(completed_cases),
        "cases": completed_cases if completed_cases else [asdict(case) for case in cases],
    }
    _write_json(results_directory / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True) + "\n", end="")


def _build_cases(
    *,
    models: tuple[str, ...],
    polynomial_degrees: tuple[int, ...],
    epsilons: tuple[float, ...],
    mesh_counts: tuple[int, ...],
    final_times: dict[str, float],
    cfl_adv: float,
) -> tuple[PaperExperimentCase, ...]:
    cases: list[PaperExperimentCase] = []
    for model in models:
        for q in polynomial_degrees:
            for epsilon in epsilons:
                lambda_max = MODEL_LAMBDA_MAX[model]
                cfl_number = float(cfl_adv) / ((2 * int(q) + 1) * lambda_max)
                cases.append(
                    PaperExperimentCase(
                        model=model,
                        experiment_name=MODEL_EXPERIMENT_NAMES[model],
                        polynomial_degree_q=int(q),
                        epsilon=float(epsilon),
                        mesh_counts=tuple(int(mesh_count) for mesh_count in mesh_counts),
                        final_time=float(final_times[model]),
                        cfl_adv=float(cfl_adv),
                        lambda_max=float(lambda_max),
                        cfl_number=float(cfl_number),
                    ),
                )
    return tuple(cases)


def _run_case(
    case: PaperExperimentCase,
    *,
    json_directory: Path,
    csv_directory: Path,
    checkpoint_directory: Path,
) -> dict[str, Any]:
    case_name = _case_name(case.model, case.polynomial_degree_q, case.epsilon)
    arguments = [
        sys.executable,
        str(RUN_EXPERIMENT_SCRIPT),
        "--model",
        case.model,
        "--polynomial-degree-q",
        str(case.polynomial_degree_q),
        "--mesh-counts",
        ",".join(str(mesh_count) for mesh_count in case.mesh_counts),
        "--epsilon",
        repr(case.epsilon),
        "--final-time",
        repr(case.final_time),
        "--cfl-number",
        repr(case.cfl_number),
        "--checkpoint-directory",
        str(checkpoint_directory),
        "--output-json",
        str(json_directory / f"{case_name}.json"),
        "--output-csv",
        str(csv_directory / f"{case_name}.csv"),
    ]
    completed = subprocess.run(
        arguments,
        check=True,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    experiment_summary = json.loads(completed.stdout)
    return {
        **asdict(case),
        "mesh_counts": experiment_summary["mesh_counts"],
    }


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _case_name(model: str, q: int, epsilon: float) -> str:
    return f"{model.replace('-', '_')}_q{int(q)}_eps{_number_token(epsilon)}"


def _number_token(value: float) -> str:
    return f"{float(value):.0e}".replace("+", "").replace("-", "m")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all paper experiments.")
    parser.add_argument("--models", type=_parse_models, default=DEFAULT_MODELS)
    parser.add_argument(
        "--polynomial-degrees",
        type=_parse_ints,
        default=DEFAULT_POLYNOMIAL_DEGREES,
    )
    parser.add_argument("--epsilons", type=_parse_floats, default=DEFAULT_EPSILONS)
    parser.add_argument("--mesh-counts", type=_parse_ints, default=DEFAULT_MESH_COUNTS)
    parser.add_argument("--cfl-adv", type=float, default=DEFAULT_CFL_ADV)
    parser.add_argument("--linear-final-time", type=float, default=DEFAULT_FINAL_TIMES["linear"])
    parser.add_argument("--burgers-final-time", type=float, default=DEFAULT_FINAL_TIMES["burgers"])
    parser.add_argument(
        "--p-system-final-time",
        type=float,
        default=DEFAULT_FINAL_TIMES["p-system"],
    )
    parser.add_argument(
        "--results-directory",
        type=Path,
        default=Path("results"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--preview", action="store_true")
    return parser.parse_args()


def _parse_models(value: str | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, tuple):
        return value
    if value == "all":
        return DEFAULT_MODELS
    models = tuple(item.strip() for item in value.split(",") if item.strip())
    invalid = sorted(set(models) - set(DEFAULT_MODELS))
    if invalid:
        raise argparse.ArgumentTypeError(f"unsupported model(s): {', '.join(invalid)}")
    if not models:
        raise argparse.ArgumentTypeError("at least one model is required")
    return models


def _parse_ints(value: str | tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(value, tuple):
        return value
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one integer value is required")
    return values


def _parse_floats(value: str | tuple[float, ...]) -> tuple[float, ...]:
    if isinstance(value, tuple):
        return value
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("at least one float value is required")
    return values


if __name__ == "__main__":
    main()
