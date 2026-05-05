#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aposteriori_cd.experiments import (  # noqa: E402
    DiffusivePSystemLevelResult,
    LinearAdvectionDiffusionLevelResult,
    ViscousBurgersLevelResult,
    run_diffusive_p_system_experiment,
    run_linear_advection_diffusion_experiment,
    run_viscous_burgers_experiment,
)


IDENTITY_FIELDS = ("experiment_name", "q", "epsilon", "final_time")


def main() -> None:
    args = _parse_args()
    spec = _model_spec(args.model)
    result = spec["run"](
        polynomial_degree_q=args.polynomial_degree_q,
        mesh_counts=args.mesh_counts,
        epsilon=args.epsilon,
        final_time=args.final_time,
        cfl_number=args.cfl_number,
        checkpoint_directory=args.checkpoint_directory,
    )
    summary = _summary(
        experiment_name=spec["experiment_name"],
        q=args.polynomial_degree_q,
        epsilon=args.epsilon,
        final_time=args.final_time,
        result=result,
    )
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        _write_text(args.output_json, text)
    if args.output_csv is not None:
        _write_csv(
            args.output_csv,
            summary,
            level_type=spec["level_type"],
        )
    print(text, end="")


def _summary(
    *,
    experiment_name: str,
    q: int,
    epsilon: float,
    final_time: float,
    result: Any,
) -> dict[str, Any]:
    return {
        "experiment_name": experiment_name,
        "q": int(q),
        "epsilon": float(epsilon),
        "final_time": float(final_time),
        "mesh_counts": [int(level.mesh_count) for level in result.levels],
        "levels": [asdict(level) for level in result.levels],
        "eoc": {name: list(values) for name, values in result.eoc.items()},
    }


def _write_csv(
    path: Path,
    summary: dict[str, Any],
    *,
    level_type: type[Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [*IDENTITY_FIELDS, *(field.name for field in fields(level_type))]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for level in summary["levels"]:
            writer.writerow(
                {
                    "experiment_name": summary["experiment_name"],
                    "q": summary["q"],
                    "epsilon": summary["epsilon"],
                    "final_time": summary["final_time"],
                    **level,
                },
            )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _model_spec(model: str) -> dict[str, Any]:
    if model == "linear":
        return {
            "experiment_name": "linear_advection_diffusion",
            "level_type": LinearAdvectionDiffusionLevelResult,
            "run": run_linear_advection_diffusion_experiment,
        }
    if model == "burgers":
        return {
            "experiment_name": "viscous_burgers",
            "level_type": ViscousBurgersLevelResult,
            "run": run_viscous_burgers_experiment,
        }
    if model == "p-system":
        return {
            "experiment_name": "diffusive_p_system",
            "level_type": DiffusivePSystemLevelResult,
            "run": run_diffusive_p_system_experiment,
        }
    raise ValueError(f"unsupported model: {model}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one paper experiment.")
    parser.add_argument(
        "--model",
        choices=("linear", "burgers", "p-system"),
        required=True,
    )
    parser.add_argument("--polynomial-degree-q", type=int, required=True)
    parser.add_argument("--mesh-counts", type=_parse_mesh_counts, required=True)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--final-time", type=float, required=True)
    parser.add_argument("--cfl-number", type=float, default=0.1)
    parser.add_argument("--checkpoint-directory")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


def _parse_mesh_counts(value: str) -> tuple[int, ...]:
    mesh_counts = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(mesh_counts) == 0:
        raise argparse.ArgumentTypeError("at least one mesh count is required")
    return mesh_counts


if __name__ == "__main__":
    main()
