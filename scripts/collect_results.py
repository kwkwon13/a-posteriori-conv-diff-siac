#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import fields
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
)


IDENTITY_FIELDS = ("experiment_name", "q", "epsilon", "final_time")
EXPERIMENT_ORDER = (
    "linear_advection_diffusion",
    "viscous_burgers",
    "diffusive_p_system",
)
LEVEL_TYPES = {
    "linear_advection_diffusion": LinearAdvectionDiffusionLevelResult,
    "viscous_burgers": ViscousBurgersLevelResult,
    "diffusive_p_system": DiffusivePSystemLevelResult,
}
EOC_FIELDS = {
    "linear_advection_diffusion": (
        "hat_u_h_t_l2_l2_error",
        "E_rec",
        "residual_1_l1_l2",
        "E_r2",
        "E_est",
    ),
    "viscous_burgers": (
        "hat_u_h_t_l2_l2_error",
        "E_rec",
        "residual_1_l1_l2",
        "E_r2",
        "E_est",
    ),
    "diffusive_p_system": (
        "hat_u_h_t_l2_l2_error",
        "E_rec",
        "residual_1_l1_l2",
        "W_second_r_u_l1_l2",
        "r_v_1_l1_l2",
        "E_r2",
        "E_est",
    ),
}


def main() -> None:
    args = _parse_args()
    experiments, resolved_source = _load_experiments(args.results_directory, args.source)
    if not experiments:
        raise SystemExit(f"no {resolved_source} records found in {args.results_directory}")
    summary = {
        "source": resolved_source,
        "experiment_count": len(experiments),
        "experiments": [
            _summarize_experiment(key, levels)
            for key, levels in sorted(experiments.items())
        ],
    }
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    output_json = args.output_json or args.results_directory / "collected.json"
    output_csv = args.output_csv or args.results_directory / "collected.csv"
    _write_text(output_json, text)
    _write_csv(output_csv, summary)
    print(text, end="")


def _load_experiments(
    results_directory: Path,
    source: str,
) -> tuple[dict[tuple[str, int, float, float], dict[int, dict[str, Any]]], str]:
    if source in ("auto", "json"):
        experiments = _load_experiment_json_data(results_directory / "json")
        if experiments or source == "json":
            return experiments, "json"
    experiments = _load_checkpoint_data(results_directory / "checkpoints")
    return experiments, "checkpoints"


def _load_experiment_json_data(
    json_directory: Path,
) -> dict[tuple[str, int, float, float], dict[int, dict[str, Any]]]:
    experiments: dict[tuple[str, int, float, float], dict[int, dict[str, Any]]] = {}
    for path in sorted(json_directory.glob("*.json")):
        experiment_data = json.loads(path.read_text(encoding="utf-8"))
        experiment_name = str(experiment_data["experiment_name"])
        key = (
            experiment_name,
            int(experiment_data["q"]),
            float(experiment_data["epsilon"]),
            float(experiment_data["final_time"]),
        )
        levels = experiments.setdefault(key, {})
        for level in experiment_data["levels"]:
            levels[int(level["mesh_count"])] = dict(level)
    return experiments


def _load_checkpoint_data(
    checkpoint_directory: Path,
) -> dict[tuple[str, int, float, float], dict[int, dict[str, Any]]]:
    experiments: dict[tuple[str, int, float, float], dict[int, dict[str, Any]]] = {}
    for path in sorted(checkpoint_directory.glob("*.json")):
        checkpoint_data = json.loads(path.read_text(encoding="utf-8"))
        identity = checkpoint_data["experiment_identity"]
        experiment_name = str(identity["experiment_name"])
        key = (
            experiment_name,
            int(identity["polynomial_degree_q"]),
            float(identity["epsilon"]),
            float(identity["final_time"]),
        )
        experiments.setdefault(key, {})[int(checkpoint_data["mesh_count"])] = dict(
            checkpoint_data["level"],
        )
    return experiments


def _summarize_experiment(
    key: tuple[str, int, float, float],
    levels_by_mesh_count: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    experiment_name, q, epsilon, final_time = key
    levels = [
        levels_by_mesh_count[mesh_count]
        for mesh_count in sorted(levels_by_mesh_count)
    ]
    return {
        "experiment_name": experiment_name,
        "q": q,
        "epsilon": epsilon,
        "final_time": final_time,
        "mesh_counts": [int(level["mesh_count"]) for level in levels],
        "levels": levels,
        "eoc": _compute_eoc(experiment_name, levels),
    }


def _compute_eoc(
    experiment_name: str,
    levels: list[dict[str, Any]],
) -> dict[str, list[float]]:
    if len(levels) < 2:
        return {}
    eoc: dict[str, list[float]] = {}
    for name in EOC_FIELDS[experiment_name]:
        values = [level.get(name) for level in levels]
        h_values = [level.get("h") for level in levels]
        if all(_is_positive_number(value) for value in values) and all(
            _is_positive_number(value) for value in h_values
        ):
            eoc[name] = [
                math.log(float(left) / float(right))
                / math.log(float(left_h) / float(right_h))
                for left, right, left_h, right_h in zip(
                    values[:-1],
                    values[1:],
                    h_values[:-1],
                    h_values[1:],
                )
            ]
    return eoc


def _write_csv(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for experiment in summary["experiments"]:
        for level in experiment["levels"]:
            rows.append(
                {
                    "experiment_name": experiment["experiment_name"],
                    "q": experiment["q"],
                    "epsilon": experiment["epsilon"],
                    "final_time": experiment["final_time"],
                    **level,
                },
            )
    fieldnames = _csv_fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _csv_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    present_experiments = {str(row["experiment_name"]) for row in rows}
    fieldnames = list(IDENTITY_FIELDS)
    seen = set(fieldnames)
    for experiment_name in EXPERIMENT_ORDER:
        if experiment_name not in present_experiments:
            continue
        for field in fields(LEVEL_TYPES[experiment_name]):
            if field.name not in seen:
                fieldnames.append(field.name)
                seen.add(field.name)
    return fieldnames


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _is_positive_number(value: Any) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value)) and float(value) > 0.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect paper reproduction results.")
    parser.add_argument("--results-directory", type=Path, default=Path("results"))
    parser.add_argument(
        "--source",
        choices=("auto", "json", "checkpoints"),
        default="auto",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()
