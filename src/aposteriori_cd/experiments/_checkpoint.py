from __future__ import annotations

import hashlib
import json
import os
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np


_EXCLUDED_PARAMETER_FIELDS = frozenset(("mesh_counts", "checkpoint_directory"))
_LevelResult = TypeVar("_LevelResult")


def checkpoint_path(
    checkpoint_directory: str,
    *,
    experiment_name: str,
    parameters: Any,
    mesh_count: int,
) -> Path:
    key = parameter_key(parameters)
    return (
        Path(checkpoint_directory)
        / f"{experiment_name}_mesh{int(mesh_count):05d}_{key}.json"
    )


def parameter_key(parameters: Any) -> str:
    parameter_data = _json_safe_dataclass(
        parameters,
        excluded_fields=_EXCLUDED_PARAMETER_FIELDS,
    )
    encoded = json.dumps(parameter_data, sort_keys=True, separators=(",", ":")).encode(
        "utf-8",
    )
    return hashlib.sha256(encoded).hexdigest()[:16]


def load_level_checkpoint(
    path: Path,
    *,
    experiment_name: str,
    parameters: Any,
    mesh_count: int,
    level_type: type[_LevelResult],
) -> _LevelResult:
    with path.open("r", encoding="utf-8") as file:
        checkpoint_data = json.load(file)

    expected_key = parameter_key(parameters)
    if checkpoint_data.get("experiment_name") != experiment_name:
        raise ValueError(f"checkpoint experiment name does not match: {path}")
    if int(checkpoint_data.get("mesh_count")) != int(mesh_count):
        raise ValueError(f"checkpoint mesh count does not match: {path}")
    if checkpoint_data.get("parameter_key") != expected_key:
        raise ValueError(f"checkpoint parameters do not match: {path}")
    return _dataclass_from_json_dict(level_type, checkpoint_data["level"])


def save_level_checkpoint(
    path: Path,
    *,
    experiment_name: str,
    parameters: Any,
    mesh_count: int,
    level: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    parameter_data = _json_safe_dataclass(
        parameters,
        excluded_fields=_EXCLUDED_PARAMETER_FIELDS,
    )
    level_data = _json_safe_dataclass(level)
    checkpoint_data = {
        "experiment_name": experiment_name,
        "mesh_count": int(mesh_count),
        "parameter_key": parameter_key(parameters),
        "experiment_identity": _experiment_identity(
            experiment_name,
            parameter_data,
            level_data,
            mesh_count,
        ),
        "parameters": parameter_data,
        "level": level_data,
    }
    staged_checkpoint_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with staged_checkpoint_path.open("w", encoding="utf-8") as file:
        json.dump(checkpoint_data, file, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(staged_checkpoint_path, path)


def _experiment_identity(
    experiment_name: str,
    parameters: dict[str, Any],
    level: dict[str, Any],
    mesh_count: int,
) -> dict[str, Any]:
    return {
        "experiment_name": experiment_name,
        "mesh_count": int(mesh_count),
        "polynomial_degree_q": parameters.get("polynomial_degree_q"),
        "epsilon": parameters.get("epsilon"),
        "final_time": parameters.get("final_time"),
        "cfl_number": parameters.get("cfl_number"),
        "velocity": parameters.get("velocity"),
        "tableau_name": level.get("tableau_name"),
        "temporal_reconstruction_type": level.get("temporal_reconstruction_type"),
        "preconditioner": level.get("preconditioner"),
    }


def _json_safe_dataclass(
    value: Any,
    *,
    excluded_fields: frozenset[str] = frozenset(),
) -> Any:
    if is_dataclass(value):
        return {
            field.name: _json_safe_dataclass(getattr(value, field.name))
            for field in fields(value)
            if field.name not in excluded_fields
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple | list):
        return [_json_safe_dataclass(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _json_safe_dataclass(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    return value


def _dataclass_from_json_dict(
    dataclass_type: type[_LevelResult],
    data: dict[str, Any],
) -> _LevelResult:
    values = {field.name: data[field.name] for field in fields(dataclass_type)}
    return dataclass_type(**values)


__all__ = [
    "checkpoint_path",
    "load_level_checkpoint",
    "parameter_key",
    "save_level_checkpoint",
]
