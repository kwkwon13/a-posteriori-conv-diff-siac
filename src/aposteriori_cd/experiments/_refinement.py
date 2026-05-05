from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ._checkpoint import checkpoint_path, load_level_checkpoint, save_level_checkpoint


def run_refinement_levels(
    parameters: Any,
    run_level: Callable[[int], Any],
    *,
    experiment_name: str,
    level_type: type[Any],
) -> tuple[Any, ...]:
    checkpoint_directory = getattr(parameters, "checkpoint_directory", None)
    return tuple(
        _run_maybe_checkpointed_level(
            parameters,
            run_level,
            experiment_name=experiment_name,
            level_type=level_type,
            mesh_count=int(mesh_count),
            checkpoint_directory=checkpoint_directory,
        )
        for mesh_count in parameters.mesh_counts
    )


def compute_eoc(
    levels: tuple[Any, ...],
    names: tuple[str, ...],
) -> dict[str, tuple[float, ...]]:
    if len(levels) < 2:
        return {}
    h_values = np.asarray([level.h for level in levels], dtype=np.float64)
    eoc: dict[str, tuple[float, ...]] = {}
    for name in names:
        values = np.asarray(
            [getattr(level, name) for level in levels],
            dtype=np.float64,
        )
        if np.all(np.isfinite(values)) and np.all(values > 0.0):
            eoc[name] = tuple(
                float(value) for value in _eoc_from_values(h_values, values)
            )
    return eoc


def _run_maybe_checkpointed_level(
    parameters: Any,
    run_level: Callable[[int], Any],
    *,
    experiment_name: str,
    level_type: type[Any],
    mesh_count: int,
    checkpoint_directory: str | None,
) -> Any:
    if checkpoint_directory is None:
        return run_level(mesh_count)

    path = checkpoint_path(
        checkpoint_directory,
        experiment_name=experiment_name,
        parameters=parameters,
        mesh_count=mesh_count,
    )
    if path.exists():
        return load_level_checkpoint(
            path,
            experiment_name=experiment_name,
            parameters=parameters,
            mesh_count=mesh_count,
            level_type=level_type,
        )
    level = run_level(mesh_count)
    save_level_checkpoint(
        path,
        experiment_name=experiment_name,
        parameters=parameters,
        mesh_count=mesh_count,
        level=level,
    )
    return level


def _eoc_from_values(h_values: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.log(values[:-1] / values[1:]) / np.log(h_values[:-1] / h_values[1:])


__all__ = [
    "compute_eoc",
    "run_refinement_levels",
]
