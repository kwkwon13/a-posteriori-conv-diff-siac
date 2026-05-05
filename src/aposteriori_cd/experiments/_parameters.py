from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LinearAdvectionDiffusionExperimentParameters:
    """Experiment parameters for the linear scalar paper experiment."""

    polynomial_degree_q: int
    mesh_counts: tuple[int, ...]
    epsilon: float
    final_time: float
    cfl_number: float = 0.1
    velocity: tuple[float, float] = (1.0, 0.5)
    checkpoint_directory: str | None = None

    def __post_init__(self) -> None:
        _validate_common_parameters(
            self.polynomial_degree_q,
            self.mesh_counts,
            self.epsilon,
            self.final_time,
            self.cfl_number,
        )
        if len(self.velocity) != 2:
            raise ValueError("velocity must have two components")


@dataclass(frozen=True, slots=True)
class ViscousBurgersExperimentParameters:
    """Experiment parameters for the viscous Burgers paper experiment."""

    polynomial_degree_q: int
    mesh_counts: tuple[int, ...]
    epsilon: float
    final_time: float = 0.1
    cfl_number: float = 0.1
    checkpoint_directory: str | None = None

    def __post_init__(self) -> None:
        _validate_common_parameters(
            self.polynomial_degree_q,
            self.mesh_counts,
            self.epsilon,
            self.final_time,
            self.cfl_number,
        )


@dataclass(frozen=True, slots=True)
class DiffusivePSystemExperimentParameters:
    """Experiment parameters for the diffusive p-system paper experiment."""

    polynomial_degree_q: int
    mesh_counts: tuple[int, ...]
    epsilon: float
    final_time: float = 0.05
    cfl_number: float = 0.1
    checkpoint_directory: str | None = None

    def __post_init__(self) -> None:
        _validate_common_parameters(
            self.polynomial_degree_q,
            self.mesh_counts,
            self.epsilon,
            self.final_time,
            self.cfl_number,
        )


def _validate_common_parameters(
    polynomial_degree_q: int,
    mesh_counts: tuple[int, ...],
    epsilon: float,
    final_time: float,
    cfl_number: float,
) -> None:
    if polynomial_degree_q < 0:
        raise ValueError("polynomial_degree_q must be non-negative")
    if len(mesh_counts) == 0:
        raise ValueError("mesh_counts must be non-empty")
    if any(int(mesh_count) <= 0 for mesh_count in mesh_counts):
        raise ValueError("mesh_counts must be positive")
    if epsilon < 0.0:
        raise ValueError("epsilon must be non-negative")
    if final_time <= 0.0:
        raise ValueError("final_time must be positive")
    if cfl_number <= 0.0:
        raise ValueError("cfl_number must be positive")


__all__ = [
    "DiffusivePSystemExperimentParameters",
    "LinearAdvectionDiffusionExperimentParameters",
    "ViscousBurgersExperimentParameters",
]
