from __future__ import annotations

from dataclasses import dataclass

import numpy as np


LEFT_ENDPOINT = 0
RIGHT_ENDPOINT = 1


@dataclass(slots=True)
class TimeSubintervalSolutionData:
    """Endpoint values and time derivatives on one time subinterval."""

    t_left: float
    t_right: float
    values: np.ndarray
    time_derivatives: np.ndarray


def allocate_time_subinterval_solution_data(
    state_shape: tuple[int, ...],
) -> TimeSubintervalSolutionData:
    values = np.zeros((2, *state_shape), dtype=np.float64)
    time_derivatives = np.zeros_like(values)
    return TimeSubintervalSolutionData(
        t_left=0.0,
        t_right=0.0,
        values=values,
        time_derivatives=time_derivatives,
    )


def fill_time_subinterval_solution_data(
    data: TimeSubintervalSolutionData,
    *,
    t_left: float,
    t_right: float,
    u_left: np.ndarray,
    u_right: np.ndarray,
    rhs_left: np.ndarray,
    rhs_right: np.ndarray,
) -> None:
    data.t_left = float(t_left)
    data.t_right = float(t_right)
    data.values[LEFT_ENDPOINT] = u_left
    data.values[RIGHT_ENDPOINT] = u_right
    data.time_derivatives[LEFT_ENDPOINT] = rhs_left
    data.time_derivatives[RIGHT_ENDPOINT] = rhs_right


__all__ = [
    "LEFT_ENDPOINT",
    "RIGHT_ENDPOINT",
    "TimeSubintervalSolutionData",
    "allocate_time_subinterval_solution_data",
    "fill_time_subinterval_solution_data",
]
