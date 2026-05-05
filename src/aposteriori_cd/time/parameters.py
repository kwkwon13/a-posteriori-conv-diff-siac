from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .tableaux import ARK3_2_4L_2_SA, ARK5_4_8L_2_SA


H000 = "H(0,0,0)"
H100 = "H(1,0,0)"


@dataclass(frozen=True, slots=True)
class FixedTimeStepParameters:
    """Fixed-step time interval."""

    t_start: float
    t_final: float
    dt: float
    num_steps: int = field(init=False)

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.t_final < self.t_start:
            raise ValueError("t_final must be greater than or equal to t_start")
        duration = self.t_final - self.t_start
        steps_float = duration / self.dt
        num_steps = int(round(steps_float))
        if not np.isclose(num_steps * self.dt, duration, rtol=1.0e-12, atol=1.0e-14):
            raise ValueError("dt must divide the time interval")
        object.__setattr__(self, "num_steps", num_steps)


def fixed_dt(final_time: float, h: float, cfl_number: float) -> tuple[float, int]:
    if final_time <= 0.0:
        raise ValueError("final_time must be positive")
    if h <= 0.0:
        raise ValueError("h must be positive")
    if cfl_number <= 0.0:
        raise ValueError("cfl_number must be positive")
    target_dt = cfl_number * h
    num_steps = max(1, math.ceil(final_time / target_dt))
    return final_time / num_steps, num_steps


def tableau_name_for_degree(polynomial_degree_q: int) -> str:
    if polynomial_degree_q <= 1:
        return ARK3_2_4L_2_SA
    return ARK5_4_8L_2_SA


def temporal_reconstruction_type_for_degree(polynomial_degree_q: int) -> str:
    if polynomial_degree_q <= 1:
        return H000
    return H100


__all__ = [
    "H000",
    "H100",
    "FixedTimeStepParameters",
    "fixed_dt",
    "tableau_name_for_degree",
    "temporal_reconstruction_type_for_degree",
]
