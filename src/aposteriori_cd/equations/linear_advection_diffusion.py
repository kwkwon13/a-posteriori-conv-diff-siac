from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._common import TWO_PI


@dataclass(frozen=True, slots=True)
class LinearAdvectionDiffusion2D:
    """Linear scalar advection-diffusion example from the paper experiments."""

    epsilon: float
    velocity: tuple[float, float] = (1.0, 0.5)
    final_time: float = 1.0
    num_components: int = field(default=1, init=False)
    source_term_is_zero: bool = field(default=True, init=False)
    diffusion_components: tuple[int, ...] = field(default=(0,), init=False)

    def __post_init__(self) -> None:
        if self.epsilon < 0.0:
            raise ValueError("epsilon must be non-negative")
        if len(self.velocity) != 2:
            raise ValueError("velocity must have two components")
        if self.final_time <= 0.0:
            raise ValueError("final_time must be positive")

    def flux(self, u: np.ndarray) -> np.ndarray:
        out = np.empty((2, 1), dtype=np.float64)
        out[0, 0] = self.velocity[0] * u[0]
        out[1, 0] = self.velocity[1] * u[0]
        return out

    def max_wavespeed(self, u: np.ndarray, normal: np.ndarray) -> float:
        return abs(normal[0] * self.velocity[0] + normal[1] * self.velocity[1])

    def exact_solution(self, t: float, x: np.ndarray) -> np.ndarray:
        velocity_x, velocity_y = self.velocity
        decay = np.exp(-8.0 * self.epsilon * np.pi * np.pi * t)
        phase_x = TWO_PI * (x[0] - velocity_x * t)
        phase_y = TWO_PI * (x[1] - velocity_y * t)
        return np.asarray([decay * np.sin(phase_x) * np.cos(phase_y)])

    def exact_gradient(self, t: float, x: np.ndarray) -> np.ndarray:
        velocity_x, velocity_y = self.velocity
        decay = np.exp(-8.0 * self.epsilon * np.pi * np.pi * t)
        phase_x = TWO_PI * (x[0] - velocity_x * t)
        phase_y = TWO_PI * (x[1] - velocity_y * t)
        out = np.empty((2, 1), dtype=np.float64)
        out[0, 0] = decay * TWO_PI * np.cos(phase_x) * np.cos(phase_y)
        out[1, 0] = -decay * TWO_PI * np.sin(phase_x) * np.sin(phase_y)
        return out

    def exact_time_derivative(self, t: float, x: np.ndarray) -> np.ndarray:
        velocity_x, velocity_y = self.velocity
        value = self.exact_solution(t, x)[0]
        gradient = self.exact_gradient(t, x)
        return np.asarray(
            [
                -8.0 * self.epsilon * np.pi * np.pi * value
                - velocity_x * gradient[0, 0]
                - velocity_y * gradient[1, 0],
            ],
        )

    def exact_flux_divergence(self, t: float, x: np.ndarray) -> np.ndarray:
        gradient = self.exact_gradient(t, x)
        return np.asarray(
            [self.velocity[0] * gradient[0, 0] + self.velocity[1] * gradient[1, 0]],
        )

    def exact_diffusion_divergence(self, t: float, x: np.ndarray) -> np.ndarray:
        value = self.exact_solution(t, x)[0]
        return np.asarray([-8.0 * np.pi * np.pi * value])

    def source(self, t: float, x: np.ndarray) -> np.ndarray:
        return np.zeros(1, dtype=np.float64)

    def source_values(self, t: float, points: np.ndarray, out: np.ndarray) -> None:
        out.fill(0.0)
