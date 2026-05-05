from __future__ import annotations

from dataclasses import dataclass, field

from numba import njit
import numpy as np

from ._common import TWO_PI


@dataclass(frozen=True, slots=True)
class ViscousBurgers2D:
    """Viscous Burgers example from the paper experiments."""

    epsilon: float
    final_time: float = 0.1
    mean_value: float = 2.0
    amplitude: float = 0.2
    phase_speed_x: float = 0.7
    phase_speed_y: float = 0.4
    num_components: int = field(default=1, init=False)
    source_term_is_zero: bool = field(default=False, init=False)
    diffusion_components: tuple[int, ...] = field(default=(0,), init=False)

    def __post_init__(self) -> None:
        if self.epsilon < 0.0:
            raise ValueError("epsilon must be non-negative")
        if self.final_time <= 0.0:
            raise ValueError("final_time must be positive")
        if self.amplitude < 0.0:
            raise ValueError("amplitude must be non-negative")

    def flux(self, u: np.ndarray) -> np.ndarray:
        value = 0.5 * u[0] * u[0]
        return np.asarray([[value], [value]], dtype=np.float64)

    def max_wavespeed(self, u: np.ndarray, normal: np.ndarray) -> float:
        return abs((normal[0] + normal[1]) * u[0])

    def exact_solution(self, t: float, x: np.ndarray) -> np.ndarray:
        value, _, _, _, _, _ = self._exact_terms(t, x)
        return np.asarray([value])

    def exact_time_derivative(self, t: float, x: np.ndarray) -> np.ndarray:
        _, dt_u, _, _, _, _ = self._exact_terms(t, x)
        return np.asarray([dt_u])

    def exact_gradient(self, t: float, x: np.ndarray) -> np.ndarray:
        _, _, u_x, u_y, _, _ = self._exact_terms(t, x)
        return np.asarray([[u_x], [u_y]], dtype=np.float64)

    def exact_flux_divergence(self, t: float, x: np.ndarray) -> np.ndarray:
        value, _, u_x, u_y, _, _ = self._exact_terms(t, x)
        return np.asarray([value * (u_x + u_y)])

    def exact_diffusion_divergence(self, t: float, x: np.ndarray) -> np.ndarray:
        _, _, _, _, u_xx, u_yy = self._exact_terms(t, x)
        return np.asarray([u_xx + u_yy])

    def source(self, t: float, x: np.ndarray) -> np.ndarray:
        value, dt_u, u_x, u_y, u_xx, u_yy = self._exact_terms(t, x)
        return np.asarray([dt_u + value * (u_x + u_y) - self.epsilon * (u_xx + u_yy)])

    def source_values(self, t: float, points: np.ndarray, out: np.ndarray) -> None:
        _viscous_burgers_source_values(
            float(t),
            points.reshape((-1, 2)),
            out.reshape((-1, 1)),
            float(self.epsilon),
            float(self.mean_value),
            float(self.amplitude),
            float(self.phase_speed_x),
            float(self.phase_speed_y),
        )

    def _exact_terms(
        self,
        t: float,
        x: np.ndarray,
    ) -> tuple[float, float, float, float, float, float]:
        phase_x = TWO_PI * (x[0] - self.phase_speed_x * t)
        phase_y = TWO_PI * (x[1] + self.phase_speed_y * t)
        sin_x = np.sin(phase_x)
        cos_x = np.cos(phase_x)
        sin_y = np.sin(phase_y)
        cos_y = np.cos(phase_y)
        oscillation = self.amplitude * sin_x * cos_y
        value = self.mean_value + oscillation
        u_x = self.amplitude * TWO_PI * cos_x * cos_y
        u_y = -self.amplitude * TWO_PI * sin_x * sin_y
        u_xx = -TWO_PI * TWO_PI * oscillation
        u_yy = -TWO_PI * TWO_PI * oscillation
        dt_u = (
            -self.amplitude * TWO_PI * self.phase_speed_x * cos_x * cos_y
            - self.amplitude * TWO_PI * self.phase_speed_y * sin_x * sin_y
        )
        return float(value), float(dt_u), float(u_x), float(u_y), float(u_xx), float(u_yy)


@njit(cache=True)
def _viscous_burgers_source_values(
    time: float,
    points: np.ndarray,
    out: np.ndarray,
    epsilon: float,
    mean_value: float,
    amplitude: float,
    phase_speed_x: float,
    phase_speed_y: float,
) -> None:
    for point_index in range(points.shape[0]):
        phase_x = TWO_PI * (points[point_index, 0] - phase_speed_x * time)
        phase_y = TWO_PI * (points[point_index, 1] + phase_speed_y * time)
        sin_x = np.sin(phase_x)
        cos_x = np.cos(phase_x)
        sin_y = np.sin(phase_y)
        cos_y = np.cos(phase_y)
        oscillation = amplitude * sin_x * cos_y
        value = mean_value + oscillation
        u_x = amplitude * TWO_PI * cos_x * cos_y
        u_y = -amplitude * TWO_PI * sin_x * sin_y
        u_xx = -TWO_PI * TWO_PI * oscillation
        u_yy = -TWO_PI * TWO_PI * oscillation
        dt_u = (
            -amplitude * TWO_PI * phase_speed_x * cos_x * cos_y
            - amplitude * TWO_PI * phase_speed_y * sin_x * sin_y
        )
        out[point_index, 0] = dt_u + value * (u_x + u_y) - epsilon * (u_xx + u_yy)
