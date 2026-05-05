from __future__ import annotations

from dataclasses import dataclass, field

from numba import njit
import numpy as np

from ._common import TWO_PI


@dataclass(frozen=True, slots=True)
class DiffusivePSystem2D:
    """Diffusive p-system example from the paper experiments."""

    epsilon: float
    final_time: float = 0.05
    tau_mean: float = 1.0
    tau_amplitude: float = 0.1
    velocity_1_amplitude: float = 0.2
    velocity_2_amplitude: float = 0.15
    phase_speed_x: float = 0.3
    phase_speed_y: float = 0.2
    num_components: int = field(default=3, init=False)
    source_term_is_zero: bool = field(default=False, init=False)
    diffusion_components: tuple[int, ...] = field(default=(1, 2), init=False)

    def __post_init__(self) -> None:
        if self.epsilon < 0.0:
            raise ValueError("epsilon must be non-negative")
        if self.final_time <= 0.0:
            raise ValueError("final_time must be positive")
        if self.tau_mean <= self.tau_amplitude:
            raise ValueError("tau_mean must exceed tau_amplitude")

    def flux(self, u: np.ndarray) -> np.ndarray:
        tau = u[0]
        pressure = 1.0 / (tau * tau)
        out = np.zeros((2, 3), dtype=np.float64)
        out[0, 0] = -u[1]
        out[0, 1] = pressure
        out[1, 0] = -u[2]
        out[1, 2] = pressure
        return out

    def max_wavespeed(self, u: np.ndarray, normal: np.ndarray) -> float:
        tau = u[0]
        normal_norm_squared = normal[0] * normal[0] + normal[1] * normal[1]
        return float(np.sqrt(2.0 * normal_norm_squared / (tau * tau * tau)))

    def potential(self, tau: float | np.ndarray) -> float | np.ndarray:
        tau_value = np.asarray(tau, dtype=np.float64)
        _validate_positive_tau(tau_value)
        return 1.0 / tau_value

    def potential_derivative(self, tau: float | np.ndarray) -> float | np.ndarray:
        tau_value = np.asarray(tau, dtype=np.float64)
        _validate_positive_tau(tau_value)
        return -1.0 / (tau_value * tau_value)

    def potential_second_derivative(
        self,
        tau: float | np.ndarray,
    ) -> float | np.ndarray:
        tau_value = np.asarray(tau, dtype=np.float64)
        _validate_positive_tau(tau_value)
        return 2.0 / (tau_value * tau_value * tau_value)

    def potential_third_derivative_abs(
        self,
        tau: float | np.ndarray,
    ) -> float | np.ndarray:
        tau_value = np.asarray(tau, dtype=np.float64)
        _validate_positive_tau(tau_value)
        return 6.0 / (tau_value**4)

    def relative_potential(
        self,
        tau: float | np.ndarray,
        tau_hat: float | np.ndarray,
    ) -> float | np.ndarray:
        tau_value = np.asarray(tau, dtype=np.float64)
        tau_hat_value = np.asarray(tau_hat, dtype=np.float64)
        _validate_positive_tau(tau_value)
        _validate_positive_tau(tau_hat_value)
        tau_difference = tau_value - tau_hat_value
        return tau_difference * tau_difference / (
            tau_value * tau_hat_value * tau_hat_value
        )

    def c_W(self, tau_min: float, tau_max: float) -> float:
        _validate_positive_tau(np.asarray([tau_min, tau_max], dtype=np.float64))
        return float(self.potential_second_derivative(max(tau_min, tau_max)))

    def C_W(self, tau_min: float, tau_max: float) -> float:
        _validate_positive_tau(np.asarray([tau_min, tau_max], dtype=np.float64))
        return float(self.potential_third_derivative_abs(min(tau_min, tau_max)))

    def exact_solution(self, t: float, x: np.ndarray) -> np.ndarray:
        phase_x, phase_y = self._phase(t, x)
        return np.asarray(
            [
                self.tau_mean + self.tau_amplitude * np.sin(phase_x) * np.cos(phase_y),
                self.velocity_1_amplitude * np.cos(phase_x) * np.cos(phase_y),
                -self.velocity_2_amplitude * np.sin(phase_x) * np.sin(phase_y),
            ],
        )

    def exact_time_derivative(self, t: float, x: np.ndarray) -> np.ndarray:
        phase_x, phase_y = self._phase(t, x)
        sin_x = np.sin(phase_x)
        cos_x = np.cos(phase_x)
        sin_y = np.sin(phase_y)
        cos_y = np.cos(phase_y)
        return np.asarray(
            [
                -self.tau_amplitude
                * TWO_PI
                * (
                    self.phase_speed_x * cos_x * cos_y
                    + self.phase_speed_y * sin_x * sin_y
                ),
                self.velocity_1_amplitude
                * TWO_PI
                * (
                    self.phase_speed_x * sin_x * cos_y
                    - self.phase_speed_y * cos_x * sin_y
                ),
                self.velocity_2_amplitude
                * TWO_PI
                * (
                    self.phase_speed_x * cos_x * sin_y
                    - self.phase_speed_y * sin_x * cos_y
                ),
            ],
            dtype=np.float64,
        )

    def exact_gradient(self, t: float, x: np.ndarray) -> np.ndarray:
        phase_x, phase_y = self._phase(t, x)
        sin_x = np.sin(phase_x)
        cos_x = np.cos(phase_x)
        sin_y = np.sin(phase_y)
        cos_y = np.cos(phase_y)
        return np.asarray(
            [
                [
                    self.tau_amplitude * TWO_PI * cos_x * cos_y,
                    -self.velocity_1_amplitude * TWO_PI * sin_x * cos_y,
                    -self.velocity_2_amplitude * TWO_PI * cos_x * sin_y,
                ],
                [
                    -self.tau_amplitude * TWO_PI * sin_x * sin_y,
                    -self.velocity_1_amplitude * TWO_PI * cos_x * sin_y,
                    -self.velocity_2_amplitude * TWO_PI * sin_x * cos_y,
                ],
            ],
            dtype=np.float64,
        )

    def exact_flux_divergence(self, t: float, x: np.ndarray) -> np.ndarray:
        tau = self.exact_solution(t, x)[0]
        gradient = self.exact_gradient(t, x)
        pressure_derivative = -2.0 / (tau * tau * tau)
        return np.asarray(
            [
                -gradient[0, 1] - gradient[1, 2],
                pressure_derivative * gradient[0, 0],
                pressure_derivative * gradient[1, 0],
            ],
            dtype=np.float64,
        )

    def exact_diffusion_divergence(self, t: float, x: np.ndarray) -> np.ndarray:
        value = self.exact_solution(t, x)
        return np.asarray(
            [
                0.0,
                -2.0 * TWO_PI * TWO_PI * value[1],
                -2.0 * TWO_PI * TWO_PI * value[2],
            ],
            dtype=np.float64,
        )

    def source(self, t: float, x: np.ndarray) -> np.ndarray:
        phase_x, phase_y = self._phase(t, x)
        sin_x = np.sin(phase_x)
        cos_x = np.cos(phase_x)
        sin_y = np.sin(phase_y)
        cos_y = np.cos(phase_y)
        tau = self.tau_mean + self.tau_amplitude * sin_x * cos_y
        tau_t = -self.tau_amplitude * TWO_PI * (
            self.phase_speed_x * cos_x * cos_y
            + self.phase_speed_y * sin_x * sin_y
        )
        v1_t = self.velocity_1_amplitude * TWO_PI * (
            self.phase_speed_x * sin_x * cos_y
            - self.phase_speed_y * cos_x * sin_y
        )
        v2_t = self.velocity_2_amplitude * TWO_PI * (
            self.phase_speed_x * cos_x * sin_y
            - self.phase_speed_y * sin_x * cos_y
        )
        tau_x = self.tau_amplitude * TWO_PI * cos_x * cos_y
        tau_y = -self.tau_amplitude * TWO_PI * sin_x * sin_y
        v1_x = -self.velocity_1_amplitude * TWO_PI * sin_x * cos_y
        v2_y = -self.velocity_2_amplitude * TWO_PI * sin_x * cos_y
        value = self.exact_solution(t, x)
        v1_xx_plus_yy = -2.0 * TWO_PI * TWO_PI * value[1]
        v2_xx_plus_yy = -2.0 * TWO_PI * TWO_PI * value[2]
        pressure_derivative = -2.0 / (tau * tau * tau)
        out = np.empty(3, dtype=np.float64)
        out[0] = tau_t - v1_x - v2_y
        out[1] = v1_t + pressure_derivative * tau_x
        out[1] -= self.epsilon * v1_xx_plus_yy
        out[2] = v2_t + pressure_derivative * tau_y
        out[2] -= self.epsilon * v2_xx_plus_yy
        return out

    def source_values(self, t: float, points: np.ndarray, out: np.ndarray) -> None:
        _diffusive_p_system_source_values(
            float(t),
            points.reshape((-1, 2)),
            out.reshape((-1, 3)),
            float(self.epsilon),
            float(self.tau_mean),
            float(self.tau_amplitude),
            float(self.velocity_1_amplitude),
            float(self.velocity_2_amplitude),
            float(self.phase_speed_x),
            float(self.phase_speed_y),
        )

    def _phase(self, t: float, x: np.ndarray) -> tuple[float, float]:
        return (
            TWO_PI * (x[0] - self.phase_speed_x * t),
            TWO_PI * (x[1] + self.phase_speed_y * t),
        )


def _validate_positive_tau(tau: np.ndarray) -> None:
    if np.any(tau <= 0.0):
        raise ValueError("tau must be positive")


@njit(cache=True)
def _diffusive_p_system_source_values(
    time: float,
    points: np.ndarray,
    out: np.ndarray,
    epsilon: float,
    tau_mean: float,
    tau_amplitude: float,
    velocity_1_amplitude: float,
    velocity_2_amplitude: float,
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
        tau = tau_mean + tau_amplitude * sin_x * cos_y
        tau_t = -tau_amplitude * TWO_PI * (
            phase_speed_x * cos_x * cos_y
            + phase_speed_y * sin_x * sin_y
        )
        v1 = velocity_1_amplitude * cos_x * cos_y
        v2 = -velocity_2_amplitude * sin_x * sin_y
        v1_t = velocity_1_amplitude * TWO_PI * (
            phase_speed_x * sin_x * cos_y
            - phase_speed_y * cos_x * sin_y
        )
        v2_t = velocity_2_amplitude * TWO_PI * (
            phase_speed_x * cos_x * sin_y
            - phase_speed_y * sin_x * cos_y
        )
        tau_x = tau_amplitude * TWO_PI * cos_x * cos_y
        tau_y = -tau_amplitude * TWO_PI * sin_x * sin_y
        v1_x = -velocity_1_amplitude * TWO_PI * sin_x * cos_y
        v2_y = -velocity_2_amplitude * TWO_PI * sin_x * cos_y
        v1_xx_plus_yy = -2.0 * TWO_PI * TWO_PI * v1
        v2_xx_plus_yy = -2.0 * TWO_PI * TWO_PI * v2
        pressure_derivative = -2.0 / (tau * tau * tau)
        out[point_index, 0] = tau_t - v1_x - v2_y
        out[point_index, 1] = (
            v1_t + pressure_derivative * tau_x - epsilon * v1_xx_plus_yy
        )
        out[point_index, 2] = (
            v2_t + pressure_derivative * tau_y - epsilon * v2_xx_plus_yy
        )
