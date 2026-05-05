"""A posteriori estimator formulas for the paper experiments."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt

import numpy as np

from .accumulation import AnalysisGridQuadrature2D


@dataclass(frozen=True, slots=True)
class PSystemInitialTerms2D:
    """Initial terms and tau range for the p-system estimator."""

    initial_W_tau_l1: float
    initial_v_l2_error_squared: float
    tau_min: float
    tau_max: float


def scalar_initial_hat_u_ts_l2_error(
    equation: object,
    quadrature: AnalysisGridQuadrature2D,
    physical_coordinates: np.ndarray,
    reconstructed_value: np.ndarray,
    *,
    time: float = 0.0,
) -> float:
    """Compute the scalar initial $$L^2$$ reconstruction error."""

    integral = 0.0
    for index in np.ndindex(quadrature.spatial_weights.shape):
        exact = equation.exact_solution(float(time), physical_coordinates[index])[0]
        difference = reconstructed_value[index][0] - exact
        integral += quadrature.spatial_weights[index] * difference * difference
    return sqrt(max(float(integral), 0.0))


def scalar_reconstruction_error(
    hat_u_ts_linf_l2_error: float,
    hat_u_ts_l2_h1_seminorm_error: float,
    epsilon: float,
) -> float:
    """Compute scalar $$E_{\\mathrm{rec}}$$."""

    linf_l2 = _nonnegative_float(hat_u_ts_linf_l2_error, "hat_u_ts_linf_l2_error")
    l2_h1 = _nonnegative_float(
        hat_u_ts_l2_h1_seminorm_error,
        "hat_u_ts_l2_h1_seminorm_error",
    )
    diffusion_coefficient = _nonnegative_float(epsilon, "epsilon")
    return sqrt(linf_l2 * linf_l2 + diffusion_coefficient * l2_h1 * l2_h1)


def linear_scalar_a_posteriori_error_estimate(
    initial_hat_u_ts_l2_error: float,
    residual_1_l1_l2: float,
    E_r2: float,
    epsilon: float,
) -> float:
    """Compute the linear scalar paper estimator."""

    initial_error = _nonnegative_float(
        initial_hat_u_ts_l2_error,
        "initial_hat_u_ts_l2_error",
    )
    residual_1 = _nonnegative_float(residual_1_l1_l2, "residual_1_l1_l2")
    residual_2 = _nonnegative_float(E_r2, "E_r2")
    diffusion_coefficient = _nonnegative_float(epsilon, "epsilon")
    return sqrt(
        2.0 * initial_error * initial_error
        + 4.0 * residual_1 * residual_1
        + 2.0 * diffusion_coefficient * residual_2 * residual_2,
    )


def viscous_burgers_lambda(
    du_dx_linf: float,
    du_dy_linf: float,
) -> float:
    """Compute the Burgers stability constant $$\\Lambda$$."""

    return (
        2.0 * _nonnegative_float(du_dx_linf, "du_dx_linf")
        + 2.0 * _nonnegative_float(du_dy_linf, "du_dy_linf")
    )


def viscous_burgers_a_posteriori_error_estimate(
    initial_hat_u_ts_l2_error: float,
    residual_1_l1_l2: float,
    E_r2: float,
    epsilon: float,
    Lambda: float,
    final_time: float,
) -> float:
    """Compute the viscous Burgers paper estimator."""

    initial_error = _nonnegative_float(
        initial_hat_u_ts_l2_error,
        "initial_hat_u_ts_l2_error",
    )
    residual_1 = _nonnegative_float(residual_1_l1_l2, "residual_1_l1_l2")
    residual_2 = _nonnegative_float(E_r2, "E_r2")
    diffusion_coefficient = _nonnegative_float(epsilon, "epsilon")
    growth_factor = _nonnegative_float(Lambda, "Lambda")
    time_value = _nonnegative_float(final_time, "final_time")
    return exp(growth_factor * time_value) * sqrt(
        4.0 * initial_error * initial_error
        + 16.0 * residual_1 * residual_1
        + 8.0 * diffusion_coefficient * residual_2 * residual_2,
    )


def p_system_reconstruction_error(
    c_W: float,
    hat_tau_ts_linf_l2_error: float,
    hat_v_ts_linf_l2_error: float,
    hat_v_ts_l2_h1_seminorm_error: float,
    epsilon: float,
) -> float:
    """Compute p-system $$E_{\\mathrm{rec}}$$."""

    convexity_constant = _positive_float(c_W, "c_W")
    tau_error = _nonnegative_float(
        hat_tau_ts_linf_l2_error,
        "hat_tau_ts_linf_l2_error",
    )
    velocity_error = _nonnegative_float(
        hat_v_ts_linf_l2_error,
        "hat_v_ts_linf_l2_error",
    )
    velocity_h1_error = _nonnegative_float(
        hat_v_ts_l2_h1_seminorm_error,
        "hat_v_ts_l2_h1_seminorm_error",
    )
    diffusion_coefficient = _nonnegative_float(epsilon, "epsilon")
    return sqrt(
        convexity_constant * tau_error * tau_error
        + velocity_error * velocity_error
        + diffusion_coefficient * velocity_h1_error * velocity_h1_error,
    )


def p_system_lambda(
    c_W: float,
    C_W: float,
    div_hat_v_linf: float,
) -> float:
    """Compute the p-system stability constant $$\\Lambda$$."""

    convexity_constant = _positive_float(c_W, "c_W")
    third_derivative_bound = _nonnegative_float(C_W, "C_W")
    divergence_bound = _nonnegative_float(div_hat_v_linf, "div_hat_v_linf")
    return 2.0 * third_derivative_bound * divergence_bound / convexity_constant


def p_system_a_posteriori_error_estimate(
    initial_W_tau_l1: float,
    initial_v_l2_error_squared: float,
    W_second_r_u_l1_l2: float,
    r_v_1_l1_l2: float,
    E_r2: float,
    epsilon: float,
    c_W: float,
    Lambda: float,
    final_time: float,
) -> float:
    """Compute the p-system paper estimator."""

    initial_W_tau = _nonnegative_float(initial_W_tau_l1, "initial_W_tau_l1")
    initial_v = _nonnegative_float(
        initial_v_l2_error_squared,
        "initial_v_l2_error_squared",
    )
    W_second_r_u = _nonnegative_float(W_second_r_u_l1_l2, "W_second_r_u_l1_l2")
    r_v_1 = _nonnegative_float(r_v_1_l1_l2, "r_v_1_l1_l2")
    residual_2 = _nonnegative_float(E_r2, "E_r2")
    diffusion_coefficient = _nonnegative_float(epsilon, "epsilon")
    convexity_constant = _positive_float(c_W, "c_W")
    growth_factor = _nonnegative_float(Lambda, "Lambda")
    time_value = _nonnegative_float(final_time, "final_time")
    return exp(0.5 * growth_factor * time_value) * sqrt(
        (16.0 / 3.0) * initial_W_tau
        + (8.0 / 3.0) * initial_v
        + (16.0 / (3.0 * convexity_constant)) * W_second_r_u * W_second_r_u
        + (16.0 / 3.0) * r_v_1 * r_v_1
        + (16.0 / 3.0) * diffusion_coefficient * residual_2 * residual_2,
    )


def _nonnegative_float(value: float, name: str) -> float:
    out = float(value)
    if out < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return out


def _positive_float(value: float, name: str) -> float:
    out = float(value)
    if out <= 0.0:
        raise ValueError(f"{name} must be positive")
    return out


__all__ = [
    "PSystemInitialTerms2D",
    "linear_scalar_a_posteriori_error_estimate",
    "p_system_a_posteriori_error_estimate",
    "p_system_lambda",
    "p_system_reconstruction_error",
    "scalar_initial_hat_u_ts_l2_error",
    "scalar_reconstruction_error",
    "viscous_burgers_a_posteriori_error_estimate",
    "viscous_burgers_lambda",
]
