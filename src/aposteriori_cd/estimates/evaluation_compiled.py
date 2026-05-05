"""Compiled estimator norm accumulation routines."""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from aposteriori_cd.equations import (
    DiffusivePSystem2D,
    LinearAdvectionDiffusion2D,
    ViscousBurgers2D,
)


SCALAR_MODEL_NONE = 0
SCALAR_MODEL_LINEAR = 1
SCALAR_MODEL_BURGERS = 2
P_SYSTEM_MODEL_NONE = 0
P_SYSTEM_MODEL_DIFFUSIVE = 1
TWO_PI = 2.0 * np.pi


def scalar_model_id_and_parameters(equation: object) -> tuple[int, np.ndarray]:
    """Return the compiled scalar model id and parameters."""

    if equation.__class__ is LinearAdvectionDiffusion2D:
        velocity_x, velocity_y = equation.velocity
        return SCALAR_MODEL_LINEAR, np.asarray(
            [
                float(equation.epsilon),
                float(velocity_x),
                float(velocity_y),
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )
    if equation.__class__ is ViscousBurgers2D:
        return SCALAR_MODEL_BURGERS, np.asarray(
            [
                float(equation.epsilon),
                float(equation.mean_value),
                float(equation.amplitude),
                float(equation.phase_speed_x),
                float(equation.phase_speed_y),
            ],
            dtype=np.float64,
        )
    return SCALAR_MODEL_NONE, np.zeros(5, dtype=np.float64)


def p_system_model_id_and_parameters(equation: object) -> tuple[int, np.ndarray]:
    """Return the compiled p-system model id and parameters."""

    if equation.__class__ is DiffusivePSystem2D:
        return P_SYSTEM_MODEL_DIFFUSIVE, np.asarray(
            [
                float(equation.tau_mean),
                float(equation.tau_amplitude),
                float(equation.velocity_1_amplitude),
                float(equation.velocity_2_amplitude),
                float(equation.phase_speed_x),
                float(equation.phase_speed_y),
                float(equation.epsilon),
            ],
            dtype=np.float64,
        )
    return P_SYSTEM_MODEL_NONE, np.zeros(7, dtype=np.float64)


@njit(cache=True)
def scalar_spatial_l2_difference(
    weights: np.ndarray,
    value: np.ndarray,
    coordinates: np.ndarray,
    time_value: float,
    model_id: int,
    parameters: np.ndarray,
) -> float:
    """Return the scalar SIAC value error in the spatial $$L^2$$ norm."""

    squared = scalar_spatial_l2_difference_squared(
        weights,
        value,
        coordinates,
        time_value,
        model_id,
        parameters,
    )
    if squared <= 0.0:
        return 0.0
    return np.sqrt(squared)


@njit(cache=True, parallel=True)
def scalar_spatial_l2_difference_squared(
    weights: np.ndarray,
    value: np.ndarray,
    coordinates: np.ndarray,
    time_value: float,
    model_id: int,
    parameters: np.ndarray,
) -> float:
    """Return the scalar SIAC value error squared in the spatial $$L^2$$ norm."""

    nx, ny, npx, npy = weights.shape
    total = nx * ny * npx * npy
    integral = 0.0
    for flat_index in prange(total):
        ex, ey, px, py = _decode_spatial_index(flat_index, ny, npx, npy)
        exact = _scalar_exact_value(
            model_id,
            parameters,
            time_value,
            coordinates[ex, ey, px, py, 0],
            coordinates[ex, ey, px, py, 1],
        )
        difference = value[ex, ey, px, py, 0] - exact
        integral += weights[ex, ey, px, py] * difference * difference
    return integral


@njit(cache=True, parallel=True)
def scalar_spatial_gradient_difference_squared(
    weights: np.ndarray,
    gradient: np.ndarray,
    coordinates: np.ndarray,
    time_value: float,
    model_id: int,
    parameters: np.ndarray,
) -> float:
    """Return the scalar SIAC gradient error squared in the spatial $$L^2$$ norm."""

    nx, ny, npx, npy = weights.shape
    total = nx * ny * npx * npy
    integral = 0.0
    for flat_index in prange(total):
        ex, ey, px, py = _decode_spatial_index(flat_index, ny, npx, npy)
        x = coordinates[ex, ey, px, py, 0]
        y = coordinates[ex, ey, px, py, 1]
        exact_x = _scalar_exact_gradient_x(model_id, parameters, time_value, x, y)
        exact_y = _scalar_exact_gradient_y(model_id, parameters, time_value, x, y)
        difference_x = gradient[ex, ey, px, py, 0, 0] - exact_x
        difference_y = gradient[ex, ey, px, py, 1, 0] - exact_y
        integral += weights[ex, ey, px, py] * (
            difference_x * difference_x + difference_y * difference_y
        )
    return integral


@njit(cache=True, parallel=True)
def scalar_temporal_l2_error_squared(
    weights: np.ndarray,
    coordinates: np.ndarray,
    temporal_value: np.ndarray,
    basis_at_quad: np.ndarray,
    time_value: float,
    model_id: int,
    parameters: np.ndarray,
) -> float:
    """Return the state-grid temporal reconstruction error squared."""

    nx, ny, npx, npy = weights.shape
    n_nodes = temporal_value.shape[2]
    total = nx * ny * npx * npy
    integral = 0.0
    for flat_index in prange(total):
        ex, ey, px, py = _decode_spatial_index(flat_index, ny, npx, npy)
        interpolated = 0.0
        for ix in range(n_nodes):
            basis_x = basis_at_quad[px, ix]
            for iy in range(n_nodes):
                interpolated += (
                    basis_x
                    * temporal_value[ex, ey, ix, iy, 0]
                    * basis_at_quad[py, iy]
                )
        exact = _scalar_exact_value(
            model_id,
            parameters,
            time_value,
            coordinates[ex, ey, px, py, 0],
            coordinates[ex, ey, px, py, 1],
        )
        difference = interpolated - exact
        integral += weights[ex, ey, px, py] * difference * difference
    return integral


@njit(cache=True)
def scalar_reconstructed_gradient_linf(gradient: np.ndarray) -> np.ndarray:
    """Return coordinate-wise maximum absolute reconstructed scalar gradient."""

    out = np.zeros((2, 1), dtype=np.float64)
    for ex in range(gradient.shape[0]):
        for ey in range(gradient.shape[1]):
            for px in range(gradient.shape[2]):
                for py in range(gradient.shape[3]):
                    value_x = abs(gradient[ex, ey, px, py, 0, 0])
                    value_y = abs(gradient[ex, ey, px, py, 1, 0])
                    if value_x > out[0, 0]:
                        out[0, 0] = value_x
                    if value_y > out[1, 0]:
                        out[1, 0] = value_y
    return out


@njit(cache=True)
def scalar_quadrature_node_terms(
    temporal_weights: np.ndarray,
    temporal_coordinates: np.ndarray,
    temporal_value: np.ndarray,
    basis_at_quad: np.ndarray,
    analysis_weights: np.ndarray,
    value: np.ndarray,
    time_derivative: np.ndarray,
    gradient: np.ndarray,
    auxiliary_derivative: np.ndarray,
    auxiliary_second_derivative: np.ndarray,
    coordinates: np.ndarray,
    time_value: float,
    model_id: int,
    parameters: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return scalar quadrature-node contributions."""

    temporal_error_squared = 0.0
    n_nodes = temporal_value.shape[2]
    for ex in range(temporal_weights.shape[0]):
        for ey in range(temporal_weights.shape[1]):
            for px in range(temporal_weights.shape[2]):
                for py in range(temporal_weights.shape[3]):
                    interpolated = 0.0
                    for ix in range(n_nodes):
                        basis_x = basis_at_quad[px, ix]
                        for iy in range(n_nodes):
                            interpolated += (
                                basis_x
                                * temporal_value[ex, ey, ix, iy, 0]
                                * basis_at_quad[py, iy]
                            )
                    exact = _scalar_exact_value(
                        model_id,
                        parameters,
                        time_value,
                        temporal_coordinates[ex, ey, px, py, 0],
                        temporal_coordinates[ex, ey, px, py, 1],
                    )
                    difference = interpolated - exact
                    temporal_error_squared += (
                        temporal_weights[ex, ey, px, py] * difference * difference
                    )

    gradient_error_squared = 0.0
    residual_1_squared = 0.0
    flux_difference_squared = 0.0
    epsilon = parameters[0]
    for ex in range(analysis_weights.shape[0]):
        for ey in range(analysis_weights.shape[1]):
            for px in range(analysis_weights.shape[2]):
                for py in range(analysis_weights.shape[3]):
                    x = coordinates[ex, ey, px, py, 0]
                    y = coordinates[ex, ey, px, py, 1]
                    exact_x = _scalar_exact_gradient_x(
                        model_id,
                        parameters,
                        time_value,
                        x,
                        y,
                    )
                    exact_y = _scalar_exact_gradient_y(
                        model_id,
                        parameters,
                        time_value,
                        x,
                        y,
                    )
                    gradient_x = gradient[ex, ey, px, py, 0, 0]
                    gradient_y = gradient[ex, ey, px, py, 1, 0]
                    gradient_difference_x = gradient_x - exact_x
                    gradient_difference_y = gradient_y - exact_y
                    weight = analysis_weights[ex, ey, px, py]
                    gradient_error_squared += weight * (
                        gradient_difference_x * gradient_difference_x
                        + gradient_difference_y * gradient_difference_y
                    )

                    residual_1 = time_derivative[ex, ey, px, py, 0]
                    residual_1 -= _scalar_source_value(
                        model_id,
                        parameters,
                        time_value,
                        x,
                        y,
                    )
                    if model_id == SCALAR_MODEL_LINEAR:
                        residual_1 += parameters[1] * gradient_x + parameters[2] * gradient_y
                    else:
                        u = value[ex, ey, px, py, 0]
                        residual_1 += u * (gradient_x + gradient_y)
                    residual_1 -= epsilon * (
                        auxiliary_second_derivative[ex, ey, px, py, 0, 0]
                        + auxiliary_second_derivative[ex, ey, px, py, 1, 0]
                    )
                    residual_1_squared += weight * residual_1 * residual_1

                    flux_difference_x = (
                        auxiliary_derivative[ex, ey, px, py, 0, 0] - gradient_x
                    )
                    flux_difference_y = (
                        auxiliary_derivative[ex, ey, px, py, 1, 0] - gradient_y
                    )
                    flux_difference_squared += weight * (
                        flux_difference_x * flux_difference_x
                        + flux_difference_y * flux_difference_y
                    )

    residual_1_l2 = 0.0
    if residual_1_squared > 0.0:
        residual_1_l2 = np.sqrt(residual_1_squared)
    return (
        temporal_error_squared,
        gradient_error_squared,
        residual_1_l2,
        flux_difference_squared,
    )


@njit(cache=True)
def p_system_initial_terms(
    weights: np.ndarray,
    value: np.ndarray,
    coordinates: np.ndarray,
    model_id: int,
    parameters: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return the p-system initial terms and tau range."""

    initial_W_tau_l1 = 0.0
    initial_v_l2_error_squared = 0.0
    tau_min = np.inf
    tau_max = -np.inf
    for ex in range(weights.shape[0]):
        for ey in range(weights.shape[1]):
            for px in range(weights.shape[2]):
                for py in range(weights.shape[3]):
                    exact_tau, exact_v1, exact_v2 = _p_system_exact_value(
                        model_id,
                        parameters,
                        0.0,
                        coordinates[ex, ey, px, py, 0],
                        coordinates[ex, ey, px, py, 1],
                    )
                    tau = value[ex, ey, px, py, 0]
                    v1 = value[ex, ey, px, py, 1]
                    v2 = value[ex, ey, px, py, 2]
                    if exact_tau <= 0.0 or tau <= 0.0:
                        raise ValueError("tau must be positive")
                    if exact_tau < tau_min:
                        tau_min = exact_tau
                    if tau < tau_min:
                        tau_min = tau
                    if exact_tau > tau_max:
                        tau_max = exact_tau
                    if tau > tau_max:
                        tau_max = tau
                    weight = weights[ex, ey, px, py]
                    tau_difference = exact_tau - tau
                    initial_W_tau_l1 += (
                        weight
                        * tau_difference
                        * tau_difference
                        / (exact_tau * tau * tau)
                    )
                    v1_difference = v1 - exact_v1
                    v2_difference = v2 - exact_v2
                    initial_v_l2_error_squared += weight * (
                        v1_difference * v1_difference
                        + v2_difference * v2_difference
                    )
    return initial_W_tau_l1, initial_v_l2_error_squared, tau_min, tau_max


@njit(cache=True, parallel=True)
def p_system_spatial_l2_errors(
    weights: np.ndarray,
    value: np.ndarray,
    coordinates: np.ndarray,
    time_value: float,
    model_id: int,
    parameters: np.ndarray,
) -> tuple[float, float]:
    """Return tau and velocity errors in the spatial $$L^2$$ norm."""

    nx, ny, npx, npy = weights.shape
    total = nx * ny * npx * npy
    tau_integral = 0.0
    velocity_integral = 0.0
    for flat_index in prange(total):
        ex, ey, px, py = _decode_spatial_index(flat_index, ny, npx, npy)
        exact_tau, exact_v1, exact_v2 = _p_system_exact_value(
            model_id,
            parameters,
            time_value,
            coordinates[ex, ey, px, py, 0],
            coordinates[ex, ey, px, py, 1],
        )
        tau_difference = value[ex, ey, px, py, 0] - exact_tau
        v1_difference = value[ex, ey, px, py, 1] - exact_v1
        v2_difference = value[ex, ey, px, py, 2] - exact_v2
        weight = weights[ex, ey, px, py]
        tau_integral += weight * tau_difference * tau_difference
        velocity_integral += weight * (
            v1_difference * v1_difference + v2_difference * v2_difference
        )
    tau_error = 0.0
    if tau_integral > 0.0:
        tau_error = np.sqrt(tau_integral)
    velocity_error = 0.0
    if velocity_integral > 0.0:
        velocity_error = np.sqrt(velocity_integral)
    return tau_error, velocity_error


@njit(cache=True, parallel=True)
def p_system_temporal_l2_error_squared(
    weights: np.ndarray,
    coordinates: np.ndarray,
    temporal_value: np.ndarray,
    basis_at_quad: np.ndarray,
    time_value: float,
    model_id: int,
    parameters: np.ndarray,
) -> float:
    """Return the p-system temporal reconstruction error squared."""

    nx, ny, npx, npy = weights.shape
    n_nodes = temporal_value.shape[2]
    total = nx * ny * npx * npy
    integral = 0.0
    for flat_index in prange(total):
        ex, ey, px, py = _decode_spatial_index(flat_index, ny, npx, npy)
        tau = 0.0
        v1 = 0.0
        v2 = 0.0
        for ix in range(n_nodes):
            basis_x = basis_at_quad[px, ix]
            for iy in range(n_nodes):
                basis_value = basis_x * basis_at_quad[py, iy]
                tau += basis_value * temporal_value[ex, ey, ix, iy, 0]
                v1 += basis_value * temporal_value[ex, ey, ix, iy, 1]
                v2 += basis_value * temporal_value[ex, ey, ix, iy, 2]
        exact_tau, exact_v1, exact_v2 = _p_system_exact_value(
            model_id,
            parameters,
            time_value,
            coordinates[ex, ey, px, py, 0],
            coordinates[ex, ey, px, py, 1],
        )
        tau_difference = tau - exact_tau
        v1_difference = v1 - exact_v1
        v2_difference = v2 - exact_v2
        integral += weights[ex, ey, px, py] * (
            tau_difference * tau_difference
            + v1_difference * v1_difference
            + v2_difference * v2_difference
        )
    return integral


@njit(cache=True, parallel=True)
def p_system_velocity_gradient_error_squared(
    weights: np.ndarray,
    gradient: np.ndarray,
    coordinates: np.ndarray,
    time_value: float,
    model_id: int,
    parameters: np.ndarray,
) -> float:
    """Return the velocity gradient error squared in the spatial $$L^2$$ norm."""

    nx, ny, npx, npy = weights.shape
    total = nx * ny * npx * npy
    integral = 0.0
    for flat_index in prange(total):
        ex, ey, px, py = _decode_spatial_index(flat_index, ny, npx, npy)
        (
            _exact_tau_x,
            exact_v1_x,
            exact_v2_x,
            _exact_tau_y,
            exact_v1_y,
            exact_v2_y,
        ) = _p_system_exact_gradient(
            model_id,
            parameters,
            time_value,
            coordinates[ex, ey, px, py, 0],
            coordinates[ex, ey, px, py, 1],
        )
        v1_x_difference = gradient[ex, ey, px, py, 0, 1] - exact_v1_x
        v1_y_difference = gradient[ex, ey, px, py, 1, 1] - exact_v1_y
        v2_x_difference = gradient[ex, ey, px, py, 0, 2] - exact_v2_x
        v2_y_difference = gradient[ex, ey, px, py, 1, 2] - exact_v2_y
        integral += weights[ex, ey, px, py] * (
            v1_x_difference * v1_x_difference
            + v1_y_difference * v1_y_difference
            + v2_x_difference * v2_x_difference
            + v2_y_difference * v2_y_difference
        )
    return integral


@njit(cache=True)
def p_system_tau_bounds_and_div_velocity(
    value: np.ndarray,
    gradient: np.ndarray,
    coordinates: np.ndarray,
    time_value: float,
    model_id: int,
    parameters: np.ndarray,
) -> tuple[float, float, float]:
    """Return tau range and reconstructed velocity divergence bound."""

    tau_min = np.inf
    tau_max = -np.inf
    div_hat_v_linf = 0.0
    for ex in range(value.shape[0]):
        for ey in range(value.shape[1]):
            for px in range(value.shape[2]):
                for py in range(value.shape[3]):
                    exact_tau, _exact_v1, _exact_v2 = _p_system_exact_value(
                        model_id,
                        parameters,
                        time_value,
                        coordinates[ex, ey, px, py, 0],
                        coordinates[ex, ey, px, py, 1],
                    )
                    tau = value[ex, ey, px, py, 0]
                    if exact_tau < tau_min:
                        tau_min = exact_tau
                    if tau < tau_min:
                        tau_min = tau
                    if exact_tau > tau_max:
                        tau_max = exact_tau
                    if tau > tau_max:
                        tau_max = tau
                    divergence = (
                        gradient[ex, ey, px, py, 0, 1]
                        + gradient[ex, ey, px, py, 1, 2]
                    )
                    abs_divergence = abs(divergence)
                    if abs_divergence > div_hat_v_linf:
                        div_hat_v_linf = abs_divergence
    return tau_min, tau_max, div_hat_v_linf


@njit(cache=True)
def p_system_residual_spatial_terms(
    weights: np.ndarray,
    value: np.ndarray,
    residual_1: np.ndarray,
) -> tuple[float, float]:
    """Return the p-system residual 1 spatial terms."""

    nx, ny, npx, npy = weights.shape
    total = nx * ny * npx * npy
    W_second_integral = 0.0
    velocity_integral = 0.0
    for flat_index in range(total):
        ex, ey, px, py = _decode_spatial_index(flat_index, ny, npx, npy)
        tau = value[ex, ey, px, py, 0]
        if tau <= 0.0:
            raise ValueError("tau must be positive")
        W_second_r_tau = 2.0 * residual_1[ex, ey, px, py, 0] / (tau * tau * tau)
        r_v1 = residual_1[ex, ey, px, py, 1]
        r_v2 = residual_1[ex, ey, px, py, 2]
        weight = weights[ex, ey, px, py]
        W_second_integral += weight * W_second_r_tau * W_second_r_tau
        velocity_integral += weight * (r_v1 * r_v1 + r_v2 * r_v2)
    W_second = 0.0
    if W_second_integral > 0.0:
        W_second = np.sqrt(W_second_integral)
    velocity = 0.0
    if velocity_integral > 0.0:
        velocity = np.sqrt(velocity_integral)
    return W_second, velocity


@njit(cache=True)
def p_system_quadrature_node_terms(
    temporal_weights: np.ndarray,
    temporal_coordinates: np.ndarray,
    temporal_value: np.ndarray,
    basis_at_quad: np.ndarray,
    analysis_weights: np.ndarray,
    value: np.ndarray,
    time_derivative: np.ndarray,
    gradient: np.ndarray,
    auxiliary_derivative: np.ndarray,
    auxiliary_second_derivative: np.ndarray,
    coordinates: np.ndarray,
    time_value: float,
    model_id: int,
    parameters: np.ndarray,
) -> tuple[float, float, float, float, float, float, float, float, float]:
    """Return p-system quadrature-node contributions."""

    temporal_error_squared = 0.0
    n_nodes = temporal_value.shape[2]
    for ex in range(temporal_weights.shape[0]):
        for ey in range(temporal_weights.shape[1]):
            for px in range(temporal_weights.shape[2]):
                for py in range(temporal_weights.shape[3]):
                    tau = 0.0
                    v1 = 0.0
                    v2 = 0.0
                    for ix in range(n_nodes):
                        basis_x = basis_at_quad[px, ix]
                        for iy in range(n_nodes):
                            basis_value = basis_x * basis_at_quad[py, iy]
                            tau += basis_value * temporal_value[ex, ey, ix, iy, 0]
                            v1 += basis_value * temporal_value[ex, ey, ix, iy, 1]
                            v2 += basis_value * temporal_value[ex, ey, ix, iy, 2]
                    exact_tau, exact_v1, exact_v2 = _p_system_exact_value(
                        model_id,
                        parameters,
                        time_value,
                        temporal_coordinates[ex, ey, px, py, 0],
                        temporal_coordinates[ex, ey, px, py, 1],
                    )
                    tau_difference = tau - exact_tau
                    v1_difference = v1 - exact_v1
                    v2_difference = v2 - exact_v2
                    temporal_error_squared += temporal_weights[ex, ey, px, py] * (
                        tau_difference * tau_difference
                        + v1_difference * v1_difference
                        + v2_difference * v2_difference
                    )

    velocity_gradient_error_squared = 0.0
    residual_1_squared = 0.0
    flux_difference_squared = 0.0
    W_second_integral = 0.0
    velocity_residual_integral = 0.0
    tau_min = np.inf
    tau_max = -np.inf
    div_hat_v_linf = 0.0
    epsilon = parameters[6]
    for ex in range(analysis_weights.shape[0]):
        for ey in range(analysis_weights.shape[1]):
            for px in range(analysis_weights.shape[2]):
                for py in range(analysis_weights.shape[3]):
                    x = coordinates[ex, ey, px, py, 0]
                    y = coordinates[ex, ey, px, py, 1]
                    exact_tau, _exact_v1, _exact_v2 = _p_system_exact_value(
                        model_id,
                        parameters,
                        time_value,
                        x,
                        y,
                    )
                    (
                        _exact_tau_x,
                        exact_v1_x,
                        exact_v2_x,
                        _exact_tau_y,
                        exact_v1_y,
                        exact_v2_y,
                    ) = _p_system_exact_gradient(
                        model_id,
                        parameters,
                        time_value,
                        x,
                        y,
                    )
                    tau = value[ex, ey, px, py, 0]
                    if tau <= 0.0:
                        raise ValueError("tau must be positive")
                    if exact_tau < tau_min:
                        tau_min = exact_tau
                    if tau < tau_min:
                        tau_min = tau
                    if exact_tau > tau_max:
                        tau_max = exact_tau
                    if tau > tau_max:
                        tau_max = tau

                    v1_x_difference = gradient[ex, ey, px, py, 0, 1] - exact_v1_x
                    v1_y_difference = gradient[ex, ey, px, py, 1, 1] - exact_v1_y
                    v2_x_difference = gradient[ex, ey, px, py, 0, 2] - exact_v2_x
                    v2_y_difference = gradient[ex, ey, px, py, 1, 2] - exact_v2_y
                    weight = analysis_weights[ex, ey, px, py]
                    velocity_gradient_error_squared += weight * (
                        v1_x_difference * v1_x_difference
                        + v1_y_difference * v1_y_difference
                        + v2_x_difference * v2_x_difference
                        + v2_y_difference * v2_y_difference
                    )

                    divergence = (
                        gradient[ex, ey, px, py, 0, 1]
                        + gradient[ex, ey, px, py, 1, 2]
                    )
                    abs_divergence = abs(divergence)
                    if abs_divergence > div_hat_v_linf:
                        div_hat_v_linf = abs_divergence

                    source_tau, source_v1, source_v2 = _p_system_source_value(
                        parameters,
                        time_value,
                        x,
                        y,
                    )
                    pressure_derivative = -2.0 / (tau * tau * tau)
                    r_tau = time_derivative[ex, ey, px, py, 0] - source_tau
                    r_tau += -gradient[ex, ey, px, py, 0, 1]
                    r_tau += -gradient[ex, ey, px, py, 1, 2]
                    r_v1 = time_derivative[ex, ey, px, py, 1] - source_v1
                    r_v1 += pressure_derivative * gradient[ex, ey, px, py, 0, 0]
                    r_v1 -= epsilon * (
                        auxiliary_second_derivative[ex, ey, px, py, 0, 1]
                        + auxiliary_second_derivative[ex, ey, px, py, 1, 1]
                    )
                    r_v2 = time_derivative[ex, ey, px, py, 2] - source_v2
                    r_v2 += pressure_derivative * gradient[ex, ey, px, py, 1, 0]
                    r_v2 -= epsilon * (
                        auxiliary_second_derivative[ex, ey, px, py, 0, 2]
                        + auxiliary_second_derivative[ex, ey, px, py, 1, 2]
                    )
                    residual_1_squared += weight * (
                        r_tau * r_tau + r_v1 * r_v1 + r_v2 * r_v2
                    )
                    W_second_r_tau = 2.0 * r_tau / (tau * tau * tau)
                    W_second_integral += weight * W_second_r_tau * W_second_r_tau
                    velocity_residual_integral += weight * (r_v1 * r_v1 + r_v2 * r_v2)

                    v1_flux_difference_x = (
                        auxiliary_derivative[ex, ey, px, py, 0, 1]
                        - gradient[ex, ey, px, py, 0, 1]
                    )
                    v1_flux_difference_y = (
                        auxiliary_derivative[ex, ey, px, py, 1, 1]
                        - gradient[ex, ey, px, py, 1, 1]
                    )
                    v2_flux_difference_x = (
                        auxiliary_derivative[ex, ey, px, py, 0, 2]
                        - gradient[ex, ey, px, py, 0, 2]
                    )
                    v2_flux_difference_y = (
                        auxiliary_derivative[ex, ey, px, py, 1, 2]
                        - gradient[ex, ey, px, py, 1, 2]
                    )
                    flux_difference_squared += weight * (
                        v1_flux_difference_x * v1_flux_difference_x
                        + v1_flux_difference_y * v1_flux_difference_y
                        + v2_flux_difference_x * v2_flux_difference_x
                        + v2_flux_difference_y * v2_flux_difference_y
                    )

    residual_1_l2 = 0.0
    if residual_1_squared > 0.0:
        residual_1_l2 = np.sqrt(residual_1_squared)
    W_second_r_u_l2 = 0.0
    if W_second_integral > 0.0:
        W_second_r_u_l2 = np.sqrt(W_second_integral)
    r_v_1_l2 = 0.0
    if velocity_residual_integral > 0.0:
        r_v_1_l2 = np.sqrt(velocity_residual_integral)
    return (
        temporal_error_squared,
        velocity_gradient_error_squared,
        residual_1_l2,
        flux_difference_squared,
        W_second_r_u_l2,
        r_v_1_l2,
        tau_min,
        tau_max,
        div_hat_v_linf,
    )


@njit(cache=True)
def _decode_spatial_index(
    flat_index: int,
    ny: int,
    npx: int,
    npy: int,
) -> tuple[int, int, int, int]:
    py = flat_index % npy
    remainder = flat_index // npy
    px = remainder % npx
    remainder = remainder // npx
    ey = remainder % ny
    ex = remainder // ny
    return ex, ey, px, py


@njit(cache=True)
def _scalar_exact_value(
    model_id: int,
    parameters: np.ndarray,
    time_value: float,
    x: float,
    y: float,
) -> float:
    if model_id == SCALAR_MODEL_LINEAR:
        epsilon = parameters[0]
        velocity_x = parameters[1]
        velocity_y = parameters[2]
        decay = np.exp(-8.0 * epsilon * np.pi * np.pi * time_value)
        phase_x = TWO_PI * (x - velocity_x * time_value)
        phase_y = TWO_PI * (y - velocity_y * time_value)
        return decay * np.sin(phase_x) * np.cos(phase_y)
    mean_value = parameters[1]
    amplitude = parameters[2]
    phase_speed_x = parameters[3]
    phase_speed_y = parameters[4]
    phase_x = TWO_PI * (x - phase_speed_x * time_value)
    phase_y = TWO_PI * (y + phase_speed_y * time_value)
    return mean_value + amplitude * np.sin(phase_x) * np.cos(phase_y)


@njit(cache=True)
def _scalar_exact_gradient_x(
    model_id: int,
    parameters: np.ndarray,
    time_value: float,
    x: float,
    y: float,
) -> float:
    if model_id == SCALAR_MODEL_LINEAR:
        epsilon = parameters[0]
        velocity_x = parameters[1]
        velocity_y = parameters[2]
        decay = np.exp(-8.0 * epsilon * np.pi * np.pi * time_value)
        phase_x = TWO_PI * (x - velocity_x * time_value)
        phase_y = TWO_PI * (y - velocity_y * time_value)
        return decay * TWO_PI * np.cos(phase_x) * np.cos(phase_y)
    amplitude = parameters[2]
    phase_speed_x = parameters[3]
    phase_speed_y = parameters[4]
    phase_x = TWO_PI * (x - phase_speed_x * time_value)
    phase_y = TWO_PI * (y + phase_speed_y * time_value)
    return amplitude * TWO_PI * np.cos(phase_x) * np.cos(phase_y)


@njit(cache=True)
def _scalar_exact_gradient_y(
    model_id: int,
    parameters: np.ndarray,
    time_value: float,
    x: float,
    y: float,
) -> float:
    if model_id == SCALAR_MODEL_LINEAR:
        epsilon = parameters[0]
        velocity_x = parameters[1]
        velocity_y = parameters[2]
        decay = np.exp(-8.0 * epsilon * np.pi * np.pi * time_value)
        phase_x = TWO_PI * (x - velocity_x * time_value)
        phase_y = TWO_PI * (y - velocity_y * time_value)
        return -decay * TWO_PI * np.sin(phase_x) * np.sin(phase_y)
    amplitude = parameters[2]
    phase_speed_x = parameters[3]
    phase_speed_y = parameters[4]
    phase_x = TWO_PI * (x - phase_speed_x * time_value)
    phase_y = TWO_PI * (y + phase_speed_y * time_value)
    return -amplitude * TWO_PI * np.sin(phase_x) * np.sin(phase_y)


@njit(cache=True)
def _scalar_source_value(
    model_id: int,
    parameters: np.ndarray,
    time_value: float,
    x: float,
    y: float,
) -> float:
    if model_id == SCALAR_MODEL_LINEAR:
        return 0.0
    epsilon = parameters[0]
    mean_value = parameters[1]
    amplitude = parameters[2]
    phase_speed_x = parameters[3]
    phase_speed_y = parameters[4]
    phase_x = TWO_PI * (x - phase_speed_x * time_value)
    phase_y = TWO_PI * (y + phase_speed_y * time_value)
    sin_x = np.sin(phase_x)
    cos_x = np.cos(phase_x)
    sin_y = np.sin(phase_y)
    cos_y = np.cos(phase_y)
    oscillation = amplitude * sin_x * cos_y
    value = mean_value + oscillation
    gradient_x = amplitude * TWO_PI * cos_x * cos_y
    gradient_y = -amplitude * TWO_PI * sin_x * sin_y
    laplacian = -2.0 * TWO_PI * TWO_PI * oscillation
    time_derivative = (
        -amplitude * TWO_PI * phase_speed_x * cos_x * cos_y
        - amplitude * TWO_PI * phase_speed_y * sin_x * sin_y
    )
    return time_derivative + value * (gradient_x + gradient_y) - epsilon * laplacian


@njit(cache=True)
def _p_system_exact_value(
    model_id: int,
    parameters: np.ndarray,
    time_value: float,
    x: float,
    y: float,
) -> tuple[float, float, float]:
    tau_mean = parameters[0]
    tau_amplitude = parameters[1]
    velocity_1_amplitude = parameters[2]
    velocity_2_amplitude = parameters[3]
    phase_speed_x = parameters[4]
    phase_speed_y = parameters[5]
    phase_x = TWO_PI * (x - phase_speed_x * time_value)
    phase_y = TWO_PI * (y + phase_speed_y * time_value)
    sin_x = np.sin(phase_x)
    cos_x = np.cos(phase_x)
    sin_y = np.sin(phase_y)
    cos_y = np.cos(phase_y)
    return (
        tau_mean + tau_amplitude * sin_x * cos_y,
        velocity_1_amplitude * cos_x * cos_y,
        -velocity_2_amplitude * sin_x * sin_y,
    )


@njit(cache=True)
def _p_system_exact_gradient(
    model_id: int,
    parameters: np.ndarray,
    time_value: float,
    x: float,
    y: float,
) -> tuple[float, float, float, float, float, float]:
    tau_amplitude = parameters[1]
    velocity_1_amplitude = parameters[2]
    velocity_2_amplitude = parameters[3]
    phase_speed_x = parameters[4]
    phase_speed_y = parameters[5]
    phase_x = TWO_PI * (x - phase_speed_x * time_value)
    phase_y = TWO_PI * (y + phase_speed_y * time_value)
    sin_x = np.sin(phase_x)
    cos_x = np.cos(phase_x)
    sin_y = np.sin(phase_y)
    cos_y = np.cos(phase_y)
    return (
        tau_amplitude * TWO_PI * cos_x * cos_y,
        -velocity_1_amplitude * TWO_PI * sin_x * cos_y,
        -velocity_2_amplitude * TWO_PI * cos_x * sin_y,
        -tau_amplitude * TWO_PI * sin_x * sin_y,
        -velocity_1_amplitude * TWO_PI * cos_x * sin_y,
        -velocity_2_amplitude * TWO_PI * sin_x * cos_y,
    )


@njit(cache=True)
def _p_system_source_value(
    parameters: np.ndarray,
    time_value: float,
    x: float,
    y: float,
) -> tuple[float, float, float]:
    tau_mean = parameters[0]
    tau_amplitude = parameters[1]
    velocity_1_amplitude = parameters[2]
    velocity_2_amplitude = parameters[3]
    phase_speed_x = parameters[4]
    phase_speed_y = parameters[5]
    epsilon = parameters[6]
    phase_x = TWO_PI * (x - phase_speed_x * time_value)
    phase_y = TWO_PI * (y + phase_speed_y * time_value)
    sin_x = np.sin(phase_x)
    cos_x = np.cos(phase_x)
    sin_y = np.sin(phase_y)
    cos_y = np.cos(phase_y)
    tau = tau_mean + tau_amplitude * sin_x * cos_y
    tau_t = -tau_amplitude * TWO_PI * (
        phase_speed_x * cos_x * cos_y + phase_speed_y * sin_x * sin_y
    )
    v1 = velocity_1_amplitude * cos_x * cos_y
    v2 = -velocity_2_amplitude * sin_x * sin_y
    v1_t = velocity_1_amplitude * TWO_PI * (
        phase_speed_x * sin_x * cos_y - phase_speed_y * cos_x * sin_y
    )
    v2_t = velocity_2_amplitude * TWO_PI * (
        phase_speed_x * cos_x * sin_y - phase_speed_y * sin_x * cos_y
    )
    tau_x = tau_amplitude * TWO_PI * cos_x * cos_y
    tau_y = -tau_amplitude * TWO_PI * sin_x * sin_y
    v1_x = -velocity_1_amplitude * TWO_PI * sin_x * cos_y
    v2_y = -velocity_2_amplitude * TWO_PI * sin_x * cos_y
    v1_xx_plus_yy = -2.0 * TWO_PI * TWO_PI * v1
    v2_xx_plus_yy = -2.0 * TWO_PI * TWO_PI * v2
    pressure_derivative = -2.0 / (tau * tau * tau)
    return (
        tau_t - v1_x - v2_y,
        v1_t + pressure_derivative * tau_x - epsilon * v1_xx_plus_yy,
        v2_t + pressure_derivative * tau_y - epsilon * v2_xx_plus_yy,
    )


__all__ = [
    "SCALAR_MODEL_BURGERS",
    "SCALAR_MODEL_LINEAR",
    "SCALAR_MODEL_NONE",
    "P_SYSTEM_MODEL_DIFFUSIVE",
    "P_SYSTEM_MODEL_NONE",
    "p_system_initial_terms",
    "p_system_model_id_and_parameters",
    "p_system_quadrature_node_terms",
    "p_system_residual_spatial_terms",
    "p_system_spatial_l2_errors",
    "p_system_tau_bounds_and_div_velocity",
    "p_system_temporal_l2_error_squared",
    "p_system_velocity_gradient_error_squared",
    "scalar_model_id_and_parameters",
    "scalar_quadrature_node_terms",
    "scalar_reconstructed_gradient_linf",
    "scalar_spatial_gradient_difference_squared",
    "scalar_spatial_l2_difference",
    "scalar_spatial_l2_difference_squared",
    "scalar_temporal_l2_error_squared",
]
