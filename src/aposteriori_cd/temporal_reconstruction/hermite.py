from __future__ import annotations

import numpy as np

from aposteriori_cd.time import (
    LEFT_ENDPOINT,
    RIGHT_ENDPOINT,
    TimeSubintervalSolutionData,
)


def cubic_hermite_basis(tau: float) -> tuple[float, float, float, float]:
    """Return the cubic Hermite basis on the unit time interval."""

    _validate_tau(tau)
    tau2 = tau * tau
    tau3 = tau2 * tau
    return (
        2.0 * tau3 - 3.0 * tau2 + 1.0,
        tau3 - 2.0 * tau2 + tau,
        -2.0 * tau3 + 3.0 * tau2,
        tau3 - tau2,
    )


def cubic_hermite_basis_time_derivative(
    tau: float,
    time_subinterval_length: float,
) -> tuple[float, float, float, float]:
    """Return time derivatives of the cubic Hermite basis."""

    _validate_tau(tau)
    if time_subinterval_length <= 0.0:
        raise ValueError("time_subinterval_length must be positive")
    tau2 = tau * tau
    return (
        (6.0 * tau2 - 6.0 * tau) / time_subinterval_length,
        3.0 * tau2 - 4.0 * tau + 1.0,
        (-6.0 * tau2 + 6.0 * tau) / time_subinterval_length,
        3.0 * tau2 - 2.0 * tau,
    )


def quintic_hermite_basis(
    tau: float,
    previous_time_subinterval_length: float,
    current_time_subinterval_length: float,
) -> tuple[float, float, float, float, float, float]:
    """Return the H(1,0,0) Hermite basis on the current time subinterval."""

    nodes, time = _three_time_level_nodes(
        tau,
        previous_time_subinterval_length,
        current_time_subinterval_length,
    )
    return _three_node_hermite_basis(nodes, time)


def quintic_hermite_basis_time_derivative(
    tau: float,
    previous_time_subinterval_length: float,
    current_time_subinterval_length: float,
) -> tuple[float, float, float, float, float, float]:
    """Return time derivatives of the H(1,0,0) Hermite basis."""

    nodes, time = _three_time_level_nodes(
        tau,
        previous_time_subinterval_length,
        current_time_subinterval_length,
    )
    return _three_node_hermite_basis_time_derivative(nodes, time)


def quintic_hermite_startup_basis(
    tau: float,
    first_time_subinterval_length: float,
    next_time_subinterval_length: float,
) -> tuple[float, float, float, float, float, float]:
    """Return the H(1,0,0) Hermite basis on the first time subinterval."""

    nodes, time = _startup_three_time_level_nodes(
        tau,
        first_time_subinterval_length,
        next_time_subinterval_length,
    )
    return _three_node_hermite_basis(nodes, time)


def quintic_hermite_startup_basis_time_derivative(
    tau: float,
    first_time_subinterval_length: float,
    next_time_subinterval_length: float,
) -> tuple[float, float, float, float, float, float]:
    """Return time derivatives of the H(1,0,0) startup Hermite basis."""

    nodes, time = _startup_three_time_level_nodes(
        tau,
        first_time_subinterval_length,
        next_time_subinterval_length,
    )
    return _three_node_hermite_basis_time_derivative(nodes, time)


def evaluate_cubic_hermite(
    tau: float,
    data: TimeSubintervalSolutionData,
    out: np.ndarray,
) -> None:
    """Evaluate the H(0,0,0) temporal reconstruction."""

    time_subinterval_length = _validate_data_and_output(data, out)
    basis_left, derivative_left, basis_right, derivative_right = cubic_hermite_basis(
        tau,
    )
    out[...] = basis_left * data.values[LEFT_ENDPOINT]
    out[...] += (
        time_subinterval_length
        * derivative_left
        * data.time_derivatives[LEFT_ENDPOINT]
    )
    out[...] += basis_right * data.values[RIGHT_ENDPOINT]
    out[...] += (
        time_subinterval_length
        * derivative_right
        * data.time_derivatives[RIGHT_ENDPOINT]
    )


def evaluate_cubic_hermite_time_derivative(
    tau: float,
    data: TimeSubintervalSolutionData,
    out: np.ndarray,
) -> None:
    """Evaluate the time derivative of the H(0,0,0) reconstruction."""

    time_subinterval_length = _validate_data_and_output(data, out)
    basis_left, derivative_left, basis_right, derivative_right = (
        cubic_hermite_basis_time_derivative(tau, time_subinterval_length)
    )
    out[...] = basis_left * data.values[LEFT_ENDPOINT]
    out[...] += derivative_left * data.time_derivatives[LEFT_ENDPOINT]
    out[...] += basis_right * data.values[RIGHT_ENDPOINT]
    out[...] += derivative_right * data.time_derivatives[RIGHT_ENDPOINT]


def evaluate_quintic_hermite(
    tau: float,
    previous_data: TimeSubintervalSolutionData,
    current_data: TimeSubintervalSolutionData,
    out: np.ndarray,
) -> None:
    """Evaluate the H(1,0,0) temporal reconstruction."""

    previous_length, current_length = _validate_adjacent_data_and_output(
        previous_data,
        current_data,
        out,
    )
    (
        value_previous,
        derivative_previous,
        value_left,
        derivative_left,
        value_right,
        derivative_right,
    ) = quintic_hermite_basis(tau, previous_length, current_length)
    out[...] = value_previous * previous_data.values[LEFT_ENDPOINT]
    out[...] += derivative_previous * previous_data.time_derivatives[LEFT_ENDPOINT]
    out[...] += value_left * current_data.values[LEFT_ENDPOINT]
    out[...] += derivative_left * current_data.time_derivatives[LEFT_ENDPOINT]
    out[...] += value_right * current_data.values[RIGHT_ENDPOINT]
    out[...] += derivative_right * current_data.time_derivatives[RIGHT_ENDPOINT]


def evaluate_quintic_hermite_time_derivative(
    tau: float,
    previous_data: TimeSubintervalSolutionData,
    current_data: TimeSubintervalSolutionData,
    out: np.ndarray,
) -> None:
    """Evaluate the time derivative of the H(1,0,0) reconstruction."""

    previous_length, current_length = _validate_adjacent_data_and_output(
        previous_data,
        current_data,
        out,
    )
    (
        value_previous,
        derivative_previous,
        value_left,
        derivative_left,
        value_right,
        derivative_right,
    ) = quintic_hermite_basis_time_derivative(tau, previous_length, current_length)
    out[...] = value_previous * previous_data.values[LEFT_ENDPOINT]
    out[...] += derivative_previous * previous_data.time_derivatives[LEFT_ENDPOINT]
    out[...] += value_left * current_data.values[LEFT_ENDPOINT]
    out[...] += derivative_left * current_data.time_derivatives[LEFT_ENDPOINT]
    out[...] += value_right * current_data.values[RIGHT_ENDPOINT]
    out[...] += derivative_right * current_data.time_derivatives[RIGHT_ENDPOINT]


def evaluate_quintic_hermite_startup(
    tau: float,
    first_data: TimeSubintervalSolutionData,
    next_data: TimeSubintervalSolutionData,
    out: np.ndarray,
) -> None:
    """Evaluate the H(1,0,0) reconstruction on the first time subinterval."""

    first_length, next_length = _validate_adjacent_data_and_output(
        first_data,
        next_data,
        out,
    )
    (
        value_left,
        derivative_left,
        value_middle,
        derivative_middle,
        value_right,
        derivative_right,
    ) = quintic_hermite_startup_basis(tau, first_length, next_length)
    out[...] = value_left * first_data.values[LEFT_ENDPOINT]
    out[...] += derivative_left * first_data.time_derivatives[LEFT_ENDPOINT]
    out[...] += value_middle * first_data.values[RIGHT_ENDPOINT]
    out[...] += derivative_middle * first_data.time_derivatives[RIGHT_ENDPOINT]
    out[...] += value_right * next_data.values[RIGHT_ENDPOINT]
    out[...] += derivative_right * next_data.time_derivatives[RIGHT_ENDPOINT]


def evaluate_quintic_hermite_startup_time_derivative(
    tau: float,
    first_data: TimeSubintervalSolutionData,
    next_data: TimeSubintervalSolutionData,
    out: np.ndarray,
) -> None:
    """Evaluate the time derivative of the H(1,0,0) startup reconstruction."""

    first_length, next_length = _validate_adjacent_data_and_output(
        first_data,
        next_data,
        out,
    )
    (
        value_left,
        derivative_left,
        value_middle,
        derivative_middle,
        value_right,
        derivative_right,
    ) = quintic_hermite_startup_basis_time_derivative(
        tau,
        first_length,
        next_length,
    )
    out[...] = value_left * first_data.values[LEFT_ENDPOINT]
    out[...] += derivative_left * first_data.time_derivatives[LEFT_ENDPOINT]
    out[...] += value_middle * first_data.values[RIGHT_ENDPOINT]
    out[...] += derivative_middle * first_data.time_derivatives[RIGHT_ENDPOINT]
    out[...] += value_right * next_data.values[RIGHT_ENDPOINT]
    out[...] += derivative_right * next_data.time_derivatives[RIGHT_ENDPOINT]


def evaluate_temporal(
    tau: float,
    data: TimeSubintervalSolutionData,
    out: np.ndarray,
) -> None:
    """Evaluate the default temporal reconstruction."""

    evaluate_cubic_hermite(tau, data, out)


def evaluate_hat_u_h_t_time_derivative(
    tau: float,
    data: TimeSubintervalSolutionData,
    out: np.ndarray,
) -> None:
    """Evaluate the time derivative of the default temporal reconstruction."""

    evaluate_cubic_hermite_time_derivative(tau, data, out)


def _validate_tau(tau: float) -> None:
    if tau < 0.0 or tau > 1.0:
        raise ValueError("tau must satisfy 0 <= tau <= 1")


def _three_time_level_nodes(
    tau: float,
    previous_time_subinterval_length: float,
    current_time_subinterval_length: float,
) -> tuple[tuple[float, float, float], float]:
    _validate_tau(tau)
    if previous_time_subinterval_length <= 0.0:
        raise ValueError("previous_time_subinterval_length must be positive")
    if current_time_subinterval_length <= 0.0:
        raise ValueError("current_time_subinterval_length must be positive")
    return (
        (-previous_time_subinterval_length, 0.0, current_time_subinterval_length),
        tau * current_time_subinterval_length,
    )


def _startup_three_time_level_nodes(
    tau: float,
    first_time_subinterval_length: float,
    next_time_subinterval_length: float,
) -> tuple[tuple[float, float, float], float]:
    _validate_tau(tau)
    if first_time_subinterval_length <= 0.0:
        raise ValueError("first_time_subinterval_length must be positive")
    if next_time_subinterval_length <= 0.0:
        raise ValueError("next_time_subinterval_length must be positive")
    return (
        (
            0.0,
            first_time_subinterval_length,
            first_time_subinterval_length + next_time_subinterval_length,
        ),
        tau * first_time_subinterval_length,
    )


def _three_node_hermite_basis(
    nodes: tuple[float, float, float],
    time: float,
) -> tuple[float, float, float, float, float, float]:
    value_0, derivative_0 = _three_node_hermite_basis_at_node(nodes, 0, time)
    value_1, derivative_1 = _three_node_hermite_basis_at_node(nodes, 1, time)
    value_2, derivative_2 = _three_node_hermite_basis_at_node(nodes, 2, time)
    return (
        value_0,
        derivative_0,
        value_1,
        derivative_1,
        value_2,
        derivative_2,
    )


def _three_node_hermite_basis_time_derivative(
    nodes: tuple[float, float, float],
    time: float,
) -> tuple[float, float, float, float, float, float]:
    value_0, derivative_0 = _three_node_hermite_time_derivative_at_node(
        nodes,
        0,
        time,
    )
    value_1, derivative_1 = _three_node_hermite_time_derivative_at_node(
        nodes,
        1,
        time,
    )
    value_2, derivative_2 = _three_node_hermite_time_derivative_at_node(
        nodes,
        2,
        time,
    )
    return (
        value_0,
        derivative_0,
        value_1,
        derivative_1,
        value_2,
        derivative_2,
    )


def _three_node_hermite_basis_at_node(
    nodes: tuple[float, float, float],
    node_index: int,
    time: float,
) -> tuple[float, float]:
    lagrange_value, _, lagrange_time_derivative_at_node = (
        _lagrange_basis_and_time_derivative(nodes, node_index, time)
    )
    offset = time - nodes[node_index]
    value_weight = (
        (1.0 - 2.0 * lagrange_time_derivative_at_node * offset)
        * lagrange_value
        * lagrange_value
    )
    derivative_weight = offset * lagrange_value * lagrange_value
    return value_weight, derivative_weight


def _three_node_hermite_time_derivative_at_node(
    nodes: tuple[float, float, float],
    node_index: int,
    time: float,
) -> tuple[float, float]:
    lagrange_value, lagrange_time_derivative, lagrange_time_derivative_at_node = (
        _lagrange_basis_and_time_derivative(nodes, node_index, time)
    )
    offset = time - nodes[node_index]
    value_factor = 1.0 - 2.0 * lagrange_time_derivative_at_node * offset
    value_weight = (
        -2.0 * lagrange_time_derivative_at_node * lagrange_value * lagrange_value
        + value_factor * 2.0 * lagrange_value * lagrange_time_derivative
    )
    derivative_weight = (
        lagrange_value * lagrange_value
        + offset * 2.0 * lagrange_value * lagrange_time_derivative
    )
    return value_weight, derivative_weight


def _lagrange_basis_and_time_derivative(
    nodes: tuple[float, float, float],
    node_index: int,
    time: float,
) -> tuple[float, float, float]:
    node = nodes[node_index]
    if node_index == 0:
        other_node_0 = nodes[1]
        other_node_1 = nodes[2]
    elif node_index == 1:
        other_node_0 = nodes[0]
        other_node_1 = nodes[2]
    else:
        other_node_0 = nodes[0]
        other_node_1 = nodes[1]
    denominator = (node - other_node_0) * (node - other_node_1)
    value = (time - other_node_0) * (time - other_node_1) / denominator
    time_derivative = (
        (time - other_node_0) + (time - other_node_1)
    ) / denominator
    time_derivative_at_node = (
        1.0 / (node - other_node_0) + 1.0 / (node - other_node_1)
    )
    return value, time_derivative, time_derivative_at_node


def _validate_data_and_output(
    data: TimeSubintervalSolutionData,
    out: np.ndarray,
) -> float:
    _validate_data(data)
    if out.shape != data.values.shape[1:]:
        raise ValueError(f"out must have shape {data.values.shape[1:]}")
    if out.dtype != np.float64:
        raise ValueError("out must have dtype numpy.float64")
    if not out.flags.c_contiguous:
        raise ValueError("out must be a C-contiguous NumPy array")
    time_subinterval_length = data.t_right - data.t_left
    if time_subinterval_length <= 0.0:
        raise ValueError("time subinterval length must be positive")
    return float(time_subinterval_length)


def _validate_adjacent_data_and_output(
    previous_data: TimeSubintervalSolutionData,
    current_data: TimeSubintervalSolutionData,
    out: np.ndarray,
) -> tuple[float, float]:
    previous_length = _validate_data_and_output(previous_data, out)
    current_length = _validate_data_and_output(current_data, out)
    if previous_data.values.shape != current_data.values.shape:
        raise ValueError("previous_data and current_data must have matching shapes")
    time_mismatch = abs(previous_data.t_right - current_data.t_left)
    time_scale = max(1.0, abs(previous_data.t_right), abs(current_data.t_left))
    if time_mismatch > 1.0e-12 * time_scale:
        raise ValueError("previous_data and current_data must be adjacent")
    return previous_length, current_length


def _validate_data(data: TimeSubintervalSolutionData) -> None:
    if data.values.dtype != np.float64:
        raise ValueError("values must have dtype numpy.float64")
    if data.time_derivatives.dtype != np.float64:
        raise ValueError("time_derivatives must have dtype numpy.float64")
    if data.values.shape != data.time_derivatives.shape:
        raise ValueError("values and time_derivatives must have matching shapes")
    if data.values.ndim != 6 or data.values.shape[0] != 2:
        raise ValueError("values must have shape (2, ex, ey, ix, iy, a)")
    if not data.values.flags.c_contiguous:
        raise ValueError("values must be a C-contiguous NumPy array")
    if not data.time_derivatives.flags.c_contiguous:
        raise ValueError("time_derivatives must be a C-contiguous NumPy array")


__all__ = [
    "cubic_hermite_basis",
    "cubic_hermite_basis_time_derivative",
    "evaluate_cubic_hermite",
    "evaluate_cubic_hermite_time_derivative",
    "evaluate_hat_u_h_t_time_derivative",
    "evaluate_quintic_hermite",
    "evaluate_quintic_hermite_startup",
    "evaluate_quintic_hermite_startup_time_derivative",
    "evaluate_quintic_hermite_time_derivative",
    "evaluate_temporal",
    "quintic_hermite_basis",
    "quintic_hermite_basis_time_derivative",
    "quintic_hermite_startup_basis",
    "quintic_hermite_startup_basis_time_derivative",
]
