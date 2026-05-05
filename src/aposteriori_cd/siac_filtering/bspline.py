from __future__ import annotations

from math import comb, factorial

import numpy as np


def central_bspline(order: int, x: float) -> float:
    """Evaluate the centered B-spline of the given order."""

    if order < 1:
        raise ValueError("order must be positive")
    half_order = 0.5 * float(order)
    if x < -half_order or x > half_order:
        return 0.0
    if order == 1:
        return 1.0

    shifted_x = float(x) + half_order
    value = 0.0
    for knot_index in range(order + 1):
        truncated = shifted_x - float(knot_index)
        if truncated > 0.0:
            value += (
                (-1.0) ** knot_index
                * float(comb(order, knot_index))
                * truncated ** (order - 1)
            )
    return value / float(factorial(order - 1))


def central_bspline_derivative(order: int, x: float) -> float:
    """Evaluate the first derivative of a centered B-spline."""

    if order < 2:
        raise ValueError("order must be at least two")
    return central_bspline(order - 1, x + 0.5) - central_bspline(
        order - 1,
        x - 0.5,
    )


def central_bspline_second_derivative(order: int, x: float) -> float:
    """Evaluate the second derivative of a centered B-spline."""

    if order < 3:
        raise ValueError("order must be at least three")
    return central_bspline_derivative(order - 1, x + 0.5) - central_bspline_derivative(
        order - 1,
        x - 0.5,
    )


def evaluate_central_bspline(order: int, points: np.ndarray) -> np.ndarray:
    """Evaluate a centered B-spline at all points."""

    out = np.empty(points.shape, dtype=np.float64)
    for index, point in np.ndenumerate(points):
        out[index] = central_bspline(order, float(point))
    return out


def evaluate_central_bspline_derivative(order: int, points: np.ndarray) -> np.ndarray:
    """Evaluate the first derivative of a centered B-spline at all points."""

    out = np.empty(points.shape, dtype=np.float64)
    for index, point in np.ndenumerate(points):
        out[index] = central_bspline_derivative(order, float(point))
    return out


def evaluate_central_bspline_second_derivative(
    order: int,
    points: np.ndarray,
) -> np.ndarray:
    """Evaluate the second derivative of a centered B-spline at all points."""

    out = np.empty(points.shape, dtype=np.float64)
    for index, point in np.ndenumerate(points):
        out[index] = central_bspline_second_derivative(order, float(point))
    return out


__all__ = [
    "central_bspline",
    "central_bspline_derivative",
    "central_bspline_second_derivative",
    "evaluate_central_bspline",
    "evaluate_central_bspline_derivative",
    "evaluate_central_bspline_second_derivative",
]
