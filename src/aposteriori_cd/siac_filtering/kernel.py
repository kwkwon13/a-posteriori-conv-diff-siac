from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np

from .bspline import (
    central_bspline,
    central_bspline_derivative,
    central_bspline_second_derivative,
)


@dataclass(frozen=True, slots=True)
class SIACKernel2D:
    """Tensor-product SIAC convolution kernel data."""

    polynomial_degree_q: int
    num_splines: int
    spline_order: int
    shifts_1d: np.ndarray
    coefficients_1d: np.ndarray
    support_radius: float


def build_siac_kernel_2d(
    polynomial_degree_q: int,
    *,
    spline_order: int | None = None,
) -> SIACKernel2D:
    """Build tensor-product SIAC convolution kernel data."""

    q = int(polynomial_degree_q)
    if q < 0:
        raise ValueError("polynomial_degree_q must be non-negative")
    if spline_order is None:
        spline_order = q + 1
    spline_order = int(spline_order)
    if spline_order < 1:
        raise ValueError("spline_order must be positive")

    num_splines = 2 * q + 1
    shifts = np.arange(-q, q + 1, dtype=np.float64)
    coefficients = _compute_coefficients(shifts, spline_order)
    support_radius = q + 0.5 * float(spline_order)
    return SIACKernel2D(
        polynomial_degree_q=q,
        num_splines=num_splines,
        spline_order=spline_order,
        shifts_1d=_readonly(shifts.astype(np.int32)),
        coefficients_1d=_readonly(coefficients),
        support_radius=float(support_radius),
    )


def evaluate_siac_kernel_1d(kernel: SIACKernel2D, x: float) -> float:
    """Evaluate the unscaled one-dimensional SIAC convolution kernel."""

    if x < -kernel.support_radius or x > kernel.support_radius:
        return 0.0
    value = 0.0
    for shift, coefficient in zip(kernel.shifts_1d, kernel.coefficients_1d):
        value += float(coefficient) * central_bspline(
            kernel.spline_order,
            x - float(shift),
        )
    return value


def evaluate_siac_kernel_derivative_1d(kernel: SIACKernel2D, x: float) -> float:
    """Evaluate the first derivative of the unscaled SIAC convolution kernel."""

    if kernel.spline_order < 2:
        raise ValueError("kernel spline_order must be at least two")
    if x < -kernel.support_radius or x > kernel.support_radius:
        return 0.0
    value = 0.0
    for shift, coefficient in zip(kernel.shifts_1d, kernel.coefficients_1d):
        value += float(coefficient) * central_bspline_derivative(
            kernel.spline_order,
            x - float(shift),
        )
    return value


def evaluate_siac_kernel_second_derivative_1d(
    kernel: SIACKernel2D,
    x: float,
) -> float:
    """Evaluate the second derivative of the unscaled SIAC convolution kernel."""

    if kernel.spline_order < 3:
        raise ValueError("kernel spline_order must be at least three")
    if x < -kernel.support_radius or x > kernel.support_radius:
        return 0.0
    value = 0.0
    for shift, coefficient in zip(kernel.shifts_1d, kernel.coefficients_1d):
        value += float(coefficient) * central_bspline_second_derivative(
            kernel.spline_order,
            x - float(shift),
        )
    return value


def siac_kernel_moment_1d(kernel: SIACKernel2D, degree: int) -> float:
    """Return the moment of the one-dimensional SIAC convolution kernel."""

    if degree < 0:
        raise ValueError("degree must be non-negative")
    moment = 0.0
    for shift, coefficient in zip(kernel.shifts_1d, kernel.coefficients_1d):
        moment += float(coefficient) * _shifted_bspline_moment(
            kernel.spline_order,
            float(shift),
            degree,
        )
    return moment


def _compute_coefficients(shifts: np.ndarray, spline_order: int) -> np.ndarray:
    reproduce_degree = shifts.shape[0] - 1
    system = np.empty((shifts.shape[0], shifts.shape[0]), dtype=np.float64)
    right_hand_side = np.zeros(shifts.shape[0], dtype=np.float64)
    right_hand_side[0] = 1.0

    for degree in range(reproduce_degree + 1):
        for column, shift in enumerate(shifts):
            system[degree, column] = _shifted_bspline_moment(
                spline_order,
                float(shift),
                degree,
            )

    return np.linalg.solve(system, right_hand_side)


def _shifted_bspline_moment(
    spline_order: int,
    shift: float,
    degree: int,
) -> float:
    central_moments = _central_bspline_moments(spline_order, degree)
    moment = 0.0
    for central_degree in range(degree + 1):
        moment += (
            float(comb(degree, central_degree))
            * shift ** (degree - central_degree)
            * central_moments[central_degree]
        )
    return moment


def _central_bspline_moments(spline_order: int, max_degree: int) -> np.ndarray:
    uniform_moments = np.zeros(max_degree + 1, dtype=np.float64)
    for degree in range(max_degree + 1):
        if degree % 2 == 0:
            uniform_moments[degree] = 1.0 / (2.0**degree * (degree + 1))

    moments = np.zeros(max_degree + 1, dtype=np.float64)
    moments[0] = 1.0
    for _ in range(spline_order):
        moments = _sum_independent_moments(moments, uniform_moments)
    return moments


def _sum_independent_moments(
    moments_a: np.ndarray,
    moments_b: np.ndarray,
) -> np.ndarray:
    max_degree = moments_a.shape[0] - 1
    out = np.zeros(max_degree + 1, dtype=np.float64)
    for degree in range(max_degree + 1):
        for left_degree in range(degree + 1):
            out[degree] += (
                float(comb(degree, left_degree))
                * moments_a[left_degree]
                * moments_b[degree - left_degree]
            )
    return out


def _readonly(array: np.ndarray) -> np.ndarray:
    out = np.ascontiguousarray(array)
    out.setflags(write=False)
    return out


__all__ = [
    "SIACKernel2D",
    "build_siac_kernel_2d",
    "evaluate_siac_kernel_derivative_1d",
    "evaluate_siac_kernel_1d",
    "evaluate_siac_kernel_second_derivative_1d",
    "siac_kernel_moment_1d",
]
