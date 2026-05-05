from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aposteriori_cd.discrete import (
    Basis2D,
    Mesh2D,
    evaluate_lagrange_basis,
    gauss_legendre_quadrature,
)

from .kernel import (
    SIACKernel2D,
    build_siac_kernel_2d,
    evaluate_siac_kernel_derivative_1d,
    evaluate_siac_kernel_1d,
    evaluate_siac_kernel_second_derivative_1d,
)


@dataclass(frozen=True, slots=True)
class SIACFilterEvaluator2D:
    """Tensor-product weights for SIAC filtered values and derivatives."""

    kernel: SIACKernel2D
    nx: int
    ny: int
    num_basis_nodes: int
    analysis_nodes_1d: np.ndarray
    offsets_x: np.ndarray
    offsets_y: np.ndarray
    weights_x: np.ndarray
    weights_y: np.ndarray
    derivative_weights_x: np.ndarray
    derivative_weights_y: np.ndarray
    second_derivative_weights_x: np.ndarray | None
    second_derivative_weights_y: np.ndarray | None


def build_siac_filter_evaluator_2d(
    mesh: Mesh2D,
    basis: Basis2D,
    *,
    analysis_nodes_1d: np.ndarray | None = None,
    quadrature_order: int | None = None,
    spline_order: int | None = None,
) -> SIACFilterEvaluator2D:
    """Build tensor-product SIAC evaluation weights on the analysis grid."""

    kernel = build_siac_kernel_2d(basis.q, spline_order=spline_order)
    if analysis_nodes_1d is None:
        analysis_nodes_1d = _analysis_nodes(basis.q)
    else:
        analysis_nodes_1d = _validate_analysis_nodes(analysis_nodes_1d)

    offsets = _offsets(kernel)
    n_quad = _quadrature_order(basis.q, quadrature_order)
    weights, derivative_weights, second_derivative_weights = _build_weights_1d(
        kernel,
        basis.nodes_1d,
        analysis_nodes_1d,
        offsets,
        n_quad,
    )
    return SIACFilterEvaluator2D(
        kernel=kernel,
        nx=mesh.nx,
        ny=mesh.ny,
        num_basis_nodes=basis.q + 1,
        analysis_nodes_1d=_readonly(analysis_nodes_1d),
        offsets_x=_readonly(offsets),
        offsets_y=_readonly(np.array(offsets, copy=True)),
        weights_x=_readonly(weights),
        weights_y=_readonly(np.array(weights, copy=True)),
        derivative_weights_x=_readonly(derivative_weights / mesh.dx),
        derivative_weights_y=_readonly(
            np.array(derivative_weights / mesh.dy, copy=True),
        ),
        second_derivative_weights_x=_readonly(second_derivative_weights / mesh.dx**2)
        if second_derivative_weights is not None
        else None,
        second_derivative_weights_y=_readonly(
            np.array(second_derivative_weights / mesh.dy**2, copy=True),
        )
        if second_derivative_weights is not None
        else None,
    )


def _analysis_nodes(polynomial_degree_q: int) -> np.ndarray:
    nodes, _ = gauss_legendre_quadrature(2 * polynomial_degree_q + 3)
    return np.array(nodes, copy=True)


def _validate_analysis_nodes(analysis_nodes_1d: np.ndarray) -> np.ndarray:
    if analysis_nodes_1d.dtype != np.float64:
        raise ValueError("analysis_nodes_1d must have dtype numpy.float64")
    if analysis_nodes_1d.ndim != 1:
        raise ValueError("analysis_nodes_1d must be one-dimensional")
    if analysis_nodes_1d.shape[0] < 1:
        raise ValueError("analysis_nodes_1d must not be empty")
    if np.any(analysis_nodes_1d < -1.0) or np.any(analysis_nodes_1d > 1.0):
        raise ValueError("analysis_nodes_1d must lie in [-1, 1]")
    return np.ascontiguousarray(analysis_nodes_1d)


def _offsets(kernel: SIACKernel2D) -> np.ndarray:
    max_offset = int(np.ceil(kernel.support_radius + 1.0))
    return np.arange(-max_offset, max_offset + 1, dtype=np.int32)


def _quadrature_order(polynomial_degree_q: int, quadrature_order: int | None) -> int:
    if quadrature_order is None:
        return max(64, 8 * polynomial_degree_q + 8)
    if quadrature_order < 1:
        raise ValueError("quadrature_order must be positive")
    return int(quadrature_order)


def _build_weights_1d(
    kernel: SIACKernel2D,
    basis_nodes: np.ndarray,
    analysis_nodes: np.ndarray,
    offsets: np.ndarray,
    quadrature_order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    quadrature_nodes, quadrature_weights = gauss_legendre_quadrature(
        quadrature_order,
    )
    weights = np.zeros(
        (analysis_nodes.shape[0], offsets.shape[0], basis_nodes.shape[0]),
        dtype=np.float64,
    )
    derivative_weights = np.zeros_like(weights)
    second_derivative_weights = None
    if kernel.spline_order >= 3:
        second_derivative_weights = np.zeros_like(weights)

    for point_index, reference_point in enumerate(analysis_nodes):
        for offset_index, offset in enumerate(offsets):
            intervals = _integration_intervals(kernel, float(reference_point), int(offset))
            for interval_left, interval_right in intervals:
                mapped_nodes = (
                    0.5 * (interval_right - interval_left) * quadrature_nodes
                    + 0.5 * (interval_left + interval_right)
                )
                mapped_weights = (
                    0.5 * (interval_right - interval_left) * quadrature_weights
                )
                basis_values = evaluate_lagrange_basis(basis_nodes, mapped_nodes)
                for basis_index in range(basis_nodes.shape[0]):
                    for quadrature_index, source_reference_point in enumerate(
                        mapped_nodes,
                    ):
                        kernel_argument = (
                            -float(offset)
                            + 0.5
                            * (float(reference_point) - float(source_reference_point))
                        )
                        basis_value = float(basis_values[quadrature_index, basis_index])
                        quadrature_weight = float(mapped_weights[quadrature_index])
                        weights[point_index, offset_index, basis_index] += (
                            0.5
                            * quadrature_weight
                            * evaluate_siac_kernel_1d(kernel, kernel_argument)
                            * basis_value
                        )
                        derivative_weights[point_index, offset_index, basis_index] += (
                            0.5
                            * quadrature_weight
                            * evaluate_siac_kernel_derivative_1d(
                                kernel,
                                kernel_argument,
                            )
                            * basis_value
                        )
                        if second_derivative_weights is not None:
                            second_derivative_weights[
                                point_index,
                                offset_index,
                                basis_index,
                            ] += (
                                0.5
                                * quadrature_weight
                                * evaluate_siac_kernel_second_derivative_1d(
                                    kernel,
                                    kernel_argument,
                                )
                                * basis_value
                            )
    return weights, derivative_weights, second_derivative_weights


def _integration_intervals(
    kernel: SIACKernel2D,
    reference_point: float,
    offset: int,
) -> list[tuple[float, float]]:
    breaks = [-1.0, 1.0]
    for shift in kernel.shifts_1d:
        for knot_index in range(kernel.spline_order + 1):
            kernel_break = (
                float(shift)
                - 0.5 * float(kernel.spline_order)
                + float(knot_index)
            )
            source_reference_point = reference_point - 2.0 * (
                kernel_break + float(offset)
            )
            if -1.0 < source_reference_point < 1.0:
                breaks.append(source_reference_point)

    intervals = []
    previous = sorted(breaks)[0]
    for point in sorted(breaks)[1:]:
        if point - previous > 1.0e-14:
            intervals.append((previous, point))
        previous = point
    return intervals


def _readonly(array: np.ndarray) -> np.ndarray:
    out = np.ascontiguousarray(array)
    out.setflags(write=False)
    return out


__all__ = [
    "SIACFilterEvaluator2D",
    "build_siac_filter_evaluator_2d",
]
