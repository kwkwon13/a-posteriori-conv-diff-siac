from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np

from aposteriori_cd.discrete import Basis2D, Mesh2D, evaluate_lagrange_basis

from .auxiliary_compiled import (
    evaluate_tilde_u_ts_directional_derivative_compiled,
    evaluate_tilde_u_ts_directional_derivatives_compiled,
    evaluate_tilde_u_ts_directional_derivatives_with_work_compiled,
    evaluate_tilde_u_ts_directional_second_derivative_compiled,
)
from .evaluator import SIACFilterEvaluator2D, build_siac_filter_evaluator_2d
from .kernel import SIACKernel2D, build_siac_kernel_2d, evaluate_siac_kernel_1d


@dataclass(frozen=True, slots=True)
class AuxiliarySIACFilterEvaluator2D:
    """Precomputed data for directional auxiliary SIAC derivatives."""

    classical_evaluator: SIACFilterEvaluator2D
    auxiliary_evaluator: SIACFilterEvaluator2D


def build_auxiliary_siac_filter_evaluator_2d(
    mesh: Mesh2D,
    basis: Basis2D,
    *,
    analysis_nodes_1d: np.ndarray | None = None,
    quadrature_order: int | None = None,
) -> AuxiliarySIACFilterEvaluator2D:
    """Build directional auxiliary SIAC derivative evaluators."""

    classical_evaluator = build_siac_filter_evaluator_2d(
        mesh,
        basis,
        analysis_nodes_1d=analysis_nodes_1d,
        quadrature_order=quadrature_order,
    )
    n_quad = _quadrature_order(basis.q, quadrature_order)
    derivative_offsets_x, derivative_weights_x = _build_thomee_axis_weights_1d(
        basis=basis,
        analysis_nodes=classical_evaluator.analysis_nodes_1d,
        derivative_order=1,
        cell_size=mesh.dx,
        quadrature_order=n_quad,
    )
    derivative_offsets_y, derivative_weights_y = _build_thomee_axis_weights_1d(
        basis=basis,
        analysis_nodes=classical_evaluator.analysis_nodes_1d,
        derivative_order=1,
        cell_size=mesh.dy,
        quadrature_order=n_quad,
    )
    second_offsets_x, second_derivative_weights_x = _build_thomee_axis_weights_1d(
        basis=basis,
        analysis_nodes=classical_evaluator.analysis_nodes_1d,
        derivative_order=2,
        cell_size=mesh.dx,
        quadrature_order=n_quad,
    )
    second_offsets_y, second_derivative_weights_y = _build_thomee_axis_weights_1d(
        basis=basis,
        analysis_nodes=classical_evaluator.analysis_nodes_1d,
        derivative_order=2,
        cell_size=mesh.dy,
        quadrature_order=n_quad,
    )

    offsets_x = np.unique(np.concatenate((derivative_offsets_x, second_offsets_x)))
    offsets_y = np.unique(np.concatenate((derivative_offsets_y, second_offsets_y)))
    aligned_derivative_weights_x = _align_weights_to_offsets(
        derivative_weights_x,
        derivative_offsets_x,
        offsets_x,
    )
    aligned_derivative_weights_y = _align_weights_to_offsets(
        derivative_weights_y,
        derivative_offsets_y,
        offsets_y,
    )
    aligned_second_derivative_weights_x = _align_weights_to_offsets(
        second_derivative_weights_x,
        second_offsets_x,
        offsets_x,
    )
    aligned_second_derivative_weights_y = _align_weights_to_offsets(
        second_derivative_weights_y,
        second_offsets_y,
        offsets_y,
    )
    auxiliary_evaluator = SIACFilterEvaluator2D(
        kernel=build_siac_kernel_2d(basis.q, spline_order=basis.q + 3),
        nx=mesh.nx,
        ny=mesh.ny,
        num_basis_nodes=basis.q + 1,
        analysis_nodes_1d=classical_evaluator.analysis_nodes_1d,
        offsets_x=_readonly(offsets_x.astype(np.int32, copy=False)),
        offsets_y=_readonly(offsets_y.astype(np.int32, copy=False)),
        weights_x=_readonly(np.zeros_like(aligned_derivative_weights_x)),
        weights_y=_readonly(np.zeros_like(aligned_derivative_weights_y)),
        derivative_weights_x=_readonly(aligned_derivative_weights_x),
        derivative_weights_y=_readonly(aligned_derivative_weights_y),
        second_derivative_weights_x=_readonly(aligned_second_derivative_weights_x),
        second_derivative_weights_y=_readonly(aligned_second_derivative_weights_y),
    )
    return AuxiliarySIACFilterEvaluator2D(
        classical_evaluator=classical_evaluator,
        auxiliary_evaluator=auxiliary_evaluator,
    )


def allocate_tilde_u_ts_directional_derivative(
    evaluator: AuxiliarySIACFilterEvaluator2D,
    num_components: int,
) -> np.ndarray:
    """Allocate values of partial_{x_beta} tilde u^{ts,beta}."""

    if num_components < 1:
        raise ValueError("num_components must be positive")
    classical = evaluator.classical_evaluator
    n_analysis = classical.analysis_nodes_1d.shape[0]
    return np.zeros(
        (classical.nx, classical.ny, n_analysis, n_analysis, 2, num_components),
        dtype=np.float64,
    )


def allocate_tilde_u_ts_directional_second_derivative(
    evaluator: AuxiliarySIACFilterEvaluator2D,
    num_components: int,
) -> np.ndarray:
    """Allocate values of partial_{x_beta x_beta} tilde u^{ts,beta}."""

    return allocate_tilde_u_ts_directional_derivative(evaluator, num_components)


def allocate_tilde_u_ts_directional_work(
    evaluator: AuxiliarySIACFilterEvaluator2D,
    num_components: int,
) -> np.ndarray:
    """Allocate reusable work values for auxiliary directional derivatives."""

    if num_components < 1:
        raise ValueError("num_components must be positive")
    classical = evaluator.classical_evaluator
    n_analysis = classical.analysis_nodes_1d.shape[0]
    return np.zeros(
        (
            classical.nx,
            classical.ny,
            n_analysis,
            classical.num_basis_nodes,
            num_components,
        ),
        dtype=np.float64,
    )


def evaluate_tilde_u_ts_directional_derivative(
    evaluator: AuxiliarySIACFilterEvaluator2D,
    u: np.ndarray,
    out: np.ndarray,
) -> None:
    """Evaluate partial_{x_beta} tilde u^{ts,beta} for beta = 0, 1."""

    classical = evaluator.classical_evaluator
    auxiliary = evaluator.auxiliary_evaluator
    evaluate_tilde_u_ts_directional_derivative_compiled(
        u,
        classical.offsets_x,
        classical.offsets_y,
        auxiliary.offsets_x,
        auxiliary.offsets_y,
        classical.weights_x,
        classical.weights_y,
        auxiliary.derivative_weights_x,
        auxiliary.derivative_weights_y,
        out,
    )


def evaluate_tilde_u_ts_directional_second_derivative(
    evaluator: AuxiliarySIACFilterEvaluator2D,
    u: np.ndarray,
    out: np.ndarray,
) -> None:
    """Evaluate partial_{x_beta x_beta} tilde u^{ts,beta} for beta = 0, 1."""

    classical = evaluator.classical_evaluator
    auxiliary = evaluator.auxiliary_evaluator
    assert auxiliary.second_derivative_weights_x is not None
    assert auxiliary.second_derivative_weights_y is not None
    evaluate_tilde_u_ts_directional_second_derivative_compiled(
        u,
        classical.offsets_x,
        classical.offsets_y,
        auxiliary.offsets_x,
        auxiliary.offsets_y,
        classical.weights_x,
        classical.weights_y,
        auxiliary.second_derivative_weights_x,
        auxiliary.second_derivative_weights_y,
        out,
    )


def evaluate_tilde_u_ts_directional_derivatives(
    evaluator: AuxiliarySIACFilterEvaluator2D,
    u: np.ndarray,
    derivative_out: np.ndarray,
    second_derivative_out: np.ndarray,
) -> None:
    """Evaluate first and second directional auxiliary SIAC derivatives."""

    classical = evaluator.classical_evaluator
    auxiliary = evaluator.auxiliary_evaluator
    assert auxiliary.second_derivative_weights_x is not None
    assert auxiliary.second_derivative_weights_y is not None
    evaluate_tilde_u_ts_directional_derivatives_compiled(
        u,
        classical.offsets_x,
        classical.offsets_y,
        auxiliary.offsets_x,
        auxiliary.offsets_y,
        classical.weights_x,
        classical.weights_y,
        auxiliary.derivative_weights_x,
        auxiliary.derivative_weights_y,
        auxiliary.second_derivative_weights_x,
        auxiliary.second_derivative_weights_y,
        derivative_out,
        second_derivative_out,
    )


def evaluate_tilde_u_ts_directional_derivatives_with_work(
    evaluator: AuxiliarySIACFilterEvaluator2D,
    u: np.ndarray,
    derivative_out: np.ndarray,
    second_derivative_out: np.ndarray,
    work: np.ndarray,
) -> None:
    """Evaluate first and second directional auxiliary derivatives with work."""

    classical = evaluator.classical_evaluator
    auxiliary = evaluator.auxiliary_evaluator
    assert auxiliary.second_derivative_weights_x is not None
    assert auxiliary.second_derivative_weights_y is not None
    evaluate_tilde_u_ts_directional_derivatives_with_work_compiled(
        u,
        classical.offsets_x,
        classical.offsets_y,
        auxiliary.offsets_x,
        auxiliary.offsets_y,
        classical.weights_x,
        classical.weights_y,
        auxiliary.derivative_weights_x,
        auxiliary.derivative_weights_y,
        auxiliary.second_derivative_weights_x,
        auxiliary.second_derivative_weights_y,
        work,
        derivative_out,
        second_derivative_out,
    )


def _quadrature_order(polynomial_degree_q: int, quadrature_order: int | None) -> int:
    if quadrature_order is None:
        return max(64, 8 * polynomial_degree_q + 8)
    if quadrature_order < 1:
        raise ValueError("quadrature_order must be positive")
    return int(quadrature_order)


def _build_thomee_axis_weights_1d(
    *,
    basis: Basis2D,
    analysis_nodes: np.ndarray,
    derivative_order: int,
    cell_size: float,
    quadrature_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    q = int(basis.q)
    if derivative_order not in (1, 2):
        raise ValueError("Thomee auxiliary weights require derivative order 1 or 2")

    coefficient_kernel = build_siac_kernel_2d(
        q,
        spline_order=q + 1 + derivative_order,
    )
    base_kernel = SIACKernel2D(
        polynomial_degree_q=q,
        num_splines=coefficient_kernel.num_splines,
        spline_order=q + 1,
        shifts_1d=coefficient_kernel.shifts_1d,
        coefficients_1d=coefficient_kernel.coefficients_1d,
        support_radius=float(q + 0.5 * (q + 1)),
    )
    finite_difference_shifts = np.asarray(
        [
            0.5 * float(derivative_order) - float(index)
            for index in range(derivative_order + 1)
        ],
        dtype=np.float64,
    )
    finite_difference_coefficients = np.asarray(
        [
            ((-1.0) ** index) * float(comb(derivative_order, index))
            for index in range(derivative_order + 1)
        ],
        dtype=np.float64,
    )

    offset_sets: list[np.ndarray] = []
    partial_weights: list[np.ndarray] = []
    for target_shift in finite_difference_shifts:
        offsets, weights = _build_shifted_value_weights_1d(
            kernel=base_kernel,
            basis_nodes=basis.nodes_1d,
            analysis_nodes=analysis_nodes,
            target_shift=float(target_shift),
            quadrature_order=quadrature_order,
        )
        offset_sets.append(offsets)
        partial_weights.append(weights)

    merged_offsets = np.unique(np.concatenate(offset_sets)).astype(
        np.int32,
        copy=False,
    )
    weights = np.zeros(
        (
            analysis_nodes.shape[0],
            merged_offsets.shape[0],
            basis.nodes_1d.shape[0],
        ),
        dtype=np.float64,
    )
    for coefficient, offsets, partial in zip(
        finite_difference_coefficients,
        offset_sets,
        partial_weights,
        strict=True,
    ):
        insert_indices = np.searchsorted(merged_offsets, offsets)
        weights[:, insert_indices, :] += coefficient * partial

    weights /= float(cell_size) ** derivative_order
    return merged_offsets, weights


def _build_shifted_value_weights_1d(
    *,
    kernel: SIACKernel2D,
    basis_nodes: np.ndarray,
    analysis_nodes: np.ndarray,
    target_shift: float,
    quadrature_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    target_points = 0.5 * analysis_nodes + float(target_shift)
    offsets = _candidate_offsets(target_points, kernel.support_radius)
    quadrature_nodes, quadrature_weights = np.polynomial.legendre.leggauss(
        quadrature_order,
    )
    quadrature_nodes = quadrature_nodes.astype(np.float64)
    quadrature_weights = quadrature_weights.astype(np.float64)
    weights = np.zeros(
        (analysis_nodes.shape[0], offsets.shape[0], basis_nodes.shape[0]),
        dtype=np.float64,
    )

    for point_index, reference_point in enumerate(analysis_nodes):
        for offset_index, offset in enumerate(offsets):
            intervals = _shifted_integration_intervals(
                kernel,
                float(reference_point),
                int(offset),
                float(target_shift),
            )
            for interval_left, interval_right in intervals:
                mapped_nodes = (
                    0.5 * (interval_right - interval_left) * quadrature_nodes
                    + 0.5 * (interval_left + interval_right)
                )
                mapped_weights = (
                    0.5 * (interval_right - interval_left) * quadrature_weights
                )
                basis_values = evaluate_lagrange_basis(basis_nodes, mapped_nodes)
                kernel_values = np.empty(mapped_nodes.shape[0], dtype=np.float64)
                for quadrature_index, source_reference_point in enumerate(
                    mapped_nodes,
                ):
                    kernel_argument = (
                        float(target_shift)
                        - float(offset)
                        + 0.5
                        * (float(reference_point) - float(source_reference_point))
                    )
                    kernel_values[quadrature_index] = evaluate_siac_kernel_1d(
                        kernel,
                        kernel_argument,
                    )
                weights[point_index, offset_index, :] += basis_values.T @ (
                    0.5 * mapped_weights * kernel_values
                )
    return offsets, weights


def _candidate_offsets(target_points: np.ndarray, support_radius: float) -> np.ndarray:
    lower = int(np.floor(np.min(target_points - support_radius - 0.5)))
    upper = int(np.ceil(np.max(target_points + support_radius + 0.5)))
    offsets = []
    for offset in range(lower, upper + 1):
        interval_left = float(offset) - 0.5
        interval_right = float(offset) + 0.5
        active = False
        for target_point in target_points:
            support_left = float(target_point) - float(support_radius)
            support_right = float(target_point) + float(support_radius)
            if interval_right > support_left and interval_left < support_right:
                active = True
                break
        if active:
            offsets.append(offset)
    return np.asarray(offsets, dtype=np.int32)


def _shifted_integration_intervals(
    kernel: SIACKernel2D,
    reference_point: float,
    offset: int,
    target_shift: float,
) -> list[tuple[float, float]]:
    breaks = [-1.0, 1.0]
    target_point = 0.5 * float(reference_point) + float(target_shift)
    for shift in kernel.shifts_1d:
        for knot_index in range(kernel.spline_order + 1):
            kernel_break = (
                float(shift)
                - 0.5 * float(kernel.spline_order)
                + float(knot_index)
            )
            source_reference_point = 2.0 * (
                target_point - kernel_break - float(offset)
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


def _align_weights_to_offsets(
    weights: np.ndarray,
    offsets: np.ndarray,
    target_offsets: np.ndarray,
) -> np.ndarray:
    out = np.zeros(
        (weights.shape[0], target_offsets.shape[0], weights.shape[2]),
        dtype=np.float64,
    )
    insert_indices = np.searchsorted(target_offsets, offsets)
    out[:, insert_indices, :] = weights
    return out


def _readonly(array: np.ndarray) -> np.ndarray:
    out = np.ascontiguousarray(array)
    out.setflags(write=False)
    return out


__all__ = [
    "AuxiliarySIACFilterEvaluator2D",
    "allocate_tilde_u_ts_directional_work",
    "allocate_tilde_u_ts_directional_derivative",
    "allocate_tilde_u_ts_directional_second_derivative",
    "build_auxiliary_siac_filter_evaluator_2d",
    "evaluate_tilde_u_ts_directional_derivative",
    "evaluate_tilde_u_ts_directional_derivatives",
    "evaluate_tilde_u_ts_directional_derivatives_with_work",
    "evaluate_tilde_u_ts_directional_second_derivative",
]
