"""Evaluation of paper reconstruction quantities used by the estimators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aposteriori_cd.discrete import Basis2D, Mesh2D
from aposteriori_cd.siac_filtering import (
    AuxiliarySIACFilterEvaluator2D,
    SIACFilterEvaluator2D,
    allocate_filtered_gradient,
    allocate_filtered_work,
    allocate_filtered_value,
    allocate_tilde_u_ts_directional_derivative,
    allocate_tilde_u_ts_directional_second_derivative,
    allocate_tilde_u_ts_directional_work,
    build_auxiliary_siac_filter_evaluator_2d,
    evaluate_filtered_gradient,
    evaluate_filtered_value,
    evaluate_filtered_value_and_gradient,
    evaluate_filtered_value_gradient_and_second_value_with_work,
    evaluate_tilde_u_ts_directional_derivatives,
    evaluate_tilde_u_ts_directional_derivatives_with_work,
)
from aposteriori_cd.temporal_reconstruction import (
    TemporalReconstructionTimeSubinterval,
)


@dataclass(frozen=True, slots=True)
class SpaceTimeReconstructionEvaluator2D:
    """Evaluation data for the space-time reconstruction."""

    classical_evaluator: SIACFilterEvaluator2D
    auxiliary_evaluator: AuxiliarySIACFilterEvaluator2D
    temporal_value: np.ndarray
    temporal_time_derivative: np.ndarray
    filtered_primary_work: np.ndarray
    filtered_secondary_work: np.ndarray
    auxiliary_work: np.ndarray


@dataclass(slots=True)
class SpaceTimeReconstructionValues2D:
    """Reconstruction values on the SIAC analysis grid."""

    reconstructed_value: np.ndarray
    reconstructed_time_derivative: np.ndarray
    reconstructed_gradient: np.ndarray
    auxiliary_directional_derivative: np.ndarray
    auxiliary_directional_second_derivative: np.ndarray


def build_space_time_reconstruction_evaluator_2d(
    mesh: Mesh2D,
    basis: Basis2D,
    num_components: int,
    *,
    analysis_nodes_1d: np.ndarray | None = None,
    quadrature_order: int | None = None,
) -> SpaceTimeReconstructionEvaluator2D:
    """Build work arrays and SIAC evaluators for space-time reconstruction."""

    if num_components < 1:
        raise ValueError("num_components must be positive")
    auxiliary_evaluator = build_auxiliary_siac_filter_evaluator_2d(
        mesh,
        basis,
        analysis_nodes_1d=analysis_nodes_1d,
        quadrature_order=quadrature_order,
    )
    state_shape = (
        mesh.nx,
        mesh.ny,
        basis.q + 1,
        basis.q + 1,
        int(num_components),
    )
    return SpaceTimeReconstructionEvaluator2D(
        classical_evaluator=auxiliary_evaluator.classical_evaluator,
        auxiliary_evaluator=auxiliary_evaluator,
        temporal_value=np.zeros(state_shape, dtype=np.float64),
        temporal_time_derivative=np.zeros(state_shape, dtype=np.float64),
        filtered_primary_work=allocate_filtered_work(
            auxiliary_evaluator.classical_evaluator,
            num_components,
        ),
        filtered_secondary_work=allocate_filtered_work(
            auxiliary_evaluator.classical_evaluator,
            num_components,
        ),
        auxiliary_work=allocate_tilde_u_ts_directional_work(
            auxiliary_evaluator,
            num_components,
        ),
    )


def allocate_space_time_reconstruction_values_2d(
    evaluator: SpaceTimeReconstructionEvaluator2D,
    num_components: int,
) -> SpaceTimeReconstructionValues2D:
    """Allocate reconstruction values on the SIAC analysis grid."""

    return SpaceTimeReconstructionValues2D(
        reconstructed_value=allocate_filtered_value(
            evaluator.classical_evaluator,
            num_components,
        ),
        reconstructed_time_derivative=allocate_filtered_value(
            evaluator.classical_evaluator,
            num_components,
        ),
        reconstructed_gradient=allocate_filtered_gradient(
            evaluator.classical_evaluator,
            num_components,
        ),
        auxiliary_directional_derivative=allocate_tilde_u_ts_directional_derivative(
            evaluator.auxiliary_evaluator,
            num_components,
        ),
        auxiliary_directional_second_derivative=(
            allocate_tilde_u_ts_directional_second_derivative(
                evaluator.auxiliary_evaluator,
                num_components,
            )
        ),
    )


def evaluate_space_time_reconstruction(
    evaluator: SpaceTimeReconstructionEvaluator2D,
    time_subinterval: TemporalReconstructionTimeSubinterval,
    tau: float,
    values: SpaceTimeReconstructionValues2D,
    *,
    value: bool = True,
    time_derivative: bool = True,
    gradient: bool = True,
    auxiliary_derivatives: bool = True,
) -> None:
    """Evaluate the requested reconstruction quantities on the SIAC grid."""

    needs_temporal_value = value or gradient or auxiliary_derivatives
    if needs_temporal_value:
        time_subinterval.evaluate(tau, evaluator.temporal_value)
    if time_derivative:
        time_subinterval.evaluate_time_derivative(
            tau,
            evaluator.temporal_time_derivative,
        )

    if value and gradient:
        if time_derivative:
            evaluate_filtered_value_gradient_and_second_value_with_work(
                evaluator.classical_evaluator,
                evaluator.temporal_value,
                evaluator.temporal_time_derivative,
                values.reconstructed_value,
                values.reconstructed_gradient,
                values.reconstructed_time_derivative,
                evaluator.filtered_primary_work,
                evaluator.filtered_secondary_work,
            )
        else:
            evaluate_filtered_value_and_gradient(
                evaluator.classical_evaluator,
                evaluator.temporal_value,
                values.reconstructed_value,
                values.reconstructed_gradient,
            )
    elif value:
        evaluate_filtered_value(
            evaluator.classical_evaluator,
            evaluator.temporal_value,
            values.reconstructed_value,
        )
    elif gradient:
        evaluate_filtered_gradient(
            evaluator.classical_evaluator,
            evaluator.temporal_value,
            values.reconstructed_gradient,
        )

    if time_derivative and not (value and gradient):
        evaluate_filtered_value(
            evaluator.classical_evaluator,
            evaluator.temporal_time_derivative,
            values.reconstructed_time_derivative,
        )
    if auxiliary_derivatives:
        evaluate_tilde_u_ts_directional_derivatives_with_work(
            evaluator.auxiliary_evaluator,
            evaluator.temporal_value,
            values.auxiliary_directional_derivative,
            values.auxiliary_directional_second_derivative,
            evaluator.auxiliary_work,
        )


__all__ = [
    "SpaceTimeReconstructionEvaluator2D",
    "SpaceTimeReconstructionValues2D",
    "allocate_space_time_reconstruction_values_2d",
    "build_space_time_reconstruction_evaluator_2d",
    "evaluate_space_time_reconstruction",
]
