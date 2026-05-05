from __future__ import annotations

import numpy as np

from .apply_compiled import (
    evaluate_filtered_gradient_compiled,
    evaluate_filtered_value_compiled,
    evaluate_filtered_value_and_gradient_compiled,
    evaluate_filtered_value_gradient_and_second_value_compiled,
    evaluate_filtered_value_gradient_and_second_value_with_work_compiled,
)
from .evaluator import SIACFilterEvaluator2D


def allocate_filtered_value(
    evaluator: SIACFilterEvaluator2D,
    num_components: int,
) -> np.ndarray:
    """Allocate SIAC filtered values on the analysis grid."""

    if num_components < 1:
        raise ValueError("num_components must be positive")
    n_analysis = evaluator.analysis_nodes_1d.shape[0]
    return np.zeros(
        (evaluator.nx, evaluator.ny, n_analysis, n_analysis, num_components),
        dtype=np.float64,
    )


def allocate_filtered_gradient(
    evaluator: SIACFilterEvaluator2D,
    num_components: int,
) -> np.ndarray:
    """Allocate SIAC filtered spatial derivatives on the analysis grid."""

    if num_components < 1:
        raise ValueError("num_components must be positive")
    n_analysis = evaluator.analysis_nodes_1d.shape[0]
    return np.zeros(
        (evaluator.nx, evaluator.ny, n_analysis, n_analysis, 2, num_components),
        dtype=np.float64,
    )


def allocate_filtered_work(
    evaluator: SIACFilterEvaluator2D,
    num_components: int,
) -> np.ndarray:
    """Allocate reusable tensor-product SIAC work values."""

    if num_components < 1:
        raise ValueError("num_components must be positive")
    n_analysis = evaluator.analysis_nodes_1d.shape[0]
    return np.zeros(
        (
            evaluator.nx,
            evaluator.ny,
            n_analysis,
            evaluator.num_basis_nodes,
            num_components,
        ),
        dtype=np.float64,
    )


def evaluate_filtered_value(
    evaluator: SIACFilterEvaluator2D,
    u: np.ndarray,
    out: np.ndarray,
) -> None:
    """Evaluate the tensor-product SIAC filtered value."""

    evaluate_filtered_value_compiled(
        u,
        evaluator.offsets_x,
        evaluator.offsets_y,
        evaluator.weights_x,
        evaluator.weights_y,
        out,
    )


def evaluate_filtered_gradient(
    evaluator: SIACFilterEvaluator2D,
    u: np.ndarray,
    out: np.ndarray,
) -> None:
    """Evaluate the tensor-product SIAC filtered spatial derivatives."""

    evaluate_filtered_gradient_compiled(
        u,
        evaluator.offsets_x,
        evaluator.offsets_y,
        evaluator.weights_x,
        evaluator.weights_y,
        evaluator.derivative_weights_x,
        evaluator.derivative_weights_y,
        out,
    )


def evaluate_filtered_value_and_gradient(
    evaluator: SIACFilterEvaluator2D,
    u: np.ndarray,
    value_out: np.ndarray,
    gradient_out: np.ndarray,
) -> None:
    """Evaluate SIAC filtered values and spatial derivatives."""

    evaluate_filtered_value_and_gradient_compiled(
        u,
        evaluator.offsets_x,
        evaluator.offsets_y,
        evaluator.weights_x,
        evaluator.weights_y,
        evaluator.derivative_weights_x,
        evaluator.derivative_weights_y,
        value_out,
        gradient_out,
    )


def evaluate_filtered_value_gradient_and_second_value(
    evaluator: SIACFilterEvaluator2D,
    u: np.ndarray,
    second_u: np.ndarray,
    value_out: np.ndarray,
    gradient_out: np.ndarray,
    second_value_out: np.ndarray,
) -> None:
    """Evaluate value/gradient of one state and filtered value of another."""

    evaluate_filtered_value_gradient_and_second_value_compiled(
        u,
        second_u,
        evaluator.offsets_x,
        evaluator.offsets_y,
        evaluator.weights_x,
        evaluator.weights_y,
        evaluator.derivative_weights_x,
        evaluator.derivative_weights_y,
        value_out,
        gradient_out,
        second_value_out,
    )


def evaluate_filtered_value_gradient_and_second_value_with_work(
    evaluator: SIACFilterEvaluator2D,
    u: np.ndarray,
    second_u: np.ndarray,
    value_out: np.ndarray,
    gradient_out: np.ndarray,
    second_value_out: np.ndarray,
    primary_work: np.ndarray,
    secondary_work: np.ndarray,
) -> None:
    """Evaluate value/gradient of one state and filtered value of another."""

    evaluate_filtered_value_gradient_and_second_value_with_work_compiled(
        u,
        second_u,
        evaluator.offsets_x,
        evaluator.offsets_y,
        evaluator.weights_x,
        evaluator.weights_y,
        evaluator.derivative_weights_x,
        evaluator.derivative_weights_y,
        primary_work,
        secondary_work,
        value_out,
        gradient_out,
        second_value_out,
    )


__all__ = [
    "allocate_filtered_gradient",
    "allocate_filtered_work",
    "allocate_filtered_value",
    "evaluate_filtered_gradient",
    "evaluate_filtered_value",
    "evaluate_filtered_value_and_gradient",
    "evaluate_filtered_value_gradient_and_second_value",
    "evaluate_filtered_value_gradient_and_second_value_with_work",
]
