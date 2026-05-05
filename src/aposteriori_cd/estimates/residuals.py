"""Full residual convention and residual splitting calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aposteriori_cd.discrete import Mesh2D
from aposteriori_cd.equations import (
    DiffusivePSystem2D,
    LinearAdvectionDiffusion2D,
    ViscousBurgers2D,
    source_values,
)

from .reconstructions import SpaceTimeReconstructionValues2D


@dataclass(frozen=True, slots=True)
class ResidualEvaluator2D:
    """Pointwise residual evaluation data on the SIAC analysis grid."""

    equation: object
    physical_coordinates: np.ndarray
    source: np.ndarray


@dataclass(slots=True)
class ResidualValues2D:
    """Pointwise residual 1 and flux-difference values on the SIAC analysis grid."""

    residual_1: np.ndarray
    residual_2_flux_difference: np.ndarray
    residual_2_bound_integrand: np.ndarray


def build_residual_evaluator_2d(
    equation: object,
    mesh: Mesh2D,
    analysis_nodes_1d: np.ndarray,
) -> ResidualEvaluator2D:
    """Build pointwise residual evaluation data."""

    physical_coordinates = _physical_coordinates(mesh, analysis_nodes_1d)
    return ResidualEvaluator2D(
        equation=equation,
        physical_coordinates=physical_coordinates,
        source=np.zeros(
            (*physical_coordinates.shape[:-1], equation.num_components),
            dtype=np.float64,
        ),
    )


def allocate_residual_values_2d(evaluator: ResidualEvaluator2D) -> ResidualValues2D:
    """Allocate pointwise residual splitting arrays."""

    prefix = evaluator.source.shape[:-1]
    num_components = evaluator.source.shape[-1]
    return ResidualValues2D(
        residual_1=np.zeros((*prefix, num_components), dtype=np.float64),
        residual_2_flux_difference=np.zeros(
            (*prefix, 2, num_components),
            dtype=np.float64,
        ),
        residual_2_bound_integrand=np.zeros(prefix, dtype=np.float64),
    )


def evaluate_residual_splitting(
    evaluator: ResidualEvaluator2D,
    reconstruction_values: SpaceTimeReconstructionValues2D,
    time: float,
    residual_values: ResidualValues2D,
) -> None:
    """Evaluate residual 1 and the $$\\widetilde G-\\widehat G$$ bound integrand."""

    if getattr(evaluator.equation, "source_term_is_zero", False):
        evaluator.source.fill(0.0)
    else:
        source_values(
            evaluator.equation,
            time,
            evaluator.physical_coordinates,
            evaluator.source,
        )
    residual_values.residual_1[...] = (
        reconstruction_values.reconstructed_time_derivative - evaluator.source
    )
    _add_convective_flux_divergence(
        evaluator.equation,
        reconstruction_values,
        residual_values.residual_1,
    )
    _subtract_auxiliary_diffusion_divergence(
        evaluator.equation,
        reconstruction_values,
        residual_values.residual_1,
    )
    _evaluate_residual_2_flux_difference(
        evaluator.equation,
        reconstruction_values,
        residual_values.residual_2_flux_difference,
    )
    residual_values.residual_2_bound_integrand[...] = np.sum(
        np.square(residual_values.residual_2_flux_difference),
        axis=(-2, -1),
    )


def _physical_coordinates(mesh: Mesh2D, analysis_nodes_1d: np.ndarray) -> np.ndarray:
    n_analysis = analysis_nodes_1d.shape[0]
    coordinates = np.empty(
        (mesh.nx, mesh.ny, n_analysis, n_analysis, 2),
        dtype=np.float64,
    )
    for ex in range(mesh.nx):
        for ey in range(mesh.ny):
            x_center = mesh.element_centers_x[ex, ey]
            y_center = mesh.element_centers_y[ex, ey]
            for px, xi in enumerate(analysis_nodes_1d):
                x = x_center + 0.5 * mesh.dx * float(xi)
                for py, eta in enumerate(analysis_nodes_1d):
                    coordinates[ex, ey, px, py, 0] = x
                    coordinates[ex, ey, px, py, 1] = (
                        y_center + 0.5 * mesh.dy * float(eta)
                    )
    return _readonly(coordinates)


def _add_convective_flux_divergence(
    equation: object,
    reconstruction_values: SpaceTimeReconstructionValues2D,
    residual_1: np.ndarray,
) -> None:
    value = reconstruction_values.reconstructed_value
    gradient = reconstruction_values.reconstructed_gradient
    if isinstance(equation, LinearAdvectionDiffusion2D):
        residual_1[..., 0] += (
            equation.velocity[0] * gradient[..., 0, 0]
            + equation.velocity[1] * gradient[..., 1, 0]
        )
        return
    if isinstance(equation, ViscousBurgers2D):
        residual_1[..., 0] += value[..., 0] * (
            gradient[..., 0, 0] + gradient[..., 1, 0]
        )
        return
    if isinstance(equation, DiffusivePSystem2D):
        tau = value[..., 0]
        pressure_derivative = -2.0 / (tau * tau * tau)
        residual_1[..., 0] += -gradient[..., 0, 1] - gradient[..., 1, 2]
        residual_1[..., 1] += pressure_derivative * gradient[..., 0, 0]
        residual_1[..., 2] += pressure_derivative * gradient[..., 1, 0]
        return
    raise TypeError(f"unsupported equation type: {type(equation).__name__}")


def _subtract_auxiliary_diffusion_divergence(
    equation: object,
    reconstruction_values: SpaceTimeReconstructionValues2D,
    residual_1: np.ndarray,
) -> None:
    epsilon = float(equation.epsilon)
    if epsilon == 0.0:
        return
    auxiliary_second = reconstruction_values.auxiliary_directional_second_derivative
    for component in equation.diffusion_components:
        residual_1[..., component] -= epsilon * (
            auxiliary_second[..., 0, component]
            + auxiliary_second[..., 1, component]
        )


def _evaluate_residual_2_flux_difference(
    equation: object,
    reconstruction_values: SpaceTimeReconstructionValues2D,
    out: np.ndarray,
) -> None:
    out[...] = 0.0
    reconstructed_gradient = reconstruction_values.reconstructed_gradient
    auxiliary_derivative = reconstruction_values.auxiliary_directional_derivative
    for component in equation.diffusion_components:
        out[..., 0, component] = (
            auxiliary_derivative[..., 0, component]
            - reconstructed_gradient[..., 0, component]
        )
        out[..., 1, component] = (
            auxiliary_derivative[..., 1, component]
            - reconstructed_gradient[..., 1, component]
        )


def _readonly(array: np.ndarray) -> np.ndarray:
    out = np.ascontiguousarray(array)
    out.setflags(write=False)
    return out


__all__ = [
    "ResidualEvaluator2D",
    "ResidualValues2D",
    "allocate_residual_values_2d",
    "build_residual_evaluator_2d",
    "evaluate_residual_splitting",
]
