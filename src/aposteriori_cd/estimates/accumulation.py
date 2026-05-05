"""Space-time norm accumulation on the SIAC analysis grid."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np

from aposteriori_cd.discrete import Mesh2D, gauss_legendre_quadrature

from .residuals import ResidualValues2D


@dataclass(frozen=True, slots=True)
class AnalysisGridQuadrature2D:
    """Spatial quadrature weights on the SIAC analysis grid."""

    analysis_nodes_1d: np.ndarray
    analysis_weights_1d: np.ndarray
    element_area_factor: float
    spatial_weights: np.ndarray


@dataclass(slots=True)
class ResidualNormAccumulator2D:
    """Accumulated norm for residual 1 and the flux-difference part of $$E_{r_2}$$."""

    residual_1_l1_l2: float = 0.0
    E_r2_squared: float = 0.0

    @property
    def E_r2(self) -> float:
        return sqrt(max(float(self.E_r2_squared), 0.0))


def build_analysis_grid_quadrature_2d(
    mesh: Mesh2D,
    analysis_nodes_1d: np.ndarray,
    analysis_weights_1d: np.ndarray | None = None,
) -> AnalysisGridQuadrature2D:
    """Build spatial quadrature weights on the SIAC analysis grid."""

    nodes = np.asarray(analysis_nodes_1d, dtype=np.float64)
    if nodes.ndim != 1:
        raise ValueError("analysis_nodes_1d must be one-dimensional")
    if nodes.shape[0] < 1:
        raise ValueError("analysis_nodes_1d must not be empty")

    if analysis_weights_1d is None:
        gauss_nodes, weights = gauss_legendre_quadrature(nodes.shape[0])
        if not np.allclose(nodes, gauss_nodes, rtol=0.0, atol=1.0e-14):
            raise ValueError(
                "analysis_nodes_1d must match Gauss--Legendre nodes "
                "when analysis_weights_1d is not provided"
            )
    else:
        weights = np.asarray(analysis_weights_1d, dtype=np.float64)

    if weights.ndim != 1:
        raise ValueError("analysis_weights_1d must be one-dimensional")
    if weights.shape != nodes.shape:
        raise ValueError("analysis_weights_1d must have the same shape as nodes")

    element_area_factor = 0.25 * mesh.dx * mesh.dy
    element_weights = element_area_factor * np.outer(weights, weights)
    spatial_weights = np.empty(
        (mesh.nx, mesh.ny, nodes.shape[0], nodes.shape[0]),
        dtype=np.float64,
    )
    spatial_weights[...] = element_weights
    return AnalysisGridQuadrature2D(
        analysis_nodes_1d=_readonly(nodes),
        analysis_weights_1d=_readonly(weights),
        element_area_factor=float(element_area_factor),
        spatial_weights=_readonly(spatial_weights),
    )


def accumulate_residual_norms(
    accumulator: ResidualNormAccumulator2D,
    quadrature: AnalysisGridQuadrature2D,
    residual_values: ResidualValues2D,
    time_weight: float,
) -> None:
    """Accumulate $$\\|r_1\\|_{L^1L^2}$$ and the $$E_{r_2}^2$$ flux-difference contribution."""

    residual_1_l2 = _spatial_l2_norm(
        quadrature.spatial_weights,
        residual_values.residual_1,
    )
    residual_2_bound = _spatial_integral(
        quadrature.spatial_weights,
        residual_values.residual_2_bound_integrand,
    )
    weight = float(time_weight)
    accumulator.residual_1_l1_l2 += weight * residual_1_l2
    accumulator.E_r2_squared += weight * residual_2_bound


def _spatial_l2_norm(spatial_weights: np.ndarray, values: np.ndarray) -> float:
    pointwise_square = np.sum(np.square(values), axis=-1)
    return sqrt(_spatial_integral(spatial_weights, pointwise_square))


def _spatial_integral(spatial_weights: np.ndarray, values: np.ndarray) -> float:
    return float(np.sum(spatial_weights * values))


def _readonly(array: np.ndarray) -> np.ndarray:
    out = np.ascontiguousarray(array, dtype=np.float64)
    out.setflags(write=False)
    return out


__all__ = [
    "AnalysisGridQuadrature2D",
    "ResidualNormAccumulator2D",
    "accumulate_residual_norms",
    "build_analysis_grid_quadrature_2d",
]
