from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aposteriori_cd.discrete import Basis2D, Mesh2D

from .convection import add_convection
from .diffusion import DEFAULT_PENALTY_PARAMETER, evaluate_diffusion
from .projection import add_source_projection
from .state import allocate_state


@dataclass(slots=True)
class DGSemidiscretization2D:
    """Reusable DG spatial semidiscretization data."""

    mesh: Mesh2D
    basis: Basis2D
    equation: object
    source_projection: np.ndarray
    diffusion_penalty_parameter: float = DEFAULT_PENALTY_PARAMETER
    diffusion_diagonal_cache: dict[tuple[tuple[int, ...], float], np.ndarray] = field(
        default_factory=dict,
    )
    diffusion_stage_stencil_cache: dict[tuple[object, ...], object] = field(
        default_factory=dict,
    )

    def evaluate_explicit_rhs(self, time: float, u: np.ndarray, rhs: np.ndarray) -> None:
        rhs.fill(0.0)
        add_convection(self.mesh, self.basis, self.equation, u, rhs)
        add_source_projection(
            self.mesh,
            self.basis,
            self.equation,
            time,
            rhs,
            self.source_projection,
        )

    def evaluate_diffusion(self, u: np.ndarray, rhs: np.ndarray) -> None:
        evaluate_diffusion(
            self.mesh,
            self.basis,
            self.equation,
            u,
            rhs,
            penalty_parameter=self.diffusion_penalty_parameter,
        )

    def evaluate_rhs(self, time: float, u: np.ndarray, rhs: np.ndarray) -> None:
        self.evaluate_explicit_rhs(time, u, rhs)
        if self.equation.epsilon != 0.0:
            diffusion = np.empty_like(rhs)
            self.evaluate_diffusion(u, diffusion)
            rhs += self.equation.epsilon * diffusion


def build_dg_semidiscretization(
    mesh: Mesh2D,
    basis: Basis2D,
    equation: object,
    *,
    diffusion_penalty_parameter: float = DEFAULT_PENALTY_PARAMETER,
) -> DGSemidiscretization2D:
    if diffusion_penalty_parameter < 0.0:
        raise ValueError("diffusion_penalty_parameter must be non-negative")
    return DGSemidiscretization2D(
        mesh=mesh,
        basis=basis,
        equation=equation,
        source_projection=allocate_state(mesh, basis, equation.num_components),
        diffusion_penalty_parameter=float(diffusion_penalty_parameter),
    )
