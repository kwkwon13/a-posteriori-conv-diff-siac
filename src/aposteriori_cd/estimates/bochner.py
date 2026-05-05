"""Time quadrature for Bochner norm accumulation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aposteriori_cd.discrete import gauss_legendre_quadrature
from aposteriori_cd.temporal_reconstruction import (
    TemporalReconstructionTimeSubinterval,
)


DEFAULT_TIME_QUADRATURE_ORDER = 4


@dataclass(frozen=True, slots=True)
class BochnerTimeQuadrature:
    """Gauss--Legendre quadrature in the time-subinterval reference variable."""

    time_quadrature_order: int = DEFAULT_TIME_QUADRATURE_ORDER
    tau_nodes: np.ndarray = field(init=False)
    tau_weights: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.time_quadrature_order <= 0:
            raise ValueError("time_quadrature_order must be positive")
        nodes, weights = gauss_legendre_quadrature(self.time_quadrature_order)
        object.__setattr__(self, "tau_nodes", _readonly(0.5 * (nodes + 1.0)))
        object.__setattr__(self, "tau_weights", _readonly(0.5 * weights))

    def linf_points(self, subinterval: TemporalReconstructionTimeSubinterval):
        """Yield time nodes used for $$L^\\infty(0,T;L^2)$$ quantities."""

        for tau in self.tau_nodes:
            tau_value = float(tau)
            yield tau_value, subinterval.time_at(tau_value)

    def quadrature_points(self, subinterval: TemporalReconstructionTimeSubinterval):
        """Yield quadrature points and physical time weights."""

        for tau, weight in zip(self.tau_nodes, self.tau_weights):
            tau_value = float(tau)
            yield (
                tau_value,
                subinterval.time_at(tau_value),
                subinterval.length * float(weight),
            )


def build_bochner_time_quadrature(
    time_quadrature_order: int = DEFAULT_TIME_QUADRATURE_ORDER,
) -> BochnerTimeQuadrature:
    """Build default time quadrature for estimator accumulation."""

    return BochnerTimeQuadrature(time_quadrature_order=int(time_quadrature_order))


def _readonly(array: np.ndarray) -> np.ndarray:
    out = np.ascontiguousarray(array, dtype=np.float64)
    out.setflags(write=False)
    return out


__all__ = [
    "DEFAULT_TIME_QUADRATURE_ORDER",
    "BochnerTimeQuadrature",
    "build_bochner_time_quadrature",
]
