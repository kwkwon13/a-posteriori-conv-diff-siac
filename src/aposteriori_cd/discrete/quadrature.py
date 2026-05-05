from __future__ import annotations

import numpy as np
from scipy.special import eval_legendre, roots_jacobi, roots_legendre

from ._arrays import readonly


def gauss_legendre_quadrature(npoints: int) -> tuple[np.ndarray, np.ndarray]:
    """Return Gauss--Legendre nodes and weights on [-1, 1]."""

    if npoints < 1:
        raise ValueError("npoints must be positive")
    nodes, weights = roots_legendre(npoints)
    return readonly(nodes.astype(np.float64)), readonly(weights.astype(np.float64))


def gauss_lobatto_legendre_nodes(npoints: int) -> np.ndarray:
    """Return Gauss--Lobatto--Legendre nodes on [-1, 1]."""

    if npoints < 1:
        raise ValueError("npoints must be positive")
    if npoints == 1:
        return readonly(np.asarray([0.0], dtype=np.float64))
    if npoints == 2:
        return readonly(np.asarray([-1.0, 1.0], dtype=np.float64))

    interior, _ = roots_jacobi(npoints - 2, 1.0, 1.0)
    nodes = np.empty(npoints, dtype=np.float64)
    nodes[0] = -1.0
    nodes[-1] = 1.0
    nodes[1:-1] = interior
    return readonly(nodes)


def gauss_lobatto_legendre_quadrature(
    npoints: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Gauss--Lobatto--Legendre nodes and weights on [-1, 1]."""

    nodes = np.array(gauss_lobatto_legendre_nodes(npoints), copy=True)
    if npoints == 1:
        return readonly(nodes), readonly(np.asarray([2.0], dtype=np.float64))

    q = npoints - 1
    legendre_values = eval_legendre(q, nodes)
    weights = 2.0 / (q * (q + 1) * np.square(legendre_values))
    return readonly(nodes), readonly(weights.astype(np.float64))
