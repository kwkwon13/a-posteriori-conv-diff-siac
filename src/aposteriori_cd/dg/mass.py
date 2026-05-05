from __future__ import annotations

import numpy as np

from aposteriori_cd.discrete import Mesh2D


def solve_tensor_product_mass_matrix(
    mass_1d: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    out = np.empty_like(rhs)
    for component in range(rhs.shape[-1]):
        work = np.linalg.solve(mass_1d, rhs[:, :, component])
        out[:, :, component] = np.linalg.solve(mass_1d, work.T).T
    return out


def solve_element_mass_matrix(
    mesh: Mesh2D,
    mass_1d: np.ndarray,
    element_rhs: np.ndarray,
) -> np.ndarray:
    jacobian = 0.25 * mesh.dx * mesh.dy
    return solve_tensor_product_mass_matrix(mass_1d, element_rhs) / jacobian
