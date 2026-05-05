from __future__ import annotations

import numpy as np

from aposteriori_cd.discrete import Basis2D, Mesh2D


def state_shape(
    mesh: Mesh2D,
    basis: Basis2D,
    num_components: int,
) -> tuple[int, int, int, int, int]:
    if num_components < 1:
        raise ValueError("num_components must be positive")
    n = basis.q + 1
    return (mesh.nx, mesh.ny, n, n, int(num_components))


def allocate_state(mesh: Mesh2D, basis: Basis2D, num_components: int) -> np.ndarray:
    return np.zeros(state_shape(mesh, basis, num_components), dtype=np.float64)
