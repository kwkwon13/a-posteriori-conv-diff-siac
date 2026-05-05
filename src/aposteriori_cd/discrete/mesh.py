from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._arrays import readonly
from .faces import FACE_BOTTOM, FACE_LEFT, FACE_RIGHT, FACE_TOP


@dataclass(frozen=True, slots=True)
class Mesh2D:
    """Periodic uniform Cartesian mesh on a rectangular domain."""

    nx: int
    ny: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    dx: float
    dy: float
    element_centers_x: np.ndarray
    element_centers_y: np.ndarray
    neighbor_ex: np.ndarray
    neighbor_ey: np.ndarray

    def neighbor(self, ex: int, ey: int, face: int) -> tuple[int, int]:
        return (
            int(self.neighbor_ex[ex, ey, face]),
            int(self.neighbor_ey[ex, ey, face]),
        )


def build_mesh_2d(
    nx: int,
    ny: int,
    *,
    x_bounds: tuple[float, float] = (0.0, 1.0),
    y_bounds: tuple[float, float] = (0.0, 1.0),
) -> Mesh2D:
    """Build the periodic uniform Cartesian mesh."""

    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be positive")
    x_min, x_max = float(x_bounds[0]), float(x_bounds[1])
    y_min, y_max = float(y_bounds[0]), float(y_bounds[1])
    if not x_min < x_max:
        raise ValueError("x_bounds must be increasing")
    if not y_min < y_max:
        raise ValueError("y_bounds must be increasing")

    dx = (x_max - x_min) / float(nx)
    dy = (y_max - y_min) / float(ny)
    element_centers_x = np.empty((nx, ny), dtype=np.float64)
    element_centers_y = np.empty((nx, ny), dtype=np.float64)
    neighbor_ex = np.empty((nx, ny, 4), dtype=np.int32)
    neighbor_ey = np.empty((nx, ny, 4), dtype=np.int32)

    for ex in range(nx):
        x_center = x_min + (float(ex) + 0.5) * dx
        for ey in range(ny):
            y_center = y_min + (float(ey) + 0.5) * dy
            element_centers_x[ex, ey] = x_center
            element_centers_y[ex, ey] = y_center
            neighbor_ex[ex, ey, FACE_LEFT] = (ex - 1) % nx
            neighbor_ey[ex, ey, FACE_LEFT] = ey
            neighbor_ex[ex, ey, FACE_RIGHT] = (ex + 1) % nx
            neighbor_ey[ex, ey, FACE_RIGHT] = ey
            neighbor_ex[ex, ey, FACE_BOTTOM] = ex
            neighbor_ey[ex, ey, FACE_BOTTOM] = (ey - 1) % ny
            neighbor_ex[ex, ey, FACE_TOP] = ex
            neighbor_ey[ex, ey, FACE_TOP] = (ey + 1) % ny

    return Mesh2D(
        nx=int(nx),
        ny=int(ny),
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        dx=float(dx),
        dy=float(dy),
        element_centers_x=readonly(element_centers_x),
        element_centers_y=readonly(element_centers_y),
        neighbor_ex=readonly(neighbor_ex),
        neighbor_ey=readonly(neighbor_ey),
    )
