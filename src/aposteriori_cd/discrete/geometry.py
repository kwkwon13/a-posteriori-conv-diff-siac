from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._arrays import readonly
from .basis import Basis2D
from .faces import FACE_BOTTOM, FACE_LEFT, FACE_RIGHT, FACE_TOP
from .mesh import Mesh2D


@dataclass(frozen=True, slots=True)
class Geometry2D:
    """Affine physical coordinates and affine map scaling factors."""

    node_x: np.ndarray
    node_y: np.ndarray
    quad_x: np.ndarray
    quad_y: np.ndarray
    jacobian: float
    dx_dxi: float
    dy_deta: float
    dxi_dx: float
    deta_dy: float
    face_scale: np.ndarray


def build_geometry_2d(mesh: Mesh2D, basis: Basis2D) -> Geometry2D:
    """Build affine coordinates and affine map scaling factors."""

    n = basis.q + 1
    nq = basis.quad_nodes_1d.shape[0]
    node_x = np.empty((mesh.nx, mesh.ny, n, n), dtype=np.float64)
    node_y = np.empty((mesh.nx, mesh.ny, n, n), dtype=np.float64)
    quad_x = np.empty((mesh.nx, mesh.ny, nq, nq), dtype=np.float64)
    quad_y = np.empty((mesh.nx, mesh.ny, nq, nq), dtype=np.float64)
    half_dx = 0.5 * mesh.dx
    half_dy = 0.5 * mesh.dy

    for ex in range(mesh.nx):
        for ey in range(mesh.ny):
            x_center = mesh.element_centers_x[ex, ey]
            y_center = mesh.element_centers_y[ex, ey]
            for ix in range(n):
                x = x_center + half_dx * basis.nodes_1d[ix]
                for iy in range(n):
                    y = y_center + half_dy * basis.nodes_1d[iy]
                    node_x[ex, ey, ix, iy] = x
                    node_y[ex, ey, ix, iy] = y
            for px in range(nq):
                x = x_center + half_dx * basis.quad_nodes_1d[px]
                for py in range(nq):
                    y = y_center + half_dy * basis.quad_nodes_1d[py]
                    quad_x[ex, ey, px, py] = x
                    quad_y[ex, ey, px, py] = y

    face_scale = np.empty(4, dtype=np.float64)
    face_scale[FACE_LEFT] = half_dy
    face_scale[FACE_RIGHT] = half_dy
    face_scale[FACE_BOTTOM] = half_dx
    face_scale[FACE_TOP] = half_dx

    return Geometry2D(
        node_x=readonly(node_x),
        node_y=readonly(node_y),
        quad_x=readonly(quad_x),
        quad_y=readonly(quad_y),
        jacobian=float(half_dx * half_dy),
        dx_dxi=float(half_dx),
        dy_deta=float(half_dy),
        dxi_dx=float(1.0 / half_dx),
        deta_dy=float(1.0 / half_dy),
        face_scale=readonly(face_scale),
    )
