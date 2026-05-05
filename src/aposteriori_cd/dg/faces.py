from __future__ import annotations

import numpy as np

from aposteriori_cd.discrete import (
    FACE_BOTTOM,
    FACE_LEFT,
    FACE_RIGHT,
    FACE_TOP,
    Basis2D,
    Mesh2D,
)


def evaluate_interior_face_trace(
    u: np.ndarray,
    basis: Basis2D,
    ex: int,
    ey: int,
    face: int,
    out: np.ndarray,
) -> None:
    if face == FACE_LEFT:
        for p in range(out.shape[0]):
            out[p, :] = basis.basis_at_quad[p, :] @ u[ex, ey, 0, :, :]
    elif face == FACE_RIGHT:
        for p in range(out.shape[0]):
            out[p, :] = basis.basis_at_quad[p, :] @ u[ex, ey, basis.q, :, :]
    elif face == FACE_BOTTOM:
        for p in range(out.shape[0]):
            out[p, :] = basis.basis_at_quad[p, :] @ u[ex, ey, :, 0, :]
    else:
        for p in range(out.shape[0]):
            out[p, :] = basis.basis_at_quad[p, :] @ u[ex, ey, :, basis.q, :]


def evaluate_exterior_face_trace(
    mesh: Mesh2D,
    u: np.ndarray,
    basis: Basis2D,
    ex: int,
    ey: int,
    face: int,
    out: np.ndarray,
) -> None:
    neighbor_ex, neighbor_ey = mesh.neighbor(ex, ey, face)
    opposite = opposite_face(face)
    evaluate_interior_face_trace(u, basis, neighbor_ex, neighbor_ey, opposite, out)


def face_normal(face: int) -> np.ndarray:
    if face == FACE_LEFT:
        return np.asarray([-1.0, 0.0], dtype=np.float64)
    if face == FACE_RIGHT:
        return np.asarray([1.0, 0.0], dtype=np.float64)
    if face == FACE_BOTTOM:
        return np.asarray([0.0, -1.0], dtype=np.float64)
    if face == FACE_TOP:
        return np.asarray([0.0, 1.0], dtype=np.float64)
    raise ValueError("unsupported face")


def face_measure_scale(mesh: Mesh2D, face: int) -> float:
    if face in (FACE_LEFT, FACE_RIGHT):
        return 0.5 * mesh.dy
    if face in (FACE_BOTTOM, FACE_TOP):
        return 0.5 * mesh.dx
    raise ValueError("unsupported face")


def opposite_face(face: int) -> int:
    if face == FACE_LEFT:
        return FACE_RIGHT
    if face == FACE_RIGHT:
        return FACE_LEFT
    if face == FACE_BOTTOM:
        return FACE_TOP
    if face == FACE_TOP:
        return FACE_BOTTOM
    raise ValueError("unsupported face")
