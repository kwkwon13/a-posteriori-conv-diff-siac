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
from aposteriori_cd.equations import (
    DiffusivePSystem2D,
    LinearAdvectionDiffusion2D,
    ViscousBurgers2D,
)

from .convection_compiled import (
    PROBLEM_BURGERS,
    PROBLEM_LINEAR,
    PROBLEM_P_SYSTEM,
    add_convection_compiled,
)
from .faces import (
    evaluate_exterior_face_trace,
    evaluate_interior_face_trace,
    face_measure_scale,
    face_normal,
)
from .mass import solve_element_mass_matrix


def evaluate_convection(
    mesh: Mesh2D,
    basis: Basis2D,
    equation: object,
    u: np.ndarray,
    out: np.ndarray,
) -> None:
    out.fill(0.0)
    add_convection(mesh, basis, equation, u, out)


def add_convection(
    mesh: Mesh2D,
    basis: Basis2D,
    equation: object,
    u: np.ndarray,
    rhs: np.ndarray,
) -> None:
    problem_id, velocity_x, velocity_y = _convection_problem_data(equation)
    add_convection_compiled(
        u,
        rhs,
        mesh.neighbor_ex,
        mesh.neighbor_ey,
        basis.basis_at_quad,
        basis.basis_derivative_at_quad,
        basis.quad_weights_1d,
        basis.mass_inverse_1d,
        mesh.dx,
        mesh.dy,
        problem_id,
        velocity_x,
        velocity_y,
    )


def _add_convection_python(
    mesh: Mesh2D,
    basis: Basis2D,
    equation: object,
    u: np.ndarray,
    rhs: np.ndarray,
) -> None:
    n = basis.q + 1
    n_quad = basis.quad_nodes_1d.shape[0]
    values_at_quad = np.empty(
        (n_quad, n_quad, equation.num_components),
        dtype=np.float64,
    )
    flux_at_quad = np.empty(
        (2, n_quad, n_quad, equation.num_components),
        dtype=np.float64,
    )
    element_rhs = np.empty((n, n, equation.num_components), dtype=np.float64)
    interior_trace = np.empty((n_quad, equation.num_components), dtype=np.float64)
    exterior_trace = np.empty_like(interior_trace)
    normal_flux = np.empty_like(interior_trace)

    for ex in range(mesh.nx):
        for ey in range(mesh.ny):
            element_rhs.fill(0.0)
            evaluate_element_values(u[ex, ey], basis, values_at_quad)
            for px in range(n_quad):
                for py in range(n_quad):
                    flux_at_quad[:, px, py, :] = equation.flux(values_at_quad[px, py])
            add_convection_volume_terms(mesh, basis, flux_at_quad, element_rhs)
            for face in (FACE_LEFT, FACE_RIGHT, FACE_BOTTOM, FACE_TOP):
                evaluate_interior_face_trace(u, basis, ex, ey, face, interior_trace)
                evaluate_exterior_face_trace(
                    mesh,
                    u,
                    basis,
                    ex,
                    ey,
                    face,
                    exterior_trace,
                )
                normal = face_normal(face)
                for point_index in range(n_quad):
                    normal_flux[point_index] = numerical_flux(
                        equation,
                        interior_trace[point_index],
                        exterior_trace[point_index],
                        normal,
                    )
                subtract_face_integral(mesh, basis, face, normal_flux, element_rhs)
            rhs[ex, ey, :, :, :] += solve_element_mass_matrix(
                mesh,
                basis.mass_1d,
                element_rhs,
            )


def numerical_flux(
    equation: object,
    u_minus: np.ndarray,
    u_plus: np.ndarray,
    normal: np.ndarray,
) -> np.ndarray:
    flux_minus = equation.flux(u_minus)
    flux_plus = equation.flux(u_plus)
    wavespeed = max(
        equation.max_wavespeed(u_minus, normal),
        equation.max_wavespeed(u_plus, normal),
    )
    return (
        0.5 * (normal[0] * (flux_minus[0] + flux_plus[0]))
        + 0.5 * (normal[1] * (flux_minus[1] + flux_plus[1]))
        - 0.5 * wavespeed * (u_plus - u_minus)
    )


def evaluate_element_values(
    element_u: np.ndarray,
    basis: Basis2D,
    out: np.ndarray,
) -> None:
    for component in range(element_u.shape[-1]):
        out[:, :, component] = (
            basis.basis_at_quad @ element_u[:, :, component] @ basis.basis_at_quad.T
        )


def add_convection_volume_terms(
    mesh: Mesh2D,
    basis: Basis2D,
    flux_at_quad: np.ndarray,
    element_rhs: np.ndarray,
) -> None:
    weights_2d = basis.quad_weights_1d[:, None] * basis.quad_weights_1d[None, :]
    for component in range(element_rhs.shape[-1]):
        weighted_flux_x = weights_2d * flux_at_quad[0, :, :, component]
        weighted_flux_y = weights_2d * flux_at_quad[1, :, :, component]
        element_rhs[:, :, component] += (
            0.5
            * mesh.dy
            * basis.basis_derivative_at_quad.T
            @ weighted_flux_x
            @ basis.basis_at_quad
        )
        element_rhs[:, :, component] += (
            0.5
            * mesh.dx
            * basis.basis_at_quad.T
            @ weighted_flux_y
            @ basis.basis_derivative_at_quad
        )


def subtract_face_integral(
    mesh: Mesh2D,
    basis: Basis2D,
    face: int,
    normal_flux: np.ndarray,
    element_rhs: np.ndarray,
) -> None:
    weighted_flux = (
        face_measure_scale(mesh, face) * basis.quad_weights_1d[:, None] * normal_flux
    )
    if face == FACE_LEFT:
                element_rhs[0, :, :] -= basis.basis_at_quad.T @ weighted_flux
    elif face == FACE_RIGHT:
        element_rhs[basis.q, :, :] -= basis.basis_at_quad.T @ weighted_flux
    elif face == FACE_BOTTOM:
        element_rhs[:, 0, :] -= basis.basis_at_quad.T @ weighted_flux
    else:
        element_rhs[:, basis.q, :] -= basis.basis_at_quad.T @ weighted_flux


def _convection_problem_data(equation: object) -> tuple[int, float, float]:
    if isinstance(equation, LinearAdvectionDiffusion2D):
        return PROBLEM_LINEAR, float(equation.velocity[0]), float(equation.velocity[1])
    if isinstance(equation, ViscousBurgers2D):
        return PROBLEM_BURGERS, 0.0, 0.0
    if isinstance(equation, DiffusivePSystem2D):
        return PROBLEM_P_SYSTEM, 0.0, 0.0
    raise TypeError("unsupported paper equation for compiled convection evaluation")
