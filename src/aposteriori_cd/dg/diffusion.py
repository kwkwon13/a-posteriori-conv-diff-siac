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

from .diffusion_compiled import (
    PROBLEM_BURGERS,
    PROBLEM_LINEAR,
    PROBLEM_P_SYSTEM,
    add_diffusion_compiled,
)
from .mass import solve_tensor_product_mass_matrix


DEFAULT_PENALTY_PARAMETER = 2.0


def evaluate_diffusion(
    mesh: Mesh2D,
    basis: Basis2D,
    equation: object,
    u: np.ndarray,
    out: np.ndarray,
    *,
    penalty_parameter: float = DEFAULT_PENALTY_PARAMETER,
) -> None:
    out.fill(0.0)
    add_diffusion(
        mesh,
        basis,
        equation,
        u,
        out,
        penalty_parameter=penalty_parameter,
    )


def add_diffusion(
    mesh: Mesh2D,
    basis: Basis2D,
    equation: object,
    u: np.ndarray,
    rhs: np.ndarray,
    *,
    penalty_parameter: float = DEFAULT_PENALTY_PARAMETER,
    scale: float = 1.0,
) -> None:
    if penalty_parameter < 0.0:
        raise ValueError("penalty_parameter must be non-negative")
    problem_id = _diffusion_problem_id(equation)
    add_diffusion_compiled(
        u,
        rhs,
        mesh.neighbor_ex,
        mesh.neighbor_ey,
        basis.basis_at_quad,
        basis.basis_derivative_at_quad,
        basis.derivative_1d,
        basis.quad_weights_1d,
        basis.mass_inverse_1d,
        mesh.dx,
        mesh.dy,
        float(penalty_parameter),
        float(scale),
        problem_id,
    )


def _add_diffusion_python(
    mesh: Mesh2D,
    basis: Basis2D,
    equation: object,
    u: np.ndarray,
    rhs: np.ndarray,
    *,
    penalty_parameter: float = DEFAULT_PENALTY_PARAMETER,
    scale: float = 1.0,
) -> None:
    if penalty_parameter < 0.0:
        raise ValueError("penalty_parameter must be non-negative")
    n = basis.q + 1
    element_scale = 0.25 * mesh.dx * mesh.dy
    element_rhs = np.empty((n, n, equation.num_components), dtype=np.float64)

    for ex in range(mesh.nx):
        for ey in range(mesh.ny):
            element_rhs.fill(0.0)
            add_diffusion_volume_terms(mesh, basis, equation, u, ex, ey, element_rhs)
            add_vertical_diffusion_face(
                mesh,
                basis,
                equation,
                u,
                ex,
                ey,
                FACE_LEFT,
                -1.0,
                penalty_parameter,
                element_rhs,
            )
            add_vertical_diffusion_face(
                mesh,
                basis,
                equation,
                u,
                ex,
                ey,
                FACE_RIGHT,
                1.0,
                penalty_parameter,
                element_rhs,
            )
            add_horizontal_diffusion_face(
                mesh,
                basis,
                equation,
                u,
                ex,
                ey,
                FACE_BOTTOM,
                -1.0,
                penalty_parameter,
                element_rhs,
            )
            add_horizontal_diffusion_face(
                mesh,
                basis,
                equation,
                u,
                ex,
                ey,
                FACE_TOP,
                1.0,
                penalty_parameter,
                element_rhs,
            )
            rhs[ex, ey, :, :, :] += (
                scale
                * solve_tensor_product_mass_matrix(basis.mass_1d, element_rhs)
                / element_scale
            )


def add_diffusion_volume_terms(
    mesh: Mesh2D,
    basis: Basis2D,
    equation: object,
    u: np.ndarray,
    ex: int,
    ey: int,
    element_rhs: np.ndarray,
) -> None:
    n = basis.q + 1
    n_quad = basis.quad_nodes_1d.shape[0]
    value = np.empty(equation.num_components, dtype=np.float64)
    gradient = np.empty((2, equation.num_components), dtype=np.float64)
    for px in range(n_quad):
        for py in range(n_quad):
            value.fill(0.0)
            gradient.fill(0.0)
            for ix in range(n):
                basis_x = basis.basis_at_quad[px, ix]
                derivative_x = basis.basis_derivative_at_quad[px, ix]
                for iy in range(n):
                    basis_y = basis.basis_at_quad[py, iy]
                    derivative_y = basis.basis_derivative_at_quad[py, iy]
                    coefficient = u[ex, ey, ix, iy, :]
                    value += basis_x * basis_y * coefficient
                    gradient[0] += derivative_x * basis_y * coefficient
                    gradient[1] += basis_x * derivative_y * coefficient
            gradient[0] *= 2.0 / mesh.dx
            gradient[1] *= 2.0 / mesh.dy
            weight = basis.quad_weights_1d[px] * basis.quad_weights_1d[py]
            for ix in range(n):
                derivative_x = basis.basis_derivative_at_quad[px, ix]
                basis_x = basis.basis_at_quad[px, ix]
                for iy in range(n):
                    x_factor = (
                        0.5
                        * mesh.dy
                        * derivative_x
                        * basis.basis_at_quad[py, iy]
                        * weight
                    )
                    y_factor = (
                        0.5
                        * mesh.dx
                        * basis_x
                        * basis.basis_derivative_at_quad[py, iy]
                        * weight
                    )
                    for component in equation.diffusion_components:
                        element_rhs[ix, iy, component] -= (
                            x_factor * gradient[0, component]
                            + y_factor * gradient[1, component]
                        )


def add_vertical_diffusion_face(
    mesh: Mesh2D,
    basis: Basis2D,
    equation: object,
    u: np.ndarray,
    ex: int,
    ey: int,
    face: int,
    normal_x: float,
    penalty_parameter: float,
    element_rhs: np.ndarray,
) -> None:
    n = basis.q + 1
    boundary_i = 0 if face == FACE_LEFT else n - 1
    neighbor_i = n - 1 if face == FACE_LEFT else 0
    neighbor_ex, neighbor_ey = mesh.neighbor(ex, ey, face)
    penalty = penalty_parameter * float(n * n) / mesh.dx
    for p in range(basis.quad_nodes_1d.shape[0]):
        basis_y_values = basis.basis_at_quad[p, :]
        derivative_y_values = basis.basis_derivative_at_quad[p, :]
        interior_value = np.zeros(equation.num_components, dtype=np.float64)
        exterior_value = np.zeros(equation.num_components, dtype=np.float64)
        interior_gradient = np.zeros((2, equation.num_components), dtype=np.float64)
        exterior_gradient = np.zeros((2, equation.num_components), dtype=np.float64)
        for iy in range(n):
            basis_y = basis_y_values[iy]
            derivative_y = derivative_y_values[iy]
            interior_value += basis_y * u[ex, ey, boundary_i, iy, :]
            exterior_value += basis_y * u[neighbor_ex, neighbor_ey, neighbor_i, iy, :]
            interior_gradient[1] += derivative_y * u[ex, ey, boundary_i, iy, :]
            exterior_gradient[1] += (
                derivative_y * u[neighbor_ex, neighbor_ey, neighbor_i, iy, :]
            )
            for ix in range(n):
                interior_gradient[0] += (
                    basis_y
                    * basis.derivative_1d[boundary_i, ix]
                    * u[ex, ey, ix, iy, :]
                )
                exterior_gradient[0] += (
                    basis_y
                    * basis.derivative_1d[neighbor_i, ix]
                    * u[neighbor_ex, neighbor_ey, ix, iy, :]
                )
        interior_gradient[0] *= 2.0 / mesh.dx
        exterior_gradient[0] *= 2.0 / mesh.dx
        interior_gradient[1] *= 2.0 / mesh.dy
        exterior_gradient[1] *= 2.0 / mesh.dy
        jump = interior_value - exterior_value
        weight = 0.5 * mesh.dy * basis.quad_weights_1d[p]
        for iy in range(n):
            boundary_weight = weight * basis_y_values[iy]
            for component in equation.diffusion_components:
                element_rhs[boundary_i, iy, component] += (
                    0.5
                    * boundary_weight
                    * normal_x
                    * (
                        interior_gradient[0, component]
                        + exterior_gradient[0, component]
                    )
                )
                element_rhs[boundary_i, iy, component] -= (
                    boundary_weight * penalty * jump[component]
                )
            for ix in range(n):
                grad_x = (
                    (2.0 / mesh.dx)
                    * basis.derivative_1d[boundary_i, ix]
                    * basis_y_values[iy]
                )
                normal_gradient = normal_x * grad_x
                for component in equation.diffusion_components:
                    element_rhs[ix, iy, component] += (
                        0.5 * weight * jump[component] * normal_gradient
                    )


def _diffusion_problem_id(equation: object) -> int:
    if isinstance(equation, LinearAdvectionDiffusion2D):
        return PROBLEM_LINEAR
    if isinstance(equation, ViscousBurgers2D):
        return PROBLEM_BURGERS
    if isinstance(equation, DiffusivePSystem2D):
        return PROBLEM_P_SYSTEM
    raise TypeError("unsupported paper equation for compiled diffusion evaluation")


def add_horizontal_diffusion_face(
    mesh: Mesh2D,
    basis: Basis2D,
    equation: object,
    u: np.ndarray,
    ex: int,
    ey: int,
    face: int,
    normal_y: float,
    penalty_parameter: float,
    element_rhs: np.ndarray,
) -> None:
    n = basis.q + 1
    boundary_j = 0 if face == FACE_BOTTOM else n - 1
    neighbor_j = n - 1 if face == FACE_BOTTOM else 0
    neighbor_ex, neighbor_ey = mesh.neighbor(ex, ey, face)
    penalty = penalty_parameter * float(n * n) / mesh.dy
    for p in range(basis.quad_nodes_1d.shape[0]):
        basis_x_values = basis.basis_at_quad[p, :]
        derivative_x_values = basis.basis_derivative_at_quad[p, :]
        interior_value = np.zeros(equation.num_components, dtype=np.float64)
        exterior_value = np.zeros(equation.num_components, dtype=np.float64)
        interior_gradient = np.zeros((2, equation.num_components), dtype=np.float64)
        exterior_gradient = np.zeros((2, equation.num_components), dtype=np.float64)
        for ix in range(n):
            basis_x = basis_x_values[ix]
            derivative_x = derivative_x_values[ix]
            interior_value += basis_x * u[ex, ey, ix, boundary_j, :]
            exterior_value += basis_x * u[neighbor_ex, neighbor_ey, ix, neighbor_j, :]
            interior_gradient[0] += derivative_x * u[ex, ey, ix, boundary_j, :]
            exterior_gradient[0] += (
                derivative_x * u[neighbor_ex, neighbor_ey, ix, neighbor_j, :]
            )
            for iy in range(n):
                interior_gradient[1] += (
                    basis_x
                    * basis.derivative_1d[boundary_j, iy]
                    * u[ex, ey, ix, iy, :]
                )
                exterior_gradient[1] += (
                    basis_x
                    * basis.derivative_1d[neighbor_j, iy]
                    * u[neighbor_ex, neighbor_ey, ix, iy, :]
                )
        interior_gradient[0] *= 2.0 / mesh.dx
        exterior_gradient[0] *= 2.0 / mesh.dx
        interior_gradient[1] *= 2.0 / mesh.dy
        exterior_gradient[1] *= 2.0 / mesh.dy
        jump = interior_value - exterior_value
        weight = 0.5 * mesh.dx * basis.quad_weights_1d[p]
        for ix in range(n):
            boundary_weight = weight * basis_x_values[ix]
            for component in equation.diffusion_components:
                element_rhs[ix, boundary_j, component] += (
                    0.5
                    * boundary_weight
                    * normal_y
                    * (
                        interior_gradient[1, component]
                        + exterior_gradient[1, component]
                    )
                )
                element_rhs[ix, boundary_j, component] -= (
                    boundary_weight * penalty * jump[component]
                )
            for iy in range(n):
                grad_y = (
                    (2.0 / mesh.dy)
                    * basis_x_values[ix]
                    * basis.derivative_1d[boundary_j, iy]
                )
                normal_gradient = normal_y * grad_y
                for component in equation.diffusion_components:
                    element_rhs[ix, iy, component] += (
                        0.5 * weight * jump[component] * normal_gradient
                    )
