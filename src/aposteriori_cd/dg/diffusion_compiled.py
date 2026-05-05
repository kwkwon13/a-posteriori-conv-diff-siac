from __future__ import annotations

import numpy as np
from numba import njit, prange


PROBLEM_LINEAR = 0
PROBLEM_BURGERS = 1
PROBLEM_P_SYSTEM = 2


@njit(cache=True, parallel=True)
def add_diffusion_compiled(
    u: np.ndarray,
    rhs: np.ndarray,
    neighbor_ex: np.ndarray,
    neighbor_ey: np.ndarray,
    basis_at_quad: np.ndarray,
    basis_derivative_at_quad: np.ndarray,
    derivative_1d: np.ndarray,
    quad_weights: np.ndarray,
    mass_inverse: np.ndarray,
    dx: float,
    dy: float,
    penalty_parameter: float,
    scale: float,
    problem_id: int,
) -> None:
    nx = u.shape[0]
    ny = u.shape[1]
    n = u.shape[2]
    n_components = u.shape[4]
    element_scale = 0.25 * dx * dy
    for element_index in prange(nx * ny):
        ex = element_index // ny
        ey = element_index - ex * ny
        element_rhs = np.zeros((n, n, n_components), dtype=np.float64)
        _add_diffusion_volume_terms(
            u,
            basis_at_quad,
            basis_derivative_at_quad,
            quad_weights,
            dx,
            dy,
            problem_id,
            ex,
            ey,
            element_rhs,
        )
        _add_vertical_diffusion_face(
            u,
            neighbor_ex,
            neighbor_ey,
            basis_at_quad,
            basis_derivative_at_quad,
            derivative_1d,
            quad_weights,
            dx,
            dy,
            problem_id,
            ex,
            ey,
            0,
            -1.0,
            penalty_parameter,
            element_rhs,
        )
        _add_vertical_diffusion_face(
            u,
            neighbor_ex,
            neighbor_ey,
            basis_at_quad,
            basis_derivative_at_quad,
            derivative_1d,
            quad_weights,
            dx,
            dy,
            problem_id,
            ex,
            ey,
            1,
            1.0,
            penalty_parameter,
            element_rhs,
        )
        _add_horizontal_diffusion_face(
            u,
            neighbor_ex,
            neighbor_ey,
            basis_at_quad,
            basis_derivative_at_quad,
            derivative_1d,
            quad_weights,
            dx,
            dy,
            problem_id,
            ex,
            ey,
            2,
            -1.0,
            penalty_parameter,
            element_rhs,
        )
        _add_horizontal_diffusion_face(
            u,
            neighbor_ex,
            neighbor_ey,
            basis_at_quad,
            basis_derivative_at_quad,
            derivative_1d,
            quad_weights,
            dx,
            dy,
            problem_id,
            ex,
            ey,
            3,
            1.0,
            penalty_parameter,
            element_rhs,
        )
        _add_mass_solved_rhs(
            element_rhs,
            mass_inverse,
            scale / element_scale,
            rhs,
            ex,
            ey,
        )


@njit(cache=True)
def _add_diffusion_volume_terms(
    u: np.ndarray,
    basis_at_quad: np.ndarray,
    basis_derivative_at_quad: np.ndarray,
    quad_weights: np.ndarray,
    dx: float,
    dy: float,
    problem_id: int,
    ex: int,
    ey: int,
    element_rhs: np.ndarray,
) -> None:
    n = u.shape[2]
    n_components = u.shape[4]
    n_quad = quad_weights.shape[0]
    for px in range(n_quad):
        for py in range(n_quad):
            gradient_x = np.zeros(n_components, dtype=np.float64)
            gradient_y = np.zeros(n_components, dtype=np.float64)
            for ix in range(n):
                basis_x = basis_at_quad[px, ix]
                derivative_x = basis_derivative_at_quad[px, ix]
                for iy in range(n):
                    basis_y = basis_at_quad[py, iy]
                    derivative_y = basis_derivative_at_quad[py, iy]
                    for component in range(n_components):
                        coefficient = u[ex, ey, ix, iy, component]
                        gradient_x[component] += derivative_x * basis_y * coefficient
                        gradient_y[component] += basis_x * derivative_y * coefficient
            for component in range(n_components):
                gradient_x[component] *= 2.0 / dx
                gradient_y[component] *= 2.0 / dy
            weight = quad_weights[px] * quad_weights[py]
            for ix in range(n):
                derivative_x = basis_derivative_at_quad[px, ix]
                basis_x = basis_at_quad[px, ix]
                for iy in range(n):
                    x_factor = (
                        0.5
                        * dy
                        * derivative_x
                        * basis_at_quad[py, iy]
                        * weight
                    )
                    y_factor = (
                        0.5
                        * dx
                        * basis_x
                        * basis_derivative_at_quad[py, iy]
                        * weight
                    )
                    for component in range(n_components):
                        if _is_diffusion_component(problem_id, component):
                            element_rhs[ix, iy, component] -= (
                                x_factor * gradient_x[component]
                                + y_factor * gradient_y[component]
                            )


@njit(cache=True)
def _add_vertical_diffusion_face(
    u: np.ndarray,
    neighbor_ex: np.ndarray,
    neighbor_ey: np.ndarray,
    basis_at_quad: np.ndarray,
    basis_derivative_at_quad: np.ndarray,
    derivative_1d: np.ndarray,
    quad_weights: np.ndarray,
    dx: float,
    dy: float,
    problem_id: int,
    ex: int,
    ey: int,
    face: int,
    normal_x: float,
    penalty_parameter: float,
    element_rhs: np.ndarray,
) -> None:
    n = u.shape[2]
    n_components = u.shape[4]
    n_quad = quad_weights.shape[0]
    boundary_i = 0
    neighbor_i = n - 1
    if face == 1:
        boundary_i = n - 1
        neighbor_i = 0
    neighbor_x = neighbor_ex[ex, ey, face]
    neighbor_y = neighbor_ey[ex, ey, face]
    penalty = penalty_parameter * float(n * n) / dx
    for p in range(n_quad):
        interior_value = np.zeros(n_components, dtype=np.float64)
        exterior_value = np.zeros(n_components, dtype=np.float64)
        interior_gradient = np.zeros((2, n_components), dtype=np.float64)
        exterior_gradient = np.zeros((2, n_components), dtype=np.float64)
        for iy in range(n):
            basis_y = basis_at_quad[p, iy]
            derivative_y = basis_derivative_at_quad[p, iy]
            for component in range(n_components):
                interior_value[component] += basis_y * u[ex, ey, boundary_i, iy, component]
                exterior_value[component] += (
                    basis_y * u[neighbor_x, neighbor_y, neighbor_i, iy, component]
                )
                interior_gradient[1, component] += (
                    derivative_y * u[ex, ey, boundary_i, iy, component]
                )
                exterior_gradient[1, component] += (
                    derivative_y
                    * u[neighbor_x, neighbor_y, neighbor_i, iy, component]
                )
            for ix in range(n):
                for component in range(n_components):
                    interior_gradient[0, component] += (
                        basis_y
                        * derivative_1d[boundary_i, ix]
                        * u[ex, ey, ix, iy, component]
                    )
                    exterior_gradient[0, component] += (
                        basis_y
                        * derivative_1d[neighbor_i, ix]
                        * u[neighbor_x, neighbor_y, ix, iy, component]
                    )
        for component in range(n_components):
            interior_gradient[0, component] *= 2.0 / dx
            exterior_gradient[0, component] *= 2.0 / dx
            interior_gradient[1, component] *= 2.0 / dy
            exterior_gradient[1, component] *= 2.0 / dy
        weight = 0.5 * dy * quad_weights[p]
        for iy in range(n):
            boundary_weight = weight * basis_at_quad[p, iy]
            for component in range(n_components):
                if _is_diffusion_component(problem_id, component):
                    jump = interior_value[component] - exterior_value[component]
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
                        boundary_weight * penalty * jump
                    )
            for ix in range(n):
                grad_x = (
                    (2.0 / dx)
                    * derivative_1d[boundary_i, ix]
                    * basis_at_quad[p, iy]
                )
                normal_gradient = normal_x * grad_x
                for component in range(n_components):
                    if _is_diffusion_component(problem_id, component):
                        jump = interior_value[component] - exterior_value[component]
                        element_rhs[ix, iy, component] += (
                            0.5 * weight * jump * normal_gradient
                        )


@njit(cache=True)
def _add_horizontal_diffusion_face(
    u: np.ndarray,
    neighbor_ex: np.ndarray,
    neighbor_ey: np.ndarray,
    basis_at_quad: np.ndarray,
    basis_derivative_at_quad: np.ndarray,
    derivative_1d: np.ndarray,
    quad_weights: np.ndarray,
    dx: float,
    dy: float,
    problem_id: int,
    ex: int,
    ey: int,
    face: int,
    normal_y: float,
    penalty_parameter: float,
    element_rhs: np.ndarray,
) -> None:
    n = u.shape[2]
    n_components = u.shape[4]
    n_quad = quad_weights.shape[0]
    boundary_j = 0
    neighbor_j = n - 1
    if face == 3:
        boundary_j = n - 1
        neighbor_j = 0
    neighbor_x = neighbor_ex[ex, ey, face]
    neighbor_y = neighbor_ey[ex, ey, face]
    penalty = penalty_parameter * float(n * n) / dy
    for p in range(n_quad):
        interior_value = np.zeros(n_components, dtype=np.float64)
        exterior_value = np.zeros(n_components, dtype=np.float64)
        interior_gradient = np.zeros((2, n_components), dtype=np.float64)
        exterior_gradient = np.zeros((2, n_components), dtype=np.float64)
        for ix in range(n):
            basis_x = basis_at_quad[p, ix]
            derivative_x = basis_derivative_at_quad[p, ix]
            for component in range(n_components):
                interior_value[component] += basis_x * u[ex, ey, ix, boundary_j, component]
                exterior_value[component] += (
                    basis_x * u[neighbor_x, neighbor_y, ix, neighbor_j, component]
                )
                interior_gradient[0, component] += (
                    derivative_x * u[ex, ey, ix, boundary_j, component]
                )
                exterior_gradient[0, component] += (
                    derivative_x
                    * u[neighbor_x, neighbor_y, ix, neighbor_j, component]
                )
            for iy in range(n):
                for component in range(n_components):
                    interior_gradient[1, component] += (
                        basis_x
                        * derivative_1d[boundary_j, iy]
                        * u[ex, ey, ix, iy, component]
                    )
                    exterior_gradient[1, component] += (
                        basis_x
                        * derivative_1d[neighbor_j, iy]
                        * u[neighbor_x, neighbor_y, ix, iy, component]
                    )
        for component in range(n_components):
            interior_gradient[0, component] *= 2.0 / dx
            exterior_gradient[0, component] *= 2.0 / dx
            interior_gradient[1, component] *= 2.0 / dy
            exterior_gradient[1, component] *= 2.0 / dy
        weight = 0.5 * dx * quad_weights[p]
        for ix in range(n):
            boundary_weight = weight * basis_at_quad[p, ix]
            for component in range(n_components):
                if _is_diffusion_component(problem_id, component):
                    jump = interior_value[component] - exterior_value[component]
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
                        boundary_weight * penalty * jump
                    )
            for iy in range(n):
                grad_y = (
                    (2.0 / dy)
                    * basis_at_quad[p, ix]
                    * derivative_1d[boundary_j, iy]
                )
                normal_gradient = normal_y * grad_y
                for component in range(n_components):
                    if _is_diffusion_component(problem_id, component):
                        jump = interior_value[component] - exterior_value[component]
                        element_rhs[ix, iy, component] += (
                            0.5 * weight * jump * normal_gradient
                        )


@njit(cache=True)
def _is_diffusion_component(problem_id: int, component: int) -> bool:
    if problem_id == PROBLEM_P_SYSTEM:
        return component == 1 or component == 2
    return component == 0


@njit(cache=True)
def _add_mass_solved_rhs(
    element_rhs: np.ndarray,
    mass_inverse: np.ndarray,
    scale: float,
    rhs: np.ndarray,
    ex: int,
    ey: int,
) -> None:
    n = element_rhs.shape[0]
    n_components = element_rhs.shape[2]
    for ix in range(n):
        for iy in range(n):
            for component in range(n_components):
                value = 0.0
                for ax in range(n):
                    for ay in range(n):
                        value += (
                            mass_inverse[ix, ax]
                            * element_rhs[ax, ay, component]
                            * mass_inverse[iy, ay]
                        )
                rhs[ex, ey, ix, iy, component] += scale * value
