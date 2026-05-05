from __future__ import annotations

import numpy as np
from numba import njit, prange


PROBLEM_LINEAR = 0
PROBLEM_BURGERS = 1
PROBLEM_P_SYSTEM = 2


@njit(cache=True, parallel=True)
def add_convection_compiled(
    u: np.ndarray,
    rhs: np.ndarray,
    neighbor_ex: np.ndarray,
    neighbor_ey: np.ndarray,
    basis_at_quad: np.ndarray,
    basis_derivative_at_quad: np.ndarray,
    quad_weights: np.ndarray,
    mass_inverse: np.ndarray,
    dx: float,
    dy: float,
    problem_id: int,
    velocity_x: float,
    velocity_y: float,
) -> None:
    nx = u.shape[0]
    ny = u.shape[1]
    n = u.shape[2]
    n_components = u.shape[4]
    n_quad = quad_weights.shape[0]
    jacobian = 0.25 * dx * dy
    for element_index in prange(nx * ny):
        ex = element_index // ny
        ey = element_index - ex * ny
        element_rhs = np.zeros((n, n, n_components), dtype=np.float64)

        for px in range(n_quad):
            for py in range(n_quad):
                value = np.zeros(n_components, dtype=np.float64)
                for ix in range(n):
                    basis_x = basis_at_quad[px, ix]
                    for iy in range(n):
                        coefficient = basis_x * basis_at_quad[py, iy]
                        for component in range(n_components):
                            value[component] += coefficient * u[ex, ey, ix, iy, component]
                flux_x = np.zeros(n_components, dtype=np.float64)
                flux_y = np.zeros(n_components, dtype=np.float64)
                _fill_flux(problem_id, velocity_x, velocity_y, value, flux_x, flux_y)
                weight = quad_weights[px] * quad_weights[py]
                for ix in range(n):
                    basis_x = basis_at_quad[px, ix]
                    derivative_x = basis_derivative_at_quad[px, ix]
                    for iy in range(n):
                        x_factor = 0.5 * dy * derivative_x * basis_at_quad[py, iy] * weight
                        y_factor = 0.5 * dx * basis_x * basis_derivative_at_quad[py, iy] * weight
                        for component in range(n_components):
                            element_rhs[ix, iy, component] += (
                                x_factor * flux_x[component]
                                + y_factor * flux_y[component]
                            )

        _subtract_vertical_face_integral(
            u,
            neighbor_ex,
            neighbor_ey,
            basis_at_quad,
            quad_weights,
            dy,
            problem_id,
            velocity_x,
            velocity_y,
            ex,
            ey,
            0,
            -1.0,
            element_rhs,
        )
        _subtract_vertical_face_integral(
            u,
            neighbor_ex,
            neighbor_ey,
            basis_at_quad,
            quad_weights,
            dy,
            problem_id,
            velocity_x,
            velocity_y,
            ex,
            ey,
            1,
            1.0,
            element_rhs,
        )
        _subtract_horizontal_face_integral(
            u,
            neighbor_ex,
            neighbor_ey,
            basis_at_quad,
            quad_weights,
            dx,
            problem_id,
            velocity_x,
            velocity_y,
            ex,
            ey,
            2,
            -1.0,
            element_rhs,
        )
        _subtract_horizontal_face_integral(
            u,
            neighbor_ex,
            neighbor_ey,
            basis_at_quad,
            quad_weights,
            dx,
            problem_id,
            velocity_x,
            velocity_y,
            ex,
            ey,
            3,
            1.0,
            element_rhs,
        )
        _add_mass_solved_rhs(element_rhs, mass_inverse, jacobian, rhs, ex, ey)


@njit(cache=True)
def _subtract_vertical_face_integral(
    u: np.ndarray,
    neighbor_ex: np.ndarray,
    neighbor_ey: np.ndarray,
    basis_at_quad: np.ndarray,
    quad_weights: np.ndarray,
    dy: float,
    problem_id: int,
    velocity_x: float,
    velocity_y: float,
    ex: int,
    ey: int,
    face: int,
    normal_x: float,
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
    for p in range(n_quad):
        value_minus = np.zeros(n_components, dtype=np.float64)
        value_plus = np.zeros(n_components, dtype=np.float64)
        for iy in range(n):
            basis_y = basis_at_quad[p, iy]
            for component in range(n_components):
                value_minus[component] += basis_y * u[ex, ey, boundary_i, iy, component]
                value_plus[component] += (
                    basis_y * u[neighbor_x, neighbor_y, neighbor_i, iy, component]
                )
        normal_flux = np.zeros(n_components, dtype=np.float64)
        _fill_normal_flux(
            problem_id,
            velocity_x,
            velocity_y,
            value_minus,
            value_plus,
            normal_x,
            0.0,
            normal_flux,
        )
        weight = 0.5 * dy * quad_weights[p]
        for iy in range(n):
            coefficient = weight * basis_at_quad[p, iy]
            for component in range(n_components):
                element_rhs[boundary_i, iy, component] -= (
                    coefficient * normal_flux[component]
                )


@njit(cache=True)
def _subtract_horizontal_face_integral(
    u: np.ndarray,
    neighbor_ex: np.ndarray,
    neighbor_ey: np.ndarray,
    basis_at_quad: np.ndarray,
    quad_weights: np.ndarray,
    dx: float,
    problem_id: int,
    velocity_x: float,
    velocity_y: float,
    ex: int,
    ey: int,
    face: int,
    normal_y: float,
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
    for p in range(n_quad):
        value_minus = np.zeros(n_components, dtype=np.float64)
        value_plus = np.zeros(n_components, dtype=np.float64)
        for ix in range(n):
            basis_x = basis_at_quad[p, ix]
            for component in range(n_components):
                value_minus[component] += basis_x * u[ex, ey, ix, boundary_j, component]
                value_plus[component] += (
                    basis_x * u[neighbor_x, neighbor_y, ix, neighbor_j, component]
                )
        normal_flux = np.zeros(n_components, dtype=np.float64)
        _fill_normal_flux(
            problem_id,
            velocity_x,
            velocity_y,
            value_minus,
            value_plus,
            0.0,
            normal_y,
            normal_flux,
        )
        weight = 0.5 * dx * quad_weights[p]
        for ix in range(n):
            coefficient = weight * basis_at_quad[p, ix]
            for component in range(n_components):
                element_rhs[ix, boundary_j, component] -= (
                    coefficient * normal_flux[component]
                )


@njit(cache=True)
def _fill_normal_flux(
    problem_id: int,
    velocity_x: float,
    velocity_y: float,
    value_minus: np.ndarray,
    value_plus: np.ndarray,
    normal_x: float,
    normal_y: float,
    out: np.ndarray,
) -> None:
    flux_minus_x = np.zeros(out.shape[0], dtype=np.float64)
    flux_minus_y = np.zeros(out.shape[0], dtype=np.float64)
    flux_plus_x = np.zeros(out.shape[0], dtype=np.float64)
    flux_plus_y = np.zeros(out.shape[0], dtype=np.float64)
    _fill_flux(problem_id, velocity_x, velocity_y, value_minus, flux_minus_x, flux_minus_y)
    _fill_flux(problem_id, velocity_x, velocity_y, value_plus, flux_plus_x, flux_plus_y)
    wavespeed_minus = _max_wavespeed(
        problem_id,
        velocity_x,
        velocity_y,
        value_minus,
        normal_x,
        normal_y,
    )
    wavespeed_plus = _max_wavespeed(
        problem_id,
        velocity_x,
        velocity_y,
        value_plus,
        normal_x,
        normal_y,
    )
    wavespeed = max(wavespeed_minus, wavespeed_plus)
    for component in range(out.shape[0]):
        out[component] = (
            0.5
            * (
                normal_x * (flux_minus_x[component] + flux_plus_x[component])
                + normal_y * (flux_minus_y[component] + flux_plus_y[component])
            )
            - 0.5 * wavespeed * (value_plus[component] - value_minus[component])
        )


@njit(cache=True)
def _fill_flux(
    problem_id: int,
    velocity_x: float,
    velocity_y: float,
    value: np.ndarray,
    flux_x: np.ndarray,
    flux_y: np.ndarray,
) -> None:
    if problem_id == PROBLEM_LINEAR:
        flux_x[0] = velocity_x * value[0]
        flux_y[0] = velocity_y * value[0]
    elif problem_id == PROBLEM_BURGERS:
        flux = 0.5 * value[0] * value[0]
        flux_x[0] = flux
        flux_y[0] = flux
    else:
        pressure = 1.0 / (value[0] * value[0])
        flux_x[0] = -value[1]
        flux_x[1] = pressure
        flux_x[2] = 0.0
        flux_y[0] = -value[2]
        flux_y[1] = 0.0
        flux_y[2] = pressure


@njit(cache=True)
def _max_wavespeed(
    problem_id: int,
    velocity_x: float,
    velocity_y: float,
    value: np.ndarray,
    normal_x: float,
    normal_y: float,
) -> float:
    if problem_id == PROBLEM_LINEAR:
        return abs(normal_x * velocity_x + normal_y * velocity_y)
    if problem_id == PROBLEM_BURGERS:
        return abs((normal_x + normal_y) * value[0])
    normal_norm_squared = normal_x * normal_x + normal_y * normal_y
    return np.sqrt(2.0 * normal_norm_squared / (value[0] * value[0] * value[0]))


@njit(cache=True)
def _add_mass_solved_rhs(
    element_rhs: np.ndarray,
    mass_inverse: np.ndarray,
    jacobian: float,
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
                rhs[ex, ey, ix, iy, component] += value / jacobian
