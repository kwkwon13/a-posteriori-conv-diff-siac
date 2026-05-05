from __future__ import annotations

from collections.abc import Callable

import numpy as np

from aposteriori_cd.discrete import Basis2D, Mesh2D
from aposteriori_cd.equations import source_values

from .mass import solve_tensor_product_mass_matrix


def project_initial_condition(
    mesh: Mesh2D,
    basis: Basis2D,
    equation: object,
    out: np.ndarray,
    *,
    time: float = 0.0,
) -> None:
    project_function_to_state(mesh, basis, equation.exact_solution, out, time=time)


def project_function_to_state(
    mesh: Mesh2D,
    basis: Basis2D,
    field: Callable[[float, np.ndarray], np.ndarray],
    out: np.ndarray,
    *,
    time: float = 0.0,
) -> None:
    n = basis.q + 1
    n_quad = basis.quad_nodes_1d.shape[0]
    values_at_quad = np.empty((n_quad, n_quad, out.shape[-1]), dtype=np.float64)
    rhs = np.empty((n, n, out.shape[-1]), dtype=np.float64)
    point = np.empty(2, dtype=np.float64)
    for ex in range(mesh.nx):
        for ey in range(mesh.ny):
            x_center = mesh.element_centers_x[ex, ey]
            y_center = mesh.element_centers_y[ex, ey]
            for px, xi in enumerate(basis.quad_nodes_1d):
                point[0] = x_center + 0.5 * mesh.dx * xi
                for py, eta in enumerate(basis.quad_nodes_1d):
                    point[1] = y_center + 0.5 * mesh.dy * eta
                    values_at_quad[px, py, :] = field(float(time), point)
            for component in range(out.shape[-1]):
                weighted_values = (
                    basis.quad_weights_1d[:, None]
                    * basis.quad_weights_1d[None, :]
                    * values_at_quad[:, :, component]
                )
                rhs[:, :, component] = (
                    basis.basis_at_quad.T @ weighted_values @ basis.basis_at_quad
                )
            out[ex, ey, :, :, :] = solve_tensor_product_mass_matrix(
                basis.mass_1d,
                rhs,
            )


def add_source_projection(
    mesh: Mesh2D,
    basis: Basis2D,
    equation: object,
    time: float,
    rhs: np.ndarray,
    source_projection: np.ndarray,
) -> None:
    if getattr(equation, "source_term_is_zero", False):
        return
    project_source_values(mesh, basis, equation, time, source_projection)
    rhs += source_projection


def project_source_values(
    mesh: Mesh2D,
    basis: Basis2D,
    equation: object,
    time: float,
    out: np.ndarray,
) -> None:
    n_quad = basis.quad_nodes_1d.shape[0]
    points = np.empty((mesh.nx, mesh.ny, n_quad, n_quad, 2), dtype=np.float64)
    values = np.empty(
        (mesh.nx, mesh.ny, n_quad, n_quad, equation.num_components),
        dtype=np.float64,
    )
    for ex in range(mesh.nx):
        for ey in range(mesh.ny):
            x_center = mesh.element_centers_x[ex, ey]
            y_center = mesh.element_centers_y[ex, ey]
            for px, xi in enumerate(basis.quad_nodes_1d):
                points[ex, ey, px, :, 0] = x_center + 0.5 * mesh.dx * xi
                for py, eta in enumerate(basis.quad_nodes_1d):
                    points[ex, ey, px, py, 1] = y_center + 0.5 * mesh.dy * eta
    source_values(equation, time, points, values)
    rhs = np.empty(
        (basis.q + 1, basis.q + 1, equation.num_components),
        dtype=np.float64,
    )
    for ex in range(mesh.nx):
        for ey in range(mesh.ny):
            for component in range(equation.num_components):
                weighted = (
                    basis.quad_weights_1d[:, None]
                    * basis.quad_weights_1d[None, :]
                    * values[ex, ey, :, :, component]
                )
                rhs[:, :, component] = (
                    basis.basis_at_quad.T @ weighted @ basis.basis_at_quad
                )
            out[ex, ey, :, :, :] = solve_tensor_product_mass_matrix(
                basis.mass_1d,
                rhs,
            )
