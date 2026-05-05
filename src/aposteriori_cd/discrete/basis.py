from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._arrays import readonly
from .quadrature import gauss_legendre_quadrature, gauss_lobatto_legendre_nodes


@dataclass(frozen=True, slots=True)
class Basis2D:
    """Tensor-product nodal basis and quadrature data."""

    q: int
    nodes_1d: np.ndarray
    mass_1d: np.ndarray
    mass_inverse_1d: np.ndarray
    derivative_1d: np.ndarray
    quad_nodes_1d: np.ndarray
    quad_weights_1d: np.ndarray
    basis_at_quad: np.ndarray
    basis_derivative_at_quad: np.ndarray


def build_basis_2d(q: int, *, quadrature_order: int | None = None) -> Basis2D:
    """Build tensor-product basis data for polynomial degree q."""

    if q < 0:
        raise ValueError("q must be non-negative")
    n = int(q) + 1
    nq = n if quadrature_order is None else int(quadrature_order)
    if nq < 1:
        raise ValueError("quadrature_order must be positive")

    nodes_1d = gauss_lobatto_legendre_nodes(n)
    quad_nodes_1d, quad_weights_1d = gauss_legendre_quadrature(nq)
    basis_at_quad = evaluate_lagrange_basis(nodes_1d, quad_nodes_1d)
    basis_derivative_at_quad = evaluate_lagrange_basis_derivative(
        nodes_1d,
        quad_nodes_1d,
    )
    derivative_1d = differentiation_matrix_1d(nodes_1d)
    weighted_basis = quad_weights_1d[:, None] * basis_at_quad
    mass_1d = basis_at_quad.T @ weighted_basis
    mass_inverse_1d = np.linalg.inv(mass_1d)

    return Basis2D(
        q=int(q),
        nodes_1d=nodes_1d,
        mass_1d=readonly(mass_1d),
        mass_inverse_1d=readonly(mass_inverse_1d),
        derivative_1d=derivative_1d,
        quad_nodes_1d=quad_nodes_1d,
        quad_weights_1d=quad_weights_1d,
        basis_at_quad=basis_at_quad,
        basis_derivative_at_quad=basis_derivative_at_quad,
    )


def evaluate_lagrange_basis(nodes: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Evaluate nodal Lagrange basis functions at target points."""

    coefficients = _lagrange_polynomial_coefficients(nodes)
    values = np.empty((targets.shape[0], nodes.shape[0]), dtype=np.float64)
    for j in range(nodes.shape[0]):
        values[:, j] = np.polynomial.polynomial.polyval(targets, coefficients[j])
    return readonly(values)


def evaluate_lagrange_basis_derivative(
    nodes: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Evaluate derivatives of nodal Lagrange basis functions at target points."""

    coefficients = _lagrange_polynomial_coefficients(nodes)
    values = np.empty((targets.shape[0], nodes.shape[0]), dtype=np.float64)
    for j in range(nodes.shape[0]):
        derivative = np.polynomial.polynomial.polyder(coefficients[j])
        values[:, j] = np.polynomial.polynomial.polyval(targets, derivative)
    return readonly(values)


def differentiation_matrix_1d(nodes: np.ndarray) -> np.ndarray:
    """Return the nodal differentiation matrix."""

    return evaluate_lagrange_basis_derivative(nodes, nodes)


def _lagrange_polynomial_coefficients(nodes: np.ndarray) -> np.ndarray:
    count = nodes.shape[0]
    coefficients = np.zeros((count, count), dtype=np.float64)
    for j in range(count):
        poly = np.asarray([1.0], dtype=np.float64)
        denominator = 1.0
        for m in range(count):
            if m == j:
                continue
            poly = np.polynomial.polynomial.polymul(
                poly,
                np.asarray([-nodes[m], 1.0], dtype=np.float64),
            )
            denominator *= nodes[j] - nodes[m]
        coefficients[j, : poly.shape[0]] = poly / denominator
    return coefficients
