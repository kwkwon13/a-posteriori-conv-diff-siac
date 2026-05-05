from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit, prange
from scipy.sparse.linalg import LinearOperator, gmres

from aposteriori_cd.dg.diffusion import evaluate_diffusion
from aposteriori_cd.discrete import build_mesh_2d
from aposteriori_cd.equations import (
    DiffusivePSystem2D,
    LinearAdvectionDiffusion2D,
    ViscousBurgers2D,
)


_STENCIL_CENTER = 0
_STENCIL_LEFT = 1
_STENCIL_RIGHT = 2
_STENCIL_BOTTOM = 3
_STENCIL_TOP = 4
_NUM_STENCIL_ENTRIES = 5


@dataclass(frozen=True, slots=True)
class IMEXRungeKuttaGMRESParameters:
    """GMRES parameters for implicit diffusion stages."""

    relative_tolerance: float = 1.0e-10
    absolute_tolerance: float = 1.0e-12
    restart: int | None = None
    max_iterations: int | None = None
    preconditioner: str = "jacobi"

    def __post_init__(self) -> None:
        if self.relative_tolerance < 0.0:
            raise ValueError("relative_tolerance must be non-negative")
        if self.absolute_tolerance < 0.0:
            raise ValueError("absolute_tolerance must be non-negative")
        if self.relative_tolerance == 0.0 and self.absolute_tolerance == 0.0:
            raise ValueError("at least one GMRES tolerance must be positive")
        if self.restart is not None and self.restart <= 0:
            raise ValueError("restart must be positive")
        if self.max_iterations is not None and self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.preconditioner not in ("none", "jacobi"):
            raise ValueError("preconditioner must be 'none' or 'jacobi'")


@dataclass(frozen=True, slots=True)
class IMEXRungeKuttaGMRESStatistics:
    """Statistics for one implicit diffusion stage solve."""

    iterations: int
    converged: bool
    info: int
    preconditioner: str
    residual_norm: float


def solve_implicit_diffusion_stage(
    semidiscretization: object,
    rhs: np.ndarray,
    out: np.ndarray,
    *,
    alpha: float,
    parameters: IMEXRungeKuttaGMRESParameters,
    initial_guess: np.ndarray | None = None,
    work_state: np.ndarray | None = None,
    work_diffusion: np.ndarray | None = None,
    work_output: np.ndarray | None = None,
) -> IMEXRungeKuttaGMRESStatistics:
    size = rhs.size
    shape = rhs.shape
    rhs_flat = rhs.ravel()
    work_state = _work_array(shape, work_state)
    work_diffusion = _work_array(shape, work_diffusion)
    work_output = _work_array(shape, work_output)
    stage_stencil = _constant_diffusion_stage_stencil_or_none(semidiscretization)
    iterations = 0

    def matvec(vector: np.ndarray) -> np.ndarray:
        work_state[...] = vector.reshape(shape)
        if stage_stencil is None:
            semidiscretization.evaluate_diffusion(work_state, work_diffusion)
            work_output[...] = work_state - alpha * work_diffusion
        else:
            _apply_constant_diffusion_stage_operator(
                work_state,
                work_output,
                stage_stencil,
                semidiscretization.mesh.neighbor_ex,
                semidiscretization.mesh.neighbor_ey,
                float(alpha),
            )
        return work_output.ravel().copy()

    preconditioner = None
    if parameters.preconditioner == "jacobi":
        diagonal = stage_operator_diagonal(semidiscretization, shape, alpha)
        inverse_diagonal = np.reciprocal(diagonal)

        def psolve(vector: np.ndarray) -> np.ndarray:
            return inverse_diagonal * vector

        preconditioner = LinearOperator((size, size), matvec=psolve, dtype=np.float64)

    def count_iteration(_residual) -> None:
        nonlocal iterations
        iterations += 1

    linear_operator = LinearOperator((size, size), matvec=matvec, dtype=np.float64)
    x0 = None
    if initial_guess is not None:
        x0 = np.asarray(initial_guess, dtype=np.float64).reshape(shape).ravel().copy()
    solution, info = gmres(
        linear_operator,
        rhs_flat,
        x0=x0,
        M=preconditioner,
        rtol=parameters.relative_tolerance,
        atol=parameters.absolute_tolerance,
        restart=parameters.restart,
        maxiter=parameters.max_iterations,
        callback=count_iteration,
        callback_type="pr_norm",
    )
    out[...] = solution.reshape(shape)
    residual_norm = float(np.linalg.norm(matvec(solution) - rhs_flat))
    return IMEXRungeKuttaGMRESStatistics(
        iterations=iterations,
        converged=info == 0,
        info=int(info),
        preconditioner=parameters.preconditioner,
        residual_norm=residual_norm,
    )


def stage_operator_diagonal(
    semidiscretization: object,
    shape: tuple[int, ...],
    alpha: float,
) -> np.ndarray:
    diagonal = np.ones(int(np.prod(shape)), dtype=np.float64)
    if alpha == 0.0:
        return diagonal
    diagonal -= alpha * diffusion_operator_diagonal(semidiscretization, shape).ravel()
    return diagonal


def diffusion_operator_diagonal(
    semidiscretization: object,
    shape: tuple[int, ...],
) -> np.ndarray:
    key = (
        tuple(int(axis_size) for axis_size in shape),
        float(semidiscretization.diffusion_penalty_parameter),
    )
    cache = getattr(semidiscretization, "diffusion_diagonal_cache", None)
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            return cached

    stencil = _constant_diffusion_stage_stencil_or_none(semidiscretization)
    if stencil is None:
        diagonal = _representative_diffusion_diagonal(semidiscretization, shape)
    else:
        diagonal = _constant_diffusion_diagonal_from_stencil(stencil, shape)
    diagonal.setflags(write=False)
    if cache is not None:
        cache[key] = diagonal
    return diagonal


def _work_array(shape: tuple[int, ...], array: np.ndarray | None) -> np.ndarray:
    if array is None:
        return np.empty(shape, dtype=np.float64)
    if array.shape != shape:
        raise ValueError("GMRES work array has incompatible shape")
    if array.dtype != np.float64:
        raise ValueError("GMRES work array must have dtype numpy.float64")
    return array


def _constant_diffusion_stage_stencil_or_none(
    semidiscretization: object,
) -> np.ndarray | None:
    equation = getattr(semidiscretization, "equation", None)
    if not isinstance(
        equation,
        (LinearAdvectionDiffusion2D, ViscousBurgers2D, DiffusivePSystem2D),
    ):
        return None
    key = (
        "constant_diffusion_stage_stencil",
        equation.__class__,
        int(equation.num_components),
        tuple(int(component) for component in equation.diffusion_components),
        int(semidiscretization.basis.q),
        float(semidiscretization.mesh.dx),
        float(semidiscretization.mesh.dy),
        float(semidiscretization.diffusion_penalty_parameter),
    )
    cache = getattr(semidiscretization, "diffusion_stage_stencil_cache", None)
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            return cached

    stencil = _build_constant_diffusion_stage_stencil(semidiscretization)
    stencil.setflags(write=False)
    if cache is not None:
        cache[key] = stencil
    return stencil


def _build_constant_diffusion_stage_stencil(semidiscretization: object) -> np.ndarray:
    mesh = semidiscretization.mesh
    basis = semidiscretization.basis
    equation = semidiscretization.equation
    n = basis.q + 1
    num_components = equation.num_components
    representative_mesh = build_mesh_2d(
        3,
        3,
        x_bounds=(mesh.x_min, mesh.x_min + 3.0 * mesh.dx),
        y_bounds=(mesh.y_min, mesh.y_min + 3.0 * mesh.dy),
    )
    state = np.zeros((3, 3, n, n, num_components), dtype=np.float64)
    diffusion = np.zeros_like(state)
    stencil = np.empty(
        (
            _NUM_STENCIL_ENTRIES,
            n,
            n,
            num_components,
            n,
            n,
            num_components,
        ),
        dtype=np.float64,
    )
    source_elements = (
        (1, 1),
        (0, 1),
        (2, 1),
        (1, 0),
        (1, 2),
    )
    for entry, (source_ex, source_ey) in enumerate(source_elements):
        for ix in range(n):
            for iy in range(n):
                for component in range(num_components):
                    state.fill(0.0)
                    diffusion.fill(0.0)
                    state[source_ex, source_ey, ix, iy, component] = 1.0
                    evaluate_diffusion(
                        representative_mesh,
                        basis,
                        equation,
                        state,
                        diffusion,
                        penalty_parameter=semidiscretization.diffusion_penalty_parameter,
                    )
                    stencil[entry, :, :, :, ix, iy, component] = diffusion[1, 1]
    return np.ascontiguousarray(stencil, dtype=np.float64)


def _constant_diffusion_diagonal_from_stencil(
    stencil: np.ndarray,
    shape: tuple[int, ...],
) -> np.ndarray:
    local = np.empty((1, 1, shape[2], shape[3], shape[4]), dtype=np.float64)
    for ix in range(shape[2]):
        for iy in range(shape[3]):
            for component in range(shape[4]):
                local[0, 0, ix, iy, component] = stencil[
                    _STENCIL_CENTER,
                    ix,
                    iy,
                    component,
                    ix,
                    iy,
                    component,
                ]
    return np.broadcast_to(local, shape).copy()


@njit(cache=True, parallel=True)
def _apply_constant_diffusion_stage_operator(
    state: np.ndarray,
    out: np.ndarray,
    stencil: np.ndarray,
    neighbor_ex: np.ndarray,
    neighbor_ey: np.ndarray,
    alpha: float,
) -> None:
    nx = state.shape[0]
    ny = state.shape[1]
    n = state.shape[2]
    num_components = state.shape[4]
    for element_index in prange(nx * ny):
        ex = element_index // ny
        ey = element_index - ex * ny
        left_ex = neighbor_ex[ex, ey, 0]
        left_ey = neighbor_ey[ex, ey, 0]
        right_ex = neighbor_ex[ex, ey, 1]
        right_ey = neighbor_ey[ex, ey, 1]
        bottom_ex = neighbor_ex[ex, ey, 2]
        bottom_ey = neighbor_ey[ex, ey, 2]
        top_ex = neighbor_ex[ex, ey, 3]
        top_ey = neighbor_ey[ex, ey, 3]

        for ix in range(n):
            for iy in range(n):
                for component in range(num_components):
                    diffusion = 0.0
                    for kx in range(n):
                        for ky in range(n):
                            for component_in in range(num_components):
                                diffusion += (
                                    stencil[
                                        _STENCIL_CENTER,
                                        ix,
                                        iy,
                                        component,
                                        kx,
                                        ky,
                                        component_in,
                                    ]
                                    * state[ex, ey, kx, ky, component_in]
                                )
                                diffusion += (
                                    stencil[
                                        _STENCIL_LEFT,
                                        ix,
                                        iy,
                                        component,
                                        kx,
                                        ky,
                                        component_in,
                                    ]
                                    * state[left_ex, left_ey, kx, ky, component_in]
                                )
                                diffusion += (
                                    stencil[
                                        _STENCIL_RIGHT,
                                        ix,
                                        iy,
                                        component,
                                        kx,
                                        ky,
                                        component_in,
                                    ]
                                    * state[right_ex, right_ey, kx, ky, component_in]
                                )
                                diffusion += (
                                    stencil[
                                        _STENCIL_BOTTOM,
                                        ix,
                                        iy,
                                        component,
                                        kx,
                                        ky,
                                        component_in,
                                    ]
                                    * state[bottom_ex, bottom_ey, kx, ky, component_in]
                                )
                                diffusion += (
                                    stencil[
                                        _STENCIL_TOP,
                                        ix,
                                        iy,
                                        component,
                                        kx,
                                        ky,
                                        component_in,
                                    ]
                                    * state[top_ex, top_ey, kx, ky, component_in]
                                )
                    out[ex, ey, ix, iy, component] = (
                        state[ex, ey, ix, iy, component] - alpha * diffusion
                    )


def _representative_diffusion_diagonal(
    semidiscretization: object,
    shape: tuple[int, ...],
) -> np.ndarray:
    mesh = semidiscretization.mesh
    basis = semidiscretization.basis
    equation = semidiscretization.equation
    n = basis.q + 1
    num_components = equation.num_components
    representative_mesh = build_mesh_2d(
        3,
        3,
        x_bounds=(mesh.x_min, mesh.x_min + 3.0 * mesh.dx),
        y_bounds=(mesh.y_min, mesh.y_min + 3.0 * mesh.dy),
    )
    state = np.zeros((3, 3, n, n, num_components), dtype=np.float64)
    diffusion = np.zeros_like(state)
    local_diagonal = np.empty((1, 1, n, n, num_components), dtype=np.float64)

    for ix in range(n):
        for iy in range(n):
            for component in range(num_components):
                state.fill(0.0)
                state[1, 1, ix, iy, component] = 1.0
                evaluate_diffusion(
                    representative_mesh,
                    basis,
                    equation,
                    state,
                    diffusion,
                    penalty_parameter=semidiscretization.diffusion_penalty_parameter,
                )
                local_diagonal[0, 0, ix, iy, component] = diffusion[
                    1,
                    1,
                    ix,
                    iy,
                    component,
                ]
    return np.broadcast_to(local_diagonal, shape).copy()


__all__ = [
    "diffusion_operator_diagonal",
    "IMEXRungeKuttaGMRESParameters",
    "IMEXRungeKuttaGMRESStatistics",
    "solve_implicit_diffusion_stage",
    "stage_operator_diagonal",
]
