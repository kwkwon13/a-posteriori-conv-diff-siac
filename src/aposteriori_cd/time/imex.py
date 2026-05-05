from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .parameters import FixedTimeStepParameters
from .gmres import (
    IMEXRungeKuttaGMRESParameters,
    IMEXRungeKuttaGMRESStatistics,
    solve_implicit_diffusion_stage,
)
from .solution_data import (
    TimeSubintervalSolutionData,
    allocate_time_subinterval_solution_data,
    fill_time_subinterval_solution_data,
)
from .tableaux import (
    ARK3_2_4L_2_SA,
    ARK5_4_8L_2_SA,
    IMEXRungeKuttaTableau,
    build_imex_runge_kutta_tableau,
)


@dataclass(slots=True)
class _IMEXRungeKuttaStageWork:
    initial_state: np.ndarray
    stage_values: np.ndarray
    explicit_rhs: np.ndarray
    diffusion_rhs: np.ndarray
    stage_rhs: np.ndarray
    gmres_state: np.ndarray
    gmres_diffusion: np.ndarray
    gmres_output: np.ndarray


def evolve_fixed_steps_imex_runge_kutta_gmres(
    semidiscretization: object,
    tableau: IMEXRungeKuttaTableau,
    config: FixedTimeStepParameters,
    u: np.ndarray,
    *,
    parameters: IMEXRungeKuttaGMRESParameters = IMEXRungeKuttaGMRESParameters(),
    on_time_subinterval: Callable[[int, TimeSubintervalSolutionData], None] | None = None,
    on_gmres_solve: Callable[[int, int, IMEXRungeKuttaGMRESStatistics], None] | None = None,
    store_time_subinterval_data: bool = True,
) -> float:
    data = allocate_time_subinterval_solution_data(u.shape)
    rhs_left = np.empty_like(u)
    rhs_right = np.empty_like(u)
    u_left = np.empty_like(u)
    stage_work = _allocate_imex_runge_kutta_stage_work(tableau, u)
    t_left = float(config.t_start)
    if config.num_steps == 0:
        return t_left
    semidiscretization.evaluate_rhs(t_left, u, rhs_left)

    for step_index in range(config.num_steps):
        u_left[:] = u
        t_right = t_left + config.dt
        _evolve_one_imex_step(
            semidiscretization,
            tableau,
            t_left,
            config.dt,
            u,
            parameters,
            step_index,
            on_gmres_solve,
            stage_work,
        )
        semidiscretization.evaluate_rhs(t_right, u, rhs_right)
        if store_time_subinterval_data:
            fill_time_subinterval_solution_data(
                data,
                t_left=t_left,
                t_right=t_right,
                u_left=u_left,
                u_right=u,
                rhs_left=rhs_left,
                rhs_right=rhs_right,
            )
            if on_time_subinterval is not None:
                on_time_subinterval(step_index, data)
        t_left = t_right
        rhs_left, rhs_right = rhs_right, rhs_left
    return t_left


def evolve_fixed_steps_ark3_2_4l_2_sa_gmres(
    semidiscretization: object,
    config: FixedTimeStepParameters,
    u: np.ndarray,
    **kwargs,
) -> float:
    return evolve_fixed_steps_imex_runge_kutta_gmres(
        semidiscretization,
        build_imex_runge_kutta_tableau(ARK3_2_4L_2_SA),
        config,
        u,
        **kwargs,
    )


def evolve_fixed_steps_ark5_4_8l_2_sa_gmres(
    semidiscretization: object,
    config: FixedTimeStepParameters,
    u: np.ndarray,
    **kwargs,
) -> float:
    return evolve_fixed_steps_imex_runge_kutta_gmres(
        semidiscretization,
        build_imex_runge_kutta_tableau(ARK5_4_8L_2_SA),
        config,
        u,
        **kwargs,
    )


def _evolve_one_imex_step(
    semidiscretization: object,
    tableau: IMEXRungeKuttaTableau,
    t_left: float,
    dt: float,
    u: np.ndarray,
    parameters: IMEXRungeKuttaGMRESParameters,
    step_index: int,
    on_gmres_solve: Callable[[int, int, IMEXRungeKuttaGMRESStatistics], None] | None,
    stage_work: _IMEXRungeKuttaStageWork,
) -> None:
    has_diffusion = semidiscretization.equation.epsilon != 0.0
    stage_work.initial_state[:] = u
    for stage in range(tableau.num_stages):
        stage_work.stage_rhs[:] = stage_work.initial_state
        for previous in range(stage):
            stage_work.stage_rhs += (
                dt
                * tableau.a_explicit[stage, previous]
                * stage_work.explicit_rhs[previous]
            )
            if has_diffusion:
                stage_work.stage_rhs += (
                    dt
                    * tableau.a_implicit[stage, previous]
                    * semidiscretization.equation.epsilon
                    * stage_work.diffusion_rhs[previous]
                )
        gamma = tableau.a_implicit[stage, stage]
        if gamma == 0.0 or not has_diffusion:
            stage_work.stage_values[stage] = stage_work.stage_rhs
        else:
            stage_work.stage_values[stage] = stage_work.initial_state
            statistics = solve_implicit_diffusion_stage(
                semidiscretization,
                stage_work.stage_rhs,
                stage_work.stage_values[stage],
                alpha=dt * gamma * semidiscretization.equation.epsilon,
                parameters=parameters,
                initial_guess=stage_work.stage_values[stage],
                work_state=stage_work.gmres_state,
                work_diffusion=stage_work.gmres_diffusion,
                work_output=stage_work.gmres_output,
            )
            if on_gmres_solve is not None:
                on_gmres_solve(step_index, stage, statistics)
        stage_time = t_left + dt * tableau.c_explicit[stage]
        semidiscretization.evaluate_explicit_rhs(
            stage_time,
            stage_work.stage_values[stage],
            stage_work.explicit_rhs[stage],
        )
        if has_diffusion:
            semidiscretization.evaluate_diffusion(
                stage_work.stage_values[stage],
                stage_work.diffusion_rhs[stage],
            )

    u[:] = stage_work.initial_state
    for stage in range(tableau.num_stages):
        u += dt * tableau.b_explicit[stage] * stage_work.explicit_rhs[stage]
        if has_diffusion:
            u += (
                dt
                * tableau.b_implicit[stage]
                * semidiscretization.equation.epsilon
                * stage_work.diffusion_rhs[stage]
            )


def _allocate_imex_runge_kutta_stage_work(
    tableau: IMEXRungeKuttaTableau,
    reference: np.ndarray,
) -> _IMEXRungeKuttaStageWork:
    stage_shape = (tableau.num_stages, *reference.shape)
    return _IMEXRungeKuttaStageWork(
        initial_state=np.empty_like(reference),
        stage_values=np.empty(stage_shape, dtype=np.float64),
        explicit_rhs=np.empty(stage_shape, dtype=np.float64),
        diffusion_rhs=np.empty(stage_shape, dtype=np.float64),
        stage_rhs=np.empty_like(reference),
        gmres_state=np.empty_like(reference),
        gmres_diffusion=np.empty_like(reference),
        gmres_output=np.empty_like(reference),
    )


__all__ = [
    "evolve_fixed_steps_ark3_2_4l_2_sa_gmres",
    "evolve_fixed_steps_ark5_4_8l_2_sa_gmres",
    "evolve_fixed_steps_imex_runge_kutta_gmres",
]
