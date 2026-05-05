from __future__ import annotations

import time
from dataclasses import dataclass, fields

import numpy as np

from aposteriori_cd.dg import (
    allocate_state,
    build_dg_semidiscretization,
    project_initial_condition,
)
from aposteriori_cd.discrete import build_basis_2d, build_mesh_2d
from aposteriori_cd.equations import (
    DiffusivePSystem2D,
    LinearAdvectionDiffusion2D,
    ViscousBurgers2D,
)
from aposteriori_cd.estimates import (
    EstimatorEvaluationTiming,
    PSystemEstimatorEvaluation2D,
    ScalarEstimatorEvaluation2D,
    build_p_system_estimator_evaluation_2d,
    build_scalar_estimator_evaluation_2d,
    linear_scalar_a_posteriori_error_estimate,
    p_system_a_posteriori_error_estimate,
    p_system_lambda,
    p_system_reconstruction_error,
    scalar_reconstruction_error,
    viscous_burgers_a_posteriori_error_estimate,
    viscous_burgers_lambda,
)
from aposteriori_cd.time import (
    FixedTimeStepParameters,
    H100,
    IMEXRungeKuttaGMRESParameters,
    IMEXRungeKuttaGMRESStatistics,
    build_imex_runge_kutta_tableau,
    evolve_fixed_steps_imex_runge_kutta_gmres,
    fixed_dt,
    tableau_name_for_degree,
    temporal_reconstruction_type_for_degree,
)

from ._parameters import (
    DiffusivePSystemExperimentParameters,
    LinearAdvectionDiffusionExperimentParameters,
    ViscousBurgersExperimentParameters,
)
from ._results import PSystemLevelResult, ScalarLevelResult, ViscousBurgersLevelResult


PAPER_GMRES_RELATIVE_TOLERANCE = 0.0
PAPER_GMRES_ABSOLUTE_TOLERANCE = 1.0e-10
PAPER_GMRES_RESTART = 40
PAPER_GMRES_MAX_ITERATIONS = 120


@dataclass(frozen=True, slots=True)
class _LevelContext:
    mesh_count: int
    h: float
    dt: float
    num_steps: int
    state: np.ndarray
    semidiscretization: object
    tableau_name: str
    temporal_reconstruction_type: str
    gmres_parameters: IMEXRungeKuttaGMRESParameters


@dataclass(frozen=True, slots=True)
class _TimeEvolutionSummary:
    mean_gmres_iterations: float
    max_gmres_iterations: int
    max_gmres_residual_norm: float
    running_time_seconds: float


def run_linear_advection_diffusion_level(
    parameters: LinearAdvectionDiffusionExperimentParameters,
    mesh_count: int,
    *,
    estimator_timing: EstimatorEvaluationTiming | None = None,
) -> ScalarLevelResult:
    equation = LinearAdvectionDiffusion2D(
        epsilon=parameters.epsilon,
        velocity=parameters.velocity,
        final_time=parameters.final_time,
    )
    context = _build_level_context(equation, parameters, mesh_count)
    initial_state = context.state.copy()
    evaluation = build_scalar_estimator_evaluation_2d(
        equation,
        context.semidiscretization.mesh,
        context.semidiscretization.basis,
        context.temporal_reconstruction_type,
        timing=estimator_timing,
    )
    initial_error = evaluation.initial_hat_u_ts_l2_error(initial_state)
    time_summary = _run_time_evolution(
        context,
        final_time=parameters.final_time,
        evaluation=evaluation,
    )
    E_rec = scalar_reconstruction_error(
        evaluation.hat_u_ts_linf_l2_error(),
        evaluation.hat_u_ts_l2_h1_seminorm_error(),
        equation.epsilon,
    )
    E_est = linear_scalar_a_posteriori_error_estimate(
        initial_error,
        evaluation.residual_1_l1_l2(),
        evaluation.E_r2(),
        equation.epsilon,
    )
    return _scalar_level_result(
        context,
        evaluation,
        initial_error,
        E_rec,
        E_est,
        time_summary,
    )


def run_viscous_burgers_level(
    parameters: ViscousBurgersExperimentParameters,
    mesh_count: int,
    *,
    estimator_timing: EstimatorEvaluationTiming | None = None,
) -> ViscousBurgersLevelResult:
    equation = ViscousBurgers2D(
        epsilon=parameters.epsilon,
        final_time=parameters.final_time,
    )
    context = _build_level_context(equation, parameters, mesh_count)
    initial_state = context.state.copy()
    evaluation = build_scalar_estimator_evaluation_2d(
        equation,
        context.semidiscretization.mesh,
        context.semidiscretization.basis,
        context.temporal_reconstruction_type,
        timing=estimator_timing,
    )
    initial_error = evaluation.initial_hat_u_ts_l2_error(initial_state)
    time_summary = _run_time_evolution(
        context,
        final_time=parameters.final_time,
        evaluation=evaluation,
    )
    E_rec = scalar_reconstruction_error(
        evaluation.hat_u_ts_linf_l2_error(),
        evaluation.hat_u_ts_l2_h1_seminorm_error(),
        equation.epsilon,
    )
    max_gradient = evaluation.max_abs_gradient()
    Lambda = viscous_burgers_lambda(max_gradient[0, 0], max_gradient[1, 0])
    E_est = viscous_burgers_a_posteriori_error_estimate(
        initial_error,
        evaluation.residual_1_l1_l2(),
        evaluation.E_r2(),
        equation.epsilon,
        Lambda,
        parameters.final_time,
    )
    scalar = _scalar_level_result(
        context,
        evaluation,
        initial_error,
        E_rec,
        E_est,
        time_summary,
    )
    return ViscousBurgersLevelResult(**_level_result_dict(scalar), Lambda=Lambda)


def run_diffusive_p_system_level(
    parameters: DiffusivePSystemExperimentParameters,
    mesh_count: int,
    *,
    estimator_timing: EstimatorEvaluationTiming | None = None,
) -> PSystemLevelResult:
    equation = DiffusivePSystem2D(
        epsilon=parameters.epsilon,
        final_time=parameters.final_time,
    )
    context = _build_level_context(equation, parameters, mesh_count)
    initial_state = context.state.copy()
    evaluation = build_p_system_estimator_evaluation_2d(
        equation,
        context.semidiscretization.mesh,
        context.semidiscretization.basis,
        context.temporal_reconstruction_type,
        timing=estimator_timing,
    )
    initial_terms = evaluation.initial_terms(initial_state)
    time_summary = _run_time_evolution(
        context,
        final_time=parameters.final_time,
        evaluation=evaluation,
    )
    c_W = equation.c_W(evaluation.tau_min, evaluation.tau_max)
    C_W = equation.C_W(evaluation.tau_min, evaluation.tau_max)
    Lambda = p_system_lambda(c_W, C_W, evaluation.div_hat_v_linf)
    E_rec = p_system_reconstruction_error(
        c_W,
        evaluation.hat_tau_ts_linf_l2_error(),
        evaluation.hat_v_ts_linf_l2_error(),
        evaluation.hat_v_ts_l2_h1_seminorm_error(),
        equation.epsilon,
    )
    E_est = p_system_a_posteriori_error_estimate(
        initial_terms.initial_W_tau_l1,
        initial_terms.initial_v_l2_error_squared,
        evaluation.W_second_r_u_l1_l2(),
        evaluation.r_v_1_l1_l2(),
        evaluation.E_r2(),
        equation.epsilon,
        c_W,
        Lambda,
        parameters.final_time,
    )
    return PSystemLevelResult(
        mesh_count=context.mesh_count,
        h=context.h,
        dt=context.dt,
        num_steps=context.num_steps,
        hat_u_h_t_l2_l2_error=evaluation.hat_u_h_t_l2_l2_error(),
        hat_tau_ts_linf_l2_error=evaluation.hat_tau_ts_linf_l2_error(),
        hat_v_ts_linf_l2_error=evaluation.hat_v_ts_linf_l2_error(),
        hat_v_ts_l2_h1_seminorm_error=(
            evaluation.hat_v_ts_l2_h1_seminorm_error()
        ),
        E_rec=E_rec,
        initial_W_tau_l1=initial_terms.initial_W_tau_l1,
        initial_v_l2_error_squared=initial_terms.initial_v_l2_error_squared,
        residual_1_l1_l2=evaluation.residual_1_l1_l2(),
        W_second_r_u_l1_l2=evaluation.W_second_r_u_l1_l2(),
        r_v_1_l1_l2=evaluation.r_v_1_l1_l2(),
        E_r2=evaluation.E_r2(),
        c_W=c_W,
        C_W=C_W,
        Lambda=Lambda,
        tau_min=evaluation.tau_min,
        tau_max=evaluation.tau_max,
        div_hat_v_linf=evaluation.div_hat_v_linf,
        E_est=E_est,
        time_at_max_tau_l2_error=evaluation.time_at_max_tau_l2_error(),
        time_at_max_velocity_l2_error=(
            evaluation.time_at_max_velocity_l2_error()
        ),
        mean_gmres_iterations=time_summary.mean_gmres_iterations,
        max_gmres_iterations=time_summary.max_gmres_iterations,
        max_gmres_residual_norm=time_summary.max_gmres_residual_norm,
        running_time_seconds=time_summary.running_time_seconds,
        preconditioner=context.gmres_parameters.preconditioner,
        tableau_name=context.tableau_name,
        temporal_reconstruction_type=context.temporal_reconstruction_type,
    )


def _build_level_context(
    equation: object,
    parameters: (
        LinearAdvectionDiffusionExperimentParameters
        | ViscousBurgersExperimentParameters
        | DiffusivePSystemExperimentParameters
    ),
    mesh_count: int,
) -> _LevelContext:
    q = parameters.polynomial_degree_q
    mesh = build_mesh_2d(int(mesh_count), int(mesh_count))
    basis = build_basis_2d(q, quadrature_order=max(q + 4, 6))
    state = allocate_state(mesh, basis, equation.num_components)
    project_initial_condition(mesh, basis, equation, state)
    semidiscretization = build_dg_semidiscretization(mesh, basis, equation)
    dt, num_steps = fixed_dt(parameters.final_time, mesh.dx, parameters.cfl_number)
    tableau_name = tableau_name_for_degree(q)
    temporal_reconstruction_type = temporal_reconstruction_type_for_degree(q)
    if temporal_reconstruction_type == H100 and num_steps < 2:
        raise ValueError(
            "H(1,0,0) reconstruction requires at least two time subintervals",
        )
    gmres_parameters = IMEXRungeKuttaGMRESParameters(
        preconditioner="jacobi",
        relative_tolerance=PAPER_GMRES_RELATIVE_TOLERANCE,
        absolute_tolerance=PAPER_GMRES_ABSOLUTE_TOLERANCE,
        restart=PAPER_GMRES_RESTART,
        max_iterations=PAPER_GMRES_MAX_ITERATIONS,
    )
    return _LevelContext(
        mesh_count=int(mesh_count),
        h=float(mesh.dx),
        dt=float(dt),
        num_steps=int(num_steps),
        state=state,
        semidiscretization=semidiscretization,
        tableau_name=tableau_name,
        temporal_reconstruction_type=temporal_reconstruction_type,
        gmres_parameters=gmres_parameters,
    )


def _run_time_evolution(
    context: _LevelContext,
    *,
    final_time: float,
    evaluation: ScalarEstimatorEvaluation2D | PSystemEstimatorEvaluation2D,
) -> _TimeEvolutionSummary:
    gmres_statistics: list[IMEXRungeKuttaGMRESStatistics] = []
    start_time = time.perf_counter()
    evolve_fixed_steps_imex_runge_kutta_gmres(
        context.semidiscretization,
        build_imex_runge_kutta_tableau(context.tableau_name),
        FixedTimeStepParameters(t_start=0.0, t_final=final_time, dt=context.dt),
        context.state,
        parameters=context.gmres_parameters,
        on_time_subinterval=evaluation.add_time_subinterval_data,
        on_gmres_solve=lambda _step, _stage, statistics: gmres_statistics.append(
            statistics,
        ),
        store_time_subinterval_data=True,
    )
    return _TimeEvolutionSummary(
        mean_gmres_iterations=_mean_gmres_iterations(gmres_statistics),
        max_gmres_iterations=_max_gmres_iterations(gmres_statistics),
        max_gmres_residual_norm=_max_gmres_residual_norm(gmres_statistics),
        running_time_seconds=float(time.perf_counter() - start_time),
    )


def _scalar_level_result(
    context: _LevelContext,
    evaluation: ScalarEstimatorEvaluation2D,
    initial_error: float,
    E_rec: float,
    E_est: float,
    time_summary: _TimeEvolutionSummary,
) -> ScalarLevelResult:
    return ScalarLevelResult(
        mesh_count=context.mesh_count,
        h=context.h,
        dt=context.dt,
        num_steps=context.num_steps,
        hat_u_h_t_l2_l2_error=evaluation.hat_u_h_t_l2_l2_error(),
        hat_u_ts_linf_l2_error=evaluation.hat_u_ts_linf_l2_error(),
        hat_u_ts_l2_h1_seminorm_error=(
            evaluation.hat_u_ts_l2_h1_seminorm_error()
        ),
        E_rec=E_rec,
        initial_hat_u_ts_l2_error=initial_error,
        residual_1_l1_l2=evaluation.residual_1_l1_l2(),
        E_r2=evaluation.E_r2(),
        E_est=E_est,
        time_at_max_l2_error=evaluation.time_at_max_l2_error(),
        mean_gmres_iterations=time_summary.mean_gmres_iterations,
        max_gmres_iterations=time_summary.max_gmres_iterations,
        max_gmres_residual_norm=time_summary.max_gmres_residual_norm,
        running_time_seconds=time_summary.running_time_seconds,
        preconditioner=context.gmres_parameters.preconditioner,
        tableau_name=context.tableau_name,
        temporal_reconstruction_type=context.temporal_reconstruction_type,
    )


def _level_result_dict(level: ScalarLevelResult) -> dict[str, object]:
    return {field.name: getattr(level, field.name) for field in fields(level)}


def _mean_gmres_iterations(
    statistics: list[IMEXRungeKuttaGMRESStatistics],
) -> float:
    if len(statistics) == 0:
        return 0.0
    return float(np.mean([entry.iterations for entry in statistics]))


def _max_gmres_iterations(
    statistics: list[IMEXRungeKuttaGMRESStatistics],
) -> int:
    if len(statistics) == 0:
        return 0
    return int(max(entry.iterations for entry in statistics))


def _max_gmres_residual_norm(
    statistics: list[IMEXRungeKuttaGMRESStatistics],
) -> float:
    if len(statistics) == 0:
        return 0.0
    return float(max(entry.residual_norm for entry in statistics))


__all__ = [
    "run_diffusive_p_system_level",
    "run_linear_advection_diffusion_level",
    "run_viscous_burgers_level",
]
