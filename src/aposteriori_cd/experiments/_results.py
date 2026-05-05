from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ScalarLevelResult:
    """Result fields shared by the scalar paper experiments."""

    mesh_count: int
    h: float
    dt: float
    num_steps: int
    hat_u_h_t_l2_l2_error: float
    hat_u_ts_linf_l2_error: float
    hat_u_ts_l2_h1_seminorm_error: float
    E_rec: float
    initial_hat_u_ts_l2_error: float
    residual_1_l1_l2: float
    E_r2: float
    E_est: float
    time_at_max_l2_error: float
    mean_gmres_iterations: float
    max_gmres_iterations: int
    max_gmres_residual_norm: float
    running_time_seconds: float
    preconditioner: str
    tableau_name: str
    temporal_reconstruction_type: str


@dataclass(frozen=True, slots=True)
class ScalarExperimentResult:
    """Refinement results for one scalar paper experiment."""

    levels: tuple[ScalarLevelResult, ...]
    eoc: dict[str, tuple[float, ...]]


@dataclass(frozen=True, slots=True)
class ViscousBurgersLevelResult(ScalarLevelResult):
    """Result fields for the viscous Burgers paper experiment."""

    Lambda: float


@dataclass(frozen=True, slots=True)
class ViscousBurgersExperimentResult:
    """Refinement results for the viscous Burgers paper experiment."""

    levels: tuple[ViscousBurgersLevelResult, ...]
    eoc: dict[str, tuple[float, ...]]


@dataclass(frozen=True, slots=True)
class PSystemLevelResult:
    """Result fields for the diffusive p-system paper experiment."""

    mesh_count: int
    h: float
    dt: float
    num_steps: int
    hat_u_h_t_l2_l2_error: float
    hat_tau_ts_linf_l2_error: float
    hat_v_ts_linf_l2_error: float
    hat_v_ts_l2_h1_seminorm_error: float
    E_rec: float
    initial_W_tau_l1: float
    initial_v_l2_error_squared: float
    residual_1_l1_l2: float
    W_second_r_u_l1_l2: float
    r_v_1_l1_l2: float
    E_r2: float
    c_W: float
    C_W: float
    Lambda: float
    tau_min: float
    tau_max: float
    div_hat_v_linf: float
    E_est: float
    time_at_max_tau_l2_error: float
    time_at_max_velocity_l2_error: float
    mean_gmres_iterations: float
    max_gmres_iterations: int
    max_gmres_residual_norm: float
    running_time_seconds: float
    preconditioner: str
    tableau_name: str
    temporal_reconstruction_type: str


@dataclass(frozen=True, slots=True)
class PSystemExperimentResult:
    """Refinement results for the diffusive p-system paper experiment."""

    levels: tuple[PSystemLevelResult, ...]
    eoc: dict[str, tuple[float, ...]]
