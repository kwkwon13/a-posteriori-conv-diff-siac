from __future__ import annotations

from ._parameters import DiffusivePSystemExperimentParameters
from ._refinement import compute_eoc, run_refinement_levels
from ._levels import run_diffusive_p_system_level
from ._results import PSystemExperimentResult, PSystemLevelResult


DiffusivePSystemLevelResult = PSystemLevelResult
DiffusivePSystemExperimentResult = PSystemExperimentResult

_EOC_FIELDS = (
    "hat_u_h_t_l2_l2_error",
    "E_rec",
    "residual_1_l1_l2",
    "W_second_r_u_l1_l2",
    "r_v_1_l1_l2",
    "E_r2",
    "E_est",
)


def run_diffusive_p_system_experiment(
    *,
    polynomial_degree_q: int,
    mesh_counts: tuple[int, ...],
    epsilon: float,
    final_time: float = 0.05,
    cfl_number: float = 0.1,
    checkpoint_directory: str | None = None,
) -> DiffusivePSystemExperimentResult:
    """Run the diffusive p-system paper experiment."""

    parameters = DiffusivePSystemExperimentParameters(
        polynomial_degree_q=polynomial_degree_q,
        mesh_counts=tuple(int(mesh_count) for mesh_count in mesh_counts),
        epsilon=float(epsilon),
        final_time=float(final_time),
        cfl_number=float(cfl_number),
        checkpoint_directory=checkpoint_directory,
    )
    levels = run_refinement_levels(
        parameters,
        lambda mesh_count: run_diffusive_p_system_level(parameters, mesh_count),
        experiment_name="diffusive_p_system",
        level_type=PSystemLevelResult,
    )
    return DiffusivePSystemExperimentResult(
        levels=levels,
        eoc=compute_eoc(levels, _EOC_FIELDS),
    )


__all__ = [
    "DiffusivePSystemExperimentResult",
    "DiffusivePSystemLevelResult",
    "run_diffusive_p_system_experiment",
]
