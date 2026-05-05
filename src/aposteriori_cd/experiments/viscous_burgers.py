from __future__ import annotations

from ._parameters import ViscousBurgersExperimentParameters
from ._refinement import compute_eoc, run_refinement_levels
from ._levels import run_viscous_burgers_level
from ._results import ViscousBurgersExperimentResult, ViscousBurgersLevelResult


_EOC_FIELDS = (
    "hat_u_h_t_l2_l2_error",
    "E_rec",
    "residual_1_l1_l2",
    "E_r2",
    "E_est",
)


def run_viscous_burgers_experiment(
    *,
    polynomial_degree_q: int,
    mesh_counts: tuple[int, ...],
    epsilon: float,
    final_time: float = 0.1,
    cfl_number: float = 0.1,
    checkpoint_directory: str | None = None,
) -> ViscousBurgersExperimentResult:
    """Run the viscous Burgers paper experiment."""

    parameters = ViscousBurgersExperimentParameters(
        polynomial_degree_q=polynomial_degree_q,
        mesh_counts=tuple(int(mesh_count) for mesh_count in mesh_counts),
        epsilon=float(epsilon),
        final_time=float(final_time),
        cfl_number=float(cfl_number),
        checkpoint_directory=checkpoint_directory,
    )
    levels = run_refinement_levels(
        parameters,
        lambda mesh_count: run_viscous_burgers_level(parameters, mesh_count),
        experiment_name="viscous_burgers",
        level_type=ViscousBurgersLevelResult,
    )
    return ViscousBurgersExperimentResult(
        levels=levels,
        eoc=compute_eoc(levels, _EOC_FIELDS),
    )


__all__ = [
    "ViscousBurgersExperimentResult",
    "ViscousBurgersLevelResult",
    "run_viscous_burgers_experiment",
]
