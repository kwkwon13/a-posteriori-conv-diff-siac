from __future__ import annotations

from ._parameters import LinearAdvectionDiffusionExperimentParameters
from ._refinement import compute_eoc, run_refinement_levels
from ._levels import run_linear_advection_diffusion_level
from ._results import ScalarExperimentResult, ScalarLevelResult


LinearAdvectionDiffusionLevelResult = ScalarLevelResult
LinearAdvectionDiffusionExperimentResult = ScalarExperimentResult

_EOC_FIELDS = (
    "hat_u_h_t_l2_l2_error",
    "E_rec",
    "residual_1_l1_l2",
    "E_r2",
    "E_est",
)


def run_linear_advection_diffusion_experiment(
    *,
    polynomial_degree_q: int,
    mesh_counts: tuple[int, ...],
    epsilon: float,
    final_time: float,
    cfl_number: float = 0.1,
    velocity: tuple[float, float] = (1.0, 0.5),
    checkpoint_directory: str | None = None,
) -> LinearAdvectionDiffusionExperimentResult:
    """Run the linear advection-diffusion paper experiment."""

    velocity_values = tuple(float(component) for component in velocity)
    parameters = LinearAdvectionDiffusionExperimentParameters(
        polynomial_degree_q=polynomial_degree_q,
        mesh_counts=tuple(int(mesh_count) for mesh_count in mesh_counts),
        epsilon=float(epsilon),
        final_time=float(final_time),
        cfl_number=float(cfl_number),
        velocity=velocity_values,
        checkpoint_directory=checkpoint_directory,
    )
    levels = run_refinement_levels(
        parameters,
        lambda mesh_count: run_linear_advection_diffusion_level(
            parameters,
            mesh_count,
        ),
        experiment_name="linear_advection_diffusion",
        level_type=ScalarLevelResult,
    )
    return LinearAdvectionDiffusionExperimentResult(
        levels=levels,
        eoc=compute_eoc(levels, _EOC_FIELDS),
    )


__all__ = [
    "LinearAdvectionDiffusionExperimentResult",
    "LinearAdvectionDiffusionLevelResult",
    "run_linear_advection_diffusion_experiment",
]
