"""Paper reproduction code for a posteriori convection-diffusion experiments."""

from .experiments import (
    DiffusivePSystemExperimentResult,
    DiffusivePSystemLevelResult,
    LinearAdvectionDiffusionExperimentResult,
    LinearAdvectionDiffusionLevelResult,
    ViscousBurgersExperimentResult,
    ViscousBurgersLevelResult,
    run_diffusive_p_system_experiment,
    run_linear_advection_diffusion_experiment,
    run_viscous_burgers_experiment,
)

__all__ = [
    "DiffusivePSystemExperimentResult",
    "DiffusivePSystemLevelResult",
    "LinearAdvectionDiffusionExperimentResult",
    "LinearAdvectionDiffusionLevelResult",
    "ViscousBurgersExperimentResult",
    "ViscousBurgersLevelResult",
    "run_diffusive_p_system_experiment",
    "run_linear_advection_diffusion_experiment",
    "run_viscous_burgers_experiment",
]
