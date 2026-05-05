"""Paper experiment entry points."""

from .diffusive_p_system import (
    DiffusivePSystemExperimentResult,
    DiffusivePSystemLevelResult,
    run_diffusive_p_system_experiment,
)
from .linear_advection_diffusion import (
    LinearAdvectionDiffusionExperimentResult,
    LinearAdvectionDiffusionLevelResult,
    run_linear_advection_diffusion_experiment,
)
from .viscous_burgers import (
    ViscousBurgersExperimentResult,
    ViscousBurgersLevelResult,
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
