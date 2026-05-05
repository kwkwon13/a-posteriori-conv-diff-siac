"""Flux, diffusion tensor, exact solution, and source data for paper equations."""

from ._common import source_values
from .diffusive_p_system import DiffusivePSystem2D
from .linear_advection_diffusion import LinearAdvectionDiffusion2D
from .viscous_burgers import ViscousBurgers2D

__all__ = [
    "DiffusivePSystem2D",
    "LinearAdvectionDiffusion2D",
    "ViscousBurgers2D",
    "source_values",
]
