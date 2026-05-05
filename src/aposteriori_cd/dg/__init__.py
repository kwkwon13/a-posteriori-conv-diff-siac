"""DG and SIP semidiscretization for the paper experiments."""

from .convection import add_convection, evaluate_convection, numerical_flux
from .diffusion import DEFAULT_PENALTY_PARAMETER, add_diffusion, evaluate_diffusion
from .faces import (
    evaluate_exterior_face_trace,
    evaluate_interior_face_trace,
    face_measure_scale,
    face_normal,
    opposite_face,
)
from .projection import (
    add_source_projection,
    project_function_to_state,
    project_initial_condition,
)
from .semidiscretization import DGSemidiscretization2D, build_dg_semidiscretization
from .state import allocate_state, state_shape

__all__ = [
    "DEFAULT_PENALTY_PARAMETER",
    "DGSemidiscretization2D",
    "add_convection",
    "add_diffusion",
    "add_source_projection",
    "allocate_state",
    "build_dg_semidiscretization",
    "evaluate_convection",
    "evaluate_diffusion",
    "evaluate_exterior_face_trace",
    "evaluate_interior_face_trace",
    "face_measure_scale",
    "face_normal",
    "numerical_flux",
    "opposite_face",
    "project_function_to_state",
    "project_initial_condition",
    "state_shape",
]
