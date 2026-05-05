"""Time integration for the paper experiments."""

from .parameters import (
    H000,
    H100,
    FixedTimeStepParameters,
    fixed_dt,
    tableau_name_for_degree,
    temporal_reconstruction_type_for_degree,
)
from .gmres import (
    IMEXRungeKuttaGMRESParameters,
    IMEXRungeKuttaGMRESStatistics,
)
from .imex import (
    evolve_fixed_steps_ark3_2_4l_2_sa_gmres,
    evolve_fixed_steps_ark5_4_8l_2_sa_gmres,
    evolve_fixed_steps_imex_runge_kutta_gmres,
)
from .solution_data import (
    LEFT_ENDPOINT,
    RIGHT_ENDPOINT,
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

__all__ = [
    "ARK3_2_4L_2_SA",
    "ARK5_4_8L_2_SA",
    "H000",
    "H100",
    "LEFT_ENDPOINT",
    "RIGHT_ENDPOINT",
    "FixedTimeStepParameters",
    "IMEXRungeKuttaGMRESParameters",
    "IMEXRungeKuttaGMRESStatistics",
    "IMEXRungeKuttaTableau",
    "TimeSubintervalSolutionData",
    "allocate_time_subinterval_solution_data",
    "build_imex_runge_kutta_tableau",
    "evolve_fixed_steps_ark3_2_4l_2_sa_gmres",
    "evolve_fixed_steps_ark5_4_8l_2_sa_gmres",
    "evolve_fixed_steps_imex_runge_kutta_gmres",
    "fill_time_subinterval_solution_data",
    "fixed_dt",
    "tableau_name_for_degree",
    "temporal_reconstruction_type_for_degree",
]
