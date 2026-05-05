from __future__ import annotations

import numpy as np

from aposteriori_cd.experiments import (
    run_diffusive_p_system_experiment,
    run_linear_advection_diffusion_experiment,
    run_viscous_burgers_experiment,
)


def test_paper_experiments_return_finite_lowest_level_results() -> None:
    experiments = (
        run_linear_advection_diffusion_experiment,
        run_viscous_burgers_experiment,
        run_diffusive_p_system_experiment,
    )

    for experiment in experiments:
        result = experiment(
            polynomial_degree_q=1,
            mesh_counts=(2,),
            epsilon=1.0e-2,
            final_time=1.0e-3,
        )
        level = result.levels[0]
        assert level.mesh_count == 2
        assert level.E_est > 0.0
        assert np.isfinite(level.E_est)


def test_paper_experiments_compute_eoc_field_lists() -> None:
    scalar_fields = {
        "hat_u_h_t_l2_l2_error",
        "E_rec",
        "residual_1_l1_l2",
        "E_r2",
        "E_est",
    }
    p_system_fields = scalar_fields | {"W_second_r_u_l1_l2", "r_v_1_l1_l2"}

    linear = run_linear_advection_diffusion_experiment(
        polynomial_degree_q=1,
        mesh_counts=(2, 4),
        epsilon=1.0e-2,
        final_time=1.0e-3,
    )
    burgers = run_viscous_burgers_experiment(
        polynomial_degree_q=1,
        mesh_counts=(2, 4),
        epsilon=1.0e-2,
        final_time=1.0e-3,
    )
    p_system = run_diffusive_p_system_experiment(
        polynomial_degree_q=1,
        mesh_counts=(2, 4),
        epsilon=1.0e-2,
        final_time=1.0e-3,
    )

    assert set(linear.eoc) == scalar_fields
    assert set(burgers.eoc) == scalar_fields
    assert set(p_system.eoc) == p_system_fields
    assert all(len(values) == 1 for values in linear.eoc.values())
    assert all(len(values) == 1 for values in burgers.eoc.values())
    assert all(len(values) == 1 for values in p_system.eoc.values())
