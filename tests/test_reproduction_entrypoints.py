from __future__ import annotations

import aposteriori_cd


def test_root_lists_only_paper_experiment_entrypoints() -> None:
    assert sorted(aposteriori_cd.__all__) == [
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


def test_root_exports_do_not_include_numerical_modules() -> None:
    assert "discrete" not in aposteriori_cd.__all__
    assert "dg" not in aposteriori_cd.__all__
    assert "time" not in aposteriori_cd.__all__
