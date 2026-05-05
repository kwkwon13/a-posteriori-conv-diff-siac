from __future__ import annotations

import csv
import json
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import pytest

from aposteriori_cd.experiments import (
    DiffusivePSystemLevelResult,
    LinearAdvectionDiffusionLevelResult,
    ViscousBurgersLevelResult,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_EXPERIMENT_SCRIPT = ROOT / "scripts" / "run_experiment.py"
ALL_EXPERIMENTS_SCRIPT = ROOT / "scripts" / "run_all_experiments.py"
COLLECT_SCRIPT = ROOT / "scripts" / "collect_results.py"
IDENTITY_FIELDS = ("experiment_name", "q", "epsilon", "final_time")


def test_run_experiment_script_writes_json_csv_and_checkpoint(tmp_path) -> None:
    output_json = tmp_path / "out" / "linear.json"
    output_csv = tmp_path / "out" / "linear.csv"
    checkpoint_directory = tmp_path / "checkpoints"

    completed = subprocess.run(
        [
            sys.executable,
            str(RUN_EXPERIMENT_SCRIPT),
            "--model",
            "linear",
            "--polynomial-degree-q",
            "1",
            "--mesh-counts",
            "2",
            "--epsilon",
            "1.0e-2",
            "--final-time",
            "1.0e-3",
            "--checkpoint-directory",
            str(checkpoint_directory),
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ],
        check=True,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    stdout_data = json.loads(completed.stdout)
    file_data = json.loads(output_json.read_text(encoding="utf-8"))
    assert stdout_data == file_data
    assert file_data["experiment_name"] == "linear_advection_diffusion"
    assert file_data["mesh_counts"] == [2]
    assert file_data["levels"][0]["E_est"] > 0.0
    checkpoint_paths = tuple(checkpoint_directory.glob("linear_advection_diffusion_*.json"))
    assert len(checkpoint_paths) == 1
    checkpoint_data = json.loads(checkpoint_paths[0].read_text(encoding="utf-8"))
    assert set(checkpoint_data) == {
        "experiment_identity",
        "experiment_name",
        "level",
        "mesh_count",
        "parameter_key",
        "parameters",
    }
    assert checkpoint_data["experiment_name"] == "linear_advection_diffusion"
    assert checkpoint_data["mesh_count"] == 2

    with output_csv.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    assert reader.fieldnames == [
        *IDENTITY_FIELDS,
        *(field.name for field in fields(LinearAdvectionDiffusionLevelResult)),
    ]
    assert len(rows) == 1
    assert rows[0]["experiment_name"] == "linear_advection_diffusion"
    assert float(rows[0]["E_est"]) > 0.0


def test_run_all_experiments_script_runs_small_experiment_set_and_reuses_checkpoints(
    tmp_path,
) -> None:
    results_directory = tmp_path / "results"
    arguments = [
        sys.executable,
        str(ALL_EXPERIMENTS_SCRIPT),
        "--models",
        "linear,burgers,p-system",
        "--polynomial-degrees",
        "1",
        "--epsilons",
        "1.0e-2",
        "--mesh-counts",
        "2",
        "--linear-final-time",
        "1.0e-3",
        "--burgers-final-time",
        "1.0e-3",
        "--p-system-final-time",
        "1.0e-3",
        "--results-directory",
        str(results_directory),
    ]

    completed = subprocess.run(
        arguments,
        check=True,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    summary = json.loads(completed.stdout)
    experiment_set = json.loads(
        (results_directory / "experiment_set.json").read_text(encoding="utf-8"),
    )
    assert set(summary) == {
        "case_count",
        "cases",
        "completed_case_count",
        "preview",
    }
    assert set(experiment_set) == {
        "case_count",
        "cases",
        "cfl_adv",
        "epsilons",
        "final_times",
        "lambda_max",
        "mesh_counts",
        "models",
        "polynomial_degrees",
    }
    assert summary["completed_case_count"] == 3
    assert experiment_set["case_count"] == 3
    assert set(experiment_set["cases"][0]) == {
        "cfl_adv",
        "cfl_number",
        "epsilon",
        "experiment_name",
        "final_time",
        "lambda_max",
        "mesh_counts",
        "model",
        "polynomial_degree_q",
    }
    assert experiment_set["cases"][0]["cfl_number"] == pytest.approx(0.1 / 3.0)
    assert len(tuple((results_directory / "json").glob("*.json"))) == 3
    assert len(tuple((results_directory / "csv").glob("*.csv"))) == 3
    checkpoint_paths = sorted((results_directory / "checkpoints").glob("*.json"))
    assert len(checkpoint_paths) == 3

    checkpoint_mtimes = {
        path.name: path.stat().st_mtime_ns
        for path in checkpoint_paths
    }
    subprocess.run(
        arguments,
        check=True,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert {
        path.name: path.stat().st_mtime_ns
        for path in checkpoint_paths
    } == checkpoint_mtimes


def test_paper_scale_pilot_preview_uses_n16_n32_experiment_set(tmp_path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ALL_EXPERIMENTS_SCRIPT),
            "--preview",
            "--mesh-counts",
            "16,32",
            "--results-directory",
            str(tmp_path / "results"),
        ],
        check=True,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    summary = json.loads(completed.stdout)
    experiment_set = json.loads(
        (tmp_path / "results" / "experiment_set.json").read_text(
            encoding="utf-8",
        ),
    )

    assert summary["preview"] is True
    assert summary["case_count"] == 30
    assert summary["completed_case_count"] == 0
    assert experiment_set["models"] == ["linear", "burgers", "p-system"]
    assert experiment_set["polynomial_degrees"] == [1, 2]
    assert experiment_set["epsilons"] == [0.0, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1]
    assert experiment_set["mesh_counts"] == [16, 32]
    assert all(case["mesh_counts"] == [16, 32] for case in experiment_set["cases"])


def test_collect_results_from_json_and_checkpoints_writes_result_csv(
    tmp_path,
) -> None:
    results_directory = tmp_path / "results"
    subprocess.run(
        [
            sys.executable,
            str(ALL_EXPERIMENTS_SCRIPT),
            "--models",
            "linear",
            "--polynomial-degrees",
            "1",
            "--epsilons",
            "1.0e-2",
            "--mesh-counts",
            "2,4",
            "--linear-final-time",
            "1.0e-3",
            "--results-directory",
            str(results_directory),
        ],
        check=True,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    json_summary = _collect(
        results_directory,
        output_json=tmp_path / "json-summary.json",
        output_csv=tmp_path / "json-summary.csv",
    )
    default_summary = _collect_with_default_outputs(results_directory)
    checkpoint_summary = _collect(
        results_directory,
        source="checkpoints",
        output_json=tmp_path / "checkpoint-summary.json",
        output_csv=tmp_path / "checkpoint-summary.csv",
    )

    assert json_summary["source"] == "json"
    assert default_summary == json_summary
    assert checkpoint_summary["source"] == "checkpoints"
    assert set(json_summary) == {"experiment_count", "experiments", "source"}
    assert (results_directory / "collected.json").is_file()
    assert (results_directory / "collected.csv").is_file()
    assert json_summary["experiments"] == checkpoint_summary["experiments"]
    experiment = json_summary["experiments"][0]
    assert experiment["mesh_counts"] == [2, 4]
    assert set(experiment["eoc"]) == {
        "hat_u_h_t_l2_l2_error",
        "E_rec",
        "residual_1_l1_l2",
        "E_r2",
        "E_est",
    }
    assert all(len(values) == 1 for values in experiment["eoc"].values())

    with (tmp_path / "json-summary.csv").open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    assert reader.fieldnames == [
        *IDENTITY_FIELDS,
        *(field.name for field in fields(LinearAdvectionDiffusionLevelResult)),
    ]
    assert len(rows) == 2
    assert rows[0]["experiment_name"] == "linear_advection_diffusion"


def test_collect_results_uses_union_of_level_fields_for_mixed_models(
    tmp_path,
) -> None:
    results_directory = tmp_path / "results"
    subprocess.run(
        [
            sys.executable,
            str(ALL_EXPERIMENTS_SCRIPT),
            "--models",
            "linear,burgers,p-system",
            "--polynomial-degrees",
            "1",
            "--epsilons",
            "1.0e-2",
            "--mesh-counts",
            "2",
            "--linear-final-time",
            "1.0e-3",
            "--burgers-final-time",
            "1.0e-3",
            "--p-system-final-time",
            "1.0e-3",
            "--results-directory",
            str(results_directory),
        ],
        check=True,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    summary = _collect(
        results_directory,
        output_json=tmp_path / "mixed-summary.json",
        output_csv=tmp_path / "mixed-summary.csv",
    )
    assert summary["experiment_count"] == 3
    assert all(experiment["eoc"] == {} for experiment in summary["experiments"])

    expected_fields = list(IDENTITY_FIELDS)
    seen = set(expected_fields)
    for level_type in (
        LinearAdvectionDiffusionLevelResult,
        ViscousBurgersLevelResult,
        DiffusivePSystemLevelResult,
    ):
        for field in fields(level_type):
            if field.name not in seen:
                expected_fields.append(field.name)
                seen.add(field.name)

    with (tmp_path / "mixed-summary.csv").open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    assert reader.fieldnames == expected_fields
    assert len(rows) == 3


def test_readme_and_scripts_use_experiment_entrypoints_only() -> None:
    paths = (
        ROOT / "README.md",
        RUN_EXPERIMENT_SCRIPT,
        ROOT / "scripts" / "run_all_experiments.py",
        ROOT / "scripts" / "collect_results.py",
    )
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    assert "from aposteriori_cd.experiments import" in text
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "scripts/run_experiment.py" in readme
    assert "scripts/run_all_experiments.py" in readme
    assert "scripts/collect_results.py" in readme
    assert "results/" in readme
    assert "experiment_set.json" in readme
    assert "summary.json" in readme
    assert "json/" in readme
    assert "csv/" in readme
    assert "checkpoints/" in readme
    assert "collected.json" in readme
    assert "collected.csv" in readme
    for prohibited in (
        "from aposteriori_cd.discrete",
        "from aposteriori_cd.dg",
        "from aposteriori_cd.time",
        "from aposteriori_cd.estimates",
        "from aposteriori_cd.siac_filtering",
        "from aposteriori_cd.temporal_reconstruction",
        "aposteriori_cd.experiments._",
    ):
        assert prohibited not in text


def _collect(
    results_directory: Path,
    *,
    output_json: Path,
    output_csv: Path,
    source: str = "auto",
) -> dict[str, object]:
    completed = subprocess.run(
        [
            sys.executable,
            str(COLLECT_SCRIPT),
            "--results-directory",
            str(results_directory),
            "--source",
            source,
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ],
        check=True,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout_data = json.loads(completed.stdout)
    file_data = json.loads(output_json.read_text(encoding="utf-8"))
    assert stdout_data == file_data
    return stdout_data


def _collect_with_default_outputs(results_directory: Path) -> dict[str, object]:
    completed = subprocess.run(
        [
            sys.executable,
            str(COLLECT_SCRIPT),
            "--results-directory",
            str(results_directory),
        ],
        check=True,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout_data = json.loads(completed.stdout)
    file_data = json.loads(
        (results_directory / "collected.json").read_text(encoding="utf-8"),
    )
    assert stdout_data == file_data
    return stdout_data
