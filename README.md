# Paper Reproduction Code

This repository contains the Python code used to reproduce the numerical
experiments for the paper
[A Posteriori Error Analysis of Runge-Kutta Discontinuous Galerkin Schemes with
SIAC Post-Processing for Nonlinear Convection-Diffusion Systems](https://doi.org/10.48550/arXiv.2604.01200).

## Installation

Use Python 3.11 or newer.

```bash
python -m pip install -e ".[test]"
```

## Run One Experiment

```bash
python scripts/run_experiment.py \
  --model linear \
  --polynomial-degree-q 1 \
  --mesh-counts 2,4 \
  --epsilon 1.0e-2 \
  --final-time 1.0e-3 \
  --checkpoint-directory checkpoints \
  --output-json output/linear.json \
  --output-csv output/linear.csv
```

Available models are `linear`, `burgers`, and `p-system`.

## Run All Paper Experiments

```bash
python scripts/run_all_experiments.py
```

The results are stored under `results/`:

```text
experiment_set.json
summary.json
json/
csv/
checkpoints/
collected.json
collected.csv
```

## Collect Results

```bash
python scripts/collect_results.py
```

By default, result collection reads the per-experiment JSON files and falls back
to checkpoints when needed.

## Tests

```bash
python3 -m pytest -q
```

## Citation And License

If you use this repository, cite the paper above. Citation information is provided
in `CITATION.cff`.

The code is released under the MIT License; see `LICENSE`.
