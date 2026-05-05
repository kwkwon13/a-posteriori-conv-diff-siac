from __future__ import annotations

import numpy as np


TWO_PI = 2.0 * np.pi


def source_values(equation: object, t: float, points: np.ndarray, out: np.ndarray) -> None:
    """Evaluate source values over a point array with final coordinate axis."""

    expected_shape = points.shape[:-1] + (equation.num_components,)
    if points.shape[-1] != 2:
        raise ValueError("points must have final axis of length 2")
    if out.shape != expected_shape:
        raise ValueError(f"out must have shape {expected_shape}")
    if getattr(equation, "source_term_is_zero", False):
        out.fill(0.0)
        return
    bulk_source_values = getattr(equation, "source_values", None)
    if bulk_source_values is not None:
        bulk_source_values(float(t), points, out)
        return
    flat_points = points.reshape((-1, 2))
    flat_out = out.reshape((-1, equation.num_components))
    for row_index in range(flat_points.shape[0]):
        flat_out[row_index] = equation.source(float(t), flat_points[row_index])
