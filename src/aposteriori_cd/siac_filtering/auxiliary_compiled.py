from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def evaluate_tilde_u_ts_directional_derivative_compiled(
    u: np.ndarray,
    classical_offsets_x: np.ndarray,
    classical_offsets_y: np.ndarray,
    auxiliary_offsets_x: np.ndarray,
    auxiliary_offsets_y: np.ndarray,
    classical_weights_x: np.ndarray,
    classical_weights_y: np.ndarray,
    auxiliary_derivative_weights_x: np.ndarray,
    auxiliary_derivative_weights_y: np.ndarray,
    out: np.ndarray,
) -> None:
    nx = out.shape[0]
    ny = out.shape[1]
    n_analysis = out.shape[2]
    n_basis = u.shape[2]
    num_components = out.shape[5]

    total_count = nx * ny * n_analysis * n_analysis * num_components
    for flat_index in prange(total_count):
        component = flat_index % num_components
        remainder = flat_index // num_components
        py = remainder % n_analysis
        remainder = remainder // n_analysis
        px = remainder % n_analysis
        remainder = remainder // n_analysis
        ey = remainder % ny
        ex = remainder // ny

        value_x = 0.0
        for offset_x_index in range(auxiliary_offsets_x.shape[0]):
            source_ex = (ex + int(auxiliary_offsets_x[offset_x_index])) % nx
            for offset_y_index in range(classical_offsets_y.shape[0]):
                source_ey = (ey + int(classical_offsets_y[offset_y_index])) % ny
                for ix in range(n_basis):
                    derivative_weight_x = auxiliary_derivative_weights_x[
                        px,
                        offset_x_index,
                        ix,
                    ]
                    if derivative_weight_x == 0.0:
                        continue
                    for iy in range(n_basis):
                        value_x += (
                            derivative_weight_x
                            * classical_weights_y[py, offset_y_index, iy]
                            * u[source_ex, source_ey, ix, iy, component]
                        )

        value_y = 0.0
        for offset_x_index in range(classical_offsets_x.shape[0]):
            source_ex = (ex + int(classical_offsets_x[offset_x_index])) % nx
            for offset_y_index in range(auxiliary_offsets_y.shape[0]):
                source_ey = (ey + int(auxiliary_offsets_y[offset_y_index])) % ny
                for ix in range(n_basis):
                    weight_x = classical_weights_x[px, offset_x_index, ix]
                    if weight_x == 0.0:
                        continue
                    for iy in range(n_basis):
                        value_y += (
                            weight_x
                            * auxiliary_derivative_weights_y[py, offset_y_index, iy]
                            * u[source_ex, source_ey, ix, iy, component]
                        )

        out[ex, ey, px, py, 0, component] = value_x
        out[ex, ey, px, py, 1, component] = value_y


@njit(cache=True, parallel=True)
def evaluate_tilde_u_ts_directional_second_derivative_compiled(
    u: np.ndarray,
    classical_offsets_x: np.ndarray,
    classical_offsets_y: np.ndarray,
    auxiliary_offsets_x: np.ndarray,
    auxiliary_offsets_y: np.ndarray,
    classical_weights_x: np.ndarray,
    classical_weights_y: np.ndarray,
    auxiliary_second_derivative_weights_x: np.ndarray,
    auxiliary_second_derivative_weights_y: np.ndarray,
    out: np.ndarray,
) -> None:
    nx = out.shape[0]
    ny = out.shape[1]
    n_analysis = out.shape[2]
    n_basis = u.shape[2]
    num_components = out.shape[5]

    total_count = nx * ny * n_analysis * n_analysis * num_components
    for flat_index in prange(total_count):
        component = flat_index % num_components
        remainder = flat_index // num_components
        py = remainder % n_analysis
        remainder = remainder // n_analysis
        px = remainder % n_analysis
        remainder = remainder // n_analysis
        ey = remainder % ny
        ex = remainder // ny

        value_x = 0.0
        for offset_x_index in range(auxiliary_offsets_x.shape[0]):
            source_ex = (ex + int(auxiliary_offsets_x[offset_x_index])) % nx
            for offset_y_index in range(classical_offsets_y.shape[0]):
                source_ey = (ey + int(classical_offsets_y[offset_y_index])) % ny
                for ix in range(n_basis):
                    second_derivative_weight_x = auxiliary_second_derivative_weights_x[
                        px,
                        offset_x_index,
                        ix,
                    ]
                    if second_derivative_weight_x == 0.0:
                        continue
                    for iy in range(n_basis):
                        value_x += (
                            second_derivative_weight_x
                            * classical_weights_y[py, offset_y_index, iy]
                            * u[source_ex, source_ey, ix, iy, component]
                        )

        value_y = 0.0
        for offset_x_index in range(classical_offsets_x.shape[0]):
            source_ex = (ex + int(classical_offsets_x[offset_x_index])) % nx
            for offset_y_index in range(auxiliary_offsets_y.shape[0]):
                source_ey = (ey + int(auxiliary_offsets_y[offset_y_index])) % ny
                for ix in range(n_basis):
                    weight_x = classical_weights_x[px, offset_x_index, ix]
                    if weight_x == 0.0:
                        continue
                    for iy in range(n_basis):
                        value_y += (
                            weight_x
                            * auxiliary_second_derivative_weights_y[
                                py,
                                offset_y_index,
                                iy,
                            ]
                            * u[source_ex, source_ey, ix, iy, component]
                        )

        out[ex, ey, px, py, 0, component] = value_x
        out[ex, ey, px, py, 1, component] = value_y


@njit(cache=True, parallel=True)
def evaluate_tilde_u_ts_directional_derivatives_compiled(
    u: np.ndarray,
    classical_offsets_x: np.ndarray,
    classical_offsets_y: np.ndarray,
    auxiliary_offsets_x: np.ndarray,
    auxiliary_offsets_y: np.ndarray,
    classical_weights_x: np.ndarray,
    classical_weights_y: np.ndarray,
    auxiliary_derivative_weights_x: np.ndarray,
    auxiliary_derivative_weights_y: np.ndarray,
    auxiliary_second_derivative_weights_x: np.ndarray,
    auxiliary_second_derivative_weights_y: np.ndarray,
    derivative_out: np.ndarray,
    second_derivative_out: np.ndarray,
) -> None:
    nx = derivative_out.shape[0]
    ny = derivative_out.shape[1]
    n_analysis = derivative_out.shape[2]
    n_basis = u.shape[2]
    num_components = derivative_out.shape[5]

    total_count = nx * ny * n_analysis * n_analysis * num_components
    for flat_index in prange(total_count):
        component = flat_index % num_components
        remainder = flat_index // num_components
        py = remainder % n_analysis
        remainder = remainder // n_analysis
        px = remainder % n_analysis
        remainder = remainder // n_analysis
        ey = remainder % ny
        ex = remainder // ny

        derivative_x = 0.0
        second_derivative_x = 0.0
        for offset_x_index in range(auxiliary_offsets_x.shape[0]):
            source_ex = (ex + int(auxiliary_offsets_x[offset_x_index])) % nx
            for offset_y_index in range(classical_offsets_y.shape[0]):
                source_ey = (ey + int(classical_offsets_y[offset_y_index])) % ny
                for ix in range(n_basis):
                    derivative_weight_x = auxiliary_derivative_weights_x[
                        px,
                        offset_x_index,
                        ix,
                    ]
                    second_derivative_weight_x = auxiliary_second_derivative_weights_x[
                        px,
                        offset_x_index,
                        ix,
                    ]
                    if derivative_weight_x == 0.0 and second_derivative_weight_x == 0.0:
                        continue
                    for iy in range(n_basis):
                        weight_y = classical_weights_y[py, offset_y_index, iy]
                        state_value = u[source_ex, source_ey, ix, iy, component]
                        derivative_x += derivative_weight_x * weight_y * state_value
                        second_derivative_x += (
                            second_derivative_weight_x * weight_y * state_value
                        )

        derivative_y = 0.0
        second_derivative_y = 0.0
        for offset_x_index in range(classical_offsets_x.shape[0]):
            source_ex = (ex + int(classical_offsets_x[offset_x_index])) % nx
            for offset_y_index in range(auxiliary_offsets_y.shape[0]):
                source_ey = (ey + int(auxiliary_offsets_y[offset_y_index])) % ny
                for ix in range(n_basis):
                    weight_x = classical_weights_x[px, offset_x_index, ix]
                    if weight_x == 0.0:
                        continue
                    for iy in range(n_basis):
                        derivative_weight_y = auxiliary_derivative_weights_y[
                            py,
                            offset_y_index,
                            iy,
                        ]
                        second_derivative_weight_y = (
                            auxiliary_second_derivative_weights_y[
                                py,
                                offset_y_index,
                                iy,
                            ]
                        )
                        state_value = u[source_ex, source_ey, ix, iy, component]
                        derivative_y += weight_x * derivative_weight_y * state_value
                        second_derivative_y += (
                            weight_x * second_derivative_weight_y * state_value
                        )

        derivative_out[ex, ey, px, py, 0, component] = derivative_x
        derivative_out[ex, ey, px, py, 1, component] = derivative_y
        second_derivative_out[ex, ey, px, py, 0, component] = second_derivative_x
        second_derivative_out[ex, ey, px, py, 1, component] = second_derivative_y


@njit(cache=True, parallel=True)
def evaluate_tilde_u_ts_directional_derivatives_with_work_compiled(
    u: np.ndarray,
    classical_offsets_x: np.ndarray,
    classical_offsets_y: np.ndarray,
    auxiliary_offsets_x: np.ndarray,
    auxiliary_offsets_y: np.ndarray,
    classical_weights_x: np.ndarray,
    classical_weights_y: np.ndarray,
    auxiliary_derivative_weights_x: np.ndarray,
    auxiliary_derivative_weights_y: np.ndarray,
    auxiliary_second_derivative_weights_x: np.ndarray,
    auxiliary_second_derivative_weights_y: np.ndarray,
    work: np.ndarray,
    derivative_out: np.ndarray,
    second_derivative_out: np.ndarray,
) -> None:
    nx = derivative_out.shape[0]
    ny = derivative_out.shape[1]
    n_analysis = derivative_out.shape[2]
    n_basis = u.shape[2]
    num_components = derivative_out.shape[5]

    work_count = nx * ny * n_analysis * n_basis * num_components
    for flat_index in prange(work_count):
        component = flat_index % num_components
        remainder = flat_index // num_components
        ix = remainder % n_basis
        remainder = remainder // n_basis
        py = remainder % n_analysis
        remainder = remainder // n_analysis
        ey = remainder % ny
        source_ex = remainder // ny

        value = 0.0
        for offset_y_index in range(classical_offsets_y.shape[0]):
            source_ey = (ey + int(classical_offsets_y[offset_y_index])) % ny
            for iy in range(n_basis):
                value += (
                    classical_weights_y[py, offset_y_index, iy]
                    * u[source_ex, source_ey, ix, iy, component]
                )
        work[source_ex, ey, py, ix, component] = value

    output_count = nx * ny * n_analysis * n_analysis * num_components
    for flat_index in prange(output_count):
        component = flat_index % num_components
        remainder = flat_index // num_components
        py = remainder % n_analysis
        remainder = remainder // n_analysis
        px = remainder % n_analysis
        remainder = remainder // n_analysis
        ey = remainder % ny
        ex = remainder // ny

        value_x = 0.0
        second_value_x = 0.0
        for offset_x_index in range(auxiliary_offsets_x.shape[0]):
            source_ex = (ex + int(auxiliary_offsets_x[offset_x_index])) % nx
            for ix in range(n_basis):
                weight_x = auxiliary_derivative_weights_x[
                    px,
                    offset_x_index,
                    ix,
                ]
                second_weight_x = auxiliary_second_derivative_weights_x[
                    px,
                    offset_x_index,
                    ix,
                ]
                if weight_x == 0.0 and second_weight_x == 0.0:
                    continue
                work_value = work[source_ex, ey, py, ix, component]
                value_x += weight_x * work_value
                second_value_x += second_weight_x * work_value
        derivative_out[ex, ey, px, py, 0, component] = value_x
        second_derivative_out[ex, ey, px, py, 0, component] = second_value_x

    for flat_index in prange(work_count):
        component = flat_index % num_components
        remainder = flat_index // num_components
        iy = remainder % n_basis
        remainder = remainder // n_basis
        px = remainder % n_analysis
        remainder = remainder // n_analysis
        source_ey = remainder % ny
        ex = remainder // ny

        value = 0.0
        for offset_x_index in range(classical_offsets_x.shape[0]):
            source_ex = (ex + int(classical_offsets_x[offset_x_index])) % nx
            for ix in range(n_basis):
                value += (
                    classical_weights_x[px, offset_x_index, ix]
                    * u[source_ex, source_ey, ix, iy, component]
                )
        work[ex, source_ey, px, iy, component] = value

    for flat_index in prange(output_count):
        component = flat_index % num_components
        remainder = flat_index // num_components
        py = remainder % n_analysis
        remainder = remainder // n_analysis
        px = remainder % n_analysis
        remainder = remainder // n_analysis
        ey = remainder % ny
        ex = remainder // ny

        value_y = 0.0
        second_value_y = 0.0
        for offset_y_index in range(auxiliary_offsets_y.shape[0]):
            source_ey = (ey + int(auxiliary_offsets_y[offset_y_index])) % ny
            for iy in range(n_basis):
                weight_y = auxiliary_derivative_weights_y[
                    py,
                    offset_y_index,
                    iy,
                ]
                second_weight_y = auxiliary_second_derivative_weights_y[
                    py,
                    offset_y_index,
                    iy,
                ]
                if weight_y == 0.0 and second_weight_y == 0.0:
                    continue
                work_value = work[ex, source_ey, px, iy, component]
                value_y += weight_y * work_value
                second_value_y += second_weight_y * work_value
        derivative_out[ex, ey, px, py, 1, component] = value_y
        second_derivative_out[ex, ey, px, py, 1, component] = second_value_y

__all__ = [
    "evaluate_tilde_u_ts_directional_derivative_compiled",
    "evaluate_tilde_u_ts_directional_derivatives_compiled",
    "evaluate_tilde_u_ts_directional_derivatives_with_work_compiled",
    "evaluate_tilde_u_ts_directional_second_derivative_compiled",
]
