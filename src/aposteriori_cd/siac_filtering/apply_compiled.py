from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, parallel=True)
def evaluate_filtered_value_compiled(
    u: np.ndarray,
    offsets_x: np.ndarray,
    offsets_y: np.ndarray,
    weights_x: np.ndarray,
    weights_y: np.ndarray,
    out: np.ndarray,
) -> None:
    nx = out.shape[0]
    ny = out.shape[1]
    n_analysis = out.shape[2]
    n_basis = u.shape[2]
    num_components = out.shape[4]

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

        value = 0.0
        for offset_x_index in range(offsets_x.shape[0]):
            source_ex = (ex + int(offsets_x[offset_x_index])) % nx
            for offset_y_index in range(offsets_y.shape[0]):
                source_ey = (ey + int(offsets_y[offset_y_index])) % ny
                for ix in range(n_basis):
                    weight_x = weights_x[px, offset_x_index, ix]
                    if weight_x == 0.0:
                        continue
                    for iy in range(n_basis):
                        weight = weight_x * weights_y[py, offset_y_index, iy]
                        if weight != 0.0:
                            value += weight * u[source_ex, source_ey, ix, iy, component]
        out[ex, ey, px, py, component] = value


@njit(cache=True, parallel=True)
def evaluate_filtered_gradient_compiled(
    u: np.ndarray,
    offsets_x: np.ndarray,
    offsets_y: np.ndarray,
    weights_x: np.ndarray,
    weights_y: np.ndarray,
    derivative_weights_x: np.ndarray,
    derivative_weights_y: np.ndarray,
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
        value_y = 0.0
        for offset_x_index in range(offsets_x.shape[0]):
            source_ex = (ex + int(offsets_x[offset_x_index])) % nx
            for offset_y_index in range(offsets_y.shape[0]):
                source_ey = (ey + int(offsets_y[offset_y_index])) % ny
                for ix in range(n_basis):
                    weight_x = weights_x[px, offset_x_index, ix]
                    derivative_weight_x = derivative_weights_x[
                        px,
                        offset_x_index,
                        ix,
                    ]
                    if weight_x == 0.0 and derivative_weight_x == 0.0:
                        continue
                    for iy in range(n_basis):
                        weight_y = weights_y[py, offset_y_index, iy]
                        derivative_weight_y = derivative_weights_y[
                            py,
                            offset_y_index,
                            iy,
                        ]
                        state_value = u[source_ex, source_ey, ix, iy, component]
                        value_x += derivative_weight_x * weight_y * state_value
                        value_y += weight_x * derivative_weight_y * state_value
        out[ex, ey, px, py, 0, component] = value_x
        out[ex, ey, px, py, 1, component] = value_y


@njit(cache=True, parallel=True)
def evaluate_filtered_value_and_gradient_compiled(
    u: np.ndarray,
    offsets_x: np.ndarray,
    offsets_y: np.ndarray,
    weights_x: np.ndarray,
    weights_y: np.ndarray,
    derivative_weights_x: np.ndarray,
    derivative_weights_y: np.ndarray,
    value_out: np.ndarray,
    gradient_out: np.ndarray,
) -> None:
    nx = value_out.shape[0]
    ny = value_out.shape[1]
    n_analysis = value_out.shape[2]
    n_basis = u.shape[2]
    num_components = value_out.shape[4]

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

        filtered_value = 0.0
        value_x = 0.0
        value_y = 0.0
        for offset_x_index in range(offsets_x.shape[0]):
            source_ex = (ex + int(offsets_x[offset_x_index])) % nx
            for offset_y_index in range(offsets_y.shape[0]):
                source_ey = (ey + int(offsets_y[offset_y_index])) % ny
                for ix in range(n_basis):
                    weight_x = weights_x[px, offset_x_index, ix]
                    derivative_weight_x = derivative_weights_x[
                        px,
                        offset_x_index,
                        ix,
                    ]
                    if weight_x == 0.0 and derivative_weight_x == 0.0:
                        continue
                    for iy in range(n_basis):
                        weight_y = weights_y[py, offset_y_index, iy]
                        derivative_weight_y = derivative_weights_y[
                            py,
                            offset_y_index,
                            iy,
                        ]
                        state_value = u[source_ex, source_ey, ix, iy, component]
                        weight = weight_x * weight_y
                        if weight != 0.0:
                            filtered_value += weight * state_value
                        value_x += derivative_weight_x * weight_y * state_value
                        value_y += weight_x * derivative_weight_y * state_value
        value_out[ex, ey, px, py, component] = filtered_value
        gradient_out[ex, ey, px, py, 0, component] = value_x
        gradient_out[ex, ey, px, py, 1, component] = value_y


@njit(cache=True, parallel=True)
def evaluate_filtered_value_gradient_and_second_value_compiled(
    u: np.ndarray,
    second_u: np.ndarray,
    offsets_x: np.ndarray,
    offsets_y: np.ndarray,
    weights_x: np.ndarray,
    weights_y: np.ndarray,
    derivative_weights_x: np.ndarray,
    derivative_weights_y: np.ndarray,
    value_out: np.ndarray,
    gradient_out: np.ndarray,
    second_value_out: np.ndarray,
) -> None:
    nx = value_out.shape[0]
    ny = value_out.shape[1]
    n_analysis = value_out.shape[2]
    n_basis = u.shape[2]
    num_components = value_out.shape[4]

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

        filtered_value = 0.0
        value_x = 0.0
        value_y = 0.0
        second_filtered_value = 0.0
        for offset_x_index in range(offsets_x.shape[0]):
            source_ex = (ex + int(offsets_x[offset_x_index])) % nx
            for offset_y_index in range(offsets_y.shape[0]):
                source_ey = (ey + int(offsets_y[offset_y_index])) % ny
                for ix in range(n_basis):
                    weight_x = weights_x[px, offset_x_index, ix]
                    derivative_weight_x = derivative_weights_x[
                        px,
                        offset_x_index,
                        ix,
                    ]
                    if weight_x == 0.0 and derivative_weight_x == 0.0:
                        continue
                    for iy in range(n_basis):
                        weight_y = weights_y[py, offset_y_index, iy]
                        derivative_weight_y = derivative_weights_y[
                            py,
                            offset_y_index,
                            iy,
                        ]
                        state_value = u[source_ex, source_ey, ix, iy, component]
                        second_state_value = second_u[
                            source_ex,
                            source_ey,
                            ix,
                            iy,
                            component,
                        ]
                        weight = weight_x * weight_y
                        if weight != 0.0:
                            filtered_value += weight * state_value
                            second_filtered_value += weight * second_state_value
                        value_x += derivative_weight_x * weight_y * state_value
                        value_y += weight_x * derivative_weight_y * state_value
        value_out[ex, ey, px, py, component] = filtered_value
        gradient_out[ex, ey, px, py, 0, component] = value_x
        gradient_out[ex, ey, px, py, 1, component] = value_y
        second_value_out[ex, ey, px, py, component] = second_filtered_value


@njit(cache=True, parallel=True)
def evaluate_filtered_value_gradient_and_second_value_with_work_compiled(
    u: np.ndarray,
    second_u: np.ndarray,
    offsets_x: np.ndarray,
    offsets_y: np.ndarray,
    weights_x: np.ndarray,
    weights_y: np.ndarray,
    derivative_weights_x: np.ndarray,
    derivative_weights_y: np.ndarray,
    primary_work: np.ndarray,
    secondary_work: np.ndarray,
    value_out: np.ndarray,
    gradient_out: np.ndarray,
    second_value_out: np.ndarray,
) -> None:
    nx = value_out.shape[0]
    ny = value_out.shape[1]
    n_analysis = value_out.shape[2]
    n_basis = u.shape[2]
    num_components = value_out.shape[4]

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

        filtered_y = 0.0
        second_filtered_y = 0.0
        for offset_y_index in range(offsets_y.shape[0]):
            source_ey = (ey + int(offsets_y[offset_y_index])) % ny
            for iy in range(n_basis):
                weight_y = weights_y[py, offset_y_index, iy]
                if weight_y == 0.0:
                    continue
                filtered_y += weight_y * u[source_ex, source_ey, ix, iy, component]
                second_filtered_y += (
                    weight_y * second_u[source_ex, source_ey, ix, iy, component]
                )
        primary_work[source_ex, ey, py, ix, component] = filtered_y
        secondary_work[source_ex, ey, py, ix, component] = second_filtered_y

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

        filtered_value = 0.0
        value_x = 0.0
        second_filtered_value = 0.0
        for offset_x_index in range(offsets_x.shape[0]):
            source_ex = (ex + int(offsets_x[offset_x_index])) % nx
            for ix in range(n_basis):
                weight_x = weights_x[px, offset_x_index, ix]
                derivative_weight_x = derivative_weights_x[
                    px,
                    offset_x_index,
                    ix,
                ]
                if weight_x == 0.0 and derivative_weight_x == 0.0:
                    continue
                filtered_y = primary_work[source_ex, ey, py, ix, component]
                if weight_x != 0.0:
                    filtered_value += weight_x * filtered_y
                    second_filtered_value += (
                        weight_x * secondary_work[source_ex, ey, py, ix, component]
                    )
                value_x += derivative_weight_x * filtered_y
        value_out[ex, ey, px, py, component] = filtered_value
        gradient_out[ex, ey, px, py, 0, component] = value_x
        second_value_out[ex, ey, px, py, component] = second_filtered_value

    for flat_index in prange(work_count):
        component = flat_index % num_components
        remainder = flat_index // num_components
        iy = remainder % n_basis
        remainder = remainder // n_basis
        px = remainder % n_analysis
        remainder = remainder // n_analysis
        source_ey = remainder % ny
        ex = remainder // ny

        filtered_x = 0.0
        for offset_x_index in range(offsets_x.shape[0]):
            source_ex = (ex + int(offsets_x[offset_x_index])) % nx
            for ix in range(n_basis):
                weight_x = weights_x[px, offset_x_index, ix]
                if weight_x == 0.0:
                    continue
                filtered_x += weight_x * u[source_ex, source_ey, ix, iy, component]
        primary_work[ex, source_ey, px, iy, component] = filtered_x

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
        for offset_y_index in range(offsets_y.shape[0]):
            source_ey = (ey + int(offsets_y[offset_y_index])) % ny
            for iy in range(n_basis):
                derivative_weight_y = derivative_weights_y[
                    py,
                    offset_y_index,
                    iy,
                ]
                if derivative_weight_y == 0.0:
                    continue
                value_y += (
                    derivative_weight_y * primary_work[ex, source_ey, px, iy, component]
                )
        gradient_out[ex, ey, px, py, 1, component] = value_y


__all__ = [
    "evaluate_filtered_gradient_compiled",
    "evaluate_filtered_value_compiled",
    "evaluate_filtered_value_and_gradient_compiled",
    "evaluate_filtered_value_gradient_and_second_value_compiled",
    "evaluate_filtered_value_gradient_and_second_value_with_work_compiled",
]
