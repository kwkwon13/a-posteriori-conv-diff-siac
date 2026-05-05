"""Spatial SIAC filtering, filtered derivatives, and auxiliary derivatives."""

from .apply import (
    allocate_filtered_gradient,
    allocate_filtered_work,
    allocate_filtered_value,
    evaluate_filtered_gradient,
    evaluate_filtered_value,
    evaluate_filtered_value_and_gradient,
    evaluate_filtered_value_gradient_and_second_value,
    evaluate_filtered_value_gradient_and_second_value_with_work,
)
from .auxiliary import (
    AuxiliarySIACFilterEvaluator2D,
    allocate_tilde_u_ts_directional_work,
    allocate_tilde_u_ts_directional_derivative,
    allocate_tilde_u_ts_directional_second_derivative,
    build_auxiliary_siac_filter_evaluator_2d,
    evaluate_tilde_u_ts_directional_derivative,
    evaluate_tilde_u_ts_directional_derivatives,
    evaluate_tilde_u_ts_directional_derivatives_with_work,
    evaluate_tilde_u_ts_directional_second_derivative,
)
from .bspline import (
    central_bspline,
    central_bspline_derivative,
    central_bspline_second_derivative,
    evaluate_central_bspline,
    evaluate_central_bspline_derivative,
    evaluate_central_bspline_second_derivative,
)
from .evaluator import SIACFilterEvaluator2D, build_siac_filter_evaluator_2d
from .kernel import (
    SIACKernel2D,
    build_siac_kernel_2d,
    evaluate_siac_kernel_derivative_1d,
    evaluate_siac_kernel_1d,
    evaluate_siac_kernel_second_derivative_1d,
    siac_kernel_moment_1d,
)

__all__ = [
    "AuxiliarySIACFilterEvaluator2D",
    "SIACFilterEvaluator2D",
    "SIACKernel2D",
    "allocate_filtered_gradient",
    "allocate_filtered_work",
    "allocate_filtered_value",
    "allocate_tilde_u_ts_directional_work",
    "allocate_tilde_u_ts_directional_derivative",
    "allocate_tilde_u_ts_directional_second_derivative",
    "build_auxiliary_siac_filter_evaluator_2d",
    "build_siac_filter_evaluator_2d",
    "build_siac_kernel_2d",
    "central_bspline",
    "central_bspline_derivative",
    "central_bspline_second_derivative",
    "evaluate_central_bspline",
    "evaluate_central_bspline_derivative",
    "evaluate_central_bspline_second_derivative",
    "evaluate_filtered_gradient",
    "evaluate_filtered_value",
    "evaluate_filtered_value_and_gradient",
    "evaluate_filtered_value_gradient_and_second_value",
    "evaluate_filtered_value_gradient_and_second_value_with_work",
    "evaluate_siac_kernel_derivative_1d",
    "evaluate_siac_kernel_1d",
    "evaluate_siac_kernel_second_derivative_1d",
    "evaluate_tilde_u_ts_directional_derivative",
    "evaluate_tilde_u_ts_directional_derivatives",
    "evaluate_tilde_u_ts_directional_derivatives_with_work",
    "evaluate_tilde_u_ts_directional_second_derivative",
    "siac_kernel_moment_1d",
]
