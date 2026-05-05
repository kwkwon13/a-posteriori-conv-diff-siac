"""Reconstruction quantities, residuals, and estimators used by experiments."""

from .accumulation import (
    AnalysisGridQuadrature2D,
    ResidualNormAccumulator2D,
    accumulate_residual_norms,
    build_analysis_grid_quadrature_2d,
)
from .bochner import (
    DEFAULT_TIME_QUADRATURE_ORDER,
    BochnerTimeQuadrature,
    build_bochner_time_quadrature,
)
from .estimators import (
    PSystemInitialTerms2D,
    linear_scalar_a_posteriori_error_estimate,
    p_system_a_posteriori_error_estimate,
    p_system_lambda,
    p_system_reconstruction_error,
    scalar_initial_hat_u_ts_l2_error,
    scalar_reconstruction_error,
    viscous_burgers_a_posteriori_error_estimate,
    viscous_burgers_lambda,
)
from .evaluation import (
    EstimatorEvaluationTiming,
    PSystemEstimatorEvaluation2D,
    ScalarEstimatorEvaluation2D,
    build_p_system_estimator_evaluation_2d,
    build_scalar_estimator_evaluation_2d,
)
from .reconstructions import (
    SpaceTimeReconstructionEvaluator2D,
    SpaceTimeReconstructionValues2D,
    allocate_space_time_reconstruction_values_2d,
    build_space_time_reconstruction_evaluator_2d,
    evaluate_space_time_reconstruction,
)
from .residuals import (
    ResidualEvaluator2D,
    ResidualValues2D,
    allocate_residual_values_2d,
    build_residual_evaluator_2d,
    evaluate_residual_splitting,
)

__all__ = [
    "AnalysisGridQuadrature2D",
    "BochnerTimeQuadrature",
    "DEFAULT_TIME_QUADRATURE_ORDER",
    "EstimatorEvaluationTiming",
    "PSystemEstimatorEvaluation2D",
    "PSystemInitialTerms2D",
    "ResidualEvaluator2D",
    "ResidualNormAccumulator2D",
    "ResidualValues2D",
    "ScalarEstimatorEvaluation2D",
    "SpaceTimeReconstructionEvaluator2D",
    "SpaceTimeReconstructionValues2D",
    "accumulate_residual_norms",
    "allocate_residual_values_2d",
    "allocate_space_time_reconstruction_values_2d",
    "build_analysis_grid_quadrature_2d",
    "build_bochner_time_quadrature",
    "build_p_system_estimator_evaluation_2d",
    "build_residual_evaluator_2d",
    "build_scalar_estimator_evaluation_2d",
    "build_space_time_reconstruction_evaluator_2d",
    "evaluate_residual_splitting",
    "evaluate_space_time_reconstruction",
    "linear_scalar_a_posteriori_error_estimate",
    "p_system_a_posteriori_error_estimate",
    "p_system_lambda",
    "p_system_reconstruction_error",
    "scalar_initial_hat_u_ts_l2_error",
    "scalar_reconstruction_error",
    "viscous_burgers_a_posteriori_error_estimate",
    "viscous_burgers_lambda",
]
