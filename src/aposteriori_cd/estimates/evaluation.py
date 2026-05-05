"""Estimator evaluation over reconstructed time subintervals."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from math import sqrt

import numpy as np

from aposteriori_cd.discrete import Basis2D, Mesh2D
from aposteriori_cd.equations import DiffusivePSystem2D
from aposteriori_cd.siac_filtering import allocate_filtered_value, evaluate_filtered_value
from aposteriori_cd.temporal_reconstruction import (
    TemporalReconstructionSequence,
    TemporalReconstructionTimeSubinterval,
)
from aposteriori_cd.time import TimeSubintervalSolutionData

from .accumulation import (
    AnalysisGridQuadrature2D,
    ResidualNormAccumulator2D,
    build_analysis_grid_quadrature_2d,
)
from .bochner import (
    DEFAULT_TIME_QUADRATURE_ORDER,
    BochnerTimeQuadrature,
    build_bochner_time_quadrature,
)
from .estimators import (
    PSystemInitialTerms2D,
    scalar_initial_hat_u_ts_l2_error,
)
from .evaluation_compiled import (
    P_SYSTEM_MODEL_NONE,
    SCALAR_MODEL_BURGERS,
    SCALAR_MODEL_NONE,
    p_system_initial_terms,
    p_system_model_id_and_parameters,
    p_system_quadrature_node_terms,
    p_system_spatial_l2_errors,
    scalar_model_id_and_parameters,
    scalar_quadrature_node_terms,
    scalar_reconstructed_gradient_linf,
    scalar_spatial_l2_difference,
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
)


@dataclass(slots=True)
class EstimatorEvaluationTiming:
    """Timing counters for reconstruction and estimator evaluation."""

    linf_time_node_count: int = 0
    time_quadrature_node_count: int = 0
    linf_time_node_seconds: float = 0.0
    time_quadrature_node_seconds: float = 0.0
    reconstruction_seconds: float = 0.0
    residual_seconds: float = 0.0
    norm_accumulation_seconds: float = 0.0


@dataclass(slots=True)
class ScalarEstimatorEvaluation2D:
    """Accumulate scalar estimator quantities on reconstruction subintervals."""

    equation: object
    mesh: Mesh2D
    basis: Basis2D
    temporal_reconstruction_type: str
    reconstruction_evaluator: SpaceTimeReconstructionEvaluator2D
    reconstruction_values: SpaceTimeReconstructionValues2D
    residual_evaluator: ResidualEvaluator2D
    residual_values: ResidualValues2D
    analysis_quadrature: AnalysisGridQuadrature2D
    bochner_quadrature: BochnerTimeQuadrature
    temporal_quadrature: AnalysisGridQuadrature2D
    temporal_quadrature_coordinates: np.ndarray
    timing: EstimatorEvaluationTiming | None = None
    temporal_sequence: TemporalReconstructionSequence = field(init=False)
    residual_norms: ResidualNormAccumulator2D = field(init=False)
    _hat_u_h_t_l2_l2_error_squared: float = field(default=0.0, init=False)
    _hat_u_ts_linf_l2_error: float = field(default=0.0, init=False)
    _time_at_max_l2_error: float = field(default=0.0, init=False)
    _hat_u_ts_l2_h1_seminorm_error_squared: float = field(default=0.0, init=False)
    _max_abs_gradient: np.ndarray = field(init=False)
    _initial_value: np.ndarray = field(init=False)
    _compiled_model_id: int = field(init=False)
    _compiled_parameters: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.temporal_sequence = TemporalReconstructionSequence(
            self.temporal_reconstruction_type,
        )
        self.residual_norms = ResidualNormAccumulator2D()
        self._max_abs_gradient = np.zeros(
            (2, self.equation.num_components),
            dtype=np.float64,
        )
        self._initial_value = allocate_filtered_value(
            self.reconstruction_evaluator.classical_evaluator,
            self.equation.num_components,
        )
        self._compiled_model_id, self._compiled_parameters = (
            scalar_model_id_and_parameters(self.equation)
        )
        if self._compiled_model_id == SCALAR_MODEL_NONE:
            raise TypeError("scalar estimator evaluation supports paper scalar equations")

    def add_time_subinterval_data(
        self,
        _step_index: int,
        data: TimeSubintervalSolutionData,
    ) -> None:
        """Add one solver time subinterval."""

        for subinterval in self.temporal_sequence.add_time_subinterval_data(data):
            self.add_reconstructible_time_subinterval(subinterval)

    def add_reconstructible_time_subinterval(
        self,
        subinterval: TemporalReconstructionTimeSubinterval,
    ) -> None:
        """Accumulate all quantities for one reconstructible subinterval."""

        self._accumulate_time_nodes(subinterval)

    def initial_hat_u_ts_l2_error(self, initial_state: np.ndarray) -> float:
        """Compute the initial SIAC reconstruction error."""

        evaluate_filtered_value(
            self.reconstruction_evaluator.classical_evaluator,
            initial_state,
            self._initial_value,
        )
        return scalar_initial_hat_u_ts_l2_error(
            self.equation,
            self.analysis_quadrature,
            self.residual_evaluator.physical_coordinates,
            self._initial_value,
        )

    def hat_u_h_t_l2_l2_error(self) -> float:
        return sqrt(max(self._hat_u_h_t_l2_l2_error_squared, 0.0))

    def hat_u_ts_linf_l2_error(self) -> float:
        return float(self._hat_u_ts_linf_l2_error)

    def hat_u_ts_l2_h1_seminorm_error(self) -> float:
        return sqrt(max(self._hat_u_ts_l2_h1_seminorm_error_squared, 0.0))

    def residual_1_l1_l2(self) -> float:
        return float(self.residual_norms.residual_1_l1_l2)

    def E_r2(self) -> float:
        return self.residual_norms.E_r2

    def max_abs_gradient(self) -> np.ndarray:
        return np.asarray(self._max_abs_gradient, dtype=np.float64)

    def time_at_max_l2_error(self) -> float:
        return float(self._time_at_max_l2_error)

    def _accumulate_time_nodes(
        self,
        subinterval: TemporalReconstructionTimeSubinterval,
    ) -> None:
        for tau, time_value, time_weight in self.bochner_quadrature.quadrature_points(
            subinterval,
        ):
            reconstruction_start = time.perf_counter()
            evaluate_space_time_reconstruction(
                self.reconstruction_evaluator,
                subinterval,
                tau,
                self.reconstruction_values,
                value=True,
                time_derivative=True,
                gradient=True,
                auxiliary_derivatives=True,
            )
            _add_timing_seconds(
                self.timing,
                "reconstruction_seconds",
                reconstruction_start,
            )

            norm_start = time.perf_counter()
            norm = scalar_spatial_l2_difference(
                self.analysis_quadrature.spatial_weights,
                self.reconstruction_values.reconstructed_value,
                self.residual_evaluator.physical_coordinates,
                time_value,
                self._compiled_model_id,
                self._compiled_parameters,
            )
            if norm > self._hat_u_ts_linf_l2_error:
                self._hat_u_ts_linf_l2_error = norm
                self._time_at_max_l2_error = time_value
            if self._compiled_model_id == SCALAR_MODEL_BURGERS:
                self._max_abs_gradient = np.maximum(
                    self._max_abs_gradient,
                    scalar_reconstructed_gradient_linf(
                        self.reconstruction_values.reconstructed_gradient,
                    ),
                )
            linf_norm_seconds = time.perf_counter() - norm_start
            _add_timing_seconds(
                self.timing,
                "norm_accumulation_seconds",
                norm_start,
            )
            if self.timing is not None:
                self.timing.linf_time_node_count += 1
                self.timing.linf_time_node_seconds += linf_norm_seconds

            norm_start = time.perf_counter()
            (
                temporal_error_squared,
                gradient_error_squared,
                residual_1_l2,
                flux_difference_squared,
            ) = scalar_quadrature_node_terms(
                self.temporal_quadrature.spatial_weights,
                self.temporal_quadrature_coordinates,
                self.reconstruction_evaluator.temporal_value,
                self.basis.basis_at_quad,
                self.analysis_quadrature.spatial_weights,
                self.reconstruction_values.reconstructed_value,
                self.reconstruction_values.reconstructed_time_derivative,
                self.reconstruction_values.reconstructed_gradient,
                self.reconstruction_values.auxiliary_directional_derivative,
                self.reconstruction_values.auxiliary_directional_second_derivative,
                self.residual_evaluator.physical_coordinates,
                time_value,
                self._compiled_model_id,
                self._compiled_parameters,
            )
            self._hat_u_h_t_l2_l2_error_squared += (
                time_weight * temporal_error_squared
            )
            self._hat_u_ts_l2_h1_seminorm_error_squared += (
                time_weight * gradient_error_squared
            )
            self.residual_norms.residual_1_l1_l2 += time_weight * residual_1_l2
            self.residual_norms.E_r2_squared += (
                time_weight * flux_difference_squared
            )
            quadrature_norm_seconds = time.perf_counter() - norm_start
            _add_timing_seconds(
                self.timing,
                "norm_accumulation_seconds",
                norm_start,
            )
            if self.timing is not None:
                self.timing.time_quadrature_node_count += 1
                self.timing.time_quadrature_node_seconds += quadrature_norm_seconds


@dataclass(slots=True)
class PSystemEstimatorEvaluation2D:
    """Accumulate p-system estimator quantities on reconstruction subintervals."""

    equation: DiffusivePSystem2D
    mesh: Mesh2D
    basis: Basis2D
    temporal_reconstruction_type: str
    reconstruction_evaluator: SpaceTimeReconstructionEvaluator2D
    reconstruction_values: SpaceTimeReconstructionValues2D
    residual_evaluator: ResidualEvaluator2D
    residual_values: ResidualValues2D
    analysis_quadrature: AnalysisGridQuadrature2D
    bochner_quadrature: BochnerTimeQuadrature
    temporal_quadrature: AnalysisGridQuadrature2D
    temporal_quadrature_coordinates: np.ndarray
    timing: EstimatorEvaluationTiming | None = None
    temporal_sequence: TemporalReconstructionSequence = field(init=False)
    residual_norms: ResidualNormAccumulator2D = field(init=False)
    _hat_u_h_t_l2_l2_error_squared: float = field(default=0.0, init=False)
    _hat_tau_ts_linf_l2_error: float = field(default=0.0, init=False)
    _hat_v_ts_linf_l2_error: float = field(default=0.0, init=False)
    _time_at_max_tau_l2_error: float = field(default=0.0, init=False)
    _time_at_max_velocity_l2_error: float = field(default=0.0, init=False)
    _hat_v_ts_l2_h1_seminorm_error_squared: float = field(default=0.0, init=False)
    _W_second_r_u_l1_l2: float = field(default=0.0, init=False)
    _r_v_1_l1_l2: float = field(default=0.0, init=False)
    tau_min: float = field(default=float("inf"), init=False)
    tau_max: float = field(default=-float("inf"), init=False)
    div_hat_v_linf: float = field(default=0.0, init=False)
    _initial_value: np.ndarray = field(init=False)
    _compiled_model_id: int = field(init=False)
    _compiled_parameters: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.temporal_sequence = TemporalReconstructionSequence(
            self.temporal_reconstruction_type,
        )
        self.residual_norms = ResidualNormAccumulator2D()
        self._initial_value = allocate_filtered_value(
            self.reconstruction_evaluator.classical_evaluator,
            self.equation.num_components,
        )
        self._compiled_model_id, self._compiled_parameters = (
            p_system_model_id_and_parameters(self.equation)
        )
        if self._compiled_model_id == P_SYSTEM_MODEL_NONE:
            raise TypeError(
                "p-system estimator evaluation supports the paper p-system equation",
            )

    def add_time_subinterval_data(
        self,
        _step_index: int,
        data: TimeSubintervalSolutionData,
    ) -> None:
        """Add one solver time subinterval."""

        for subinterval in self.temporal_sequence.add_time_subinterval_data(data):
            self.add_reconstructible_time_subinterval(subinterval)

    def add_reconstructible_time_subinterval(
        self,
        subinterval: TemporalReconstructionTimeSubinterval,
    ) -> None:
        """Accumulate all quantities for one reconstructible subinterval."""

        self._accumulate_time_nodes(subinterval)

    def initial_terms(self, initial_state: np.ndarray) -> PSystemInitialTerms2D:
        """Compute initial p-system estimator terms."""

        evaluate_filtered_value(
            self.reconstruction_evaluator.classical_evaluator,
            initial_state,
            self._initial_value,
        )
        (
            initial_W_tau_l1,
            initial_v_l2_error_squared,
            tau_min,
            tau_max,
        ) = p_system_initial_terms(
            self.analysis_quadrature.spatial_weights,
            self._initial_value,
            self.residual_evaluator.physical_coordinates,
            self._compiled_model_id,
            self._compiled_parameters,
        )
        terms = PSystemInitialTerms2D(
            initial_W_tau_l1=float(initial_W_tau_l1),
            initial_v_l2_error_squared=float(initial_v_l2_error_squared),
            tau_min=float(tau_min),
            tau_max=float(tau_max),
        )
        self.tau_min = min(self.tau_min, terms.tau_min)
        self.tau_max = max(self.tau_max, terms.tau_max)
        return terms

    def hat_u_h_t_l2_l2_error(self) -> float:
        return sqrt(max(self._hat_u_h_t_l2_l2_error_squared, 0.0))

    def hat_tau_ts_linf_l2_error(self) -> float:
        return float(self._hat_tau_ts_linf_l2_error)

    def hat_v_ts_linf_l2_error(self) -> float:
        return float(self._hat_v_ts_linf_l2_error)

    def hat_v_ts_l2_h1_seminorm_error(self) -> float:
        return sqrt(max(self._hat_v_ts_l2_h1_seminorm_error_squared, 0.0))

    def residual_1_l1_l2(self) -> float:
        return float(self.residual_norms.residual_1_l1_l2)

    def W_second_r_u_l1_l2(self) -> float:
        return float(self._W_second_r_u_l1_l2)

    def r_v_1_l1_l2(self) -> float:
        return float(self._r_v_1_l1_l2)

    def E_r2(self) -> float:
        return self.residual_norms.E_r2

    def time_at_max_tau_l2_error(self) -> float:
        return float(self._time_at_max_tau_l2_error)

    def time_at_max_velocity_l2_error(self) -> float:
        return float(self._time_at_max_velocity_l2_error)

    def _accumulate_time_nodes(
        self,
        subinterval: TemporalReconstructionTimeSubinterval,
    ) -> None:
        for tau, time_value, time_weight in self.bochner_quadrature.quadrature_points(
            subinterval,
        ):
            reconstruction_start = time.perf_counter()
            evaluate_space_time_reconstruction(
                self.reconstruction_evaluator,
                subinterval,
                tau,
                self.reconstruction_values,
                value=True,
                time_derivative=True,
                gradient=True,
                auxiliary_derivatives=True,
            )
            _add_timing_seconds(
                self.timing,
                "reconstruction_seconds",
                reconstruction_start,
            )

            norm_start = time.perf_counter()
            tau_norm, velocity_norm = p_system_spatial_l2_errors(
                self.analysis_quadrature.spatial_weights,
                self.reconstruction_values.reconstructed_value,
                self.residual_evaluator.physical_coordinates,
                time_value,
                self._compiled_model_id,
                self._compiled_parameters,
            )
            if tau_norm > self._hat_tau_ts_linf_l2_error:
                self._hat_tau_ts_linf_l2_error = tau_norm
                self._time_at_max_tau_l2_error = time_value
            if velocity_norm > self._hat_v_ts_linf_l2_error:
                self._hat_v_ts_linf_l2_error = velocity_norm
                self._time_at_max_velocity_l2_error = time_value
            linf_norm_seconds = time.perf_counter() - norm_start
            _add_timing_seconds(
                self.timing,
                "norm_accumulation_seconds",
                norm_start,
            )
            if self.timing is not None:
                self.timing.linf_time_node_count += 1
                self.timing.linf_time_node_seconds += linf_norm_seconds

            norm_start = time.perf_counter()
            (
                temporal_error_squared,
                velocity_gradient_error_squared,
                residual_1_l2,
                flux_difference_squared,
                W_second_r_u_l2,
                r_v_1_l2,
                tau_min,
                tau_max,
                div_hat_v_linf,
            ) = p_system_quadrature_node_terms(
                self.temporal_quadrature.spatial_weights,
                self.temporal_quadrature_coordinates,
                self.reconstruction_evaluator.temporal_value,
                self.basis.basis_at_quad,
                self.analysis_quadrature.spatial_weights,
                self.reconstruction_values.reconstructed_value,
                self.reconstruction_values.reconstructed_time_derivative,
                self.reconstruction_values.reconstructed_gradient,
                self.reconstruction_values.auxiliary_directional_derivative,
                self.reconstruction_values.auxiliary_directional_second_derivative,
                self.residual_evaluator.physical_coordinates,
                time_value,
                self._compiled_model_id,
                self._compiled_parameters,
            )
            self._hat_u_h_t_l2_l2_error_squared += (
                time_weight * temporal_error_squared
            )
            self._hat_v_ts_l2_h1_seminorm_error_squared += (
                time_weight * velocity_gradient_error_squared
            )
            self.residual_norms.residual_1_l1_l2 += time_weight * residual_1_l2
            self.residual_norms.E_r2_squared += (
                time_weight * flux_difference_squared
            )
            self._W_second_r_u_l1_l2 += time_weight * W_second_r_u_l2
            self._r_v_1_l1_l2 += time_weight * r_v_1_l2
            self.tau_min = min(self.tau_min, float(tau_min))
            self.tau_max = max(self.tau_max, float(tau_max))
            self.div_hat_v_linf = max(
                self.div_hat_v_linf,
                float(div_hat_v_linf),
            )
            quadrature_norm_seconds = time.perf_counter() - norm_start
            _add_timing_seconds(
                self.timing,
                "norm_accumulation_seconds",
                norm_start,
            )
            if self.timing is not None:
                self.timing.time_quadrature_node_count += 1
                self.timing.time_quadrature_node_seconds += quadrature_norm_seconds


def build_scalar_estimator_evaluation_2d(
    equation: object,
    mesh: Mesh2D,
    basis: Basis2D,
    temporal_reconstruction_type: str,
    *,
    time_quadrature_order: int = DEFAULT_TIME_QUADRATURE_ORDER,
    timing: EstimatorEvaluationTiming | None = None,
) -> ScalarEstimatorEvaluation2D:
    """Build scalar estimator evaluation data."""

    common = _build_common_evaluation_data(
        equation,
        mesh,
        basis,
        temporal_reconstruction_type,
        time_quadrature_order=time_quadrature_order,
    )
    return ScalarEstimatorEvaluation2D(**common, timing=timing)


def build_p_system_estimator_evaluation_2d(
    equation: DiffusivePSystem2D,
    mesh: Mesh2D,
    basis: Basis2D,
    temporal_reconstruction_type: str,
    *,
    time_quadrature_order: int = DEFAULT_TIME_QUADRATURE_ORDER,
    timing: EstimatorEvaluationTiming | None = None,
) -> PSystemEstimatorEvaluation2D:
    """Build p-system estimator evaluation data."""

    common = _build_common_evaluation_data(
        equation,
        mesh,
        basis,
        temporal_reconstruction_type,
        time_quadrature_order=time_quadrature_order,
    )
    return PSystemEstimatorEvaluation2D(**common, timing=timing)


def _build_common_evaluation_data(
    equation: object,
    mesh: Mesh2D,
    basis: Basis2D,
    temporal_reconstruction_type: str,
    *,
    time_quadrature_order: int,
) -> dict[str, object]:
    reconstruction_evaluator = build_space_time_reconstruction_evaluator_2d(
        mesh,
        basis,
        equation.num_components,
    )
    reconstruction_values = allocate_space_time_reconstruction_values_2d(
        reconstruction_evaluator,
        equation.num_components,
    )
    analysis_nodes = reconstruction_evaluator.classical_evaluator.analysis_nodes_1d
    residual_evaluator = build_residual_evaluator_2d(equation, mesh, analysis_nodes)
    analysis_quadrature = build_analysis_grid_quadrature_2d(mesh, analysis_nodes)
    temporal_quadrature = build_analysis_grid_quadrature_2d(
        mesh,
        basis.quad_nodes_1d,
        basis.quad_weights_1d,
    )
    return {
        "equation": equation,
        "mesh": mesh,
        "basis": basis,
        "temporal_reconstruction_type": temporal_reconstruction_type,
        "reconstruction_evaluator": reconstruction_evaluator,
        "reconstruction_values": reconstruction_values,
        "residual_evaluator": residual_evaluator,
        "residual_values": allocate_residual_values_2d(residual_evaluator),
        "analysis_quadrature": analysis_quadrature,
        "bochner_quadrature": build_bochner_time_quadrature(time_quadrature_order),
        "temporal_quadrature": temporal_quadrature,
        "temporal_quadrature_coordinates": _physical_coordinates(
            mesh,
            basis.quad_nodes_1d,
        ),
    }


def _physical_coordinates(mesh: Mesh2D, nodes_1d: np.ndarray) -> np.ndarray:
    n_nodes = nodes_1d.shape[0]
    coordinates = np.empty((mesh.nx, mesh.ny, n_nodes, n_nodes, 2), dtype=np.float64)
    for ex in range(mesh.nx):
        for ey in range(mesh.ny):
            x_center = mesh.element_centers_x[ex, ey]
            y_center = mesh.element_centers_y[ex, ey]
            for px, xi in enumerate(nodes_1d):
                coordinates[ex, ey, px, :, 0] = x_center + 0.5 * mesh.dx * float(xi)
                for py, eta in enumerate(nodes_1d):
                    coordinates[ex, ey, px, py, 1] = (
                        y_center + 0.5 * mesh.dy * float(eta)
                    )
    return np.ascontiguousarray(coordinates, dtype=np.float64)


def _add_timing_seconds(
    timing: EstimatorEvaluationTiming | None,
    field_name: str,
    start_time: float,
) -> None:
    if timing is None:
        return
    setattr(
        timing,
        field_name,
        getattr(timing, field_name) + time.perf_counter() - start_time,
    )


__all__ = [
    "EstimatorEvaluationTiming",
    "PSystemEstimatorEvaluation2D",
    "ScalarEstimatorEvaluation2D",
    "build_p_system_estimator_evaluation_2d",
    "build_scalar_estimator_evaluation_2d",
]
