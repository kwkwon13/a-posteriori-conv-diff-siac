from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from aposteriori_cd.time import TimeSubintervalSolutionData

from .hermite import (
    evaluate_cubic_hermite,
    evaluate_cubic_hermite_time_derivative,
    evaluate_quintic_hermite,
    evaluate_quintic_hermite_startup,
    evaluate_quintic_hermite_startup_time_derivative,
    evaluate_quintic_hermite_time_derivative,
)
from .types import TemporalReconstructionType


@dataclass(frozen=True, slots=True)
class TemporalReconstructionTimeSubinterval:
    """A time subinterval on which the temporal reconstruction is defined."""

    temporal_reconstruction_type: TemporalReconstructionType
    current_data: TimeSubintervalSolutionData
    previous_data: TimeSubintervalSolutionData | None = None
    next_data: TimeSubintervalSolutionData | None = None
    is_startup: bool = False

    def __post_init__(self) -> None:
        temporal_reconstruction_type = TemporalReconstructionType(
            self.temporal_reconstruction_type,
        )
        object.__setattr__(
            self,
            "temporal_reconstruction_type",
            temporal_reconstruction_type,
        )
        if temporal_reconstruction_type is TemporalReconstructionType.H000:
            return
        if self.is_startup:
            if self.next_data is None:
                raise ValueError("next_data is required for H(1,0,0) startup")
            return
        if self.previous_data is None:
            raise ValueError("previous_data is required for H(1,0,0)")

    @property
    def t_left(self) -> float:
        return float(self.current_data.t_left)

    @property
    def t_right(self) -> float:
        return float(self.current_data.t_right)

    @property
    def length(self) -> float:
        length = self.t_right - self.t_left
        if length <= 0.0:
            raise ValueError("time subinterval length must be positive")
        return float(length)

    def time_at(self, tau: float) -> float:
        return float(self.t_left + float(tau) * self.length)

    def evaluate(self, tau: float, out: np.ndarray) -> None:
        """Evaluate the temporal reconstruction on this time subinterval."""

        if self.temporal_reconstruction_type is TemporalReconstructionType.H000:
            evaluate_cubic_hermite(tau, self.current_data, out)
            return
        if self.is_startup:
            assert self.next_data is not None
            evaluate_quintic_hermite_startup(
                tau,
                self.current_data,
                self.next_data,
                out,
            )
            return
        assert self.previous_data is not None
        evaluate_quintic_hermite(tau, self.previous_data, self.current_data, out)

    def evaluate_time_derivative(self, tau: float, out: np.ndarray) -> None:
        """Evaluate the time derivative on this time subinterval."""

        if self.temporal_reconstruction_type is TemporalReconstructionType.H000:
            evaluate_cubic_hermite_time_derivative(tau, self.current_data, out)
            return
        if self.is_startup:
            assert self.next_data is not None
            evaluate_quintic_hermite_startup_time_derivative(
                tau,
                self.current_data,
                self.next_data,
                out,
            )
            return
        assert self.previous_data is not None
        evaluate_quintic_hermite_time_derivative(
            tau,
            self.previous_data,
            self.current_data,
            out,
        )


@dataclass(slots=True)
class TemporalReconstructionSequence:
    """Collect time subintervals when the chosen reconstruction becomes defined."""

    temporal_reconstruction_type: TemporalReconstructionType
    _previous_data: TimeSubintervalSolutionData | None = field(
        default=None,
        init=False,
    )
    _startup_done: bool = field(default=False, init=False)
    _received_time_subinterval_count: int = field(default=0, init=False)
    _reconstructible_time_subinterval_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.temporal_reconstruction_type = TemporalReconstructionType(
            self.temporal_reconstruction_type,
        )

    def add_time_subinterval_data(
        self,
        data: TimeSubintervalSolutionData,
    ) -> tuple[TemporalReconstructionTimeSubinterval, ...]:
        """Add one time-subinterval data object and return defined subintervals."""

        current_data = _copy_time_subinterval_solution_data(data)
        self._received_time_subinterval_count += 1

        if self.temporal_reconstruction_type is TemporalReconstructionType.H000:
            return self._mark_reconstructible(
                TemporalReconstructionTimeSubinterval(
                    temporal_reconstruction_type=self.temporal_reconstruction_type,
                    current_data=current_data,
                ),
            )

        previous_data = self._previous_data
        self._previous_data = current_data
        if previous_data is None:
            return ()

        if not self._startup_done:
            self._startup_done = True
            return self._mark_reconstructible(
                TemporalReconstructionTimeSubinterval(
                    temporal_reconstruction_type=self.temporal_reconstruction_type,
                    current_data=previous_data,
                    next_data=current_data,
                    is_startup=True,
                ),
                TemporalReconstructionTimeSubinterval(
                    temporal_reconstruction_type=self.temporal_reconstruction_type,
                    previous_data=previous_data,
                    current_data=current_data,
                ),
            )

        return self._mark_reconstructible(
            TemporalReconstructionTimeSubinterval(
                temporal_reconstruction_type=self.temporal_reconstruction_type,
                previous_data=previous_data,
                current_data=current_data,
            ),
        )

    @property
    def received_time_subinterval_count(self) -> int:
        return self._received_time_subinterval_count

    @property
    def reconstructible_time_subinterval_count(self) -> int:
        return self._reconstructible_time_subinterval_count

    def _mark_reconstructible(
        self,
        *subintervals: TemporalReconstructionTimeSubinterval,
    ) -> tuple[TemporalReconstructionTimeSubinterval, ...]:
        self._reconstructible_time_subinterval_count += len(subintervals)
        return subintervals


def _copy_time_subinterval_solution_data(
    data: TimeSubintervalSolutionData,
) -> TimeSubintervalSolutionData:
    return TimeSubintervalSolutionData(
        t_left=float(data.t_left),
        t_right=float(data.t_right),
        values=np.ascontiguousarray(data.values.copy(), dtype=np.float64),
        time_derivatives=np.ascontiguousarray(
            data.time_derivatives.copy(),
            dtype=np.float64,
        ),
    )


__all__ = [
    "TemporalReconstructionSequence",
    "TemporalReconstructionTimeSubinterval",
]
