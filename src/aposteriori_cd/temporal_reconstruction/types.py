from __future__ import annotations

from enum import StrEnum


class TemporalReconstructionType(StrEnum):
    """Temporal reconstruction type in the notation H(p,d,r)."""

    H000 = "H(0,0,0)"
    H100 = "H(1,0,0)"


__all__ = ["TemporalReconstructionType"]
