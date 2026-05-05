from __future__ import annotations

import numpy as np


def readonly(array: np.ndarray) -> np.ndarray:
    out = np.ascontiguousarray(array)
    out.setflags(write=False)
    return out
