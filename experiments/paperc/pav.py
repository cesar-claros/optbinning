"""Pool-adjacent-violators block search (torch-free).

Shared by the torch OT layer (PAV penalty Jacobian blocks) and the
numpy-only calibration harness; kept import-light so torch-free
pipelines can use it.
"""

# Cesar Claros <cesar.claros@outlook.com>
# Copyright (C) 2026

from __future__ import annotations

import numpy as np


def _pav_blocks(y: np.ndarray, w: np.ndarray) -> list[list[int]]:
    """Pooled blocks of the weighted increasing isotonic regression."""
    vals: list[float] = []
    wts: list[float] = []
    idx: list[list[int]] = []
    for i, (yy, ww) in enumerate(zip(y, w)):
        vals.append(float(yy))
        wts.append(float(ww))
        idx.append([i])
        while len(vals) > 1 and vals[-2] > vals[-1] + 1e-15:
            merged = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / (
                wts[-2] + wts[-1])
            wts[-2] += wts[-1]
            vals[-2] = merged
            idx[-2] += idx[-1]
            vals.pop()
            wts.pop()
            idx.pop()
    return idx
