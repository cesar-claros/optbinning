"""Reference utilities for Paper C: exact optima, grid summaries, and the
reduced-space polish shared by the torch and numpy recovery experiments."""

# Cesar Claros <cesar.claros@outlook.com>
# Copyright (C) 2026

from __future__ import annotations

from itertools import combinations

import numpy as np


def grid_summary(x: np.ndarray, y: np.ndarray,
                 n_prebins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantile pre-bin summary: representatives and per-class counts."""
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_prebins + 1)))
    inner = edges[1:-1]
    cell = np.searchsorted(inner, x)
    n_cells = len(inner) + 1
    ne = np.bincount(cell[y == 0], minlength=n_cells).astype(float)
    ev = np.bincount(cell[y == 1], minlength=n_cells).astype(float)
    total = np.bincount(cell, weights=x, minlength=n_cells)
    keep = (ne + ev) > 0
    reps = total[keep] / (ne + ev)[keep]
    return reps, ne[keep], ev[keep]


def iv_monotone(bounds: np.ndarray, ne: np.ndarray,
                ev: np.ndarray) -> float:
    """IV of a contiguous partition after monotone (PAV) pooling."""
    a = np.add.reduceat(ne, bounds[:-1])
    b = np.add.reduceat(ev, bounds[:-1])
    rate = b / np.maximum(a + b, 1e-12)
    order_blocks = _pav_blocks(rate, a + b)
    p = np.array([a[blk].sum() for blk in order_blocks]) / ne.sum()
    q = np.array([b[blk].sum() for blk in order_blocks]) / ev.sum()
    keep = (p > 0) & (q > 0)
    if not keep.any():
        return -np.inf
    return float(np.sum((p[keep] - q[keep]) * np.log(p[keep] / q[keep])))


def exact_monotone_optimum(ne: np.ndarray, ev: np.ndarray,
                           n_bins: int) -> float:
    """Exhaustive maximum of monotone-feasible IV over n_bins partitions."""
    n = len(ne)
    best = -np.inf
    for cuts in combinations(range(1, n), n_bins - 1):
        bounds = np.array((0,) + cuts + (n,))
        best = max(best, iv_monotone(bounds, ne, ev))
    return best


def polish(bounds: np.ndarray, ne: np.ndarray, ev: np.ndarray,
           n_bins: int) -> tuple[np.ndarray, float]:
    """Reduced-space exact refinement: boundary hill-climbing plus greedy
    cut insertion up to ``n_bins`` (the warm-start role of the layer)."""
    n = len(ne)
    bounds = bounds.copy()
    best = iv_monotone(bounds, ne, ev)
    improved = True
    while improved:
        improved = False
        for j in range(1, len(bounds) - 1):
            for step in (-1, 1):
                cand = bounds.copy()
                cand[j] += step
                if cand[j] <= cand[j - 1] or cand[j] >= cand[j + 1]:
                    continue
                value = iv_monotone(cand, ne, ev)
                if value > best + 1e-12:
                    bounds, best, improved = cand, value, True
    while len(bounds) - 1 < n_bins:
        options = [(np.sort(np.append(bounds, c)), c)
                   for c in range(1, n) if c not in bounds]
        if not options:
            break
        values = [iv_monotone(b, ne, ev) for b, _ in options]
        top = int(np.argmax(values))
        if values[top] <= best + 1e-12:
            break
        bounds, best = options[top][0], values[top]
    return bounds, best


def cuts_to_bounds(cuts: np.ndarray, reps: np.ndarray) -> np.ndarray:
    """Map continuous cut values to pre-bin boundary indices."""
    idx = np.searchsorted(reps, cuts)
    idx = np.unique(np.clip(idx, 1, len(reps) - 1))
    return np.concatenate(([0], idx, [len(reps)]))


def _pav_blocks(y: np.ndarray, w: np.ndarray) -> list[list[int]]:
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
