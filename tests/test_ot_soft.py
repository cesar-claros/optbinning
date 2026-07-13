"""
Soft OT-binning layer testing (OT-WoE extension; project note P6).
"""

# Cesar Claros <cesar.claros@outlook.com>
# Copyright (C) 2026

import numpy as np

from itertools import combinations

from optbinning.binning.soft import SoftBinning, _pav_isotonic


def _gen_data(seed, n=30, n0=2000, n1=2000):
    rng = np.random.default_rng(seed)
    u = np.sort(rng.uniform(0, 1, n))
    dens = rng.dirichlet(np.ones(n) * 3)
    rate = 1 / (1 + np.exp(-(4 * (u - 0.5) + rng.normal(0, 0.35, n))))
    ne = np.maximum(1, np.round(dens * (1 - rate)
                                / (dens * (1 - rate)).sum() * n0))
    e = np.maximum(1, np.round(dens * rate / (dens * rate).sum() * n1))
    return u, ne, e


def _exact_mono_optimum(ne, e, m):
    """Exhaustive maximum of monotone-feasible IV over m-bin partitions."""
    n = len(ne)
    cp = np.concatenate([[0], np.cumsum(ne / ne.sum())])
    cq = np.concatenate([[0], np.cumsum(e / e.sum())])
    cn = np.concatenate([[0], np.cumsum(ne)])
    ce = np.concatenate([[0], np.cumsum(e)])
    best = -np.inf
    for cuts in combinations(range(1, n), m - 1):
        bd = np.array((0,) + cuts + (n,))
        p = cp[bd[1:]] - cp[bd[:-1]]
        q = cq[bd[1:]] - cq[bd[:-1]]
        if np.any(p <= 0) or np.any(q <= 0):
            continue
        ev = ce[bd[1:]] - ce[bd[:-1]]
        nv = cn[bd[1:]] - cn[bd[:-1]]
        rate = ev / (ev + nv)
        if np.any(np.diff(rate) < -1e-12):
            continue
        best = max(best, float(np.sum((p - q) * np.log(p / q))))
    return best


def test_contiguity_and_recovery_gap():
    # P6 Thm. 3.1: hardened assignments are contiguous; with the
    # reduced-space polish the recovery gap to the exact monotone optimum
    # is small (E1 experiment: mean 1.6%, max ~8%).
    for seed in (0, 1):
        u, ne, e = _gen_data(seed)
        sb = SoftBinning(n_bins=5, n_steps=200, n_restarts=3,
                         random_state=seed).fit(u, ne, e)
        assert sb.contiguous_
        assert np.all(np.diff(sb.bounds_) >= 1)

        exact = _exact_mono_optimum(ne, e, 5)
        gap = (exact - sb.iv_) / exact
        assert gap <= 0.10


def test_polish_never_hurts():
    u, ne, e = _gen_data(3)
    raw = SoftBinning(n_bins=5, n_steps=120, n_restarts=1, polish=False,
                      random_state=0).fit(u, ne, e)
    pol = SoftBinning(n_bins=5, n_steps=120, n_restarts=1, polish=True,
                      random_state=0).fit(u, ne, e)
    assert pol.iv_ >= raw.iv_ - 1e-12


def test_pav_fixed_points():
    # PAV is the identity exactly on monotone vectors (Thm. 3.1(iv)).
    y = np.array([0.1, 0.2, 0.35, 0.5])
    w = np.ones(4)
    out, blocks = _pav_isotonic(y, w)
    assert np.allclose(out, y)
    assert len(blocks) == 4

    y2 = np.array([0.3, 0.1, 0.2])
    out2, blocks2 = _pav_isotonic(y2, w[:3])
    assert np.all(np.diff(out2) >= -1e-15)
    assert len(blocks2) < 3


def test_validation():
    from pytest import raises
    with raises(ValueError):
        SoftBinning(n_bins=1)
