"""
Flat metric (bounded-Lipschitz distance) testing (OT-WoE extension).

Pinned regression tests porting the certified checks of project note P3
(Thm. A: TV-L1 structure; level-set DP; regime propositions) against LP
ground truth.
"""

# Cesar Claros <cesar.claros@outlook.com>
# Copyright (C) 2026

import numpy as np

from pytest import approx, raises
from scipy.optimize import linprog

from optbinning.binning.metrics import flat_metric_1d
from optbinning.binning.metrics import wasserstein_1d


def _fm_coupling_lp(atoms, a, b, lam):
    """Ground truth: LP over transport plan + destroy/create masses."""
    m = len(atoms)
    nv = m * m + 2 * m
    cost = np.concatenate([np.abs(atoms[:, None] - atoms[None, :]).ravel(),
                           lam * np.ones(2 * m)])
    a_eq = np.zeros((2 * m, nv))
    b_eq = np.concatenate([a, b])
    for k in range(m):
        a_eq[k, k * m:(k + 1) * m] = 1
        a_eq[k, m * m + k] = 1
    for c in range(m):
        a_eq[m + c, c:m * m:m] = 1
        a_eq[m + c, m * m + m + c] = 1
    res = linprog(cost, A_eq=a_eq, b_eq=b_eq, bounds=[(0, None)] * nv,
                  method="highs")
    return res.fun


def _fm_dual_lp(atoms, a, b, lam):
    """Bounded-Lipschitz dual: max <f, a-b>, |f| <= lam, |df| <= datoms."""
    m = len(atoms)
    a_ub = []
    b_ub = []
    for k in range(m - 1):
        row = np.zeros(m)
        row[k + 1], row[k] = 1, -1
        a_ub.append(row.copy())
        b_ub.append(atoms[k + 1] - atoms[k])
        a_ub.append(-row)
        b_ub.append(atoms[k + 1] - atoms[k])
    res = linprog(-(a - b), A_ub=np.array(a_ub), b_ub=np.array(b_ub),
                  bounds=[(-lam, lam)] * m, method="highs")
    return -res.fun


def test_dp_equals_coupling_lp():
    # P3 Thm. A + level-set property: the DP is exact.
    rng = np.random.default_rng(0)
    for _ in range(40):
        m = rng.integers(1, 9)
        atoms = np.sort(rng.uniform(0, 3, m))
        while m > 1 and np.min(np.diff(atoms)) < 1e-6:
            atoms = np.sort(rng.uniform(0, 3, m))
        a = rng.uniform(0, 3, m) * (rng.random(m) > 0.2)
        b = rng.uniform(0, 3, m) * (rng.random(m) > 0.2)
        lam = rng.uniform(0.05, 2.0)
        assert flat_metric_1d(a, b, atoms, lam) == approx(
            _fm_coupling_lp(atoms, a, b, lam), abs=1e-9)


def test_dp_equals_dual_lp():
    rng = np.random.default_rng(1)
    for _ in range(10):
        m = rng.integers(2, 8)
        atoms = np.sort(rng.uniform(0, 2, m))
        while np.min(np.diff(atoms)) < 1e-6:
            atoms = np.sort(rng.uniform(0, 2, m))
        a = rng.uniform(0.1, 3, m)
        b = rng.uniform(0.1, 3, m)
        lam = rng.uniform(0.05, 1.5)
        assert flat_metric_1d(a, b, atoms, lam) == approx(
            _fm_dual_lp(atoms, a, b, lam), abs=1e-9)


def test_two_dirac_closed_form():
    # FM(a delta_x, b delta_y) = min(d, 2 lam) min(a, b) + lam |a - b|.
    rng = np.random.default_rng(2)
    for _ in range(30):
        va, vb = rng.uniform(0.2, 3, 2)
        d = rng.uniform(0.01, 3)
        lam = rng.uniform(0.05, 1.5)
        val = flat_metric_1d(np.array([va, 0.]), np.array([0., vb]),
                             np.array([0., d]), lam)
        ref = min(d, 2 * lam) * min(va, vb) + lam * abs(va - vb)
        assert val == approx(ref, abs=1e-12)


def test_small_lambda_tv_regime():
    # P3 Prop. 3.2: all gaps > 2 lam => FM = lam * sum |a_k - b_k|.
    rng = np.random.default_rng(3)
    for _ in range(20):
        m = rng.integers(2, 7)
        atoms = np.cumsum(rng.uniform(1.0, 2.0, m))
        a = rng.uniform(0, 3, m)
        b = rng.uniform(0, 3, m)
        lam = rng.uniform(0.01, 0.49)  # 2 lam < 1 <= min gap
        assert flat_metric_1d(a, b, atoms, lam) == approx(
            lam * np.sum(np.abs(a - b)), abs=1e-12)


def test_large_lambda_w1_limit():
    # P3 Prop. 3.3: equal masses, lam >= diam / 2 => FM = W1.
    rng = np.random.default_rng(4)
    for _ in range(20):
        m = rng.integers(2, 8)
        atoms = np.sort(rng.uniform(0, 3, m))
        while np.min(np.diff(atoms)) < 1e-6:
            atoms = np.sort(rng.uniform(0, 3, m))
        a = rng.uniform(0.1, 3, m)
        b = rng.dirichlet(np.ones(m)) * a.sum()
        assert flat_metric_1d(a, b, atoms, 100.0) == approx(
            wasserstein_1d(a, b, atoms), abs=1e-9)


def test_cluster_additivity():
    # P3 Prop. 3.1: a gap wider than 2 lam splits the problem.
    rng = np.random.default_rng(5)
    for _ in range(15):
        lam = rng.uniform(0.1, 0.8)
        left = np.sort(rng.uniform(0, 0.5, 3))
        right = np.sort(rng.uniform(0, 0.5, 3)) + 0.5 + 2 * lam * 1.05
        atoms = np.concatenate([left, right])
        a = rng.uniform(0.1, 3, 6)
        b = rng.uniform(0.1, 3, 6)
        total = flat_metric_1d(a, b, atoms, lam)
        parts = (flat_metric_1d(a[:3], b[:3], left, lam)
                 + flat_metric_1d(a[3:], b[3:], right, lam))
        assert total == approx(parts, abs=1e-10)


def test_batch_consistency_and_validation():
    rng = np.random.default_rng(6)
    m = 7
    atoms = np.sort(rng.uniform(0, 2, m))
    A = rng.uniform(0, 2, (5, m))
    B = rng.uniform(0, 2, (5, m))
    batch = flat_metric_1d(A, B, atoms, 0.5)
    for i in range(5):
        assert batch[i] == approx(flat_metric_1d(A[i], B[i], atoms, 0.5),
                                  abs=1e-12)

    with raises(ValueError):
        flat_metric_1d(-A, B, atoms, 0.5, validate=True)

    with raises(ValueError):
        flat_metric_1d(A, B, atoms, -1.0, validate=True)
