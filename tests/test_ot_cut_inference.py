"""
Cut-point inference testing (OT-WoE extension; Paper D).

Pinned constants from the Paper D verification log (normal design:
class 0 ~ N(0,1), class 1 ~ N(1, 1.3), rho = 1/2):
c* = 1.26598, V = 0.34867, sigma^2 = 15.86115, scale = 8.05117;
oracle-limit CI coverage 0.945 at n = 8000.
"""

# Cesar Claros <cesar.claros@outlook.com>
# Copyright (C) 2026

import numpy as np

from pytest import approx
from scipy.stats import norm

from optbinning.binning.cut_inference import chernoff_sample
from optbinning.binning.cut_inference import iv_cut_constants
from optbinning.binning.cut_inference import psi_iv
from optbinning.binning.cut_inference import subagged_cuts
from optbinning.binning.cut_inference import subsampling_ci

M0, S0, M1, S1, RHO = 0.0, 1.0, 1.0, 1.3, 0.5


def _F0(c):
    return norm.cdf((c - M0) / S0)


def _f0(c):
    return norm.pdf((c - M0) / S0) / S0


def _F1(c):
    return norm.cdf((c - M1) / S1)


def _f1(c):
    return norm.pdf((c - M1) / S1) / S1


def _c_star():
    cs = np.linspace(-1, 3, 2001)
    c = cs[np.argmax(psi_iv(_F0(cs), _F1(cs)))]
    for _ in range(50):
        h = 1e-6
        iv = [psi_iv(_F0(c + d), _F1(c + d)) for d in (-h, 0, h)]
        g = (iv[2] - iv[0]) / (2 * h)
        hess = (iv[2] - 2 * iv[1] + iv[0]) / h ** 2
        c -= g / hess
    return c


def test_pinned_limit_constants():
    c_star = _c_star()
    assert c_star == approx(1.26598, abs=1e-3)

    V, sigma2, scale = iv_cut_constants(c_star, _f0, _F0, _f1, _F1, RHO)
    assert V == approx(0.34867, rel=2e-3)
    assert sigma2 == approx(15.86115, rel=2e-3)
    assert scale == approx(8.05117, rel=2e-3)


def test_chernoff_sample():
    z = chernoff_sample(n_paths=3000, random_state=0)
    assert 0.45 < z.std() < 0.58        # known sd(Z) ~ 0.52
    assert abs(np.median(z)) < 0.05     # symmetric


def _fit_single_cut(x, y):
    """cheap single-cut IV argmax over a quantile grid. The candidate
    range is restricted to the [0.10, 0.90] quantiles: without a
    min-bin-mass guard the small-subsample argmax rides the boundary IV
    blow-up (the spike pathology of P1 Sec. 6.2 in miniature) -- the same
    reason the exact MIP carries Property-3 / min-bin-size constraints."""
    cands = np.quantile(x, np.linspace(0.10, 0.90, 80))
    x0 = np.sort(x[y == 0])
    x1 = np.sort(x[y == 1])
    a = np.clip(np.searchsorted(x0, cands) / len(x0), 1e-4, 1 - 1e-4)
    b = np.clip(np.searchsorted(x1, cands) / len(x1), 1e-4, 1 - 1e-4)
    return np.array([cands[np.argmax(psi_iv(a, b))]])


def test_subsampling_ci_and_subagging():
    rng = np.random.default_rng(3)
    n = 3000
    y = (rng.uniform(0, 1, n) < RHO).astype(int)
    x = np.where(y == 0, rng.normal(M0, S0, n), rng.normal(M1, S1, n))

    cuts, lo, hi = subsampling_ci(x, y, _fit_single_cut, n_subsamples=60,
                                  random_state=0)
    assert lo[0] < cuts[0] < hi[0]
    assert 0 < hi[0] - lo[0] < 1.5

    bag = subagged_cuts(x, y, _fit_single_cut, m=int(n ** 0.8),
                        n_subsamples=60, random_state=0)
    assert lo[0] - 0.3 < bag[0] < hi[0] + 0.3
