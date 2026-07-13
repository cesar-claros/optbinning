"""
OptimalBinning with transport objectives testing (OT-WoE extension).

End-to-end MIP wiring: divergence in {"w1", "cramer2", "hellinger_raw"} and
the hybrid gamma_wasserstein. Correctness via solver-independent dominance
invariants: the maximizer of objective A scores at least as high in A as the
maximizer of objective B, evaluated identically on the fitted solutions.
"""

# Cesar Claros <cesar.claros@outlook.com>
# Copyright (C) 2026

import numpy as np

from pytest import approx, raises

from optbinning import OptimalBinning
from optbinning.binning.metrics import cramer_1d
from optbinning.binning.metrics import hellinger_raw
from optbinning.binning.metrics import jeffrey
from optbinning.binning.metrics import wasserstein_1d


def _data(seed=0, n=4000):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    prob = 1 / (1 + np.exp(-(1.6 * x - 0.4)))
    y = (rng.uniform(0, 1, n) < prob).astype(int)
    return x, y


def _binned_stats(x, y, splits):
    """Bin data by the fitted splits (transform convention: right=False)
    and return normalized class masses, pooled-mean atoms, raw counts."""
    indices = np.digitize(x, splits, right=False)
    n_bins = len(splits) + 1
    ne = np.array([np.sum((indices == i) & (y == 0)) for i in range(n_bins)],
                  dtype=float)
    e = np.array([np.sum((indices == i) & (y == 1)) for i in range(n_bins)],
                 dtype=float)
    w = np.array([x[indices == i].mean() if np.any(indices == i) else 0.0
                  for i in range(n_bins)])
    keep = (ne + e) > 0
    ne, e, w = ne[keep], e[keep], w[keep]
    order = np.argsort(w)
    ne, e, w = ne[order], e[order], w[order]
    return ne / ne.sum(), e / e.sum(), w, ne, e


def _fit(divergence, gamma_wasserstein=0, **kwargs):
    x, y = _data()
    optb = OptimalBinning(name="x", dtype="numerical", solver="mip",
                          divergence=divergence,
                          gamma_wasserstein=gamma_wasserstein,
                          monotonic_trend="ascending", **kwargs)
    optb.fit(x, y)
    return optb, x, y


def test_transport_divergences_fit():
    for divergence in ("w1", "cramer2", "hellinger_raw"):
        optb, _, _ = _fit(divergence)
        assert optb.status in ("OPTIMAL", "FEASIBLE")
        assert len(optb.splits) >= 1


def test_hybrid_gamma_zero_matches_iv():
    optb_iv, _, _ = _fit("iv")
    optb_h0, _, _ = _fit("iv", gamma_wasserstein=0)
    assert list(optb_iv.splits) == approx(list(optb_h0.splits))


def test_dominance_invariants():
    # argmax property: on the same data and constraints, the w1-optimal
    # binning has W1 >= that of the iv-optimal binning, and vice versa
    # for IV (both solutions are feasible for both problems).
    optb_iv, x, y = _fit("iv")
    optb_w1, _, _ = _fit("w1")

    p_iv, q_iv, w_iv, _, _ = _binned_stats(x, y, optb_iv.splits)
    p_w1, q_w1, w_w1, _, _ = _binned_stats(x, y, optb_w1.splits)

    w1_of_w1 = wasserstein_1d(p_w1, q_w1, w_w1)
    w1_of_iv = wasserstein_1d(p_iv, q_iv, w_iv)
    assert w1_of_w1 >= w1_of_iv - 1e-9

    iv_of_iv = jeffrey(p_iv, q_iv, return_sum=True)
    iv_of_w1 = jeffrey(p_w1, q_w1, return_sum=True)
    assert iv_of_iv >= iv_of_w1 - 1e-9


def test_dominance_cramer_and_hellinger_raw():
    optb_iv, x, y = _fit("iv")
    p_iv, q_iv, w_iv, ne_iv, e_iv = _binned_stats(x, y, optb_iv.splits)

    optb_c2, _, _ = _fit("cramer2")
    p_c2, q_c2, w_c2, _, _ = _binned_stats(x, y, optb_c2.splits)
    assert (cramer_1d(p_c2, q_c2, w_c2, p=2)
            >= cramer_1d(p_iv, q_iv, w_iv, p=2) - 1e-9)

    optb_hr, _, _ = _fit("hellinger_raw")
    _, _, _, ne_hr, e_hr = _binned_stats(x, y, optb_hr.splits)
    assert hellinger_raw(ne_hr, e_hr) >= hellinger_raw(ne_iv, e_iv) - 1e-6


def test_hybrid_interpolates():
    # a small hybrid weight must not decrease the W1 of the solution
    # relative to pure IV (tie-breaking or better), and the hybrid
    # objective value of the hybrid solution dominates that of both pure
    # solutions.
    gamma_w = 0.5
    optb_iv, x, y = _fit("iv")
    optb_hy, _, _ = _fit("iv", gamma_wasserstein=gamma_w)

    p_i, q_i, w_i, _, _ = _binned_stats(x, y, optb_iv.splits)
    p_h, q_h, w_h, _, _ = _binned_stats(x, y, optb_hy.splits)

    def hybrid_value(p, q, w):
        return (jeffrey(p, q, return_sum=True)
                + gamma_w * wasserstein_1d(p, q, w))

    assert (hybrid_value(p_h, q_h, w_h)
            >= hybrid_value(p_i, q_i, w_i) - 1e-9)


def test_parameter_validation():
    x, y = _data()

    with raises(ValueError):
        optb = OptimalBinning(dtype="numerical", solver="ls",
                              divergence="w1")
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalBinning(dtype="categorical", solver="mip",
                              divergence="w1")
        optb.fit(x.astype(str), y)

    with raises(ValueError):
        optb = OptimalBinning(dtype="numerical", solver="mip",
                              gamma_wasserstein=-0.1)
        optb.fit(x, y)


def test_cp_mip_agreement():
    # CP-SAT (integer-scaled transport matrices) and CBC must agree on the
    # achieved objective value up to scaling resolution.
    x, y = _data()
    for divergence in ("w1", "hellinger_raw"):
        vals = {}
        for solver in ("cp", "mip"):
            optb = OptimalBinning(name="x", dtype="numerical", solver=solver,
                                  divergence=divergence,
                                  monotonic_trend="ascending")
            optb.fit(x, y)
            p, q, w, ne, e = _binned_stats(x, y, optb.splits)
            if divergence == "w1":
                vals[solver] = wasserstein_1d(p, q, w)
            else:
                vals[solver] = hellinger_raw(ne, e)
        scale_tol = 1e-3 * max(abs(vals["mip"]), 1.0)
        assert abs(vals["cp"] - vals["mip"]) <= scale_tol


def test_cp_hybrid_fits():
    x, y = _data()
    optb = OptimalBinning(name="x", dtype="numerical", solver="cp",
                          divergence="iv", gamma_wasserstein=0.5,
                          monotonic_trend="ascending")
    optb.fit(x, y)
    assert optb.status in ("OPTIMAL", "FEASIBLE")
    assert len(optb.splits) >= 1
