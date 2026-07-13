"""
Flat-metric MIP block testing (OT-WoE extension; project note P3, Thm. B).

The decisive check is EXACTNESS: the trust constraint fm_tau accepts a
binning if and only if its true flat metric — recomputed independently from
the fitted bins with the certified level-set DP — meets the threshold.
"""

# Cesar Claros <cesar.claros@outlook.com>
# Copyright (C) 2026

import numpy as np

from pytest import raises

from optbinning import OptimalBinning
from optbinning.binning.metrics import flat_metric_1d
from optbinning.binning.metrics import jeffrey

LAM = 0.5


def _data(seed=0, n=4000):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    prob = 1 / (1 + np.exp(-(1.6 * x - 0.4)))
    y = (rng.uniform(0, 1, n) < prob).astype(int)
    return x, y


def _binned(x, y, splits):
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
    return ne[order] / ne.sum(), e[order] / e.sum(), w[order]


def _fm_of(x, y, splits, lam=LAM):
    p, q, w = _binned(x, y, splits)
    return flat_metric_1d(p, q, w, lam)


def _fit(**kwargs):
    x, y = _data()
    optb = OptimalBinning(name="x", dtype="numerical", solver="mip",
                          mip_solver="cbc", monotonic_trend="ascending",
                          **kwargs)
    optb.fit(x, y)
    return optb, x, y


def test_trust_constraint_exactness():
    # binding tau: the returned binning's independently recomputed flat
    # metric must satisfy the constraint (Theorem B exactness), at an IV
    # no greater than the unconstrained optimum.
    optb_iv, x, y = _fit(divergence="iv")
    f_iv = _fm_of(x, y, optb_iv.splits)

    optb_hi, _, _ = _fit(divergence="iv", fm_lambda=LAM, fm_mu=100.0)
    f_hi = _fm_of(x, y, optb_hi.splits)

    tau = 0.5 * (f_iv + f_hi) if f_hi > f_iv + 1e-6 else 0.9 * f_iv

    optb_tau, _, _ = _fit(divergence="iv", fm_lambda=LAM, fm_tau=tau)
    assert optb_tau.status in ("OPTIMAL", "FEASIBLE")
    assert _fm_of(x, y, optb_tau.splits) >= tau - 1e-7

    p_i, q_i, _ = _binned(x, y, optb_iv.splits)
    p_t, q_t, _ = _binned(x, y, optb_tau.splits)
    assert (jeffrey(p_t, q_t, return_sum=True)
            <= jeffrey(p_i, q_i, return_sum=True) + 1e-9)


def test_trust_constraint_inactive():
    # tau far below the unconstrained solution's flat metric: constraint
    # inactive, same splits as pure IV.
    optb_iv, x, y = _fit(divergence="iv")
    f_iv = _fm_of(x, y, optb_iv.splits)

    optb_lo, _, _ = _fit(divergence="iv", fm_lambda=LAM, fm_tau=0.1 * f_iv)
    assert np.allclose(optb_lo.splits, optb_iv.splits)


def test_trust_constraint_infeasible():
    # FM <= 2 * lambda for normalized masses: tau = 3 * lambda cannot hold.
    optb, _, _ = _fit(divergence="iv", fm_lambda=LAM, fm_tau=3.0 * LAM)
    assert optb.status == "INFEASIBLE"


def test_hybrid_dominance():
    mu = 2.0
    optb_iv, x, y = _fit(divergence="iv")
    optb_hy, _, _ = _fit(divergence="iv", fm_lambda=LAM, fm_mu=mu)

    def hybrid_value(splits):
        p, q, _ = _binned(x, y, splits)
        return (jeffrey(p, q, return_sum=True)
                + mu * _fm_of(x, y, splits))

    assert (hybrid_value(optb_hy.splits)
            >= hybrid_value(optb_iv.splits) - 1e-9)


def test_parameter_validation():
    x, y = _data()

    with raises(ValueError):
        optb = OptimalBinning(dtype="numerical", solver="mip",
                              mip_solver="cbc", fm_mu=1.0)
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalBinning(dtype="numerical", solver="mip",
                              mip_solver="bop", fm_lambda=LAM, fm_mu=1.0)
        optb.fit(x, y)

    with raises(ValueError):
        optb = OptimalBinning(dtype="numerical", solver="mip",
                              mip_solver="cbc", fm_lambda=-1.0, fm_mu=1.0)
        optb.fit(x, y)
