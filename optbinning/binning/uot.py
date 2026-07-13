"""
Certified Hellinger-Kantorovich (unbalanced optimal transport) solver
(OT-WoE extension; project note P2 / Paper A, Sec. 7).

Every value is returned with a rigorous certificate: a feasible primal
plan gives an upper bound, a feasible dual potential pair a lower bound
(logarithmic entropy-transport duality); the reported value is the
midpoint and `gap` the bracket width. Structure exploited: additivity over
clusters separated by gaps >= kappa*pi/2 (P2, Thm. 2.3) with the closed
Hellinger form for singleton clusters (P2, Thm. 2.4). HK^2 is NOT
extent-additive (P2, Thm. 3.1 — certified rectangle-identity violation),
so no V_ij-style MILP encoding exists; use this solver inside local-search
loops or for kappa-continuation analyses.
"""

# Cesar Claros <cesar.claros@outlook.com>
# Copyright (C) 2026

import numpy as np

from scipy.optimize import LinearConstraint
from scipy.optimize import minimize

_PI2 = np.pi / 2


def hk_cost_matrix(wa, wb, kappa):
    """LET cost -log cos^2(d / kappa), +inf beyond the kappa*pi/2 horizon."""
    d = np.abs(np.asarray(wa)[:, None] - np.asarray(wb)[None, :]) / kappa
    dm = np.minimum(d, _PI2 * 0.9999999)
    return np.where(d < _PI2 * 0.9999999, -2 * np.log(np.cos(dm)), np.inf)


def hk_activation_margin(distance, kappa, odds_ratio):
    """Transport-activation criterion between two adjacent bins (P2,
    Prop. 2.5): transport activates iff cos^4(d/kappa) * OR > 1, with OR
    the (direction-consistent) between-bin odds ratio >= 1. Returns
    cos^4(d/kappa) * OR - 1 (positive = active)."""
    if distance >= kappa * _PI2:
        return -1.0
    return float(np.cos(distance / kappa) ** 4 * odds_ratio - 1.0)


def cluster_blocks(w, kappa):
    """Maximal blocks of consecutive atoms separated by gaps < kappa*pi/2
    (HK^2 is additive across blocks; P2, Thm. 2.3)."""
    w = np.asarray(w, dtype=float)
    if len(w) == 0:
        return []
    cutpoints = np.where(np.diff(w) >= kappa * _PI2)[0]
    bounds = np.concatenate(([0], cutpoints + 1, [len(w)]))
    return [np.arange(bounds[i], bounds[i + 1])
            for i in range(len(bounds) - 1)]


def _hk2_block(w, a, b, kappa):
    """Certified HK^2 on one cluster (finite cross-costs possible)."""
    ia = np.where(a > 0)[0]
    ib = np.where(b > 0)[0]
    if len(ia) == 0 and len(ib) == 0:
        return 0.0, 0.0
    if len(ia) == 0:
        return float(b.sum()), 0.0
    if len(ib) == 0:
        return float(a.sum()), 0.0

    wa, aa = w[ia], a[ia]
    wb, bb = w[ib], b[ib]
    C = hk_cost_matrix(wa, wb, kappa)
    fin = np.isfinite(C)
    idx = np.argwhere(fin)
    cv = C[fin]
    ka, kb = len(aa), len(bb)
    if len(idx) == 0:
        return float(aa.sum() + bb.sum()), 0.0

    # Primal upper bound: L-BFGS on log-parametrized plans.
    def obj_grad(th):
        gv = np.exp(th)
        g = np.zeros((ka, kb))
        g[fin] = gv
        row = g.sum(1)
        col = g.sum(0)
        f = float((gv * cv).sum())
        f += float(np.sum(np.where(
            row > 1e-300,
            row * np.log(np.maximum(row, 1e-300) / aa) - row + aa, aa)))
        f += float(np.sum(np.where(
            col > 1e-300,
            col * np.log(np.maximum(col, 1e-300) / bb) - col + bb, bb)))
        gr = (cv + np.log(np.maximum(row, 1e-300) / aa)[idx[:, 0]]
              + np.log(np.maximum(col, 1e-300) / bb)[idx[:, 1]])
        return f, gr * gv

    upper = np.inf
    for scale in (1.0, 0.2):
        g0 = np.sqrt(aa[idx[:, 0]] * bb[idx[:, 1]]) * np.exp(-cv / 2) * scale
        res = minimize(obj_grad, np.log(np.maximum(g0, 1e-14)), jac=True,
                       method="L-BFGS-B",
                       options=dict(maxiter=50000, maxfun=100000,
                                    ftol=1e-16, gtol=1e-14))
        upper = min(upper, res.fun)

    # Dual lower bound: max sum a (1 - e^-phi) + sum b (1 - e^-psi)
    # subject to phi_k + psi_l <= C_kl (linear constraints).
    A = np.zeros((len(idx), ka + kb))
    for r, (k, l) in enumerate(idx):
        A[r, k] = 1
        A[r, ka + l] = 1

    def neg_dual(z):
        ph, ps = z[:ka], z[ka:]
        return -(np.sum(aa * (1 - np.exp(-ph)))
                 + np.sum(bb * (1 - np.exp(-ps))))

    def neg_dual_jac(z):
        ph, ps = z[:ka], z[ka:]
        return np.concatenate((-aa * np.exp(-ph), -bb * np.exp(-ps)))

    res = minimize(neg_dual, np.zeros(ka + kb), jac=neg_dual_jac,
                   method="trust-constr",
                   constraints=[LinearConstraint(A, -np.inf, cv)],
                   bounds=[(-50, 50)] * (ka + kb),
                   options=dict(maxiter=600, gtol=1e-10, xtol=1e-13))
    ph, ps = res.x[:ka], res.x[ka:]
    slack = float(np.min(cv - A @ res.x))
    if slack < 0:
        ph = ph + slack  # uniform shift restores feasibility
    lower = float(np.sum(aa * (1 - np.exp(-ph)))
                  + np.sum(bb * (1 - np.exp(-ps))))

    return float((upper + lower) / 2), float(upper - lower)


def hk2(w, a, b, kappa=1.0):
    """Certified squared Hellinger-Kantorovich distance between the
    nonnegative atomic measures sum a_k delta_{w_k} and sum b_k delta_{w_k}
    on a shared ordered grid.

    Parameters
    ----------
    w : array-like, shape = (n_atoms,)
        Increasing atom positions.

    a, b : array-like, shape = (n_atoms,)
        Nonnegative masses (raw counts welcome; no normalization needed;
        zero-count cells cost at most their own mass).

    kappa : float, optional (default=1.0)
        Transport scale; kappa*pi/2 is the hard horizon. Below
        2 * min-gap / pi the value is exactly the raw-count squared
        Hellinger distance (P2, Thm. 2.4).

    Returns
    -------
    value : float

    gap : float
        Certified bracket width (primal-minus-dual); the true value lies
        within [value - gap/2, value + gap/2].
    """
    w = np.asarray(w, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    value = 0.0
    gap = 0.0
    for block in cluster_blocks(w, kappa):
        if len(block) == 1:
            k = block[0]
            value += (np.sqrt(a[k]) - np.sqrt(b[k])) ** 2
        else:
            v, g = _hk2_block(w[block], a[block], b[block], kappa)
            value += v
            gap += g
    return float(value), float(gap)
