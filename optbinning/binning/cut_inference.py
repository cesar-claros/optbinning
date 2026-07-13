"""
Sampling theory for optimal-binning cut points (OT-WoE extension).

Cut points of IV-optimal binnings are cube-root (Kim-Pollard) estimators
with a scaled Chernoff limit (Paper D): n^(1/3) (c_hat - c*) converges to
(2 sigma / V)^(2/3) * argmax_t {W(t) - t^2}. This module provides the limit
constants, simulated Chernoff quantiles, symmetric Politis-Romano
subsampling confidence intervals (the validated inference route; naive
bootstrap is inconsistent in this regime), and the subagged point
estimator.
"""

# Cesar Claros <cesar.claros@outlook.com>
# Copyright (C) 2026

import numpy as np


def psi_iv(a, b):
    """Two-bin IV as a function of the class CDF values (a, b) at the cut."""
    return ((a - b) * np.log(a / b)
            + (b - a) * np.log((1 - a) / (1 - b)))


def iv_cut_constants(c_star, f0, F0, f1, F1, rho, step=1e-4):
    """Limit-law constants for the single-cut IV maximizer (Paper D,
    Thm. 3.1).

    Parameters
    ----------
    c_star : float
        Population maximizer of the two-bin IV.

    f0, F0, f1, F1 : callables
        Class-conditional densities and CDFs.

    rho : float
        Event proportion, n_event / n -> rho in (0, 1).

    step : float, optional (default=1e-4)
        Central-difference step for the curvature.

    Returns
    -------
    V : float
        Curvature -IV''(c*).

    sigma2 : float
        Local diffusion constant
        Psi_a^2 f0(c*) / (1 - rho) + Psi_b^2 f1(c*) / rho.

    scale : float
        (2 sqrt(sigma2) / V)^(2/3); the limit is scale * Chernoff.
    """
    def iv(c):
        return psi_iv(F0(c), F1(c))

    V = -(iv(c_star + step) - 2 * iv(c_star) + iv(c_star - step)) / step ** 2

    a, b = F0(c_star), F1(c_star)
    L = np.log(a / b) - np.log((1 - a) / (1 - b))
    psi_a = L + (a - b) / (a * (1 - a))
    psi_b = -L - (a - b) / (b * (1 - b))

    sigma2 = (psi_a ** 2 * f0(c_star) / (1 - rho)
              + psi_b ** 2 * f1(c_star) / rho)

    scale = (2 * np.sqrt(sigma2) / V) ** (2. / 3.)
    return float(V), float(sigma2), float(scale)


def chernoff_sample(n_paths=20000, t_max=2.5, step=0.002, random_state=None):
    """Simulate Chernoff's distribution Z = argmax_t {W(t) - t^2} by
    discretized two-sided Brownian paths. Known sd(Z) ~ 0.52.

    Returns
    -------
    z : numpy.ndarray, shape = (n_paths,)
    """
    rng = np.random.default_rng(random_state)
    tg = np.arange(-t_max, t_max + step, step)
    k = len(tg)
    mid = k // 2
    out = []
    chunk = max(1, min(n_paths, int(5e7 // k)))
    done = 0
    while done < n_paths:
        p = min(chunk, n_paths - done)
        w = np.zeros((p, k))
        inc = rng.normal(0, np.sqrt(step), (p, k - 1))
        w[:, mid + 1:] = np.cumsum(inc[:, mid:], axis=1)
        w[:, :mid] = -np.cumsum(inc[:, :mid][:, ::-1], axis=1)[:, ::-1]
        out.append(tg[np.argmax(w - tg[None, :] ** 2, axis=1)])
        done += p
    return np.concatenate(out)


def _sorted_cuts(fit_cuts, x, y):
    s = np.asarray(fit_cuts(x, y), dtype=float)
    return np.sort(s)


def subsampling_ci(x, y, fit_cuts, m=None, n_subsamples=200, alpha=0.05,
                   random_state=None):
    """Symmetric Politis-Romano subsampling confidence intervals for cut
    points, at the cube-root rate (Paper D, Sec. 5: the symmetric variant
    is essential; the naive n-out-of-n bootstrap is inconsistent here).

    Parameters
    ----------
    x, y : array-like
        Data.

    fit_cuts : callable
        ``fit_cuts(x, y) -> array of cut points`` (the estimator).

    m : int or None, optional (default=None)
        Subsample size; default round(n^0.7).

    n_subsamples : int, optional (default=200)

    alpha : float, optional (default=0.05)

    random_state : int or None

    Returns
    -------
    cuts : numpy.ndarray
        Full-sample cuts.

    lower, upper : numpy.ndarray
        Per-cut confidence bounds. Subsamples returning a different number
        of cuts than the full sample are skipped (their frequency is
        itself an instability diagnostic; see ConsensusBinning).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    if m is None:
        m = int(round(n ** 0.7))
    rng = np.random.default_rng(random_state)

    cuts = _sorted_cuts(fit_cuts, x, y)
    k = len(cuts)

    deviations = []
    for _ in range(n_subsamples):
        idx = rng.choice(n, size=m, replace=False)
        cs = _sorted_cuts(fit_cuts, x[idx], y[idx])
        if len(cs) == k:
            deviations.append(m ** (1. / 3.) * np.abs(cs - cuts))

    if not deviations:
        raise RuntimeError("No subsample matched the full-sample number of "
                           "cuts; increase m or fix the bin count.")

    q = np.quantile(np.vstack(deviations), 1 - alpha, axis=0)
    half = q / n ** (1. / 3.)
    return cuts, cuts - half, cuts + half


def subagged_cuts(x, y, fit_cuts, m=None, n_subsamples=200,
                  random_state=None):
    """Subagged (subsample-aggregated) cut estimator: the barycentric mean
    of sorted subsample cuts. Typically 2-3x more accurate than the raw
    argmax in the cube-root regime (Paper D, Sec. 5(iii); P7, Sec. 3).

    Returns
    -------
    cuts_subagged : numpy.ndarray
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    if m is None:
        m = int(round(n ** 0.75))
    rng = np.random.default_rng(random_state)

    cuts = _sorted_cuts(fit_cuts, x, y)
    k = len(cuts)

    coll = []
    for _ in range(n_subsamples):
        idx = rng.choice(n, size=m, replace=False)
        cs = _sorted_cuts(fit_cuts, x[idx], y[idx])
        if len(cs) == k:
            coll.append(cs)

    if not coll:
        raise RuntimeError("No subsample matched the full-sample number of "
                           "cuts.")

    return np.vstack(coll).mean(axis=0)
