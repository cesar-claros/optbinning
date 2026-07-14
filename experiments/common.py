"""
Shared harness utilities: arm construction, binning evaluation, bootstrap
stability, result IO (E0.3).
"""

# Cesar Claros <cesar.claros@outlook.com>
# Copyright (C) 2026

from pathlib import Path

import numpy as np
import pandas as pd

from optbinning import OptimalBinning
from optbinning.binning.metrics import jeffrey
from optbinning.binning.metrics import wasserstein_1d


def make_arm(arm, lam=None, gamma=None, fm_tau=None, monotonic="auto",
             max_n_prebins=20, max_n_bins=None, time_limit=30):
    """OptimalBinning configured for a benchmark arm.

    ``iv`` uses the CP solver (upstream default); ``iv_mip`` is the same IV
    objective under the MIP/CBC solver, a control that isolates objective
    effects from solver tie-breaking when compared against the transport arms.
    ``max_n_bins`` (when set) caps the number of bins, which is what forces
    the fine structure of refinement-monotone objectives (iv, hellinger_raw)
    to diverge; without it they saturate to the same finest feasible binning.
    """
    base = dict(dtype="numerical", monotonic_trend=monotonic,
                max_n_prebins=max_n_prebins, time_limit=time_limit)
    if max_n_bins is not None:
        base["max_n_bins"] = max_n_bins
    if arm == "iv":
        return OptimalBinning(solver="cp", divergence="iv", **base)
    if arm == "iv_mip":
        return OptimalBinning(solver="mip", mip_solver="cbc",
                              divergence="iv", **base)
    if arm == "iv_w1":
        return OptimalBinning(solver="mip", mip_solver="cbc",
                              divergence="iv", gamma_wasserstein=gamma,
                              **base)
    if arm == "fm_tau":
        return OptimalBinning(solver="mip", mip_solver="cbc",
                              divergence="iv", fm_lambda=lam, fm_tau=fm_tau,
                              **base)
    if arm == "w1":
        return OptimalBinning(solver="mip", mip_solver="cbc",
                              divergence="w1", **base)
    if arm == "cramer2":
        return OptimalBinning(solver="mip", mip_solver="cbc",
                              divergence="cramer2", **base)
    if arm == "hellinger_raw":
        return OptimalBinning(solver="mip", mip_solver="cbc",
                              divergence="hellinger_raw", **base)
    raise ValueError("Unknown arm: {}".format(arm))


def binned_stats(x, y, splits):
    indices = np.digitize(x, splits, right=False)
    n_bins = len(splits) + 1
    ne = np.array([np.sum((indices == i) & (y == 0))
                   for i in range(n_bins)], dtype=float)
    e = np.array([np.sum((indices == i) & (y == 1))
                  for i in range(n_bins)], dtype=float)
    w = np.array([x[indices == i].mean() if np.any(indices == i) else np.nan
                  for i in range(n_bins)])
    keep = (ne + e) > 0
    ne, e, w = ne[keep], e[keep], w[keep]
    order = np.argsort(w)
    return ne[order], e[order], w[order]


def eval_binning(splits, x_test, y_test, lam=None):
    """Out-of-sample metrics of a fixed binning."""
    ne, e, w = binned_stats(x_test, y_test, splits)
    p = ne / max(ne.sum(), 1.0)
    q = e / max(e.sum(), 1.0)
    ok = (p > 0) & (q > 0)
    oos_iv = float(jeffrey(p[ok], q[ok], return_sum=True)) if ok.any() \
        else 0.0
    oos_w1 = float(wasserstein_1d(p, q, w)) if len(w) > 1 else 0.0
    woe = np.zeros(len(p))
    woe[ok] = np.log(p[ok] / q[ok])
    spike = int(np.sum((np.minimum(ne, e) <= 5) & (np.abs(woe) > 2)))
    return dict(oos_iv=oos_iv, oos_w1=oos_w1, n_bins=len(p),
                spike_bins=spike)


def bootstrap_cut_sd(fit_factory, x, y, n_boot=10, seed=0):
    """Mean per-cut sd across bootstrap refits (sorted-matched; refits
    with a different cut count are skipped and counted)."""
    rng = np.random.default_rng(seed)
    base = fit_factory().fit(x, y)
    k = len(base.splits)
    if k == 0:
        return np.nan, 1.0
    coll = []
    mismatched = 0
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        try:
            s = fit_factory().fit(x[idx], y[idx]).splits
        except Exception:
            mismatched += 1
            continue
        if len(s) == k:
            coll.append(np.sort(np.asarray(s, dtype=float)))
        else:
            mismatched += 1
    if len(coll) < 2:
        return np.nan, mismatched / n_boot
    arr = np.vstack(coll)
    return float(arr.std(axis=0).mean()), mismatched / n_boot


def save_results(rows, out_path):
    """Append-style results writer; parquet with csv fallback."""
    df = pd.DataFrame(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(out_path.with_suffix(".parquet"), index=False)
        return out_path.with_suffix(".parquet")
    except Exception:
        df.to_csv(out_path.with_suffix(".csv"), index=False)
        return out_path.with_suffix(".csv")


def load_results(path):
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)
