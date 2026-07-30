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


def prepare_features(ds, special_handling="expand", return_names=False):
    """Numerical feature matrix with sentinel-code handling.

    special_handling:
      "ignore" -- sentinels flow through as numeric values (the harness
        behavior of all pre-HELOC campaigns). WRONG for datasets like HELOC
        whose -7/-8/-9 codes have no ordinal relation to the scale; kept only
        for reproducing those runs.
      "expand" (default) -- sentinel values are removed from the numeric scale
        (median-imputed) and per-(feature, code) indicator columns are
        appended, so sentinel-ness is explicit binary information and every arm
        sees identical features. Constant indicators dropped. A no-op on
        datasets without ``special_codes``.

    With ``return_names=True`` also returns the column names (numeric feature
    names followed by ``"feat==code"`` indicator names), so callers can label
    the expanded columns.
    """
    x = ds.X[ds.numerical].to_numpy(dtype=float)
    names = list(ds.numerical)
    extra = []
    if special_handling == "expand" and ds.special_codes:
        codes = list(ds.special_codes)
        for j in range(x.shape[1]):
            for c in codes:
                mask = x[:, j] == c
                if 0 < mask.sum() < len(x):
                    extra.append(mask.astype(float))
                    names.append("{}=={}".format(ds.numerical[j], c))
            x[np.isin(x[:, j], codes), j] = np.nan
    elif special_handling not in ("ignore", "expand"):
        raise ValueError(
            "special_handling must be 'ignore' or 'expand'; got "
            "{}.".format(special_handling))
    med = np.nanmedian(x, axis=0)
    x = np.where(np.isfinite(x), x, med)
    if extra:
        x = np.column_stack([x] + extra)
    return (x, names) if return_names else x


def expanded_features(ds, base=None, special_handling="expand"):
    """Binnable column names for the per-feature drivers.

    Returns the numeric feature names and, with ``special_handling='expand'``
    (default) on a dataset that declares ``special_codes``, one extra
    ``"feat==code"`` name per occurring (feature, sentinel) pair, so sentinel-
    ness is binned and reported as its own indicator while the numeric column
    (see :func:`feature_array`) is cleaned. ``'ignore'`` returns the base names
    unchanged. Pairs with the array loader ``feature_array``.
    """
    if special_handling not in ("expand", "ignore"):
        raise ValueError("special_handling must be 'expand' or 'ignore'; got "
                         "{}.".format(special_handling))
    base = list(base) if base is not None else list(ds.numerical)
    codes = list(ds.special_codes) if special_handling == "expand" else []
    names = []
    for feat in base:
        names.append(feat)
        for c in codes:
            m = ds.X[feat].to_numpy(dtype=float) == c
            if 0 < int(m.sum()) < len(m):
                names.append("{}=={}".format(feat, c))
    return names


def feature_array(ds, name, special_handling="expand"):
    """Values for a name from :func:`expanded_features`.

    An indicator name ``"feat==code"`` yields its 0/1 column; a plain feature
    yields its numeric values with the declared sentinels median-imputed
    (removed from the numeric scale) when ``special_handling='expand'``, so the
    numeric binning never sees a sentinel as an ordinary low value.
    """
    codes = list(ds.special_codes) if special_handling == "expand" else []
    if codes and "==" in name:
        feat, code = name.rsplit("==", 1)
        return (ds.X[feat].to_numpy(dtype=float) == float(code)).astype(float)
    col = ds.X[name].to_numpy(dtype=float)
    if codes:
        is_special = np.isin(col, codes)
        if is_special.any():
            clean = np.where(is_special, np.nan, col)
            col = np.where(np.isfinite(clean), clean, np.nanmedian(clean))
    return col


def sentinel_split(ds):
    """Sentinel routing inputs for special_handling='token': the numeric
    matrix with sentinels replaced by NaN (caller imputes; the imputed
    value never reaches the encoder -- token routing overrides it) and a
    code-index matrix (0 = clean, k = position of the code in
    ds.special_codes, 1-based)."""
    x = ds.X[ds.numerical].to_numpy(dtype=float)
    codes = np.zeros(x.shape, dtype=np.int64)
    for k, c in enumerate(ds.special_codes, start=1):
        mask = x == c
        codes[mask] = k
        x[mask] = np.nan
    return x, codes


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


def save_results(rows, out_path, cfg=None):
    """Results writer; parquet with csv fallback. OVERWRITES the target
    (rerun semantics) -- partial reruns (e.g. a single-arm top-up) must
    use a fresh out dir and be merged at analysis time, or they clobber
    the full-arm files (near-miss logged 2026-07; hence the warning).

    With ``cfg`` (a Hydra/OmegaConf config), the RESOLVED configuration is
    persisted next to the results as ``<out>.config.yaml`` so every
    artifact records its own protocol (split sizes, arms, solver options;
    reviewer P0 -- the 65/35-vs-60/40 discrepancy was unresolvable from
    the parquet alone)."""
    import logging
    df = pd.DataFrame(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.with_suffix(".parquet").exists():
        logging.getLogger(__name__).warning(
            "overwriting existing results file %s",
            out_path.with_suffix(".parquet"))
    if cfg is not None:
        try:
            from omegaconf import OmegaConf
            out_path.with_suffix(".config.yaml").write_text(
                OmegaConf.to_yaml(cfg, resolve=True))
        except Exception:
            logging.getLogger(__name__).exception(
                "could not persist resolved config for %s", out_path)
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
