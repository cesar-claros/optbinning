"""Experiment MATCHED — G3 (matched-cost stability) and G4 (downstream
noninferiority) of the practical-narrative plan.

Per dataset x seed, with nested fit/validation/test roles (chronological
split when the dataset declares a time column, e.g. bank -- the G4
out-of-time leg):

1. FEATURE PANEL: nondegenerate features only (a quick Gamma_j probe;
   degenerate features have identical partitions across methods by
   construction), capped at ``max_features`` by seeded draw.
2. BUDGET-MATCHED SELECTION (per feature, per budget delta): each
   method's hyperparameter is the strongest setting whose VALIDATION IV
   loss against the IV argmax is <= delta. Methods:
     iv          -- the unconstrained argmax reference;
     iv_maxbins  -- validation-selected bin-count cap (simplicity
                    control);
     min_size / min_event -- feasibility controls;
     smoothed_iv -- add-alpha pseudocount control;
     w1_tau      -- largest rho on the G-grid within budget (the plan's
                    rho_hat(delta) rule);
     hybrid      -- gamma selected on validation within budget.
3. G3 STABILITY: n_resamples subsample refits (0.7, frozen
   hyperparameters); D_assign = fraction of TEST points whose
   order-aligned bin assignment differs across refit pairs; WoE drift =
   mean sd of per-point WoE across refits; bin count.
4. G4 DOWNSTREAM: one WoE-logistic scorecard per method over the panel
   features (frozen hyperparameters, fit on fit+val), evaluated once on
   test: AUC, log-loss, Brier, ECE, calibration slope/intercept.

Local smoke test:
    python experiments/run_matched.py dataset=german n_seeds=1 \
        n_resamples=5 max_features=4
"""

# Cesar Claros <cesar.claros@outlook.com>
# Copyright (C) 2026

import sys

from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra                                            # noqa: E402
from omegaconf import DictConfig                        # noqa: E402
from sklearn.linear_model import LogisticRegression     # noqa: E402
from sklearn.metrics import log_loss, roc_auc_score     # noqa: E402

from experiments import datasets                        # noqa: E402
from experiments.common import (binned_stats, expanded_features,  # noqa: E402
                                feature_array, make_arm, save_results,
                                splits_hash, to_coordinate)

BUDGETS = (0.0025, 0.005, 0.01)


def _iv_of(splits, x, y):
    ne, e, _ = binned_stats(x, y, splits)
    p = np.maximum(ne, 0.5) / max(ne.sum(), 0.5)
    q = np.maximum(e, 0.5) / max(e.sum(), 0.5)
    return float(((p - q) * np.log(p / q)).sum())


def _fit_method(name, spec, x, y, monotonic):
    kw = dict(monotonic=monotonic)
    if name == "iv_maxbins":
        return make_arm("iv_mip", **kw, max_n_bins=int(spec)).fit(x, y)
    if name == "min_size":
        m = make_arm("iv_mip", **kw)
        m.set_params(min_bin_size=float(spec))
        return m.fit(x, y)
    if name == "min_event":
        m = make_arm("iv_mip", **kw)
        m.set_params(min_bin_n_event=int(spec))
        return m.fit(x, y)
    if name == "smoothed_iv":
        alpha = float(spec)
        qs = np.quantile(x, np.linspace(0, 1, 21))
        mids = (qs[1:] + qs[:-1]) / 2
        reps = int(np.ceil(alpha))
        xa = np.concatenate([x, np.repeat(mids, 2 * reps)])
        ya = np.concatenate([y, np.tile(np.repeat([0, 1], reps),
                                        len(mids))])
        return make_arm("iv_mip", **kw).fit(xa, ya)
    if name == "w1_tau":
        return make_arm("w1_tau", fm_tau=float(spec), **kw).fit(x, y)
    if name == "hybrid":
        return make_arm("iv_w1", gamma=float(spec), **kw).fit(x, y)
    return make_arm("iv_mip", **kw).fit(x, y)          # iv reference


def _grids(w1_floor, w1_ceil, scale):
    """Candidate hyperparameter grids per method (strongest first)."""
    rho = [w1_floor + f * (w1_ceil - w1_floor)
           for f in (1.0, 0.9, 0.75, 0.5, 0.25, 0.1)]
    return {
        "iv_maxbins": [3, 4, 5, 8, 12],
        "min_size": [0.20, 0.10, 0.05, 0.02],
        "min_event": [100, 25, 5],
        "smoothed_iv": [4.0, 2.0, 0.5],
        "w1_tau": rho,
        "hybrid": [g / max(abs(scale), 1e-9)
                   for g in (4.0, 2.0, 1.0, 0.5, 0.25, 0.1)],
    }


def _select_budget(name, grid, xf, yf, xv, yv, iv_val_ref, delta,
                   monotonic):
    """Strongest setting whose validation IV loss <= delta; None if no
    setting qualifies (method excluded at this budget)."""
    for spec in grid:
        try:
            fit = _fit_method(name, spec, xf, yf, monotonic)
            if fit.status != "OPTIMAL":
                continue
        except Exception:                               # noqa: BLE001
            continue
        if iv_val_ref - _iv_of(fit.splits, xv, yv) <= delta:
            return spec, fit
    return None, None


def _assign(splits, x):
    return np.searchsorted(np.asarray(splits, float), x, side="right")


def _woe_map(splits, x, y):
    idx = _assign(splits, x)
    k = len(splits) + 1
    e = np.bincount(idx, weights=y, minlength=k) + 0.5
    ne = np.bincount(idx, weights=1 - y, minlength=k) + 0.5
    return np.log((e / e.sum()) / (ne / ne.sum()))


def _stability(name, spec, xf, yf, x_eval, monotonic, n_res, frac, rng):
    """D_assign + WoE drift over subsample refits, on the frozen test
    grid (order-aligned by construction: assignments are interval
    indices on one axis)."""
    assigns, woes = [], []
    m = int(frac * len(yf))
    for _ in range(n_res):
        idx = rng.permutation(len(yf))[:m]
        try:
            fit = _fit_method(name, spec, xf[idx], yf[idx], monotonic)
        except Exception:                               # noqa: BLE001
            continue
        a = _assign(fit.splits, x_eval)
        assigns.append(a)
        woes.append(_woe_map(fit.splits, xf[idx], yf[idx])[a])
    if len(assigns) < 2:
        return np.nan, np.nan, 0
    pairs = list(combinations(range(len(assigns)), 2))
    d = float(np.mean([(assigns[i] != assigns[j]).mean()
                       for i, j in pairs]))
    drift = float(np.mean(np.std(np.stack(woes), axis=0)))
    return d, drift, len(assigns)


def _ece(prob, y, n_bins=10):
    idx = np.clip((prob * n_bins).astype(int), 0, n_bins - 1)
    return float(sum((idx == b).mean()
                     * abs(prob[idx == b].mean() - y[idx == b].mean())
                     for b in range(n_bins) if (idx == b).any()))


def run(cfg):
    ds = datasets.load(cfg.dataset)
    feats_all = expanded_features(
        ds, None, cfg.get("special_handling", "expand"))
    chrono = ds.time_column is not None
    n = len(ds.y)
    rows = []
    for seed in range(cfg.seed_offset, cfg.seed_offset + cfg.n_seeds):
        if chrono:
            order = np.argsort(
                ds.X[ds.time_column].to_numpy())      # out-of-time split
            n_te = int(cfg.test_size * n)
            tr, te = order[:n - n_te], order[n - n_te:]
        else:
            tr, te = datasets.split_indices(n, cfg.test_size, seed)
        rng = np.random.default_rng(seed)

        # feature panel: nondegenerate probe, seeded cap
        panel = []
        for feat in feats_all:
            x = feature_array(ds, feat, cfg.get("special_handling",
                                                "expand"))
            mask = np.isfinite(x)
            xt, yt = x[tr][mask[tr]], ds.y[tr][mask[tr]]
            if len(np.unique(xt)) < 6:
                continue
            xt, _ = to_coordinate(xt, xt, yt, kind="rank")
            try:
                ivf = make_arm("iv_mip", monotonic=cfg.monotonic
                               ).fit(xt, yt)
                w1f = make_arm("w1", monotonic=cfg.monotonic
                               ).fit(xt, yt)
            except Exception:                           # noqa: BLE001
                continue
            from experiments.run_w1tau import _w1_of
            if _w1_of(w1f.splits, xt, yt) - _w1_of(ivf.splits, xt, yt) \
                    > 1e-9:
                panel.append(feat)
        rng.shuffle(panel)
        panel = panel[:cfg.max_features]

        # per-feature selection + stability; collect cuts for scorecards
        cuts = {}                                       # (budget, method, feat)
        for feat in panel:
            x = feature_array(ds, feat, cfg.get("special_handling",
                                                "expand"))
            med = np.nanmedian(x[tr])
            x = np.where(np.isfinite(x), x, med)
            xtr_all, ytr_all = x[tr], ds.y[tr]
            fit_i, val_i = datasets.split_indices(
                len(ytr_all), cfg.get("val_size", 0.25), seed + 10_000)
            xf, yf = xtr_all[fit_i], ytr_all[fit_i]
            xv, yv = xtr_all[val_i], ytr_all[val_i]
            xf, xte_c = to_coordinate(xf, x[te], yf, kind="rank")
            _, xv = to_coordinate(xtr_all[fit_i], xv, yf, kind="rank")

            iv_ref = make_arm("iv_mip", monotonic=cfg.monotonic
                              ).fit(xf, yf)
            iv_val_ref = _iv_of(iv_ref.splits, xv, yv)
            from experiments.run_w1tau import _w1_of
            w1_floor = _w1_of(iv_ref.splits, xf, yf)
            w1_ceil = _w1_of(make_arm("w1", monotonic=cfg.monotonic
                                      ).fit(xf, yf).splits, xf, yf)
            scale = np.subtract(*np.nanquantile(xf, [0.95, 0.05]))
            grids = _grids(w1_floor, w1_ceil, scale)

            for delta in BUDGETS:
                for name in ("iv", "iv_maxbins", "min_size", "min_event",
                             "smoothed_iv", "w1_tau", "hybrid"):
                    if name == "iv":
                        spec, fit = 0.0, iv_ref
                    else:
                        spec, fit = _select_budget(
                            name, grids[name], xf, yf, xv, yv,
                            iv_val_ref, delta, cfg.monotonic)
                    if fit is None:
                        rows.append(dict(
                            dataset=ds.name, seed=seed, feature=feat,
                            budget=delta, method=name,
                            status="NO_QUALIFYING_SETTING"))
                        continue
                    d, drift, n_ok = _stability(
                        name, spec, xf, yf, xte_c, cfg.monotonic,
                        cfg.n_resamples, cfg.subsample_frac,
                        np.random.default_rng(seed * 7 + 1))
                    cuts[(delta, name, feat)] = np.asarray(
                        fit.splits, float)
                    rows.append(dict(
                        dataset=ds.name, seed=seed, feature=feat,
                        budget=delta, method=name, spec=float(spec),
                        status="OK", chrono=chrono,
                        n_bins=len(fit.splits) + 1,
                        splits_hash=splits_hash(fit.splits),
                        val_iv_loss=iv_val_ref
                        - _iv_of(fit.splits, xv, yv),
                        d_assign=d, woe_drift=drift,
                        n_refits=n_ok))

        # G4 downstream scorecards per (budget, method)
        for delta in BUDGETS:
            for name in ("iv", "iv_maxbins", "min_size", "min_event",
                         "smoothed_iv", "w1_tau", "hybrid"):
                w_tr, w_te = [], []
                for feat in panel:
                    c = cuts.get((delta, name, feat))
                    if c is None:
                        continue
                    x = feature_array(ds, feat,
                                      cfg.get("special_handling",
                                              "expand"))
                    med = np.nanmedian(x[tr])
                    x = np.where(np.isfinite(x), x, med)
                    xt_c, xe_c = to_coordinate(x[tr], x[te], ds.y[tr],
                                               kind="rank")
                    woe = _woe_map(c, xt_c, ds.y[tr].astype(float))
                    w_tr.append(woe[_assign(c, xt_c)])
                    w_te.append(woe[_assign(c, xe_c)])
                if len(w_tr) < 2:
                    continue
                clf = LogisticRegression(max_iter=1000).fit(
                    np.column_stack(w_tr), ds.y[tr])
                prob = clf.predict_proba(np.column_stack(w_te))[:, 1]
                yte = ds.y[te]
                lo = np.log(np.clip(prob, 1e-12, 1 - 1e-12)
                            / np.clip(1 - prob, 1e-12, 1))
                cal = LogisticRegression(max_iter=1000).fit(
                    lo[:, None], yte)
                rows.append(dict(
                    dataset=ds.name, seed=seed, feature="__scorecard__",
                    budget=delta, method=name, status="OK",
                    chrono=chrono, n_features=len(w_tr),
                    auc=float(roc_auc_score(yte, prob)),
                    logloss=float(log_loss(yte, prob)),
                    brier=float(np.mean((prob - yte) ** 2)),
                    ece=_ece(prob, yte),
                    cal_slope=float(cal.coef_[0, 0]),
                    cal_intercept=float(cal.intercept_[0])))
        print("seed", seed, "done", flush=True)

    out = Path(cfg.out) / "matched_{}_{}".format(cfg.dataset,
                                                 cfg.seed_offset)
    path = save_results(rows, out, cfg=cfg)
    print("MATCHED: wrote {} rows -> {}".format(len(rows), path))
    return path


@hydra.main(version_base=None, config_path="../conf",
            config_name="matched")
def main(cfg: DictConfig):
    return run(cfg)


if __name__ == "__main__":
    main()
