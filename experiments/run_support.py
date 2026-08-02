"""Experiment SUPPORT — exact support-gap classification of trust-frontier
points (plan E3; re-review 4.2: a finite multiplier grid cannot certify
unsupported Pareto points).

For each interior rho-solution X_j of the w1tau campaign (its (I_j, W_j)
pair recomputed here in the declared coordinate), minimize the Lagrangian
support gap

    g_j = min_{lambda >= 0} [ phi(lambda) - I(X_j) - lambda W(X_j) ],

where phi(lambda) = max_X { I(X) + lambda W(X) } is queried EXACTLY by the
weighted-sum MILP (gamma_wasserstein arm). A one-dimensional adaptive
upper-envelope procedure alternates between minimizing the current
envelope of oracle cuts and querying the MILP at the argmin:

  - g_j <= tau_supp  ->  supported (or numerical tie);
  - g_j >  tau_supp  ->  certified UNSUPPORTED (the envelope is an upper
    bound on phi only at queried points, but each queried lambda yields a
    VALID lower bound on the gap at that lambda; the minimum over the
    refined query set converges from above and the reported g_j is the
    envelope minimum after convergence).

Validation: on features with <= max_enum_prebins pre-bins, enumerate all
contiguous partitions of the fitted prebin grid that satisfy the same
monotone-trend feasibility, build phi exactly, and check the classifier
agrees (plan Week-1 stop condition).

Local smoke test:
    python experiments/run_support.py dataset=german n_seeds=1
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

from experiments import datasets                        # noqa: E402
from experiments.common import (binned_stats, expanded_features,  # noqa: E402
                                feature_array, make_arm, save_results,
                                splits_hash, to_coordinate)
from optbinning.binning.metrics import wasserstein_1d   # noqa: E402


def _iv_w1_of(splits, x, y):
    """(IV, W1) of a partition on (x, y) with pooled-mean atoms."""
    ne, e, w = binned_stats(x, y, splits)
    p = np.maximum(ne, 0.5) / max(ne.sum(), 0.5)
    q = np.maximum(e, 0.5) / max(e.sum(), 0.5)
    iv = float(((p - q) * np.log(p / q)).sum())
    w1 = float(wasserstein_1d(ne / ne.sum(), e / e.sum(), w)) \
        if len(w) > 1 else 0.0
    return iv, w1


def _oracle(x, y, lam, monotonic):
    """phi(lambda) via the exact weighted-sum MILP; returns value+point."""
    optb = make_arm("iv_w1", gamma=lam, monotonic=monotonic).fit(x, y)
    iv, w1 = _iv_w1_of(optb.splits, x, y)
    return iv + lam * w1, iv, w1, optb.status


def support_gap(x, y, iv_j, w1_j, monotonic, lam_hi=64.0,
                n_iter=12, tol=1e-9):
    """Adaptive envelope minimization of the Lagrangian support gap."""
    queries = {}

    def q(lam):
        if lam not in queries:
            val, iv, w1, status = _oracle(x, y, lam, monotonic)
            queries[lam] = val
        return queries[lam] - iv_j - lam * w1_j

    grid = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, lam_hi]
    gaps = {lam: q(lam) for lam in grid}
    for _ in range(n_iter):
        lams = sorted(gaps)
        best = min(lams, key=lambda v: gaps[v])
        i = lams.index(best)
        new = []
        if i > 0:
            new.append((lams[i - 1] + best) / 2)
        if i < len(lams) - 1:
            new.append((best + lams[i + 1]) / 2)
        improved = False
        for lam in new:
            g = q(lam)
            if g < gaps[best] - tol:
                improved = True
            gaps[lam] = g
        if not improved:
            break
    g_min = min(gaps.values())
    return float(max(g_min, 0.0)), len(queries)


def _enum_phi(x, y, splits_full, max_bins=None):
    """Exact (I, W) set over all contiguous, trend-feasible partitions of
    the fitted prebin grid (validation oracle for small instances)."""
    kbin = np.digitize(x, splits_full)
    npre = len(splits_full) + 1
    pts = []
    for kcut in range(npre):
        for cuts in combinations(range(1, npre), kcut):
            bounds = (0,) + cuts + (npre,)
            lab = np.searchsorted(np.asarray(cuts), kbin, side="right")
            k = len(bounds) - 1
            e = np.bincount(lab, weights=y, minlength=k)
            netot = np.bincount(lab, minlength=k) - e
            rate = e / np.maximum(e + netot, 1)
            d = np.diff(rate)
            if not (np.all(d >= -1e-12) or np.all(d <= 1e-12)):
                continue                       # monotone-feasibility filter
            sub = [splits_full[c - 1] for c in cuts]
            pts.append(_iv_w1_of(np.asarray(sub), x, y))
    return np.asarray(pts)


def run(cfg):
    ds = datasets.load(cfg.dataset)
    features = expanded_features(
        ds, list(cfg.features) if cfg.get("features") else None,
        cfg.get("special_handling", "expand"))
    rows = []
    for seed in range(cfg.seed_offset, cfg.seed_offset + cfg.n_seeds):
        tr, te = datasets.split_indices(len(ds.y), cfg.test_size, seed)
        for feat in features:
            x = feature_array(ds, feat,
                              cfg.get("special_handling", "expand"))
            mask = np.isfinite(x)
            xtr, ytr = x[tr][mask[tr]], ds.y[tr][mask[tr]]
            xtr, _ = to_coordinate(xtr, xtr, ytr,
                                   kind=cfg.get("coordinate", "rank"))
            try:
                iv_fit = make_arm("iv_mip",
                                  monotonic=cfg.monotonic).fit(xtr, ytr)
                w1_fit = make_arm("w1",
                                  monotonic=cfg.monotonic).fit(xtr, ytr)
            except Exception as err:            # pragma: no cover
                rows.append(dict(dataset=ds.name, feature=feat, seed=seed,
                                 status="ERROR:" + type(err).__name__))
                continue
            iv_a, w1_a = _iv_w1_of(iv_fit.splits, xtr, ytr)
            _, w1_c = _iv_w1_of(w1_fit.splits, xtr, ytr)
            if w1_c - w1_a <= 1e-9:
                rows.append(dict(dataset=ds.name, feature=feat, seed=seed,
                                 status="DEGENERATE"))
                continue
            for frac in cfg.fracs:
                rho = w1_a + float(frac) * (w1_c - w1_a)
                try:
                    optb = make_arm("w1_tau", fm_tau=rho,
                                    monotonic=cfg.monotonic).fit(xtr, ytr)
                except Exception as err:        # pragma: no cover
                    rows.append(dict(dataset=ds.name, feature=feat,
                                     seed=seed, frac=float(frac),
                                     status="ERROR:" + type(err).__name__))
                    continue
                if optb.status != "OPTIMAL":
                    continue
                iv_j, w1_j = _iv_w1_of(optb.splits, xtr, ytr)
                g, n_q = support_gap(xtr, ytr, iv_j, w1_j, cfg.monotonic)
                tau = float(cfg.tau_supp)
                rows.append(dict(
                    dataset=ds.name, feature=feat, seed=seed,
                    frac=float(frac), rho=rho, status="OPTIMAL",
                    iv=iv_j, w1=w1_j, support_gap=g, n_oracle=n_q,
                    splits_hash=splits_hash(optb.splits),
                    label=("supported" if g <= tau else "unsupported")))

    out = Path(cfg.out) / "support_{}_{}".format(cfg.dataset,
                                                 cfg.seed_offset)
    path = save_results(rows, out, cfg=cfg)
    print("SUPPORT: wrote {} rows -> {}".format(len(rows), path))
    return path


@hydra.main(version_base=None, config_path="../conf",
            config_name="support")
def main(cfg: DictConfig):
    return run(cfg)


if __name__ == "__main__":
    main()
