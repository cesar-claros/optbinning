"""Experiment AUDIT — the COMPAS case study (Paper C Sec. 6).

Not a benchmark: a demonstration that the layer's audit artifacts carry
the recidivism transparency debate's full checklist. Trains the C4
scorecard (OT layer + linear head, ple_interp) on non-protected
predictors, then produces three tables:

  _audit     : the extracted point table -- per-feature bin edges in
               raw units plus partial log-odds contributions probed at
               bin midpoints (exactly additive under the linear head).
  _fairness  : the FM group-disparity certificate between
               group-conditional score distributions (default the two
               largest race groups): FM_lambda, permutation p, the
               certified bound on |Delta E h| for hard-threshold
               policies ((1/2lambda) * FM; ||h - 1/2||_inf <= 1/2) and
               for Lipschitz policies, the observed mean-PD gap, and a
               per-feature group-FM attribution on the layer's bins.
  _consensus : P7 barycentric consensus cuts per feature (subsample
               folds -- Paper D: bootstrap is inconsistent for
               cube-root cut estimators) with cut dispersion and the
               n^(2/3)-calibrated stability index, so the published
               table carries per-cut uncertainty.

Torch-free pipeline check:  python experiments/run_compas.py model=logreg
HPC:                        python experiments/run_compas.py -m \
                                'seed=range(0,5)' device=cuda
"""

# Cesar Claros <cesar.claros@outlook.com>
# Copyright (C) 2026

import logging
import sys

from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra                                            # noqa: E402
from omegaconf import DictConfig                        # noqa: E402
from sklearn.metrics import roc_auc_score               # noqa: E402

from optbinning import ConsensusBinning                 # noqa: E402
from optbinning.binning.metrics import flat_metric_1d   # noqa: E402

from experiments import datasets                        # noqa: E402
from experiments.common import save_results             # noqa: E402
from experiments.run_l1 import (_ecdf_apply, _ecdf_fit,  # noqa: E402
                                _fit_logreg, _fit_ot_scorecard,
                                _fm_perm, _impute)

logger = logging.getLogger(__name__)


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _audit_rows(score_fn, bin_edges, srt, feats):
    """Point table by per-feature probes at bin midpoints (rank space),
    all other features held at their median rank; exactly the additive
    component under the linear head, up to a shared constant."""
    n_feat = len(feats)
    base = np.full((1, n_feat), 0.5)
    base_logit = float(_logit(score_fn(base))[0])
    rows = []
    for j, name in enumerate(feats):
        edges = np.concatenate(([0.0], np.asarray(bin_edges[j]), [1.0]))
        mids = (edges[:-1] + edges[1:]) / 2
        probes = np.tile(base, (len(mids), 1))
        probes[:, j] = mids
        pts = _logit(score_fn(probes)) - base_logit
        raw = np.quantile(srt[j], np.clip(edges, 0, 1))
        for k in range(len(mids)):
            rows.append(dict(feature=name, bin=k,
                             lo=float(raw[k]), hi=float(raw[k + 1]),
                             points=float(pts[k])))
    return rows


def run(cfg):
    ds = datasets.load("compas")
    x = _impute(ds.X[ds.numerical].to_numpy(dtype=float),
                np.nanmedian(ds.X[ds.numerical].to_numpy(dtype=float),
                             axis=0))
    y, groups = ds.y, ds.X[cfg.group_column].values

    rng = np.random.default_rng(cfg.seed)
    tr, te = datasets.split_indices(len(y), cfg.test_size, cfg.seed)
    srt = _ecdf_fit(x[tr])
    q_tr, q_te = _ecdf_apply(srt, x[tr]), _ecdf_apply(srt, x[te])

    if cfg.model == "ot_ple":
        score_fn, bin_edges = _fit_ot_scorecard(q_tr, y[tr], cfg)
    else:
        score_fn, bin_edges = _fit_logreg(q_tr, y[tr], cfg)
    s_te = score_fn(q_te)
    auc = float(roc_auc_score(y[te], s_te))
    logger.info("COMPAS %s: test auc=%.4f", cfg.model, auc)

    audit_rows = _audit_rows(score_fn, bin_edges, srt, ds.numerical)

    g1, g2 = cfg.groups
    m1, m2 = groups[te] == g1, groups[te] == g2
    fair_rows = []
    fm, p = _fm_perm(s_te[m1], s_te[m2], cfg.lam, cfg.n_grid,
                     cfg.n_permutations, rng)
    fair_rows.append(dict(
        kind="score", feature="__score__", group_a=g1, group_b=g2,
        n_a=int(m1.sum()), n_b=int(m2.sum()), fm=fm, fm_p=p,
        bound_threshold=fm / (2 * cfg.lam),
        bound_lipschitz=max(1.0, 0.5 / cfg.lam) * fm,
        observed_gap=float(s_te[m1].mean() - s_te[m2].mean())))
    for j, name in enumerate(ds.numerical):
        e = np.asarray(bin_edges[j])
        atoms = 0.5 * (np.concatenate([[0.0], e])
                       + np.concatenate([e, [1.0]]))
        a = np.bincount(np.searchsorted(e, q_te[m1, j]),
                        minlength=len(atoms)) / max(m1.sum(), 1)
        b = np.bincount(np.searchsorted(e, q_te[m2, j]),
                        minlength=len(atoms)) / max(m2.sum(), 1)
        fair_rows.append(dict(
            kind="feature", feature=name, group_a=g1, group_b=g2,
            n_a=int(m1.sum()), n_b=int(m2.sum()),
            fm=float(flat_metric_1d(a, b, atoms, cfg.lam)),
            fm_p=np.nan, bound_threshold=np.nan,
            bound_lipschitz=np.nan, observed_gap=np.nan))

    cons_rows = []
    for j, name in enumerate(ds.numerical):
        try:
            cb = ConsensusBinning(
                n_folds=cfg.n_folds, resampling="subsample",
                subsample_fraction=cfg.subsample_fraction,
                random_state=cfg.seed,
                max_n_bins=cfg.n_bins).fit(x[tr][:, j], y[tr])
            for k, (cut, disp) in enumerate(
                    zip(np.atleast_1d(cb.splits_),
                        np.atleast_1d(cb.cut_dispersion_))):
                cons_rows.append(dict(
                    feature=name, cut=k, consensus_cut=float(cut),
                    dispersion=float(disp),
                    stability_index=float(cb.stability_index_)))
        except Exception:
            logger.exception("consensus failed on %s", name)

    common = dict(seed=cfg.seed, model=cfg.model, lam=cfg.lam,
                  auc=auc)
    for r in audit_rows + fair_rows + cons_rows:
        r.update(common)
    out = Path(cfg.out)
    tag = "compas_{}_{}".format(cfg.model, cfg.seed)
    paths = [save_results(audit_rows, out / (tag + "_audit")),
             save_results(fair_rows, out / (tag + "_fairness")),
             save_results(cons_rows, out / (tag + "_consensus"))]
    logger.info("AUDIT: score FM=%.4f (p=%.3f) bound_thr=%.4f "
                "observed=%.4f; wrote %s", fm, p, fm / (2 * cfg.lam),
                fair_rows[0]["observed_gap"], paths[0])
    return paths


@hydra.main(version_base=None, config_path="../conf",
            config_name="compas")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
