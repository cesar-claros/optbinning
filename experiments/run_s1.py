"""Experiment S1 — cut stability: three binning arms under resampling.

Tests the Paper D rate separation on real data: two-stage IV-optimal
cuts are cube-root (Kim-Pollard) estimators, smoothed end-to-end cuts
should be sqrt-n, quantile edges are the sqrt-n stability ceiling
(target-blind). Three arms produce nothing but per-feature cut sets:

  optbinning : OptimalBinning per feature (two-stage IV-optimal)
  ot_ple     : the Paper C layer (ple_interp; cuts = learned bin edges)
  quantile   : equal-frequency edges (stability ceiling, no target)

Measured per arm: (a) cut movement across resamples of the training set
-- mean pairwise Hausdorff distance and matched-cut sd on the reference
train rank scale (movement of optbinning cuts is transform-free there:
its splits depend on sample order/counts only); (b) retention -- test
IV and test AUC of an identical WoE-logistic downstream built on each
resample's cuts, so binning quality is isolated from head differences.

Resampling defaults to subsampling without replacement: Paper D shows
the naive bootstrap is inconsistent for cube-root functionals, so
bootstrap movement of the optbinning arm would carry an inflation bias.

Torch-free smoke (no ot_ple arm):
    python experiments/run_s1.py dataset=synthetic-smooth \
        'arms=[optbinning,quantile]' n_resamples=8
HPC:
    python experiments/run_s1.py -m dataset=german,taiwan,gmsc \
        'seed=range(0,5)' device=cuda out=outputs/s1
"""

# Cesar Claros <cesar.claros@outlook.com>
# Copyright (C) 2026

import logging
import sys
import time

from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra                                            # noqa: E402
from omegaconf import DictConfig                        # noqa: E402
from sklearn.linear_model import LogisticRegression     # noqa: E402
from sklearn.metrics import roc_auc_score               # noqa: E402

from optbinning import OptimalBinning                   # noqa: E402

from experiments import datasets                        # noqa: E402
from experiments.common import save_results             # noqa: E402

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Cut extraction per arm (raw feature units)
# --------------------------------------------------------------------- #

def _cuts_optbinning(x, y, n_bins):
    """Per-feature IV-optimal splits (the two-stage procedure)."""
    cuts = []
    for j in range(x.shape[1]):
        try:
            optb = OptimalBinning(dtype="numerical", solver="cp",
                                  max_n_bins=n_bins).fit(x[:, j], y)
            cuts.append(np.asarray(optb.splits, dtype=float))
        except Exception:                                # noqa: BLE001
            logger.exception("optbinning failed on feature %d", j)
            cuts.append(np.array([]))
    return cuts


def _cuts_quantile(x, n_bins):
    return [np.unique(np.quantile(
        x[:, j], np.linspace(0, 1, n_bins + 1)[1:-1]))
        for j in range(x.shape[1])]


def _cuts_ot(x, y, cfg):
    """One joint layer fit for all features (Paper C winning setup);
    learned rank-space edges mapped back to raw units via the
    resample's own quantile function (production-faithful)."""
    import torch
    from experiments.run_c3 import TokenizedNet

    device = torch.device(cfg.device)
    # fixed optimization seed across resamples: cut movement should
    # measure data variation, not batch-shuffling noise.
    torch.manual_seed(cfg.seed)
    srt = [np.sort(x[:, j]) for j in range(x.shape[1])]
    q = np.empty_like(x, dtype=float)
    for j, s in enumerate(srt):
        q[:, j] = np.searchsorted(s, x[:, j], side="right") / len(s)

    xtr = torch.as_tensor(q, dtype=torch.float32, device=device)
    ytr = torch.as_tensor(y, dtype=torch.float32, device=device)
    edges0 = [np.linspace(0, 1, cfg.n_bins + 1)] * x.shape[1]
    net = TokenizedNet("ot_ple", edges0, cfg.n_bins, "linear", cfg.hidden,
                       token_mode=cfg.token_mode,
                       sinkhorn_iters=cfg.sinkhorn_iters).to(device)
    net.ot.set_range(xtr.min(dim=0).values, xtr.max(dim=0).values)
    optim = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    bce = torch.nn.BCEWithLogitsLoss()
    n = len(ytr)
    for epoch in range(cfg.epochs):
        frac = epoch / max(cfg.epochs - 1, 1)
        eps = cfg.eps_start * (cfg.eps_end / cfg.eps_start) ** frac
        perm = torch.randperm(n, device=device)
        for lo in range(0, n, cfg.batch_size):
            idx = perm[lo:lo + cfg.batch_size]
            if len(idx) < cfg.n_bins * 4:
                continue
            logits, assign = net(xtr[idx], eps=eps,
                                 need_assign=cfg.aux_iv > 0)
            loss = bce(logits, ytr[idx])
            if assign is not None and cfg.aux_iv > 0:
                from experiments.paperc.otlayer import (pav_penalty_multi,
                                                        soft_iv_multi)
                loss = loss - cfg.aux_iv * soft_iv_multi(assign, ytr[idx])
                loss = loss + cfg.aux_iv * pav_penalty_multi(assign,
                                                             ytr[idx])
            optim.zero_grad()
            loss.backward()
            optim.step()

    rank_edges = net.ot.bin_edges().detach().cpu().numpy()
    return [np.quantile(x[:, j], np.clip(rank_edges[j], 0, 1))
            for j in range(x.shape[1])]


# --------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------- #

def _to_rank(ref_sorted, cuts):
    """Map raw cut positions to the frozen reference rank scale."""
    return [np.searchsorted(ref_sorted[j], c, side="right")
            / len(ref_sorted[j]) for j, c in enumerate(cuts)]


def _hausdorff(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.nan if len(a) != len(b) else 0.0
    d = np.abs(a[:, None] - b[None, :])
    return float(max(d.min(axis=1).max(), d.min(axis=0).max()))


def _movement_rows(rank_cuts, arm):
    """Per-feature movement stats across resamples."""
    rows = []
    n_feat = len(rank_cuts[0])
    for j in range(n_feat):
        sets = [rc[j] for rc in rank_cuts]
        counts = np.array([len(s) for s in sets])
        pair_h = [_hausdorff(a, b) for a, b in combinations(sets, 2)]
        modal = np.bincount(counts).argmax()
        matched = np.array([s for s in sets if len(s) == modal])
        rows.append(dict(
            arm=arm, feature=j,
            hausdorff=float(np.nanmean(pair_h)) if pair_h else np.nan,
            cut_sd=float(matched.std(axis=0).mean())
            if modal > 0 and len(matched) > 1 else np.nan,
            n_cuts_mean=float(counts.mean()),
            n_cuts_sd=float(counts.std()),
            frac_modal=float((counts == modal).mean())))
    return rows


def _woe_auc_iv(cuts, x_fit, y_fit, x_test, y_test):
    """Identical downstream for every arm: WoE transform learned on the
    resample, logistic scorecard, evaluated on the frozen test set.
    Also the total test IV of the cut sets themselves."""
    def digitize(x):
        return [np.digitize(x[:, j], cuts[j]) for j in range(x.shape[1])]

    w_fit, w_test, iv_total = [], [], 0.0
    d_fit, d_test = digitize(x_fit), digitize(x_test)
    for j in range(x_fit.shape[1]):
        k = len(cuts[j]) + 1
        e = np.bincount(d_fit[j], weights=y_fit, minlength=k) + 0.5
        ne = np.bincount(d_fit[j], weights=1 - y_fit, minlength=k) + 0.5
        woe = np.log((e / e.sum()) / (ne / ne.sum()))
        w_fit.append(woe[d_fit[j]])
        w_test.append(woe[d_test[j]])

        te = np.bincount(d_test[j], weights=y_test, minlength=k) + 0.5
        tn = np.bincount(d_test[j], weights=1 - y_test, minlength=k) + 0.5
        p, qq = te / te.sum(), tn / tn.sum()
        iv_total += float(((p - qq) * np.log(p / qq)).sum())

    model = LogisticRegression(max_iter=1000).fit(
        np.column_stack(w_fit), y_fit)
    prob = model.predict_proba(np.column_stack(w_test))[:, 1]
    return float(roc_auc_score(y_test, prob)), iv_total


# --------------------------------------------------------------------- #

def run(cfg):
    ds = datasets.load(cfg.dataset, n=cfg.get("n", 6000),
                       seed=cfg.get("data_seed", 0)) \
        if str(cfg.dataset).startswith("synthetic") \
        else datasets.load(cfg.dataset)
    x = ds.X[ds.numerical].to_numpy(dtype=float)
    med = np.nanmedian(x, axis=0)
    x = np.where(np.isfinite(x), x, med)

    tr, te = datasets.split_indices(len(ds.y), cfg.test_size, cfg.seed)
    x_tr, y_tr = x[tr], ds.y[tr]
    x_te, y_te = x[te], ds.y[te]
    ref_sorted = [np.sort(x_tr[:, j]) for j in range(x_tr.shape[1])]

    rng = np.random.default_rng(cfg.seed)
    m = int(cfg.subsample_frac * len(y_tr))

    rank_cuts = {arm: [] for arm in cfg.arms}
    perf_rows = []
    for b in range(cfg.n_resamples):
        idx = (rng.choice(len(y_tr), len(y_tr), replace=True)
               if cfg.resampling == "bootstrap"
               else rng.permutation(len(y_tr))[:m])
        xb, yb = x_tr[idx], y_tr[idx]
        for arm in cfg.arms:
            start = time.perf_counter()
            if arm == "optbinning":
                cuts = _cuts_optbinning(xb, yb, cfg.n_bins)
            elif arm == "quantile":
                cuts = _cuts_quantile(xb, cfg.n_bins)
            elif arm == "ot_ple":
                cuts = _cuts_ot(xb, yb, cfg)
            else:
                raise ValueError("unknown arm: {}".format(arm))
            rank_cuts[arm].append(_to_rank(ref_sorted, cuts))
            auc, iv = _woe_auc_iv(cuts, xb, yb, x_te, y_te)
            perf_rows.append(dict(
                arm=arm, resample=b, auc=auc, iv_test=iv,
                n_cuts_mean=float(np.mean([len(c) for c in cuts])),
                fit_time=time.perf_counter() - start))
        logger.info("resample %d/%d done", b + 1, cfg.n_resamples)

    cut_rows = []
    for arm in cfg.arms:
        cut_rows.extend(_movement_rows(rank_cuts[arm], arm))

    common = dict(dataset=str(cfg.dataset), seed=cfg.seed,
                  n_bins=cfg.n_bins, resampling=cfg.resampling,
                  n_resamples=cfg.n_resamples,
                  subsample_frac=cfg.subsample_frac)
    for r in perf_rows + cut_rows:
        r.update(common)

    out = Path(cfg.out)
    tag = "s1_{}_{}".format(cfg.dataset, cfg.seed)
    paths = [save_results(cut_rows, out / (tag + "_cuts")),
             save_results(perf_rows, out / (tag + "_perf"))]
    for arm in cfg.arms:
        h = np.nanmean([r["hausdorff"] for r in cut_rows
                        if r["arm"] == arm])
        a = np.mean([r["auc"] for r in perf_rows if r["arm"] == arm])
        logger.info("S1 %s/%s: hausdorff=%.5f auc=%.4f",
                    cfg.dataset, arm, h, a)
    return paths


@hydra.main(version_base=None, config_path="../conf", config_name="s1")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
