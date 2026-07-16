"""Experiment L1 — lifecycle: OT-binned scorecard + FM drift monitoring.

The Paper B x Paper C bridge on temporally shifting data: train the
end-to-end OT-binning scorecard (Paper C, winning BAF configuration:
ple_interp tokens, linear head) on early months, hold out half of the
training window as the monitoring reference, then surveil the remaining
months with the flat-metric stack of Paper B on the SAME representation:
batch FM with permutation p-values and PD certificates, PSI/KS industry
baselines, the anytime-valid tolerance e-process, and per-feature FM
attribution on the layer's own audit-table bins. One lambda across the
lifecycle.

Pipeline check without torch (synthetic drift ramp, logistic model):
    python experiments/run_l1.py dataset=synthetic-smooth model=logreg
HPC (real experiment):
    python experiments/run_l1.py -m dataset=baf 'seed=range(0,5)' \
        device=cuda out=outputs/l1
"""

# Cesar Claros <cesar.claros@outlook.com>
# Copyright (C) 2026

import logging
import sys
import time

from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra                                            # noqa: E402
from omegaconf import DictConfig                        # noqa: E402
from sklearn.metrics import roc_auc_score               # noqa: E402

from optbinning import SequentialMonitor                # noqa: E402
from optbinning.binning.metrics import flat_metric_1d   # noqa: E402

from experiments import datasets                        # noqa: E402
from experiments.common import save_results             # noqa: E402

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Monitoring primitives (score space = PD in [0, 1])
# --------------------------------------------------------------------- #

def _grid_cells(pooled, n_grid):
    """Quantile-grid compression: edges, per-cell atom positions."""
    edges = np.unique(np.quantile(pooled, np.linspace(0, 1, n_grid + 1)))
    edges = edges[1:-1]
    cells = np.searchsorted(edges, pooled)
    n_cells = len(edges) + 1
    cnt = np.bincount(cells, minlength=n_cells).astype(float)
    atoms = np.where(
        cnt > 0,
        np.bincount(cells, weights=pooled,
                    minlength=n_cells) / np.maximum(cnt, 1),
        0.5 * (np.concatenate([[pooled.min()], edges]) +
               np.concatenate([edges, [pooled.max()]])))
    return edges, atoms


def _fm_perm(ref, cur, lam, n_grid, n_perm, rng):
    """Grid-compressed FM(ref, cur) and its permutation p-value (all
    permutation replicates evaluated in one batched level-set DP)."""
    pooled = np.concatenate([ref, cur])
    edges, atoms = _grid_cells(pooled, n_grid)
    cells = np.searchsorted(edges, pooled)
    n_cells = len(atoms)
    n_ref = len(ref)

    def masses(idx_ref, idx_cur):
        a = np.bincount(cells[idx_ref], minlength=n_cells) / len(idx_ref)
        b = np.bincount(cells[idx_cur], minlength=n_cells) / len(idx_cur)
        return a, b

    A = np.empty((n_perm + 1, n_cells))
    B = np.empty((n_perm + 1, n_cells))
    A[0], B[0] = masses(np.arange(n_ref), np.arange(n_ref, len(pooled)))
    for i in range(n_perm):
        perm = rng.permutation(len(pooled))
        A[i + 1], B[i + 1] = masses(perm[:n_ref], perm[n_ref:])
    fm = np.asarray(flat_metric_1d(A, B, atoms, lam), dtype=float)
    obs = float(fm[0])
    return obs, (1 + int((fm[1:] >= obs).sum())) / (1 + n_perm)


def _fm_null_q(ref, lam, n_grid, n_rep, q, rng):
    """In-control FM scale: q-quantile of half-vs-half reference splits
    (shared grid on the full reference; batched DP)."""
    edges, atoms = _grid_cells(ref, n_grid)
    cells = np.searchsorted(edges, ref)
    n_cells = len(atoms)
    half = len(ref) // 2
    A = np.empty((n_rep, n_cells))
    B = np.empty((n_rep, n_cells))
    for i in range(n_rep):
        perm = rng.permutation(len(ref))
        A[i] = np.bincount(cells[perm[:half]], minlength=n_cells) / half
        B[i] = np.bincount(cells[perm[half:]],
                           minlength=n_cells) / (len(ref) - half)
    vals = np.asarray(flat_metric_1d(A, B, atoms, lam), dtype=float)
    return float(np.quantile(vals, q))


def _psi(ref, cur, n_bins=10, eps=1e-4):
    """Population stability index on reference deciles."""
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)))
    edges = edges[1:-1]
    a = np.bincount(np.searchsorted(edges, ref),
                    minlength=len(edges) + 1) + eps
    b = np.bincount(np.searchsorted(edges, cur),
                    minlength=len(edges) + 1) + eps
    a, b = a / a.sum(), b / b.sum()
    return float(np.sum((b - a) * np.log(b / a)))


def _ks(ref, cur):
    """Two-sample Kolmogorov-Smirnov statistic."""
    grid = np.sort(np.concatenate([ref, cur]))
    fa = np.searchsorted(np.sort(ref), grid, side="right") / len(ref)
    fb = np.searchsorted(np.sort(cur), grid, side="right") / len(cur)
    return float(np.abs(fa - fb).max())


# --------------------------------------------------------------------- #
# Feature pipeline
# --------------------------------------------------------------------- #

def _ecdf_fit(x_fit):
    """Per-feature sorted arrays of the fit half (train ECDF)."""
    return [np.sort(x_fit[:, j]) for j in range(x_fit.shape[1])]

def _ecdf_apply(srt, x):
    q = np.empty_like(x, dtype=float)
    for j, s in enumerate(srt):
        q[:, j] = np.searchsorted(s, x[:, j], side="right") / len(s)
    return q


def _impute(x, med):
    return np.where(np.isfinite(x), x, med)


# --------------------------------------------------------------------- #
# Models (score = predicted PD)
# --------------------------------------------------------------------- #

def _fit_ot_scorecard(q_fit, y_fit, cfg):
    """Winning BAF configuration: ple_interp tokens + linear head.
    Returns (score_fn, per-feature rank-space bin edges)."""
    import torch
    from experiments.run_c3 import TokenizedNet

    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    xtr = torch.as_tensor(q_fit, dtype=torch.float32, device=device)
    ytr = torch.as_tensor(y_fit, dtype=torch.float32, device=device)
    edges0 = [np.linspace(0, 1, cfg.n_bins + 1)] * q_fit.shape[1]
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

    net.eval()

    def score_fn(q):
        with torch.no_grad():
            xt = torch.as_tensor(q, dtype=torch.float32, device=device)
            probs = []
            for lo in range(0, len(xt), cfg.batch_size):
                logits, _ = net(xt[lo:lo + cfg.batch_size],
                                eps=cfg.eps_end, need_assign=False)
                probs.append(torch.sigmoid(logits))
            return torch.cat(probs).cpu().numpy()

    bin_edges = net.ot.bin_edges().detach().cpu().numpy()
    return score_fn, [bin_edges[j] for j in range(bin_edges.shape[0])]


def _fit_logreg(q_fit, y_fit, cfg):
    """Torch-free pipeline check: logistic model on rank features,
    equal-frequency attribution bins."""
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000).fit(q_fit, y_fit)

    def score_fn(q):
        return model.predict_proba(q)[:, 1]

    bin_edges = [np.unique(np.quantile(
        q_fit[:, j], np.linspace(0, 1, cfg.n_bins + 1)[1:-1]))
        for j in range(q_fit.shape[1])]
    return score_fn, bin_edges


# --------------------------------------------------------------------- #
# Data: month -> (X_numerical, y)
# --------------------------------------------------------------------- #

def _monthly_data(cfg):
    """Real data: split on the time column. Synthetic: months 0-2 are
    in-control draws; monitor months carry a drift-magnitude ramp
    (first monitor month has magnitude 0 = size check)."""
    name = str(cfg.dataset)
    if name.startswith("synthetic"):
        design = name.split("-", 1)[1] if "-" in name else "smooth"
        months = {}
        for m in range(3):
            ds = datasets.make_synthetic(design, n=cfg.n_month,
                                         seed=cfg.seed * 1000 + m)
            months[m] = (ds.X[ds.numerical].to_numpy(dtype=float), ds.y)
        for i, m in enumerate(range(3, 8)):
            mag = cfg.drift_step * i
            drift = ({"kind": cfg.drift_kind, "magnitude": mag}
                     if mag > 0 else None)
            ds = datasets.make_synthetic(design, n=cfg.n_month,
                                         seed=cfg.seed * 1000 + 100 + m,
                                         drift=drift)
            months[m] = (ds.X[ds.numerical].to_numpy(dtype=float), ds.y)
        return months, [0, 1, 2]

    ds = datasets.load(name)
    if ds.time_column is None:
        raise ValueError("dataset {} has no time column.".format(name))
    t = ds.X[ds.time_column].to_numpy()
    x = ds.X[ds.numerical].to_numpy(dtype=float)
    months = {int(m): (x[t == m], ds.y[t == m]) for m in np.unique(t)}
    return months, list(cfg.train_months)


# --------------------------------------------------------------------- #

def run(cfg):
    rng = np.random.default_rng(cfg.seed)
    months, train_months = _monthly_data(cfg)
    monitor_months = sorted(m for m in months if m not in train_months)

    # fit half / reference half of the training window (B2 protocol:
    # in-sample references read as spurious drift and break permutation
    # exchangeability).
    x_tr = np.vstack([months[m][0] for m in train_months])
    y_tr = np.concatenate([months[m][1] for m in train_months])
    perm = rng.permutation(len(y_tr))
    half = len(y_tr) // 2
    fit_idx, ref_idx = perm[:half], perm[half:]

    med = np.nanmedian(x_tr[fit_idx], axis=0)
    srt = _ecdf_fit(_impute(x_tr[fit_idx], med))
    q_fit = _ecdf_apply(srt, _impute(x_tr[fit_idx], med))
    q_ref = _ecdf_apply(srt, _impute(x_tr[ref_idx], med))

    start = time.perf_counter()
    if cfg.model == "ot_ple":
        score_fn, bin_edges = _fit_ot_scorecard(q_fit, y_tr[fit_idx], cfg)
    else:
        score_fn, bin_edges = _fit_logreg(q_fit, y_tr[fit_idx], cfg)
    fit_time = time.perf_counter() - start

    s_ref = score_fn(q_ref)
    auc_ref = float(roc_auc_score(y_tr[ref_idx], s_ref))

    # tolerance: calibrated in-control FM (q95 of half-vs-half splits)
    tol = cfg.tolerance
    if tol is None:
        tol = _fm_null_q(s_ref, cfg.lam, cfg.n_grid, cfg.n_null, 0.95,
                         np.random.default_rng(cfg.seed + 1))
    logger.info("L1 %s/%s: ref auc=%.4f, tolerance=%.5f",
                cfg.dataset, cfg.model, auc_ref, tol)

    seq = SequentialMonitor(s_ref, lam=cfg.lam, alpha=cfg.alpha,
                            tolerance=tol,
                            restart_every=cfg.restart_every)

    pd_factor = max(1.0, 0.5 / cfg.lam)   # |dE h| <= max(Lip, ||h-c||/lam)
    month_rows, seq_rows, attr_rows = [], [], []
    for m in monitor_months:
        x_m, y_m = months[m]
        q_m = _ecdf_apply(srt, _impute(x_m, med))
        s_m = score_fn(q_m)

        fm, p = _fm_perm(s_ref, s_m, cfg.lam, cfg.n_grid,
                         cfg.n_permutations, rng)
        month_rows.append(dict(
            month=m, n=len(s_m),
            fm=fm, fm_p=p, psi=_psi(s_ref, s_m), ks=_ks(s_ref, s_m),
            auc=float(roc_auc_score(y_m, s_m)) if 0 < y_m.mean() < 1
            else np.nan,
            mean_pd=float(s_m.mean()),
            delta_mean_pd=float(s_m.mean() - s_ref.mean()),
            pd_bound=pd_factor * fm,
            pd_bound_ok=bool(abs(s_m.mean() - s_ref.mean())
                             <= pd_factor * fm + 1e-12)))

        for lo in range(0, len(s_m), cfg.seq_batch):
            status = seq.update(s_m[lo:lo + cfg.seq_batch])
            seq_rows.append(dict(month=m, **status))

        for j, e in enumerate(bin_edges):
            atoms = 0.5 * (np.concatenate([[0.0], e])
                           + np.concatenate([e, [1.0]]))
            a = np.bincount(np.searchsorted(e, q_ref[:, j]),
                            minlength=len(atoms)) / len(q_ref)
            b = np.bincount(np.searchsorted(e, q_m[:, j]),
                            minlength=len(atoms)) / len(q_m)
            attr_rows.append(dict(
                month=m, feature=j,
                fm=float(flat_metric_1d(a, b, atoms, cfg.lam))))

    common = dict(dataset=str(cfg.dataset), model=cfg.model, seed=cfg.seed,
                  lam=cfg.lam, tolerance=float(tol), auc_ref=auc_ref,
                  fit_time=fit_time)
    for rows in (month_rows, seq_rows, attr_rows):
        for r in rows:
            r.update(common)

    out = Path(cfg.out)
    tag = "l1_{}_{}_{}".format(cfg.dataset, cfg.model, cfg.seed)
    paths = [save_results(month_rows, out / (tag + "_months")),
             save_results(seq_rows, out / (tag + "_seq")),
             save_results(attr_rows, out / (tag + "_attr"))]
    alarms = [r for r in seq_rows if r["alarm"]]
    logger.info("L1 %s: first alarm %s; wrote %s",
                cfg.dataset,
                "month {} (n={})".format(alarms[0]["month"],
                                         alarms[0]["n_seen"])
                if alarms else "none", paths[0])
    return paths


@hydra.main(version_base=None, config_path="../conf", config_name="l1")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
