"""Experiment E-SURV — learned time grids for discrete-time hazards.

Discrete-time survival needs a time grid; ad hoc grids give
event-sparse intervals. Arms differ ONLY in the grid: equal-width,
event-quantile, and the OT grid -- interior edges by ordered cumulative
softplus, soft exposures via clipped time ramps (the PLE encoding of
time, differentiable in the edges; interval membership = adjacent-ramp
differences), proportional discrete hazard, soft NLL + an event-mass
floor penalty (minimum events per interval -- the credibility
constraint). Every arm's grid is then fit with the identical
person-period logistic downstream. Metrics at fixed horizons: AUC and
Brier on the at-risk subset (IPCW-free simplification, noted), minimum
events per interval, and grid stability (edge Hausdorff over training
resamples, on the event-time rank scale).

Torch-free smoke (fixed grids only):
    python experiments/run_surv.py 'arms=[width,quantile]' n=4000
HPC:
    python experiments/run_surv.py -m 'seed=range(0,5)' device=cuda \
        out=outputs/surv
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

from experiments.common import save_results             # noqa: E402
from experiments.run_s1 import _hausdorff               # noqa: E402

logger = logging.getLogger(__name__)


def _synthetic(n, seed, censor_scale):
    """Weibull survival with two covariates and uniform censoring."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (n, 2))
    scale = np.exp(0.8 * x[:, 0] + 0.5 * x[:, 1])
    t_event = scale * rng.weibull(1.5, n)
    c = rng.uniform(0, censor_scale * np.median(scale), n)
    t = np.minimum(t_event, c)
    delta = (t_event <= c).astype(float)
    return x, t, delta


# --------------------------------------------------------------------- #
# Grids
# --------------------------------------------------------------------- #

def _grid_fixed(arm, t, delta, k):
    if arm == "width":
        return np.linspace(0, t.max(), k + 1)[1:-1]
    ev = np.sort(t[delta > 0])
    return np.unique(np.quantile(ev, np.linspace(0, 1, k + 1)[1:-1]))


def _grid_ot(x, t, delta, cfg):
    """Learned grid: soft exposures via clipped time ramps; hazard
    sigma(alpha_k + x beta); soft discrete NLL + event-mass floor."""
    import torch
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    k = cfg.n_intervals
    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    tt = torch.as_tensor(t, dtype=torch.float32, device=device)
    dt = torch.as_tensor(delta, dtype=torch.float32, device=device)
    t_max = float(t.max()) * 1.001

    theta = torch.nn.Parameter(torch.zeros(k, device=device))
    alpha = torch.nn.Parameter(torch.full((k,), -1.5, device=device))
    beta = torch.nn.Parameter(torch.zeros(x.shape[1], device=device))
    optim = torch.optim.Adam([theta, alpha, beta], lr=cfg.lr)

    for _ in range(cfg.steps):
        inc = torch.nn.functional.softplus(theta) + 0.05
        cum = torch.cumsum(inc, 0)
        edges = torch.cat([torch.zeros(1, device=device),
                           t_max * cum / cum[-1]])       # (K+1,)
        width = (edges[1:] - edges[:-1]).clamp_min(1e-6)
        r = ((tt[:, None] - edges[None, :-1])
             / width[None]).clamp(0.0, 1.0)              # (n, K) ramps
        m = r - torch.cat([r[:, 1:],
                           torch.zeros_like(r[:, :1])], 1)
        surv = torch.cat([r[:, 1:], torch.zeros_like(r[:, :1])], 1)

        h = torch.sigmoid(alpha[None, :]
                          + (xt @ beta)[:, None]).clamp(1e-6, 1 - 1e-6)
        log1m = torch.log(1 - h)
        nll = -(surv * log1m).sum(1) \
            - dt * torch.log((m * h).sum(1).clamp_min(1e-12)) \
            - (1 - dt) * (m * log1m).sum(1)
        ev = (dt[:, None] * m).sum(0)
        floor = torch.relu(cfg.min_events - ev).pow(2).sum()
        loss = nll.mean() + cfg.floor_weight * floor
        optim.zero_grad()
        loss.backward()
        optim.step()

    inc = torch.nn.functional.softplus(theta.detach()) + 0.05
    cum = torch.cumsum(inc, 0)
    return (t_max * cum / cum[-1]).cpu().numpy()[:-1]    # interior


# --------------------------------------------------------------------- #
# Identical downstream: person-period logistic on any grid
# --------------------------------------------------------------------- #

def _person_period(edges, x, t, delta):
    idx = np.digitize(t, edges)
    rows_x, rows_k, rows_y = [], [], []
    for i in range(len(t)):
        for k in range(idx[i] + 1):
            rows_x.append(x[i])
            rows_k.append(k)
            rows_y.append(1.0 if (k == idx[i] and delta[i]) else 0.0)
    return (np.asarray(rows_x), np.asarray(rows_k),
            np.asarray(rows_y))


def _fit_hazard(edges, x, t, delta):
    px, pk, py = _person_period(edges, x, t, delta)
    k = len(edges) + 1
    onehot = np.eye(k)[pk]
    model = LogisticRegression(max_iter=2000).fit(
        np.column_stack([onehot, px]), py)

    def survival(xq, horizon):
        n_full = int(np.searchsorted(edges, horizon))
        s = np.ones(len(xq))
        for kk in range(n_full + 1):
            feats = np.column_stack([np.tile(np.eye(k)[kk],
                                             (len(xq), 1)), xq])
            s *= 1 - model.predict_proba(feats)[:, 1]
        return s
    return survival


def _horizon_metrics(survival, x, t, delta, horizon):
    """AUC/Brier of risk = 1 - S(horizon | x); censored-before-horizon
    excluded (IPCW-free simplification, noted in the P9 spec)."""
    event_by_h = (t <= horizon) & (delta > 0)
    known = event_by_h | (t > horizon)
    if event_by_h[known].mean() in (0.0, 1.0):
        return np.nan, np.nan
    risk = 1 - survival(x[known], horizon)
    y = event_by_h[known].astype(float)
    return (float(roc_auc_score(y, risk)),
            float(np.mean((risk - y) ** 2)))


def _min_events(edges, t, delta):
    idx = np.digitize(t[delta > 0], edges)
    return int(np.bincount(idx, minlength=len(edges) + 1).min())


# --------------------------------------------------------------------- #

def run(cfg):
    x, t, delta = _synthetic(cfg.n, cfg.seed, cfg.censor_scale)
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(len(t))
    n_te = int(cfg.test_size * len(t))
    te, tr = perm[:n_te], perm[n_te:]
    x_tr, t_tr, d_tr = x[tr], t[tr], delta[tr]
    horizons = np.quantile(t_tr[d_tr > 0], list(cfg.horizon_quantiles))
    ev_sorted = np.sort(t_tr[d_tr > 0])

    def fit_grid(arm, xs, ts, ds):
        return (_grid_ot(xs, ts, ds, cfg) if arm == "ot"
                else _grid_fixed(arm, ts, ds, cfg.n_intervals))

    perf_rows, stab_rows = [], []
    for arm in cfg.arms:
        start = time.perf_counter()
        edges = fit_grid(arm, x_tr, t_tr, d_tr)
        survival = _fit_hazard(edges, x_tr, t_tr, d_tr)
        for hq, h in zip(cfg.horizon_quantiles, horizons):
            auc, brier = _horizon_metrics(survival, x[te], t[te],
                                          delta[te], h)
            perf_rows.append(dict(
                arm=arm, horizon_q=float(hq), auc=auc, brier=brier,
                min_events=_min_events(edges, t_tr, d_tr),
                n_intervals=len(edges) + 1,
                fit_time=time.perf_counter() - start))

        m = int(cfg.stab_frac * len(tr))
        edge_sets = []
        for _ in range(cfg.n_resamples):
            sub = rng.permutation(len(tr))[:m]
            e_b = fit_grid(arm, x_tr[sub], t_tr[sub], d_tr[sub])
            edge_sets.append(np.searchsorted(ev_sorted, e_b)
                             / len(ev_sorted))            # rank scale
        pair_h = [_hausdorff(a, b)
                  for a, b in combinations(edge_sets, 2)]
        stab_rows.append(dict(
            arm=arm, hausdorff=float(np.nanmean(pair_h))))
        logger.info("%s: auc@median=%.4f min_events=%d hausdorff=%.4f",
                    arm, perf_rows[-2]["auc"] if len(horizons) > 1
                    else perf_rows[-1]["auc"],
                    perf_rows[-1]["min_events"], stab_rows[-1]["hausdorff"])

    common = dict(dataset="synthetic-weibull", seed=cfg.seed, n=cfg.n)
    for r in perf_rows + stab_rows:
        r.update(common)
    out = Path(cfg.out)
    tag = "surv_{}".format(cfg.seed)
    paths = [save_results(perf_rows, out / (tag + "_perf")),
             save_results(stab_rows, out / (tag + "_stab"))]
    logger.info("E-SURV: wrote %s", paths[0])
    return paths


@hydra.main(version_base=None, config_path="../conf", config_name="surv")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
