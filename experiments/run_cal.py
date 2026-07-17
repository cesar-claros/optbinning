"""Experiment E-CAL — the OT layer as a trainable isotonic calibrator.

Post-hoc calibration of a base classifier: uncalibrated, Platt,
isotonic, equal-frequency histogram binning, and the OT calibrator
(single-feature layer on scores, structurally monotone rates via
cumulative softplus, annealed NLL training, hardened to an auditable
step table with PAV-pooled empirical rates). Reports ECE both ways
(equal-mass and equal-width -- ECE is bin-sensitive), Brier, NLL, and
calibration-map stability over calibration-set resamples (sd of the
deployed map on a fixed score grid; edge Hausdorff for binned arms).

Torch-free smoke (no ot arm):
    python experiments/run_cal.py dataset=synthetic-smooth \
        'arms=[uncal,platt,isotonic,hist]' base=logreg
HPC:
    python experiments/run_cal.py -m dataset=gmsc,taiwan,adult \
        'seed=range(0,5)' device=cuda out=outputs/cal
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
from sklearn.isotonic import IsotonicRegression         # noqa: E402
from sklearn.linear_model import LogisticRegression     # noqa: E402
from sklearn.metrics import brier_score_loss, log_loss  # noqa: E402

from experiments import datasets                        # noqa: E402
from experiments.common import save_results             # noqa: E402
from experiments.common import prepare_features         # noqa: E402
from experiments.paperc.pav import _pav_blocks          # noqa: E402
from experiments.run_s1 import _hausdorff               # noqa: E402

logger = logging.getLogger(__name__)

_GRID = np.linspace(0.01, 0.99, 99)


# --------------------------------------------------------------------- #
# Calibrator arms: fit(s, y) -> deployed map p(s); edges or None
# --------------------------------------------------------------------- #

def _fit_platt(s, y, cfg):
    z = np.log(np.clip(s, 1e-6, 1 - 1e-6) / np.clip(1 - s, 1e-6, 1))
    model = LogisticRegression(max_iter=1000).fit(z[:, None], y)

    def cal(t):
        zt = np.log(np.clip(t, 1e-6, 1 - 1e-6) / np.clip(1 - t, 1e-6, 1))
        return model.predict_proba(zt[:, None])[:, 1]
    return cal, None


def _fit_isotonic(s, y, cfg):
    iso = IsotonicRegression(out_of_bounds="clip",
                             y_min=1e-6, y_max=1 - 1e-6).fit(s, y)
    jumps = np.unique(iso.predict(_GRID))
    return iso.predict, np.asarray(jumps[:0])   # no comparable edge set


def _step_calibrator(edges, s_cal, y_cal):
    """Empirical event rate per bin, PAV-pooled to enforce monotonicity;
    the deployed calibrator is the (edges, rates) table."""
    idx = np.digitize(s_cal, edges)
    k = len(edges) + 1
    e = np.bincount(idx, weights=y_cal, minlength=k) + 0.5
    n = np.bincount(idx, minlength=k) + 1.0
    rate = e / n
    mass = n / n.sum()
    pooled = rate.copy()
    for block in _pav_blocks(rate, mass):
        b = np.asarray(block)
        pooled[b] = float((rate[b] * mass[b]).sum() / mass[b].sum())

    def cal(t):
        return pooled[np.digitize(t, edges)]
    return cal


def _fit_hist(s, y, cfg):
    edges = np.unique(np.quantile(
        s, np.linspace(0, 1, cfg.n_cal_bins + 1)[1:-1]))
    return _step_calibrator(edges, s, y), edges


def _fit_ot(s, y, cfg):
    """Single-feature OT layer on RANK-transformed scores (base-model
    scores concentrate heavily -- e.g. PDs near zero -- and range-space
    bin geometry is quantile-blind under concentration, the Paper C
    Sec. 5.4 failure mode; the v1 run of this experiment reproduced it
    as edge wander in empty score regions). Monotone rates by cumulative
    softplus; edges map back through the calibration quantile function;
    hardened to a step table."""
    import torch
    from experiments.paperc.otlayer import MultiOTBinningLayer

    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    m = cfg.n_cal_bins
    layer = MultiOTBinningLayer(1, n_bins=m,
                                sinkhorn_iters=cfg.sinkhorn_iters)
    layer.set_range(torch.zeros(1), torch.ones(1))
    rho0 = torch.nn.Parameter(torch.tensor(-2.0))
    eta = torch.nn.Parameter(torch.full((m - 1,), -2.0))
    layer.to(device)
    rho0.data = rho0.data.to(device)
    eta.data = eta.data.to(device)

    srt = np.sort(s)
    u = np.searchsorted(srt, s, side="right") / len(s)
    st = torch.as_tensor(u, dtype=torch.float32, device=device)[:, None]
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    optim = torch.optim.Adam(
        list(layer.parameters()) + [rho0, eta], lr=cfg.lr)
    for step in range(cfg.steps):
        frac = step / max(cfg.steps - 1, 1)
        eps = cfg.eps_start * (cfg.eps_end / cfg.eps_start) ** frac
        rates = torch.sigmoid(torch.cat(
            [rho0[None], rho0 + torch.cumsum(
                torch.nn.functional.softplus(eta), 0)]))
        assign = layer(st, eps=eps)[:, 0, :]
        p = (assign * rates[None]).sum(1).clamp(1e-6, 1 - 1e-6)
        loss = torch.nn.functional.binary_cross_entropy(p, yt)
        optim.zero_grad()
        loss.backward()
        optim.step()

    edges_rank = layer.bin_edges().detach().cpu().numpy()[0]
    edges = np.quantile(s, np.clip(edges_rank, 0, 1))
    return _step_calibrator(edges, s, y), edges


_ARMS = {"platt": _fit_platt, "isotonic": _fit_isotonic,
         "hist": _fit_hist, "ot": _fit_ot}


# --------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------- #

def _ece(p, y, n_bins, scheme):
    edges = (np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)[1:-1]))
             if scheme == "eqmass"
             else np.linspace(0, 1, n_bins + 1)[1:-1])
    idx = np.digitize(p, edges)
    k = len(edges) + 1
    n = np.bincount(idx, minlength=k)
    conf = np.bincount(idx, weights=p, minlength=k)
    acc = np.bincount(idx, weights=y, minlength=k)
    mask = n > 0
    return float(np.sum(np.abs(conf[mask] - acc[mask])) / len(p))


def _perf(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return dict(ece_eqmass=_ece(p, y, 15, "eqmass"),
                ece_eqwidth=_ece(p, y, 15, "eqwidth"),
                brier=float(brier_score_loss(y, p)),
                nll=float(log_loss(y, p)))


# --------------------------------------------------------------------- #

def _base_scores(cfg, x, y, tr_base, rng):
    model = None
    if cfg.base == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                   random_state=cfg.seed, verbose=-1)
        except (ImportError, OSError):
            # missing wheel or missing libgomp runtime: fall back so the
            # calibration comparison still runs (base model is not the
            # object under study).
            logger.warning("lightgbm unavailable; falling back to "
                           "logistic base model")
    if model is None:
        model = LogisticRegression(max_iter=1000)
    model.fit(x[tr_base], y[tr_base])
    base_used = type(model).__name__
    return (lambda idx: model.predict_proba(x[idx])[:, 1]), base_used


def run(cfg):
    ds = datasets.load(cfg.dataset, n=cfg.get("n", 12000),
                       seed=cfg.get("data_seed", 0)) \
        if str(cfg.dataset).startswith("synthetic") \
        else datasets.load(cfg.dataset)
    x = prepare_features(ds, cfg.get("special_handling", "ignore"))
    y = ds.y

    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(len(y))
    n1, n2 = int(0.4 * len(y)), int(0.7 * len(y))
    tr_base, cal_idx, te = perm[:n1], perm[n1:n2], perm[n2:]
    score, base_used = _base_scores(cfg, x, y, tr_base, rng)
    s_cal, y_cal = score(cal_idx), y[cal_idx]
    s_te, y_te = score(te), y[te]

    perf_rows, stab_rows = [], []
    for arm in cfg.arms:
        start = time.perf_counter()
        if arm == "uncal":
            cal, edges = (lambda t: t), None
        else:
            cal, edges = _ARMS[arm](s_cal, y_cal, cfg)
        row = _perf(cal(s_te), y_te)
        row.update(arm=arm, fit_time=time.perf_counter() - start,
                   n_edges=len(edges) if edges is not None else np.nan)
        perf_rows.append(row)

        if arm == "uncal":
            continue
        maps, edge_sets = [], []
        m = int(cfg.stab_frac * len(cal_idx))
        for _ in range(cfg.n_resamples):
            sub = rng.permutation(len(cal_idx))[:m]
            c_b, e_b = _ARMS[arm](s_cal[sub], y_cal[sub], cfg)
            maps.append(c_b(_GRID))
            if e_b is not None and len(e_b):
                edge_sets.append(np.asarray(e_b))
        sd = np.std(np.array(maps), axis=0)
        curve_sd = float(sd.mean())
        # density-weighted variant: instability where scores actually
        # live (uniform grids overweight empty score regions -- the v1
        # metric artifact). Weight = calibration-score mass near each
        # grid point; applied identically to every arm.
        srt = np.sort(s_cal)
        h = (_GRID[1] - _GRID[0]) / 2
        w = (np.searchsorted(srt, _GRID + h)
             - np.searchsorted(srt, _GRID - h)).astype(float)
        curve_sd_w = float((sd * w).sum() / max(w.sum(), 1.0))
        # edge movement on the calibration-score rank scale (score-unit
        # Hausdorff is dominated by score concentration).
        rank_sets = [np.searchsorted(srt, e) / len(srt)
                     for e in edge_sets]
        pair_h = [_hausdorff(a, b)
                  for a, b in combinations(rank_sets, 2)]
        stab_rows.append(dict(
            arm=arm, curve_sd=curve_sd, curve_sd_w=curve_sd_w,
            hausdorff=float(np.nanmean(pair_h)) if pair_h else np.nan))
        logger.info("%s: ece=%.4f brier=%.4f curve_sd_w=%.4f",
                    arm, row["ece_eqmass"], row["brier"], curve_sd_w)

    common = dict(dataset=str(cfg.dataset), base=base_used,
                  seed=cfg.seed, n_cal=len(cal_idx))
    for r in perf_rows + stab_rows:
        r.update(common)
    out = Path(cfg.out)
    tag = "cal_{}_{}".format(cfg.dataset, cfg.seed)
    paths = [save_results(perf_rows, out / (tag + "_perf")),
             save_results(stab_rows, out / (tag + "_stab"))]
    logger.info("E-CAL: wrote %s", paths[0])
    return paths


@hydra.main(version_base=None, config_path="../conf", config_name="cal")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
