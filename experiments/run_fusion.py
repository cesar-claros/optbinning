"""Experiment: a learnable transform UPSTREAM of the binning (P6 Sec. 4.3 on
real feature groups).

This is the one regime the two-stage optbinning pipeline cannot reach: when the
feature the bins serve is itself produced by a learnable map, learning that map
JOINTLY with the binning (end-to-end, through the differentiable OT layer) beats
fixing it first, and approaches an expensive oracle search. On classic credit
tables the effect is invisible because each column is binned as-is; here we make
it visible by fusing a natural, correlated feature GROUP (e.g. Taiwan's six
repayment-status months) into one auditable score and binning that.

Every arm is scored by the SAME certified metric -- the out-of-sample IV of the
exact optbinning of the fused score u = Xg @ w -- so the only thing that varies
is the fusion weight w. That isolates the value of learning w:

  equal        two-stage naive default: equal weights over the group
  pca1         two-stage: first principal component of the group
  best_single  two-stage: the single group feature with the best marginal IV
  endtoend     w learned by the differentiable OT-binning layer (ours), then
               the exact binning is re-fit at that w (isolates the fusion
               direction from OT-vs-exact binning quality)
  oracle       w = argmax train-IV by black-box (Powell) search -- the ceiling

The honest read is the gap ``endtoend - equal`` (did joint learning help over
the naive default?) and the ratio ``endtoend / oracle`` (how close to the best
attainable fusion). If the group's naive fusion is already near-optimal the gap
is small, which is itself a finding (see the language in the paper: the edge
grows with feature redundancy and with the complexity of the upstream map).

    python experiments/run_fusion.py -m dataset=taiwan group_key=pay,bill,payamt
    python experiments/run_fusion.py -m dataset=gmsc group_key=delinq
"""

# Cesar Claros <cesar.claros@outlook.com>
# Copyright (C) 2026

import logging
import sys

from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra                                            # noqa: E402
import torch                                            # noqa: E402
from omegaconf import DictConfig                        # noqa: E402
from scipy.optimize import minimize                     # noqa: E402
from sklearn.metrics import roc_auc_score               # noqa: E402
from torch import nn                                    # noqa: E402

from optbinning import OptimalBinning                   # noqa: E402

from experiments import datasets                        # noqa: E402
from experiments.common import save_results             # noqa: E402
from experiments.paperc.otlayer import (OTBinningLayer,  # noqa: E402
                                        pav_penalty, soft_iv)

logger = logging.getLogger(__name__)

# Natural correlated feature groups per dataset (a fusion has something to do
# only when the columns are redundant views of one underlying signal). Taiwan
# X6..X11 = PAY_0,PAY_2..PAY_6 (repayment status), X12..X17 = BILL_AMT1..6,
# X18..X23 = PAY_AMT1..6; GMSC's three past-due counts; HMEQ credit-stress and
# collateral-value blocks.
GROUPS = {
    "taiwan": {
        "pay": ["X6", "X7", "X8", "X9", "X10", "X11"],
        "bill": ["X12", "X13", "X14", "X15", "X16", "X17"],
        "payamt": ["X18", "X19", "X20", "X21", "X22", "X23"],
    },
    "gmsc": {
        "delinq": ["NumberOfTime30-59DaysPastDueNotWorse",
                   "NumberOfTimes90DaysLate",
                   "NumberOfTime60-89DaysPastDueNotWorse"],
        "afford": ["DebtRatio", "MonthlyIncome"],
    },
    "hmeq": {
        "stress": ["DEROG", "DELINQ", "NINQ"],
        "value": ["MORTDUE", "VALUE", "LOAN"],
    },
}


def _unit(w):
    """L2-normalize a fusion direction (optbinning is scale-invariant, so only
    the direction of ``w`` matters; a fixed norm makes weights comparable)."""
    w = np.asarray(w, dtype=float)
    n = np.linalg.norm(w)
    return w / n if n > 1e-12 else np.full_like(w, 1.0 / np.sqrt(len(w)))


def _fit_splits(u, y, n_bins, solver):
    """Exact optbinning cut points of a scalar composite ``u`` against ``y``."""
    ob = OptimalBinning(dtype="numerical", solver=solver,
                        max_n_bins=n_bins).fit(np.ascontiguousarray(u), y)
    return np.asarray(ob.splits, dtype=float)


def _iv_auc(splits, u_tr, y_tr, u_te, y_te):
    """Out-of-sample IV and AUC of a fixed binning: bins and WoE are defined on
    train, then IV is recomputed from the TEST class masses and the train WoE is
    used to score the test points (Jeffreys 0.5 smoothing avoids empty-bin
    blow-ups)."""
    tr_idx = np.digitize(u_tr, splits)
    te_idx = np.digitize(u_te, splits)
    n0tr, n1tr = max(int((y_tr == 0).sum()), 1), max(int((y_tr == 1).sum()), 1)
    n0te, n1te = max(int((y_te == 0).sum()), 1), max(int((y_te == 1).sum()), 1)
    iv = 0.0
    woe = {}
    for k in range(len(splits) + 1):
        mtr, mte = tr_idx == k, te_idx == k
        e1te = int((y_te[mte] == 1).sum())
        e0te = int(mte.sum()) - e1te
        p_te, q_te = (e0te + 0.5) / n0te, (e1te + 0.5) / n1te
        iv += (p_te - q_te) * np.log(p_te / q_te)
        e1tr = int((y_tr[mtr] == 1).sum())
        e0tr = int(mtr.sum()) - e1tr
        woe[k] = np.log(((e0tr + 0.5) / n0tr) / ((e1tr + 0.5) / n1tr))
    # score by risk: WoE ranks low-risk high, so negate to orient AUC above 0.5
    score = np.fromiter((-woe.get(int(k), 0.0) for k in te_idx), dtype=float,
                        count=len(te_idx))
    try:
        auc = float(roc_auc_score(y_te, score))
    except ValueError:                        # single occupied bin -> no order
        auc = float("nan")
    return float(iv), auc


def _train_iv(w, xg, y, n_bins, solver):
    """In-sample IV of the exact binning of ``u = Xg @ w`` (oracle objective)."""
    u = xg @ _unit(w)
    splits = _fit_splits(u, y, n_bins, solver)
    iv, _ = _iv_auc(splits, u, y, u, y)
    return iv


def _naive_weights(xg, y, kind, n_bins, solver):
    """Two-stage fusion directions that do not look at the joint binning."""
    k = xg.shape[1]
    if kind == "equal":
        return _unit(np.ones(k))
    if kind == "pca1":
        _, _, vt = np.linalg.svd(xg - xg.mean(axis=0), full_matrices=False)
        return _unit(vt[0])
    if kind == "best_single":
        ivs = [_train_iv(np.eye(k)[j], xg, y, n_bins, solver) for j in range(k)]
        return _unit(np.eye(k)[int(np.argmax(ivs))])
    raise ValueError("unknown naive kind: {}".format(kind))


def _oracle_weights(xg, y, n_bins, solver, inits, cfg):
    """Best fusion direction found by black-box (Powell) search on train IV,
    from several starts. Search on a capped subsample for speed; the returned
    ``w`` is scored on the full split like every other arm."""
    rng = np.random.default_rng(0)
    if cfg.oracle_search_n and len(y) > cfg.oracle_search_n:
        sub = rng.choice(len(y), int(cfg.oracle_search_n), replace=False)
        xs, ys = xg[sub], y[sub]
    else:
        xs, ys = xg, y
    best_w, best_iv = _unit(inits[0]), -np.inf
    for w0 in inits:
        res = minimize(lambda w: -_train_iv(w, xs, ys, n_bins, solver),
                       _unit(w0), method="Powell",
                       options={"maxiter": int(cfg.oracle_maxiter),
                                "maxfev": int(cfg.oracle_maxfev)})
        w = _unit(res.x)
        iv = _train_iv(w, xg, y, n_bins, solver)
        if iv > best_iv:
            best_iv, best_w = iv, w
    return best_w


class FusionNet(nn.Module):
    """Learnable linear fusion of a feature group, then a differentiable
    OT-binning of the fused scalar and a linear scorecard head. The fused score
    is standardized by detached batch statistics before the OT layer, so the
    binning sees a stable range and the fusion weight matters only up to
    direction (the monotone-rescaling invariance of P6 Sec. 4.3)."""

    def __init__(self, group_size: int, n_bins: int, sinkhorn_iters: int,
                 init_w=None) -> None:
        super().__init__()
        self.fuse = nn.Linear(group_size, 1, bias=False)
        if init_w is not None:
            with torch.no_grad():
                self.fuse.weight.copy_(torch.as_tensor(
                    _unit(init_w), dtype=torch.float32).reshape(1, -1))
        self.ot = OTBinningLayer(n_bins, sinkhorn_iters)
        self.ot.set_range(-3.0, 3.0)
        self.head = nn.Linear(n_bins, 1)

    def _norm_u(self, xg):
        u = self.fuse(xg).squeeze(-1)
        mu = u.mean().detach()
        sd = u.std().detach().clamp_min(1e-6)
        return ((u - mu) / sd).clamp(-3.5, 3.5)

    def forward(self, xg, eps):
        assign = self.ot(self._norm_u(xg), eps=eps)
        logit = self.head(assign).squeeze(-1)
        return logit, assign

    def weight(self):
        return _unit(self.fuse.weight.detach().cpu().numpy().reshape(-1))


def _endtoend_weights(xg_tr, y_tr, cfg, seed, init_w):
    """Learn the fusion direction jointly with the OT binning (BCE plus an
    IV/PAV auxiliary), annealing the temperature; start from ``init_w`` so any
    gain over the two-stage default is attributable to joint learning."""
    torch.manual_seed(seed)
    device = torch.device(cfg.device)
    xg = torch.as_tensor(xg_tr, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y_tr, dtype=torch.float32, device=device)
    net = FusionNet(xg.shape[1], cfg.n_bins, cfg.sinkhorn_iters,
                    init_w=init_w).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    bce = nn.BCEWithLogitsLoss()
    n = len(yt)
    for epoch in range(cfg.epochs):
        frac = epoch / max(cfg.epochs - 1, 1)
        eps = cfg.eps_start * (cfg.eps_end / cfg.eps_start) ** frac
        perm = torch.randperm(n, device=device)
        for lo in range(0, n, cfg.batch_size):
            idx = perm[lo:lo + cfg.batch_size]
            if len(idx) < cfg.n_bins * 4:
                continue
            logit, assign = net(xg[idx], eps=eps)
            loss = bce(logit, yt[idx])
            loss = loss - cfg.aux_iv * soft_iv(assign, yt[idx])
            loss = loss + cfg.aux_iv * pav_penalty(assign, yt[idx])
            optim.zero_grad()
            loss.backward()
            optim.step()
    return net.weight()


def run(cfg):
    dataset = str(cfg.dataset)
    if dataset not in GROUPS:
        raise SystemExit("run_fusion needs a dataset with defined groups: {}"
                         .format(sorted(GROUPS)))
    group_key = str(cfg.group_key)
    if group_key not in GROUPS[dataset]:
        raise SystemExit("group_key '{}' not in {} groups {}".format(
            group_key, dataset, sorted(GROUPS[dataset])))
    group = GROUPS[dataset][group_key]

    ds = datasets.load(dataset)
    xg_all = ds.X[group].to_numpy(dtype=float)
    # sentinels (HELOC -7/-8/-9, BAF -1) are median-imputed, not fused as
    # ordinary numbers; a fused group is a numeric combination, so no indicators
    if ds.special_codes:
        xg_all = np.where(np.isin(xg_all, list(ds.special_codes)), np.nan,
                          xg_all)
    xg_all = np.where(np.isfinite(xg_all), xg_all, np.nanmedian(xg_all, axis=0))
    solver = cfg.get("ob_solver", "mip")

    rows = []
    for seed in range(cfg.seed_offset, cfg.seed_offset + cfg.n_seeds):
        tr, te = datasets.split_indices(len(ds.y), cfg.test_size, seed)
        mu, sd = xg_all[tr].mean(axis=0), xg_all[tr].std(axis=0) + 1e-9
        xg_tr, xg_te = (xg_all[tr] - mu) / sd, (xg_all[te] - mu) / sd
        y_tr, y_te = ds.y[tr], ds.y[te]

        w_equal = _naive_weights(xg_tr, y_tr, "equal", cfg.n_bins, solver)
        w_pca1 = _naive_weights(xg_tr, y_tr, "pca1", cfg.n_bins, solver)
        w_best = _naive_weights(xg_tr, y_tr, "best_single", cfg.n_bins, solver)
        w_e2e = _endtoend_weights(xg_tr, y_tr, cfg, seed, init_w=w_equal)
        w_oracle = _oracle_weights(xg_tr, y_tr, cfg.n_bins, solver,
                                   [w_equal, w_pca1, w_best], cfg)

        for arm, w in [("equal", w_equal), ("pca1", w_pca1),
                       ("best_single", w_best), ("endtoend", w_e2e),
                       ("oracle", w_oracle)]:
            u_tr, u_te = xg_tr @ w, xg_te @ w
            splits = _fit_splits(u_tr, y_tr, cfg.n_bins, solver)
            iv, auc = _iv_auc(splits, u_tr, y_tr, u_te, y_te)
            iv_tr, _ = _iv_auc(splits, u_tr, y_tr, u_tr, y_tr)
            rows.append(dict(
                dataset=dataset, group=group_key, group_size=len(group),
                seed=seed, arm=arm, test_iv=iv, test_auc=auc, train_iv=iv_tr,
                n_bins=len(splits) + 1, w=";".join("{:.4f}".format(v)
                                                   for v in w)))

    _report(dataset, group_key, rows)
    out = Path(cfg.out) / "fusion_{}_{}_{}".format(dataset, group_key,
                                                   cfg.seed_offset)
    path = save_results(rows, out, cfg=cfg)
    logger.info("fusion: wrote %d rows -> %s", len(rows), path)
    return path


def _report(dataset, group_key, rows):
    """Print the per-arm test IV / AUC and the end-to-end gap over the naive
    default and its fraction of the oracle ceiling."""
    import pandas as pd
    df = pd.DataFrame(rows)
    order = ["equal", "pca1", "best_single", "endtoend", "oracle"]
    g = (df.groupby("arm").agg(
        test_iv=("test_iv", "mean"), test_iv_sd=("test_iv", "std"),
        test_auc=("test_auc", "mean"), train_iv=("train_iv", "mean"),
        n_bins=("n_bins", "mean")).reindex(order).round(4))
    print("\n===== feature-fusion, upstream of binning: {} / {} ({} seeds) ====="
          .format(dataset, group_key, df["seed"].nunique()))
    print(g.to_string())
    orc = g.loc["oracle", "test_iv"]
    eq = g.loc["equal", "test_iv"]
    e2e = g.loc["endtoend", "test_iv"]
    print("\nend-to-end vs two-stage(equal): {:+.4f} IV  ({:+.1%} relative)"
          .format(e2e - eq, (e2e - eq) / abs(eq) if eq else float("nan")))
    if orc:
        print("end-to-end / oracle:  {:.1%}     two-stage(equal) / oracle: {:.1%}"
              .format(e2e / orc, eq / orc))


@hydra.main(version_base=None, config_path="../conf", config_name="fusion")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
