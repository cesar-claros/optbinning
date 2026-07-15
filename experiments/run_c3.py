"""Experiment C3/C4 — tokenizer benchmark (Paper C).

Compares numerical-feature tokenizers under a shared backbone: raw values,
quantile piecewise-linear encoding (PLE), target-aware PLE (frozen
optbinning bins), and the end-to-end OT-binning layer — plus a LightGBM
reference. ``backbone=linear`` is the C4 self-explaining scorecard head.

Local smoke test:
    python experiments/run_c3.py dataset=synthetic-smooth epochs=8 \
        "arms=[raw,quantile_ple,ot_ple]" "backbones=[linear]"
HPC:
    python experiments/run_c3.py -m hydra/launcher=submitit_slurm \
        dataset=german,taiwan,gmsc seed=range(0,5)
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
import torch                                            # noqa: E402
from omegaconf import DictConfig                        # noqa: E402
from sklearn.metrics import log_loss, roc_auc_score     # noqa: E402
from torch import Tensor, nn                            # noqa: E402

from optbinning import OptimalBinning                   # noqa: E402

from experiments import datasets                        # noqa: E402
from experiments.common import save_results             # noqa: E402
from experiments.paperc.otlayer import (MultiOTBinningLayer,  # noqa: E402
                                        pav_penalty_multi, soft_iv_multi)

logger = logging.getLogger(__name__)


def _ple_encode(x: Tensor, edges: Tensor) -> Tensor:
    """Piecewise-linear encoding of Gorishniy et al. (NeurIPS 2022)."""
    lo = edges[:-1]
    width = (edges[1:] - lo).clamp_min(1e-9)
    return ((x[:, None] - lo[None, :]) / width[None, :]).clamp(0.0, 1.0)


class TokenizedNet(nn.Module):
    """Per-feature tokenizer + shared backbone binary classifier."""

    def __init__(self, arm: str, edges: list[np.ndarray], n_bins: int,
                 backbone: str, hidden: int,
                 token_mode: str = "cumulative",
                 sinkhorn_iters: int = 15) -> None:
        super().__init__()
        self.arm = arm
        self.token_mode = token_mode
        self.n_features = len(edges)
        if arm == "ot_ple":
            self.ot = MultiOTBinningLayer(len(edges), n_bins=n_bins,
                                          sinkhorn_iters=sinkhorn_iters)
            token_dim = n_bins + (1 if token_mode == "cumulative_plus_raw"
                                  else 0)
        elif arm in ("quantile_ple", "target_ple"):
            for i, e in enumerate(edges):
                self.register_buffer(f"edges_{i}",
                                     torch.as_tensor(e, dtype=torch.float32))
            token_dim = max(len(e) - 1 for e in edges)
            self._dims = [len(e) - 1 for e in edges]
        else:                                            # raw
            token_dim = 1
        in_dim = self.n_features * token_dim
        self.token_dim = token_dim
        if backbone == "linear":
            self.head = nn.Linear(in_dim, 1)
        else:
            self.head = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def tokens(self, x: Tensor,
               eps: float) -> tuple[Tensor, Tensor | None]:
        """Token matrix and, for ot_ple, the raw soft assignment (reused
        by the auxiliary loss to avoid a second Sinkhorn pass)."""
        if self.arm == "ot_ple":
            assign = self.ot(x, eps=eps)
            tok = assign
            if self.token_mode.startswith("cumulative"):
                # soft analogue of the PLE ramp encoding: with a linear
                # head, cumulative tokens span (smoothed) monotone step
                # bases rather than localized bumps.
                tok = torch.cumsum(assign, dim=2)
            tok = tok.reshape(len(x), -1)
            if self.token_mode == "cumulative_plus_raw":
                # lossless tokenization: step tokens destroy within-bin
                # position (PLE keeps it via interpolation); appending
                # the raw feature restores it at one extra dim/feature.
                tok = torch.cat([tok, x], dim=1)
            return tok, assign
        cols = []
        for i in range(self.n_features):
            xi = x[:, i]
            if self.arm in ("quantile_ple", "target_ple"):
                enc = _ple_encode(xi, getattr(self, f"edges_{i}"))
                pad = self.token_dim - enc.shape[1]
                if pad:
                    enc = nn.functional.pad(enc, (0, pad))
                cols.append(enc)
            else:
                cols.append(xi[:, None])
        return torch.cat(cols, dim=1), None

    def forward(self, x: Tensor,
                eps: float = 0.05) -> tuple[Tensor, Tensor | None]:
        tok, assign = self.tokens(x, eps)
        return self.head(tok).squeeze(-1), assign


def _edges_for_arm(arm: str, xtr: np.ndarray, ytr: np.ndarray,
                   n_bins: int) -> list[np.ndarray]:
    edges = []
    for i in range(xtr.shape[1]):
        col = xtr[:, i]
        if arm == "target_ple":
            optb = OptimalBinning(dtype="numerical", solver="cp",
                                  max_n_bins=n_bins).fit(col, ytr)
            inner = np.asarray(optb.splits, dtype=float)
        else:
            inner = np.unique(np.quantile(
                col, np.linspace(0, 1, n_bins + 1)[1:-1]))
        edges.append(np.concatenate(([col.min() - 1e-6], inner,
                                     [col.max() + 1e-6])))
    return edges


def _quantile_transform(xtr: np.ndarray,
                        xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature train-ECDF transform. Puts the OT layer's bin geometry
    in rank space: range-based bin placement is quantile-blind under
    heavy-tailed features (the GMSC failure mode), while ranks give the
    layer the same footing quantile-PLE gets from its edges. Cuts map
    back through the train quantile function for the audit table."""
    qtr = np.empty_like(xtr)
    qte = np.empty_like(xte)
    for j in range(xtr.shape[1]):
        srt = np.sort(xtr[:, j])
        qtr[:, j] = np.searchsorted(srt, xtr[:, j], side="right") / len(srt)
        qte[:, j] = np.searchsorted(srt, xte[:, j], side="right") / len(srt)
    return qtr, qte


def _train_eval(arm: str, backbone: str, data: dict,
                cfg: DictConfig) -> dict:
    device = torch.device(cfg.device)
    if arm == "ot_ple" and cfg.get("ot_input", "quantile") == "quantile":
        data = dict(data)
        data["xtr"], data["xte"] = _quantile_transform(data["xtr"],
                                                       data["xte"])
    xtr = torch.as_tensor(data["xtr"], dtype=torch.float32, device=device)
    ytr = torch.as_tensor(data["ytr"], dtype=torch.float32, device=device)
    xte = torch.as_tensor(data["xte"], dtype=torch.float32, device=device)

    edges = _edges_for_arm(arm, data["xtr"], data["ytr"], cfg.n_bins)
    net = TokenizedNet(arm, edges, cfg.n_bins, backbone, cfg.hidden,
                       token_mode=cfg.get("token_mode", "cumulative"),
                       sinkhorn_iters=cfg.get("sinkhorn_iters", 15))
    net.to(device)
    if arm == "ot_ple":
        net.ot.set_range(xtr.min(dim=0).values, xtr.max(dim=0).values)
    optim = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    bce = nn.BCEWithLogitsLoss()

    start = time.perf_counter()
    n = len(ytr)
    for epoch in range(cfg.epochs):
        frac = epoch / max(cfg.epochs - 1, 1)
        eps = cfg.eps_start * (cfg.eps_end / cfg.eps_start) ** frac
        perm = torch.randperm(n, device=device)
        for lo in range(0, n, cfg.batch_size):
            idx = perm[lo:lo + cfg.batch_size]
            if len(idx) < cfg.n_bins * 4:
                continue
            logits, assign = net(xtr[idx], eps=eps)
            loss = bce(logits, ytr[idx])
            if assign is not None and cfg.aux_iv > 0:
                loss = loss - cfg.aux_iv * soft_iv_multi(assign, ytr[idx])
                loss = loss + cfg.aux_iv * pav_penalty_multi(assign,
                                                             ytr[idx])
            optim.zero_grad()
            loss.backward()
            optim.step()
    fit_time = time.perf_counter() - start

    with torch.no_grad():
        logits, _ = net(xte, eps=cfg.eps_end)
        prob = torch.sigmoid(logits).cpu().numpy()
    row = dict(auc=float(roc_auc_score(data["yte"], prob)),
               logloss=float(log_loss(data["yte"], prob)),
               fit_time=fit_time)
    if arm == "ot_ple":
        hard = net.ot.harden(xtr)
        row["contiguous_frac"] = float(np.mean([h["contiguous"]
                                                for h in hard]))
        row["mean_n_cuts"] = float(np.mean([len(h["cuts"]) for h in hard]))
    return row


def _lightgbm_row(data: dict, seed: int) -> dict:
    from lightgbm import LGBMClassifier
    start = time.perf_counter()
    model = LGBMClassifier(n_estimators=300, learning_rate=0.05,
                           random_state=seed, verbose=-1)
    model.fit(data["xtr"], data["ytr"])
    prob = model.predict_proba(data["xte"])[:, 1]
    return dict(auc=float(roc_auc_score(data["yte"], prob)),
                logloss=float(log_loss(data["yte"], prob)),
                fit_time=time.perf_counter() - start)


def run(cfg: DictConfig) -> Path:
    ds = datasets.load(cfg.dataset, n=cfg.get("n", 6000),
                       seed=cfg.get("data_seed", 0)) \
        if str(cfg.dataset).startswith("synthetic") \
        else datasets.load(cfg.dataset)
    x = ds.X[ds.numerical].to_numpy(dtype=float)
    med = np.nanmedian(x, axis=0)
    x = np.where(np.isfinite(x), x, med)

    rows = []
    tr, te = datasets.split_indices(len(ds.y), cfg.test_size, cfg.seed)
    mu, sd = x[tr].mean(axis=0), x[tr].std(axis=0) + 1e-9
    data = dict(xtr=(x[tr] - mu) / sd, ytr=ds.y[tr],
                xte=(x[te] - mu) / sd, yte=ds.y[te])

    torch.manual_seed(cfg.seed)
    for arm in cfg.arms:
        if arm == "lightgbm":
            try:
                row = _lightgbm_row(data, cfg.seed)
            except Exception:
                logger.exception("arm %s failed", arm)
                row = dict(auc=np.nan, logloss=np.nan, fit_time=np.nan)
            row.update(dataset=ds.name, arm=arm, backbone="gbdt",
                       seed=cfg.seed)
            rows.append(row)
            continue
        for backbone in cfg.backbones:
            try:
                row = _train_eval(arm, backbone, data, cfg)
                logger.info("%s/%s: auc=%.4f", arm, backbone, row["auc"])
            except Exception:
                logger.exception("arm %s/%s failed", arm, backbone)
                row = dict(auc=np.nan, logloss=np.nan, fit_time=np.nan)
            row.update(dataset=ds.name, arm=arm, backbone=backbone,
                       seed=cfg.seed)
            rows.append(row)

    out = Path(cfg.out) / f"c3_{cfg.dataset}_{cfg.seed}"
    path = save_results(rows, out)
    logger.info("C3: wrote %d rows -> %s", len(rows), path)
    return path


@hydra.main(version_base=None, config_path="../conf", config_name="c3")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
