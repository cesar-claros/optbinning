"""Experiment E-TOK — learned numeric vocabularies (P9 note).

Foundation-model regime: numeric features enter as DISCRETE tokens
(embedding lookup -- lossy, no within-bin position), unlike Paper C's
continuous PLE vectors. All arms share the FeatureTokenTransformer
backbone fed one-hot tokens (its per-feature embedding weight IS the
vocabulary table): equal-width vocab, quantile vocab, and the OT vocab
trained end-to-end (soft assignment during training, hardened interval
lookup at eval -- the deployed tokenizer is a static edge table).
Evaluated over a train-size grid: sample efficiency is the
foundation-model question (which vocabulary wastes least capacity).

HPC:
    python experiments/run_tok.py -m dataset=gmsc,adult,higgs-small \
        'seed=range(0,5)' device=cuda out=outputs/tok
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
from sklearn.metrics import roc_auc_score               # noqa: E402
from torch import nn                                    # noqa: E402

from experiments import datasets                        # noqa: E402
from experiments.common import save_results             # noqa: E402
from experiments.paperc.backbones import FeatureTokenTransformer  # noqa: E402
from experiments.paperc.otlayer import MultiOTBinningLayer  # noqa: E402
from experiments.run_c3 import _quantile_transform      # noqa: E402

logger = logging.getLogger(__name__)


class VocabNet(nn.Module):
    """One-hot numeric vocabulary + FT backbone. Fixed arms bucketize
    against static edges; the ot arm learns the vocabulary end-to-end
    (soft assignment in training, interval one-hot at eval)."""

    def __init__(self, arm, n_features, n_bins, cfg, edges=None):
        super().__init__()
        self.arm = arm
        self.n_bins = n_bins
        if arm == "ot":
            self.ot = MultiOTBinningLayer(
                n_features, n_bins=n_bins,
                sinkhorn_iters=cfg.sinkhorn_iters)
        else:
            self.register_buffer(
                "edges", torch.as_tensor(np.stack(edges),
                                         dtype=torch.float32))
        self.backbone = FeatureTokenTransformer(
            n_features, n_bins, d_model=cfg.hidden,
            n_layers=cfg.ft_layers, n_heads=cfg.ft_heads)

    def tokens(self, x, eps=0.05):
        if self.arm == "ot":
            if self.training:
                return self.ot(x, eps=eps)               # soft vocab
            edges = self.ot.bin_edges().detach()
            idx = torch.searchsorted(edges, x.t().contiguous()).t()
        else:
            idx = torch.searchsorted(self.edges,
                                     x.t().contiguous()).t()
        return nn.functional.one_hot(
            idx.clamp(0, self.n_bins - 1), self.n_bins).float()

    def forward(self, x, eps=0.05):
        return self.backbone(self.tokens(x, eps))


def _fixed_edges(arm, xtr, n_bins):
    if arm == "width":
        return [np.linspace(xtr[:, j].min(), xtr[:, j].max(),
                            n_bins + 1)[1:-1] for j in range(xtr.shape[1])]
    return [np.unique(np.quantile(
        xtr[:, j], np.linspace(0, 1, n_bins + 1)[1:-1]))
        for j in range(xtr.shape[1])]


def _pad_edges(edges, n_bins):
    """searchsorted needs a rectangular edge tensor; pad with +inf."""
    out = np.full((len(edges), n_bins - 1), np.inf)
    for j, e in enumerate(edges):
        out[j, :len(e)] = e
    return list(out)


def _train_eval(arm, data, n_train, cfg):
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    sub = (rng.permutation(len(data["ytr"]))[:n_train]
           if n_train else np.arange(len(data["ytr"])))
    qtr, qte = _quantile_transform(data["xtr"][sub], data["xte"])
    ytr = data["ytr"][sub]

    edges = (None if arm == "ot" else
             _pad_edges(_fixed_edges(arm, qtr, cfg.n_bins), cfg.n_bins))
    net = VocabNet(arm, qtr.shape[1], cfg.n_bins, cfg, edges).to(device)
    if arm == "ot":
        xt = torch.as_tensor(qtr, dtype=torch.float32, device=device)
        net.ot.set_range(xt.min(dim=0).values, xt.max(dim=0).values)
    optim = torch.optim.AdamW(net.parameters(), lr=cfg.lr,
                              weight_decay=1e-5)
    bce = nn.BCEWithLogitsLoss()
    xtr = torch.as_tensor(qtr, dtype=torch.float32, device=device)
    ytr_t = torch.as_tensor(ytr, dtype=torch.float32, device=device)
    xte = torch.as_tensor(qte, dtype=torch.float32, device=device)

    start = time.perf_counter()
    net.train()
    n = len(ytr_t)
    for epoch in range(cfg.epochs):
        frac = epoch / max(cfg.epochs - 1, 1)
        eps = cfg.eps_start * (cfg.eps_end / cfg.eps_start) ** frac
        perm = torch.randperm(n, device=device)
        for lo in range(0, n, cfg.batch_size):
            idx = perm[lo:lo + cfg.batch_size]
            if len(idx) < cfg.n_bins * 4:
                continue
            loss = bce(net(xtr[idx], eps=eps), ytr_t[idx])
            optim.zero_grad()
            loss.backward()
            optim.step()
    fit_time = time.perf_counter() - start

    net.eval()
    with torch.no_grad():
        probs, used = [], torch.zeros(qtr.shape[1], cfg.n_bins,
                                      device=device)
        for lo in range(0, len(xte), cfg.batch_size):
            xb = xte[lo:lo + cfg.batch_size]
            probs.append(torch.sigmoid(net(xb)))
            used += net.tokens(xb).sum(dim=0)
        prob = torch.cat(probs).cpu().numpy()
        gap = np.nan
        if arm == "ot":
            net.train()
            soft = []
            for lo in range(0, len(xte), cfg.batch_size):
                soft.append(torch.sigmoid(
                    net(xte[lo:lo + cfg.batch_size], eps=cfg.eps_end)))
            net.eval()
            gap = float(roc_auc_score(
                data["yte"], torch.cat(soft).cpu().numpy())
                - roc_auc_score(data["yte"], prob))
    util = float((used > 0).float().mean())
    return dict(auc=float(roc_auc_score(data["yte"], prob)),
                soft_hard_gap=gap, utilization=util, fit_time=fit_time)


def run(cfg):
    ds = datasets.load(cfg.dataset, n=cfg.get("n", 12000),
                       seed=cfg.get("data_seed", 0)) \
        if str(cfg.dataset).startswith("synthetic") \
        else datasets.load(cfg.dataset)
    x = ds.X[ds.numerical].to_numpy(dtype=float)
    med = np.nanmedian(x, axis=0)
    x = np.where(np.isfinite(x), x, med)
    tr, te = datasets.split_indices(len(ds.y), cfg.test_size, cfg.seed)
    data = dict(xtr=x[tr], ytr=ds.y[tr], xte=x[te], yte=ds.y[te])

    rows = []
    for n_train in cfg.train_sizes:
        for arm in cfg.arms:
            try:
                row = _train_eval(arm, data, n_train, cfg)
                logger.info("n=%s %s: auc=%.4f util=%.2f",
                            n_train or "full", arm, row["auc"],
                            row["utilization"])
            except Exception:
                logger.exception("arm %s (n=%s) failed", arm, n_train)
                row = dict(auc=np.nan)
            row.update(dataset=ds.name, arm=arm, seed=cfg.seed,
                       n_train=n_train or len(data["ytr"]),
                       n_bins=cfg.n_bins)
            rows.append(row)

    path = save_results(rows, Path(cfg.out)
                        / "tok_{}_{}".format(cfg.dataset, cfg.seed))
    logger.info("E-TOK: wrote %s", path)
    return path


@hydra.main(version_base=None, config_path="../conf", config_name="tok")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
