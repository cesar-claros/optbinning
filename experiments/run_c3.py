"""Experiment C3/C4 — tokenizer benchmark (Paper C).

Compares numerical-feature tokenizers under a shared backbone: raw values,
quantile piecewise-linear encoding (PLE), target-aware PLE (frozen
optbinning bins), and the end-to-end OT-binning layer — plus a LightGBM
reference. ``backbone=linear`` is the C4 self-explaining scorecard head;
``backbone=ft`` is the camera-ready FT-Transformer comparison (per-feature
token embeddings + attention + CLS). ``token_mode=ple_interp`` is the
learned-knot PLE variant: spline tokens on the OT layer's own bin edges
(lossless + basis-rich; interval bins by construction).

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
from experiments.common import prepare_features, save_results  # noqa: E402
from experiments.paperc.backbones import FeatureTokenTransformer  # noqa: E402
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
                 sinkhorn_iters: int = 15, ft_layers: int = 2,
                 ft_heads: int = 4) -> None:
        super().__init__()
        self.arm = arm
        self.backbone = backbone
        self.token_mode = token_mode
        self.n_features = len(edges)
        if arm == "ot_ple":
            self.ot = MultiOTBinningLayer(len(edges), n_bins=n_bins,
                                          sinkhorn_iters=sinkhorn_iters)
            token_dim = n_bins + (1 if token_mode == "cumulative_plus_raw"
                                  else 0)
        elif arm in ("quantile_ple", "target_ple", "ot_frozen"):
            for i, e in enumerate(edges):
                self.register_buffer(f"edges_{i}",
                                     torch.as_tensor(e, dtype=torch.float32))
            token_dim = max(len(e) - 1 for e in edges)
            self._dims = [len(e) - 1 for e in edges]
        else:                                            # raw
            token_dim = 1
        self.token_dim = token_dim
        if backbone == "ft":
            # FT-Transformer pattern: per-feature tokens + attention +
            # CLS readout; consumes tokens as (batch, features, dim).
            self.head: nn.Module = FeatureTokenTransformer(
                self.n_features, token_dim, d_model=hidden,
                n_layers=ft_layers, n_heads=ft_heads)
        elif backbone == "linear":
            self.head = nn.Linear(self.n_features * token_dim, 1)
        else:
            in_dim = self.n_features * token_dim
            self.head = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def tokens(self, x: Tensor, eps: float,
               need_assign: bool = True) -> tuple[Tensor, Tensor | None]:
        """Per-feature tokens ``(batch, n_features, token_dim)`` and, for
        ot_ple, the soft assignment reused by the auxiliary loss (one
        Sinkhorn pass total; None when not needed -- ple_interp tokens
        skip Sinkhorn entirely outside auxiliary training)."""
        if self.arm == "ot_ple":
            if self.token_mode == "ple_interp":
                # learned-knot PLE: spline ramps on the layer's own bin
                # edges -- lossless AND basis-rich, the reconciliation
                # of the Sec. 5.4 residual. Bins are intervals by
                # construction; the audit table is the edge vector.
                tok = self.ot.interp_tokens(x)
                assign = self.ot(x, eps=eps) if need_assign else None
                return tok, assign
            assign = self.ot(x, eps=eps)
            tok = assign
            if self.token_mode.startswith("cumulative"):
                # soft analogue of the PLE ramp encoding: with a linear
                # head, cumulative tokens span (smoothed) monotone step
                # bases rather than localized bumps.
                tok = torch.cumsum(assign, dim=2)
            if self.token_mode == "cumulative_plus_raw":
                # lossless tokenization: step tokens destroy within-bin
                # position; the raw feature restores it as one extra
                # per-feature channel (flat layout is a permutation of
                # the earlier global concat -- same model class).
                tok = torch.cat([tok, x[:, :, None]], dim=2)
            return tok, assign
        cols = []
        for i in range(self.n_features):
            xi = x[:, i]
            if self.arm in ("quantile_ple", "target_ple", "ot_frozen"):
                enc = _ple_encode(xi, getattr(self, f"edges_{i}"))
                pad = self.token_dim - enc.shape[1]
                if pad:
                    enc = nn.functional.pad(enc, (0, pad))
            else:
                enc = xi[:, None]
            cols.append(enc)
        return torch.stack(cols, dim=1), None

    def forward(self, x: Tensor, eps: float = 0.05,
                need_assign: bool = True) -> tuple[Tensor, Tensor | None]:
        tok, assign = self.tokens(x, eps, need_assign)
        if self.backbone == "ft":
            return self.head(tok), assign
        return self.head(tok.reshape(len(x), -1)).squeeze(-1), assign


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


def _make_optim(net: nn.Module, backbone: str,
                cfg: DictConfig) -> torch.optim.Optimizer:
    if backbone == "ft":
        # transformers want a lower lr + decoupled weight decay than the
        # flat heads; lr_ft keeps mixed-backbone sweeps fair.
        return torch.optim.AdamW(
            net.parameters(), lr=cfg.get("lr_ft") or cfg.lr,
            weight_decay=cfg.get("ft_weight_decay", 1e-5))
    return torch.optim.Adam(net.parameters(), lr=cfg.lr)


def _run_epochs(net: nn.Module, optim: torch.optim.Optimizer,
                xtr: Tensor, ytr: Tensor, cfg: DictConfig,
                use_aux: bool) -> None:
    bce = nn.BCEWithLogitsLoss()
    n = len(ytr)
    for epoch in range(cfg.epochs):
        frac = epoch / max(cfg.epochs - 1, 1)
        eps = cfg.eps_start * (cfg.eps_end / cfg.eps_start) ** frac
        perm = torch.randperm(n, device=xtr.device)
        for lo in range(0, n, cfg.batch_size):
            idx = perm[lo:lo + cfg.batch_size]
            if len(idx) < cfg.n_bins * 4:
                continue
            logits, assign = net(xtr[idx], eps=eps, need_assign=use_aux)
            loss = bce(logits, ytr[idx])
            if assign is not None and use_aux:
                loss = loss - cfg.aux_iv * soft_iv_multi(assign, ytr[idx])
                loss = loss + cfg.aux_iv * pav_penalty_multi(assign,
                                                             ytr[idx])
            optim.zero_grad()
            loss.backward()
            optim.step()


def _train_eval(arm: str, backbone: str, data: dict,
                cfg: DictConfig) -> dict:
    device = torch.device(cfg.device)
    if arm in ("ot_ple", "ot_frozen") \
            and cfg.get("ot_input", "quantile") == "quantile":
        data = dict(data)
        data["xtr"], data["xte"] = _quantile_transform(data["xtr"],
                                                       data["xte"])
    xtr = torch.as_tensor(data["xtr"], dtype=torch.float32, device=device)
    ytr = torch.as_tensor(data["ytr"], dtype=torch.float32, device=device)
    xte = torch.as_tensor(data["xte"], dtype=torch.float32, device=device)

    kwargs = dict(token_mode=cfg.get("token_mode", "cumulative"),
                  sinkhorn_iters=cfg.get("sinkhorn_iters", 15),
                  ft_layers=cfg.get("ft_layers", 2),
                  ft_heads=cfg.get("ft_heads", 4))
    start = time.perf_counter()
    if arm == "ot_frozen":
        # two-stage control isolating JOINTNESS from the estimator:
        # stage 1 trains the layer end-to-end exactly as ot_ple; the
        # learned edges are then frozen and a fresh head is trained on
        # their PLE encoding. (ot_ple - ot_frozen) = value of joint
        # training; (ot_frozen - target_ple) = value of the smoothed
        # estimator at fixed two-stage protocol.
        placeholder = [np.linspace(0, 1, cfg.n_bins + 1)] * \
            data["xtr"].shape[1]
        pre = TokenizedNet("ot_ple", placeholder, cfg.n_bins, backbone,
                           cfg.hidden, **kwargs).to(device)
        pre.ot.set_range(xtr.min(dim=0).values, xtr.max(dim=0).values)
        _run_epochs(pre, _make_optim(pre, backbone, cfg), xtr, ytr, cfg,
                    use_aux=cfg.aux_iv > 0)
        eb = pre.ot.bin_edges().detach().cpu().numpy()
        edges = [np.concatenate(([data["xtr"][:, j].min() - 1e-6], eb[j],
                                 [data["xtr"][:, j].max() + 1e-6]))
                 for j in range(eb.shape[0])]
        net = TokenizedNet(arm, edges, cfg.n_bins, backbone, cfg.hidden,
                           **kwargs).to(device)
    else:
        edges = _edges_for_arm(arm, data["xtr"], data["ytr"], cfg.n_bins)
        net = TokenizedNet(arm, edges, cfg.n_bins, backbone, cfg.hidden,
                           **kwargs).to(device)
        if arm == "ot_ple":
            net.ot.set_range(xtr.min(dim=0).values,
                             xtr.max(dim=0).values)
    _run_epochs(net, _make_optim(net, backbone, cfg), xtr, ytr, cfg,
                use_aux=cfg.aux_iv > 0)
    fit_time = time.perf_counter() - start

    net.eval()
    with torch.no_grad():
        # chunked eval: exact (attention is across features, not batch);
        # a single 350k-row forward overflows the fused transformer
        # kernel's launch config on large test sets (BAF).
        probs = []
        for lo in range(0, len(xte), cfg.batch_size):
            logits, _ = net(xte[lo:lo + cfg.batch_size], eps=cfg.eps_end,
                            need_assign=False)
            probs.append(torch.sigmoid(logits))
        prob = torch.cat(probs).cpu().numpy()
    row = dict(auc=float(roc_auc_score(data["yte"], prob)),
               logloss=float(log_loss(data["yte"], prob)),
               fit_time=fit_time)
    if arm == "ot_ple":
        if net.token_mode == "ple_interp":
            # interval bins by construction: contiguity is structural
            # and the audit table is the learned edge vector itself.
            edges_np = net.ot.bin_edges().detach().cpu().numpy()
            row["contiguous_frac"] = 1.0
            row["mean_n_cuts"] = float(np.mean(
                [len(np.unique(np.round(e, 6))) for e in edges_np]))
        else:
            hard = net.ot.harden(xtr)
            row["contiguous_frac"] = float(np.mean([h["contiguous"]
                                                    for h in hard]))
            row["mean_n_cuts"] = float(np.mean([len(h["cuts"])
                                                for h in hard]))
    elif arm == "ot_frozen":
        row["contiguous_frac"] = 1.0
        row["mean_n_cuts"] = float(np.mean(
            [len(np.unique(np.round(e[1:-1], 6))) for e in edges]))
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
    x = prepare_features(ds, cfg.get("special_handling", "ignore"))

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
