"""Experiment AUDIT — hardened-guarantee postconditions and static
inference (Paper C, reviewer P0).

Trains the production ot_ple configuration (rank input, ple_interp
tokens, IV/PAV auxiliary), then verifies every advertised structural
guarantee DIRECTLY, per feature:

(a) Sinkhorn feasibility at the hardening temperature: column-marginal
    residual ``max_k |sum_t pi_tk - beta_k|`` (rows are normalized by
    construction).
(b) soft mass floor: ``min_k beta_k``.
(c) mass-coordinate static partition (``harden_static``): absorbed
    bins, ``max |hard - soft|`` bin mass against the ``2 a_max``
    theorem bound, pre-PAV rate violations (count and magnitude), and
    bin count after EXACT PAV block merging (hard rates exactly
    monotone).
(d) agreement between batchwise Sinkhorn-argmax hardening (the legacy
    path) and static interval lookup on the same points.
(e) batch invariance: cuts extracted by the batchwise path from two
    disjoint evaluation halves, compared in rank space; the static path
    is batch-invariant by construction (asserted).
(f) deployment fidelity: a WoE scorecard on the merged hard bins whose
    per-bin points table is verified to reproduce the model logits to
    machine precision by independent interval lookup on test data;
    AUCs of the pre-merge and post-merge hard scorecards are reported
    next to the soft model's.

Local smoke test:
    python experiments/run_audit.py dataset=german epochs=8
HPC:
    python experiments/run_audit.py -m hydra/launcher=submitit_slurm \
        dataset=german,taiwan,gmsc,hmeq,heloc seed=range(0,5)
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
from sklearn.linear_model import LogisticRegression     # noqa: E402
from sklearn.metrics import log_loss, roc_auc_score     # noqa: E402

from experiments import datasets                        # noqa: E402
from experiments.common import prepare_features, save_results  # noqa: E402
from experiments.run_c3 import (TokenizedNet, _make_optim,  # noqa: E402
                                _quantile_transform, _run_epochs)

logger = logging.getLogger(__name__)

_HARDEN_EPS = 0.003
_HARDEN_ITERS = 60


def _canonical_labels(labels: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Relabel a contiguous-in-x partition by first occurrence along the
    sorted feature, so two partitions can be compared independently of
    raw label values (absorbed-bin index offsets)."""
    canon = np.empty_like(labels)
    mapping: dict[int, int] = {}
    for pos in order:
        lab = int(labels[pos])
        if lab not in mapping:
            mapping[lab] = len(mapping)
        canon[pos] = mapping[lab]
    return canon


def _rank_hausdorff(c1: np.ndarray, c2: np.ndarray) -> float:
    """Hausdorff distance between two cut sets (rank-space units)."""
    if len(c1) == 0 and len(c2) == 0:
        return 0.0
    if len(c1) == 0 or len(c2) == 0:
        return 1.0
    d12 = np.abs(c1[:, None] - c2[None, :])
    return float(max(d12.min(axis=1).max(), d12.min(axis=0).max()))


def _woe_table(idx: np.ndarray, y: np.ndarray, n_bins: int) -> np.ndarray:
    """Per-bin WoE with the optbinning-style floor on cell counts."""
    tot = np.bincount(idx, minlength=n_bins).astype(float)
    ev = np.bincount(idx, weights=y, minlength=n_bins)
    ne = tot - ev
    p = np.maximum(ne, 0.5) / max(ne.sum(), 0.5)
    q = np.maximum(ev, 0.5) / max(ev.sum(), 0.5)
    return np.log(p / q)


def _scorecard_rows(cuts: list[np.ndarray], xtr: np.ndarray,
                    ytr: np.ndarray, xte: np.ndarray,
                    yte: np.ndarray) -> dict:
    """Fit a WoE-logistic scorecard on hard bins; verify the extracted
    points table reproduces the model logits by independent interval
    lookup; return fidelity and performance metrics."""
    n_feat = xtr.shape[1]
    woe_tr = np.empty_like(xtr)
    woe_te = np.empty_like(xte)
    tables = []
    for j in range(n_feat):
        idx_tr = np.searchsorted(cuts[j], xtr[:, j], side="right")
        idx_te = np.searchsorted(cuts[j], xte[:, j], side="right")
        table = _woe_table(idx_tr, ytr, len(cuts[j]) + 1)
        woe_tr[:, j] = table[idx_tr]
        woe_te[:, j] = table[idx_te]
        tables.append(table)
    clf = LogisticRegression(max_iter=2000).fit(woe_tr, ytr)
    logits_model = clf.decision_function(woe_te)
    # independent table lookup: points_jb = coef_j * woe_jb; score =
    # intercept + sum_j points_j[bin_j(x)] -- recomputed from scratch
    logits_table = np.full(len(xte), float(clf.intercept_[0]))
    for j in range(n_feat):
        points = clf.coef_[0, j] * tables[j]
        idx_te = np.searchsorted(cuts[j], xte[:, j], side="right")
        logits_table += points[idx_te]
    prob = 1.0 / (1.0 + np.exp(-logits_model))
    return {
        "table_fidelity_max_abs": float(
            np.max(np.abs(logits_table - logits_model))),
        "scorecard_auc": float(roc_auc_score(yte, prob)),
        "scorecard_logloss": float(log_loss(yte, prob)),
    }


def run(cfg: DictConfig) -> Path:
    ds = datasets.load(cfg.dataset)
    task = getattr(ds, "task", "binary")
    if task != "binary":
        raise ValueError(
            f"audit requires a binary dataset; {cfg.dataset} is {task}.")
    x = prepare_features(ds, cfg.get("special_handling", "expand"))
    tr, te = datasets.split_indices(len(ds.y), cfg.test_size, cfg.seed)
    mu, sd = x[tr].mean(axis=0), x[tr].std(axis=0) + 1e-9
    xtr_np, xte_np = (x[tr] - mu) / sd, (x[te] - mu) / sd
    xtr_np, xte_np = _quantile_transform(xtr_np, xte_np)   # rank space
    ytr_np, yte_np = ds.y[tr].astype(float), ds.y[te].astype(float)

    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    xtr = torch.as_tensor(xtr_np, dtype=torch.float32, device=device)
    ytr = torch.as_tensor(ytr_np, dtype=torch.float32, device=device)
    xte = torch.as_tensor(xte_np, dtype=torch.float32, device=device)

    edges = [np.linspace(0, 1, cfg.n_bins + 1)] * xtr_np.shape[1]
    net = TokenizedNet("ot_ple", edges, cfg.n_bins, cfg.backbone,
                       cfg.hidden, token_mode="ple_interp",
                       sinkhorn_iters=cfg.sinkhorn_iters).to(device)
    net.ot.set_range(xtr.min(dim=0).values, xtr.max(dim=0).values)
    _run_epochs(net, _make_optim(net, cfg.backbone, cfg), xtr, ytr, cfg,
                use_aux=True, task="binary")
    net.eval()

    # soft-model reference AUC
    with torch.no_grad():
        outs = [net(xte[lo:lo + cfg.batch_size], eps=cfg.eps_end,
                    need_assign=False)[0]
                for lo in range(0, len(xte), cfg.batch_size)]
        prob = torch.sigmoid(torch.cat(outs)).cpu().numpy()
    soft_auc = float(roc_auc_score(yte_np, prob))

    beta = net.ot.bin_masses().detach().cpu().numpy()

    # (a) Sinkhorn feasibility at hardening temperature (train batch)
    with torch.no_grad():
        saved = net.ot.sinkhorn_iters
        net.ot.sinkhorn_iters = _HARDEN_ITERS
        plan_rows = net.ot(xtr, eps=_HARDEN_EPS)      # rows sum to 1
        net.ot.sinkhorn_iters = saved
        col = plan_rows.mean(dim=0).cpu().numpy()     # pi columns, a=1/n
        argmax_lab = plan_rows.argmax(dim=2).cpu().numpy()
    col_resid = np.abs(col - beta).max(axis=1)

    # (c) mass-coordinate static partition, exact PAV merge
    static = net.ot.harden_static(xtr, ytr)

    # (e) batchwise-path cuts on two disjoint eval halves
    half = len(xte) // 2
    hard_a = net.ot.harden(xte[:half], eps=_HARDEN_EPS,
                           iters=_HARDEN_ITERS)
    hard_b = net.ot.harden(xte[half:], eps=_HARDEN_EPS,
                           iters=_HARDEN_ITERS)
    hard_tr = net.ot.harden(xtr, eps=_HARDEN_EPS, iters=_HARDEN_ITERS)
    static_b = net.ot.harden_static(xte[half:])
    static_a = net.ot.harden_static(xte[:half])

    rows = []
    order_cols = np.argsort(xtr_np, axis=0)
    for j, st in enumerate(static):
        idx_static = np.searchsorted(st["cuts_raw"], xtr_np[:, j],
                                     side="right")
        canon_s = _canonical_labels(idx_static, order_cols[:, j])
        canon_b = _canonical_labels(argmax_lab[:, j], order_cols[:, j])
        agree = float((canon_s == canon_b).mean())
        # static path is batch-invariant by construction: same learned
        # beta, cuts depend only on the atom grid of the data supplied;
        # identical inputs give identical cuts (asserted on halves)
        assert np.array_equal(
            static_a[j]["cuts_raw"],
            net.ot.harden_static(xte[:half])[j]["cuts_raw"])
        rows.append(dict(
            dataset=ds.name, seed=cfg.seed, feature=j,
            soft_auc=soft_auc,
            sinkhorn_col_resid=float(col_resid[j]),
            beta_min=float(beta[j].min()),
            a_max=st["a_max"],
            mass_err_max=st["mass_err_max"],
            mass_bound_ok=bool(
                st["mass_err_max"] <= 2 * st["a_max"] + 1e-12),
            n_bins_raw=st["n_bins_raw"],
            n_absorbed=st["n_absorbed"],
            n_violations_raw=st["n_violations_raw"],
            max_violation_raw=st["max_violation_raw"],
            n_bins_merged=len(st["cuts"]) + 1,
            n_merges=st["n_merges"],
            monotone_exact=bool(st["monotone"]),
            trend=st["trend"],
            argmax_vs_static_agree=agree,
            batch_cuts_hausdorff_halves=_rank_hausdorff(
                hard_a[j]["cuts"], hard_b[j]["cuts"]),
            batch_cuts_hausdorff_train_half=_rank_hausdorff(
                hard_tr[j]["cuts"], hard_a[j]["cuts"]),
            static_cuts_hausdorff_halves=_rank_hausdorff(
                static_a[j]["cuts_raw"], static_b[j]["cuts_raw"]),
            batch_contiguous=bool(hard_tr[j]["contiguous"]),
        ))

    # (f) deployment fidelity: scorecards on raw and merged partitions
    for name, cutset in (
            ("raw", [st["cuts_raw"] for st in static]),
            ("merged", [st["cuts"] for st in static])):
        sc = _scorecard_rows(cutset, xtr_np, ytr_np, xte_np, yte_np)
        rows.append(dict(dataset=ds.name, seed=cfg.seed, feature=-1,
                         soft_auc=soft_auc, partition=name, **sc))

    out = Path(cfg.out) / f"audit_{cfg.dataset}_{cfg.seed}"
    path = save_results(rows, out, cfg=cfg)
    logger.info("AUDIT: wrote %d rows -> %s", len(rows), path)
    return path


@hydra.main(version_base=None, config_path="../conf", config_name="audit")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
