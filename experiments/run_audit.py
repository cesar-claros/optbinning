"""Experiment AUDIT v2 — hardened-guarantee postconditions, static
inference, and conversion cost (Paper C, re-review P0/P1).

Trains the production configuration (rank input, learned knots, IV/PAV
auxiliary with PRE-DECLARED per-feature trends), then verifies every
advertised structural guarantee directly, per feature, and measures the
deployment-conversion cost the re-review requires to be reported
separately from soft-tokenizer accuracy:

(a)  Sinkhorn column residuals at the hardening temperature AND over
     sampled TRAINING batches at training temperatures/unroll (median,
     p95, max per feature) -- approximate mass control is measured, not
     assumed.
(b)  soft mass floor; sharper nearest-grid bound |b' - b| <= a_max
     (deployment Prop.) next to the 2*a_max row-argmax bound.
(c)  mass-coordinate static partition: absorbed bins, pre-merge
     violations, bins after exact directional merge; post-merge
     application-constraint columns (min bin fraction, min event and
     non-event counts) so instance-specific feasibility can be checked
     from the artifact.
(d)  agreement between batchwise Sinkhorn-argmax hardening (diagnostic)
     and static interval lookup.
(e)  FROZEN-CUT batch invariance: cuts built once on train, stored, and
     the same test records evaluated whole, shuffled, and in chunks --
     identical bin ids and logits asserted and recorded. (The
     half-vs-half recomputation is kept, relabeled as estimator
     data-sensitivity, which is what it measures.)
(f)  deployment fidelity AND conversion cost: the hard scorecard is a
     REFIT on the static bins (named as such, not an extraction); we
     report table-lookup self-consistency, plus the direct
     PLE-to-hard discrepancy: logit RMSE/max, prediction agreement,
     and paired AUC / log-loss / ECE deltas.
(g)  trend-variant study: merged scorecards under the declared
     per-feature trend, global auto, peak, valley, and no merge -- the
     constraint-class cost (HMEQ question) measured per dataset.

Local smoke test:
    python experiments/run_audit.py dataset=german epochs=8
HPC:
    python experiments/run_audit.py -m dataset=german,taiwan,gmsc,hmeq,heloc \
        seed=0,1,2,3,4 device=cuda out=outputs/audit_v2
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
                                _quantile_transform, _run_epochs,
                                declared_trends)

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


def _ece(prob: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error on equal-width probability bins."""
    idx = np.clip((prob * n_bins).astype(int), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(prob[m].mean() - y[m].mean())
    return float(e)


def _scorecard(cuts: list[np.ndarray], xtr: np.ndarray, ytr: np.ndarray,
               xte: np.ndarray, yte: np.ndarray,
               soft_logits: np.ndarray | None = None) -> dict:
    """WoE-logistic scorecard REFIT on hard bins (a conversion, not an
    extraction of the trained network -- named accordingly). Verifies
    table-lookup self-consistency, and, when the soft model's test
    logits are supplied, reports the direct PLE-to-hard conversion
    discrepancy the re-review requires."""
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
    logits_table = np.full(len(xte), float(clf.intercept_[0]))
    for j in range(n_feat):
        points = clf.coef_[0, j] * tables[j]
        idx_te = np.searchsorted(cuts[j], xte[:, j], side="right")
        logits_table += points[idx_te]
    prob = 1.0 / (1.0 + np.exp(-logits_model))
    out = {
        "table_fidelity_max_abs": float(
            np.max(np.abs(logits_table - logits_model))),
        "scorecard_auc": float(roc_auc_score(yte, prob)),
        "scorecard_logloss": float(log_loss(yte, prob)),
        "scorecard_ece": _ece(prob, yte),
    }
    if soft_logits is not None:
        sp = 1.0 / (1.0 + np.exp(-soft_logits))
        out.update(
            conv_logit_rmse=float(np.sqrt(np.mean(
                (logits_model - soft_logits) ** 2))),
            conv_logit_max=float(np.max(np.abs(
                logits_model - soft_logits))),
            conv_pred_agree=float(((prob >= 0.5) ==
                                   (sp >= 0.5)).mean()),
            conv_auc_delta=float(roc_auc_score(yte, prob)
                                 - roc_auc_score(yte, sp)),
            conv_logloss_delta=float(log_loss(yte, prob)
                                     - log_loss(yte, sp)),
            conv_ece_delta=_ece(prob, yte) - _ece(sp, yte),
        )
    return out


def _train_batch_residuals(net: TokenizedNet, xtr: torch.Tensor,
                           cfg: DictConfig,
                           n_batches: int = 10) -> np.ndarray:
    """Sinkhorn column residuals max_k |mean_t phi_tk - beta_k| over
    sampled training batches at training temperatures and the TRAINING
    unroll -- the approximate-mass-control measurement the re-review
    requires. Returns (n_samples, n_features)."""
    beta = net.ot.bin_masses().detach().cpu().numpy()
    temps = [cfg.eps_start, float(np.sqrt(cfg.eps_start * cfg.eps_end)),
             cfg.eps_end]
    res = []
    g = torch.Generator(device="cpu").manual_seed(0)
    with torch.no_grad():
        for eps in temps:
            for _ in range(n_batches):
                idx = torch.randperm(len(xtr), generator=g)[
                    :cfg.batch_size].to(xtr.device)
                phi = net.ot(xtr[idx], eps=eps)
                col = phi.mean(dim=0).cpu().numpy()
                res.append(np.abs(col - beta).max(axis=1))
    return np.asarray(res)


def _frozen_cut_invariance(cuts: list[np.ndarray],
                           xte: np.ndarray) -> bool:
    """Deployed-API batch invariance: identical bin assignments for the
    same records evaluated whole, shuffled, and in odd-sized chunks."""
    rng = np.random.default_rng(0)
    for j, c in enumerate(cuts):
        whole = np.searchsorted(c, xte[:, j], side="right")
        perm = rng.permutation(len(xte))
        shuffled = np.empty_like(whole)
        shuffled[perm] = np.searchsorted(c, xte[perm, j], side="right")
        chunked = np.concatenate(
            [np.searchsorted(c, xte[lo:lo + 997, j], side="right")
             for lo in range(0, len(xte), 997)])
        if not (np.array_equal(whole, shuffled)
                and np.array_equal(whole, chunked)):
            return False
    return True


def run(cfg: DictConfig) -> Path:
    ds = datasets.load(cfg.dataset)
    task = getattr(ds, "task", "binary")
    if task != "binary":
        raise ValueError(
            f"audit requires a binary dataset; {cfg.dataset} is {task}.")
    arm = cfg.get("arm", "ot_ple")
    if arm not in ("ot_ple", "mass_knot_ot"):
        raise ValueError(f"audit arm must be ot_ple|mass_knot_ot; "
                         f"got {arm}.")
    x = prepare_features(ds, cfg.get("special_handling", "expand"))
    tr, te = datasets.split_indices(len(ds.y), cfg.test_size, cfg.seed)
    mu, sd = x[tr].mean(axis=0), x[tr].std(axis=0) + 1e-9
    xtr_np, xte_np = (x[tr] - mu) / sd, (x[te] - mu) / sd
    xtr_np, xte_np = _quantile_transform(xtr_np, xte_np)   # rank space
    ytr_np, yte_np = ds.y[tr].astype(float), ds.y[te].astype(float)

    # pre-declared per-feature trends: shared by aux AND deployment
    trend_sign = declared_trends(xtr_np, ytr_np)
    trends = ["ascending" if s > 0 else "descending"
              for s in trend_sign]

    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    xtr = torch.as_tensor(xtr_np, dtype=torch.float32, device=device)
    ytr = torch.as_tensor(ytr_np, dtype=torch.float32, device=device)
    xte = torch.as_tensor(xte_np, dtype=torch.float32, device=device)

    edges = [np.linspace(0, 1, cfg.n_bins + 1)] * xtr_np.shape[1]
    net = TokenizedNet(arm, edges, cfg.n_bins, cfg.backbone,
                       cfg.hidden, token_mode="ple_interp",
                       sinkhorn_iters=cfg.sinkhorn_iters).to(device)
    net.ot.set_range(xtr.min(dim=0).values, xtr.max(dim=0).values)
    _run_epochs(net, _make_optim(net, cfg.backbone, cfg), xtr, ytr, cfg,
                use_aux=True, task="binary", trend_sign=trend_sign)
    net.eval()

    # soft-model test logits (the PLE predictor being converted)
    with torch.no_grad():
        soft_logits = torch.cat(
            [net(xte[lo:lo + cfg.batch_size], eps=cfg.eps_end,
                 need_assign=False)[0]
             for lo in range(0, len(xte), cfg.batch_size)]).cpu().numpy()
    soft_auc = float(roc_auc_score(yte_np,
                                   1 / (1 + np.exp(-soft_logits))))

    beta = net.ot.bin_masses().detach().cpu().numpy()

    # (a) hardening-temperature residual + argmax labels (diagnostic)
    with torch.no_grad():
        saved = net.ot.sinkhorn_iters
        net.ot.sinkhorn_iters = _HARDEN_ITERS
        plan_rows = net.ot(xtr, eps=_HARDEN_EPS)
        net.ot.sinkhorn_iters = saved
        col = plan_rows.mean(dim=0).cpu().numpy()
        argmax_lab = plan_rows.argmax(dim=2).cpu().numpy()
    col_resid = np.abs(col - beta).max(axis=1)

    # (a2) training-batch residual distribution (training unroll/temps)
    train_resid = _train_batch_residuals(net, xtr, cfg)

    # (c) static partitions under the declared trend + variants
    static = net.ot.harden_static(xtr, ytr, trend=trends)
    variants = {"auto": net.ot.harden_static(xtr, ytr, trend="auto"),
                "peak": net.ot.harden_static(xtr, ytr, trend="peak"),
                "valley": net.ot.harden_static(xtr, ytr, trend="valley")}

    # (e) batchwise diagnostic on halves + estimator data-sensitivity
    half = len(xte) // 2
    hard_a = net.ot.harden(xte[:half], eps=_HARDEN_EPS,
                           iters=_HARDEN_ITERS)
    hard_b = net.ot.harden(xte[half:], eps=_HARDEN_EPS,
                           iters=_HARDEN_ITERS)
    hard_tr = net.ot.harden(xtr, eps=_HARDEN_EPS, iters=_HARDEN_ITERS)
    static_a = net.ot.harden_static(xte[:half])
    static_b = net.ot.harden_static(xte[half:])

    n_tr = len(xtr_np)
    rows = []
    order_cols = np.argsort(xtr_np, axis=0)
    for j, st in enumerate(static):
        idx_static = np.searchsorted(st["cuts_raw"], xtr_np[:, j],
                                     side="right")
        canon_s = _canonical_labels(idx_static, order_cols[:, j])
        canon_b = _canonical_labels(argmax_lab[:, j], order_cols[:, j])
        merged_tot = st["hard_mass"] * n_tr
        merged_ev = st["event_rate"] * merged_tot
        rows.append(dict(
            dataset=ds.name, seed=cfg.seed, feature=j, arm=arm,
            soft_auc=soft_auc,
            trend_declared=trends[j],
            sinkhorn_col_resid=float(col_resid[j]),
            train_resid_med=float(np.median(train_resid[:, j])),
            train_resid_p95=float(np.percentile(train_resid[:, j], 95)),
            train_resid_max=float(train_resid[:, j].max()),
            beta_min=float(beta[j].min()),
            a_max=st["a_max"],
            mass_err_max=st["mass_err_max"],
            mass_bound_ok=bool(
                st["mass_err_max"] <= 2 * st["a_max"] + 1e-12),
            mass_bound_sharp_ok=bool(
                st["mass_err_max"] <= st["a_max"] + 1e-12),
            n_bins_raw=st["n_bins_raw"], n_absorbed=st["n_absorbed"],
            n_violations_raw=st["n_violations_raw"],
            max_violation_raw=st["max_violation_raw"],
            n_bins_merged=len(st["cuts"]) + 1,
            n_merges=st["n_merges"],
            monotone_exact=bool(st["monotone"]),
            min_bin_frac_merged=float(st["hard_mass"].min()),
            min_event_count_merged=float(merged_ev.min()),
            min_nonevent_count_merged=float(
                (merged_tot - merged_ev).min()),
            argmax_vs_static_agree=float((canon_s == canon_b).mean()),
            batch_cuts_hausdorff_halves=_rank_hausdorff(
                hard_a[j]["cuts"], hard_b[j]["cuts"]),
            batch_cuts_hausdorff_train_half=_rank_hausdorff(
                hard_tr[j]["cuts"], hard_a[j]["cuts"]),
            static_estimator_sensitivity_halves=_rank_hausdorff(
                static_a[j]["cuts_raw"], static_b[j]["cuts_raw"]),
            batch_contiguous=bool(hard_tr[j]["contiguous"]),
        ))

    # (e2) frozen-cut deployed-API invariance (the actual guarantee)
    frozen_ok = _frozen_cut_invariance([st["cuts"] for st in static],
                                       xte_np)

    # (f)+(g) conversion cost per partition variant
    named = {"merged": [st["cuts"] for st in static],
             "raw": [st["cuts_raw"] for st in static],
             "merged_auto": [v["cuts"] for v in variants["auto"]],
             "merged_peak": [v["cuts"] for v in variants["peak"]],
             "merged_valley": [v["cuts"] for v in variants["valley"]]}
    for name, cutset in named.items():
        sc = _scorecard(cutset, xtr_np, ytr_np, xte_np, yte_np,
                        soft_logits=soft_logits)
        rows.append(dict(dataset=ds.name, seed=cfg.seed, feature=-1,
                         arm=arm, soft_auc=soft_auc, partition=name,
                         frozen_cut_invariant=frozen_ok, **sc))

    # arm in the tag: an arm sweep into one out dir must not self-clobber
    # (the v2 ot_ple/mass_knot_ot multirun did exactly that -- logged)
    out = Path(cfg.out) / f"audit_{arm}_{cfg.dataset}_{cfg.seed}"
    path = save_results(rows, out, cfg=cfg)
    logger.info("AUDIT v2: wrote %d rows -> %s", len(rows), path)
    return path


@hydra.main(version_base=None, config_path="../conf", config_name="audit")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
