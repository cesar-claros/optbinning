"""Experiment C4-audit — interpretability substantiation (Paper C §5.6).

Trains the C4 self-explaining scorecard (OT-binning layer + linear head) and
measures the properties the interpretability claim rests on, none of which the
C3/C4 accuracy tables report:

  * additive-extraction fidelity: the linear head makes the logit an exact sum
    of per-feature contributions, ``b + sum_f g_f(x_f)``, so the explanation is
    the model (verified to machine precision, an architectural identity);
  * hardening gap: replacing the soft assignment with the hardened one-bin
    lookup (the deployed scorecard) costs auc_soft - auc_hard;
  * cut agreement: how far the end-to-end cuts sit from the certified two-stage
    optbinning cuts, feature by feature;
  * seed stability: spread of the learned cuts across seeds (is the audit
    reproducible);
  * an extracted example scorecard (bins, WoE, points) for the most influential
    feature.

``ot_input=quantile`` (default) audits the rank-space geometry the C3/C4 tables
recommend (the range-to-rank fix of §5.4); ``ot_input=standard`` audits the raw
standardized geometry instead. Either way the learned cuts are mapped back to
standardized feature units before they are compared with the optbinning cuts,
so cut-agreement and stability are reported in the same units. Token mode
``assign`` gives the classic step points table.

    python experiments/run_c4_audit.py -m dataset=german,taiwan,gmsc n_seeds=5
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
from sklearn.metrics import roc_auc_score               # noqa: E402
from torch import nn                                    # noqa: E402

from optbinning import OptimalBinning                   # noqa: E402

from experiments import datasets                        # noqa: E402
from experiments.common import (prepare_features,       # noqa: E402
                                save_results)
from experiments.paperc.otlayer import (pav_penalty_multi,  # noqa: E402
                                        soft_iv_multi)
from experiments.run_c3 import (TokenizedNet,           # noqa: E402
                                _edges_for_arm,
                                _quantile_transform)

logger = logging.getLogger(__name__)


def _fit(data, cfg, seed):
    """Train an OT-binning layer + linear (scorecard) head; return the net and
    the standardized train/test tensors."""
    torch.manual_seed(seed)
    device = torch.device(cfg.device)
    xtr = torch.as_tensor(data["xtr"], dtype=torch.float32, device=device)
    ytr = torch.as_tensor(data["ytr"], dtype=torch.float32, device=device)
    xte = torch.as_tensor(data["xte"], dtype=torch.float32, device=device)

    edges = _edges_for_arm("ot_ple", data["xtr"], data["ytr"], cfg.n_bins)
    net = TokenizedNet("ot_ple", edges, cfg.n_bins, "linear", cfg.hidden,
                       token_mode="assign",
                       sinkhorn_iters=cfg.sinkhorn_iters).to(device)
    net.ot.set_range(xtr.min(dim=0).values, xtr.max(dim=0).values)
    optim = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    bce = nn.BCEWithLogitsLoss()

    n = len(ytr)
    for epoch in range(cfg.epochs):
        frac = epoch / max(cfg.epochs - 1, 1)
        eps = cfg.eps_start * (cfg.eps_end / cfg.eps_start) ** frac
        perm = torch.randperm(n, device=device)
        for lo in range(0, n, cfg.batch_size):
            idx = perm[lo:lo + cfg.batch_size]
            if len(idx) < cfg.n_bins * 4:
                continue
            logits, assign = net(xtr[idx], eps=eps, need_assign=True)
            loss = bce(logits, ytr[idx])
            loss = loss - cfg.aux_iv * soft_iv_multi(assign, ytr[idx])
            loss = loss + cfg.aux_iv * pav_penalty_multi(assign, ytr[idx])
            optim.zero_grad()
            loss.backward()
            optim.step()
    net.eval()
    return net, xtr, ytr, xte


def _fidelity_and_auc(net, xte, yte, eps):
    """Additive reconstruction error and soft-vs-hardened scorecard AUC."""
    n_f, n_b = net.n_features, net.token_dim
    with torch.no_grad():
        logits, _ = net(xte, eps=eps, need_assign=False)
        tok, _ = net.tokens(xte, eps, need_assign=False)
        w = net.head.weight.detach().reshape(n_f, n_b)
        contrib = torch.einsum("bft,ft->bf", tok, w)          # (B, features)
        recon = contrib.sum(dim=1) + net.head.bias.detach().squeeze()
        fidelity = float((logits - recon).abs().max())

        assign = net.ot(xte, eps=eps)
        hard_bin = assign.argmax(dim=2)                       # (B, features)
        hard_contrib = torch.gather(
            w.unsqueeze(0).expand(len(xte), -1, -1), 2,
            hard_bin.unsqueeze(2)).squeeze(2)
        logit_hard = hard_contrib.sum(dim=1) + net.head.bias.detach().squeeze()
    auc_soft = float(roc_auc_score(yte, torch.sigmoid(logits).cpu().numpy()))
    auc_hard = float(roc_auc_score(yte, torch.sigmoid(logit_hard).cpu().numpy()))
    return fidelity, auc_soft, auc_hard


def _optbinning_cuts(xtr, ytr, n_bins, solver):
    """Certified two-stage optbinning cuts per feature (the target_ple
    reference); solver is configurable ('cp' matches C3/C4, 'mip' avoids a
    CP-SAT stall in some environments)."""
    out = []
    for j in range(xtr.shape[1]):
        ob = OptimalBinning(dtype="numerical", solver=solver,
                            max_n_bins=n_bins).fit(xtr[:, j], ytr)
        out.append(np.asarray(ob.splits, dtype=float))
    return out


def _cut_agreement(ot_cuts, ob_cuts, tol):
    """Mean nearest-neighbor distance from optbinning cuts to OT cuts, and the
    fraction of optbinning cuts matched within ``tol``."""
    if not len(ot_cuts) or not len(ob_cuts):
        return np.nan, np.nan
    d = np.min(np.abs(ob_cuts[:, None] - ot_cuts[None, :]), axis=1)
    return float(d.mean()), float((d <= tol).mean())


def _exhibit(net, xin, xstd, y, ot_cuts, feats, mu, sd, eps):
    """Extract and print the scorecard of the most influential feature: bin
    ranges (original units), event rate, WoE, and points (the feature's own
    contribution to the logit, read straight off the linear head).

    ``xin`` is the model's input (rank or standardized); ``ot_cuts`` are the
    learned cuts already mapped to standardized units, and bin membership is
    taken in those units on ``xstd``."""
    device = next(net.parameters()).device
    xt = torch.as_tensor(xin, dtype=torch.float32, device=device)
    with torch.no_grad():
        tok, _ = net.tokens(xt, eps, need_assign=False)
        w = net.head.weight.detach().reshape(net.n_features, net.token_dim)
        g = torch.einsum("nft,ft->nf", tok, w).cpu().numpy()   # contributions
    f = int(np.argmax(g.max(axis=0) - g.min(axis=0)))          # widest span
    cuts = ot_cuts[f]
    edges = np.concatenate(([-np.inf], cuts, [np.inf]))
    idx = np.digitize(xstd[:, f], cuts)
    n0, n1 = max((y == 0).sum(), 1), max((y == 1).sum(), 1)
    print("\n--- extracted scorecard: feature '{}' ---".format(feats[f]))
    print("bin | x in original units          |    n | rate  |   WoE  | points")
    for k in range(len(cuts) + 1):
        m = idx == k
        if not m.any():
            continue
        e1 = int((y[m] == 1).sum())
        e0 = int(m.sum()) - e1
        woe = np.log(((e0 + 0.5) / n0) / ((e1 + 0.5) / n1))
        lo = mu[f] + sd[f] * edges[k] if np.isfinite(edges[k]) else -np.inf
        hi = mu[f] + sd[f] * edges[k + 1] if np.isfinite(edges[k + 1]) \
            else np.inf
        print("{:>3} | [{:>10.4g}, {:>10.4g}) | {:>4} | {:.3f} | {:+.3f} | "
              "{:+.3f}".format(k, lo, hi, int(m.sum()), e1 / m.sum(), woe,
                               float(g[m, f].mean())))
    print("intercept: {:+.3f}".format(float(net.head.bias.detach().item())))


def run(cfg):
    ds = datasets.load(cfg.dataset, n=cfg.get("n", 6000),
                       seed=cfg.get("data_seed", 0)) \
        if str(cfg.dataset).startswith("synthetic") \
        else datasets.load(cfg.dataset)
    # sentinel-code handling (HELOC/BAF): median-impute + append indicator
    # features, with names for the per-feature audit. A no-op without codes.
    x, feats = prepare_features(ds, cfg.get("special_handling", "expand"),
                                return_names=True)
    ot_input = cfg.get("ot_input", "quantile")

    rows = []
    cuts_by_feature = {f: [] for f in feats}
    for seed in range(cfg.seed_offset, cfg.seed_offset + cfg.n_seeds):
        tr, te = datasets.split_indices(len(ds.y), cfg.test_size, seed)
        mu, sd = x[tr].mean(axis=0), x[tr].std(axis=0) + 1e-9
        xstd_tr, xstd_te = (x[tr] - mu) / sd, (x[te] - mu) / sd
        if ot_input == "quantile":
            xin_tr, xin_te = _quantile_transform(xstd_tr, xstd_te)
        else:
            xin_tr, xin_te = xstd_tr, xstd_te
        train_data = dict(xtr=xin_tr, ytr=ds.y[tr], xte=xin_te, yte=ds.y[te])

        net, xtr_t, _, xte_t = _fit(train_data, cfg, seed)
        fidelity, auc_soft, auc_hard = _fidelity_and_auc(
            net, xte_t, ds.y[te], cfg.eps_end)

        hard = net.ot.harden(xtr_t)
        ot_cuts = []                     # learned cuts in standardized units
        for i in range(len(feats)):
            c = np.asarray(hard[i]["cuts"], dtype=float)
            if ot_input == "quantile" and len(c):
                c = np.quantile(xstd_tr[:, i], np.clip(c, 0.0, 1.0))
            ot_cuts.append(c)

        if seed == cfg.seed_offset:
            _exhibit(net, xin_tr, xstd_tr, ds.y[tr], ot_cuts, feats, mu, sd,
                     cfg.eps_end)

        ob_cuts = _optbinning_cuts(xstd_tr, ds.y[tr], cfg.n_bins,
                                   cfg.get("ob_solver", "cp"))
        for i, feat in enumerate(feats):
            cuts_by_feature[feat].append(ot_cuts[i])
            dist, matched = _cut_agreement(ot_cuts[i], ob_cuts[i], cfg.cut_tol)
            rows.append(dict(
                dataset=ds.name, seed=seed, feature=feat,
                n_cuts=len(ot_cuts[i]),
                contiguous=bool(hard[i]["contiguous"]),
                fidelity_max=fidelity, auc_soft=auc_soft, auc_hard=auc_hard,
                cut_dist=dist, cut_matched=matched))

    _report(cfg, ds.name, rows, cuts_by_feature)
    out = Path(cfg.out) / "c4audit_{}_{}_{}".format(
        cfg.dataset, ot_input, cfg.seed_offset)
    path = save_results(rows, out, cfg=cfg)
    logger.info("c4audit: wrote %d rows -> %s", len(rows), path)
    return path


def _report(cfg, name, rows, cuts_by_feature):
    """Print the aggregate audit and one extracted example scorecard."""
    import pandas as pd
    df = pd.DataFrame(rows)
    per_seed = df.groupby("seed").first()
    print("\n===== C4 interpretability audit: {} ({} seeds, ot_input={}) ====="
          .format(name, df["seed"].nunique(), cfg.get("ot_input", "quantile")))
    print("additive-extraction fidelity (max |logit - sum_f g_f|): {:.2e}"
          .format(df["fidelity_max"].max()))
    print("scorecard AUC  soft {:.4f} +/- {:.4f}   hardened {:.4f} +/- {:.4f}"
          "   gap {:.4f}".format(
              per_seed["auc_soft"].mean(), per_seed["auc_soft"].std(),
              per_seed["auc_hard"].mean(), per_seed["auc_hard"].std(),
              (per_seed["auc_soft"] - per_seed["auc_hard"]).mean()))
    print("contiguous fits: {:.1%}   cuts/feature: {:.1f}".format(
        df["contiguous"].mean(), df["n_cuts"].mean()))
    print("cut agreement with certified optbinning: mean dist {:.3f} sd-units,"
          "  matched within {}: {:.1%}".format(
              df["cut_dist"].mean(), cfg.cut_tol, df["cut_matched"].mean()))

    stab = []
    for cuts in cuts_by_feature.values():
        counts = [len(c) for c in cuts]
        mode = max(set(counts), key=counts.count)
        keep = np.array([c for c in cuts if len(c) == mode])
        if len(keep) >= 2 and mode > 0:
            stab.append(keep.std(axis=0).mean())
    if stab:
        print("seed stability: mean cut sd across seeds {:.3f} sd-units "
              "(features with a stable cut count)".format(float(np.mean(stab))))


@hydra.main(version_base=None, config_path="../conf", config_name="c4_audit")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
