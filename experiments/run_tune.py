"""Experiment TUNE — Bayesian-optimized arm comparison (one dataset).

The Gorishniy-protocol answer to "your table is untuned": every
compared tokenizer arm receives the SAME Optuna TPE budget on a
64/16/20 train/val/test split with per-epoch early stopping on the
validation score (patience-limited, best-state restore). Selection
uses validation only; the incumbent configuration is then refit
`eval_seeds` times (fresh torch seeds, same splits) and reported on
test. Search spaces share the optimization dimensions (lr, weight
decay, width/depth) and add each arm's own structural knobs -- for the
periodic arm this finally includes sigma, whose tuning its literature
assumes; for the layer: bin count, anneal endpoints, Sinkhorn depth.

HPC:
    python experiments/run_tune.py -m dataset=gesture \
        'arms=[ot_ple,quantile_ple,target_ple,periodic]' \
        'backbones=[mlp,ft]' device=cuda out=outputs/tune
"""

# Cesar Claros <cesar.claros@outlook.com>
# Copyright (C) 2026

import copy
import json
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

from experiments import datasets                        # noqa: E402
from experiments.common import prepare_features, save_results  # noqa: E402
from experiments.run_c3 import (TokenizedNet, _edges_for_arm,  # noqa: E402
                                _loss_fn, _quantile_transform)

logger = logging.getLogger(__name__)


def _score(task, y, out, y_mu=0.0, y_sd=1.0):
    """Higher-is-better validation/test score per task."""
    if task == "multiclass":
        return float((out.argmax(1) == y).mean())
    if task == "regression":
        pred = out.squeeze() * y_sd + y_mu
        return -float(np.sqrt(np.mean((pred - y) ** 2)))
    return float(roc_auc_score(y, out.squeeze()))


def _suggest(trial, arm, backbone):
    """Shared + arm-specific search space (identical budget per arm)."""
    p = {
        "lr": trial.suggest_float("lr", 1e-4, 3e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3,
                                            log=True),
        "hidden": trial.suggest_categorical("hidden", [32, 64, 128, 256]),
    }
    if backbone == "ft":
        p["ft_layers"] = trial.suggest_int("ft_layers", 1, 3)
        p["hidden"] = trial.suggest_categorical("d_model", [32, 64, 128])
    if arm in ("quantile_ple", "target_ple"):
        p["n_bins"] = trial.suggest_categorical("n_bins", [4, 8, 16, 32])
    elif arm == "ot_ple":
        p["n_bins"] = trial.suggest_categorical("n_bins", [4, 8, 16])
        p["eps_start"] = trial.suggest_float("eps_start", 0.05, 0.5,
                                             log=True)
        p["eps_end"] = trial.suggest_float("eps_end", 0.005, 0.05,
                                           log=True)
        p["sinkhorn_iters"] = trial.suggest_categorical(
            "sinkhorn_iters", [10, 15, 25])
    elif arm == "periodic":
        p["periodic_k"] = trial.suggest_categorical("periodic_k",
                                                    [4, 8, 16, 32])
        p["periodic_sigma"] = trial.suggest_float("periodic_sigma",
                                                  0.01, 10.0, log=True)
    return p


def _train_once(arm, backbone, p, D, cfg, torch_seed):
    """One training run with early stopping on val; returns
    (best val score, test score at the best-val state)."""
    device = torch.device(cfg.device)
    torch.manual_seed(torch_seed)
    task, n_out = D["task"], D["n_out"]
    n_bins = int(p.get("n_bins", 8))
    edges = D["edges_cache"](arm, n_bins)
    xtr = D["x"][arm]
    net = TokenizedNet(
        arm, edges, n_bins, backbone, int(p["hidden"]),
        token_mode="ple_interp",
        sinkhorn_iters=int(p.get("sinkhorn_iters", 15)),
        ft_layers=int(p.get("ft_layers", 2)), ft_heads=4,
        n_out=n_out, periodic_k=int(p.get("periodic_k", 8)),
        periodic_sigma=float(p.get("periodic_sigma", 1.0))).to(device)
    if arm == "ot_ple":
        net.ot.set_range(xtr["tr"].min(dim=0).values,
                         xtr["tr"].max(dim=0).values)
    optim = torch.optim.AdamW(net.parameters(), lr=p["lr"],
                              weight_decay=p["weight_decay"])
    loss_fn = _loss_fn(task)
    ytr = D["y_t"]["tr"]

    best_val, best_state, bad = -np.inf, None, 0
    n = len(ytr)
    for epoch in range(cfg.max_epochs):
        frac = epoch / max(cfg.max_epochs - 1, 1)
        eps = float(p.get("eps_start", 0.15)) * (
            float(p.get("eps_end", 0.02))
            / float(p.get("eps_start", 0.15))) ** frac
        net.train()
        perm = torch.randperm(n, device=xtr["tr"].device)
        for lo in range(0, n, cfg.batch_size):
            idx = perm[lo:lo + cfg.batch_size]
            if len(idx) < 16:
                continue
            logits, _ = net(xtr["tr"][idx], eps=eps, need_assign=False)
            loss = loss_fn(logits, ytr[idx])
            optim.zero_grad()
            loss.backward()
            optim.step()
        net.eval()
        with torch.no_grad():
            out_v = net(xtr["va"], eps=float(p.get("eps_end", 0.02)),
                        need_assign=False)[0].cpu().numpy()
        val = _score(task, D["y"]["va"], out_v, D["y_mu"], D["y_sd"])
        if val > best_val + 1e-6:
            best_val, bad = val, 0
            best_state = copy.deepcopy(net.state_dict())
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        out_t = net(xtr["te"], eps=float(p.get("eps_end", 0.02)),
                    need_assign=False)[0].cpu().numpy()
    return best_val, _score(task, D["y"]["te"], out_t, D["y_mu"],
                            D["y_sd"])


def _prepare(cfg):
    ds = datasets.load(cfg.dataset)
    x = prepare_features(ds, cfg.get("special_handling", "ignore"))
    task = getattr(ds, "task", "binary")
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(len(ds.y))
    n_tr, n_va = int(0.64 * len(ds.y)), int(0.16 * len(ds.y))
    idx = {"tr": perm[:n_tr], "va": perm[n_tr:n_tr + n_va],
           "te": perm[n_tr + n_va:]}
    mu, sd = x[idx["tr"]].mean(0), x[idx["tr"]].std(0) + 1e-9
    xs = {k: (x[v] - mu) / sd for k, v in idx.items()}
    q = dict(zip(["tr", "va", "te"], (
        list(_quantile_transform(xs["tr"], xs["va"]))
        + [_quantile_transform(xs["tr"], xs["te"])[1]])))
    y = {k: ds.y[v] for k, v in idx.items()}
    y_mu = float(y["tr"].mean()) if task == "regression" else 0.0
    y_sd = float(y["tr"].std()) if task == "regression" else 1.0

    device = torch.device(cfg.device)

    def to_t(a):
        return torch.as_tensor(a, dtype=torch.float32, device=device)

    x_std = {k: to_t(v) for k, v in xs.items()}
    x_rank = {k: to_t(v) for k, v in q.items()}
    n_out = int(ds.y.max()) + 1 if task == "multiclass" else 1
    if task == "multiclass":
        y_tr_t = torch.as_tensor(y["tr"], dtype=torch.long,
                                 device=device)
    else:
        y_tr_t = to_t((y["tr"] - y_mu) / y_sd)

    edge_memo = {}

    def edges_cache(arm, n_bins):
        key = (arm if arm == "target_ple" else "q", n_bins)
        if key not in edge_memo:
            base = _edges_for_arm(
                "target_ple" if arm == "target_ple" else "quantile_ple",
                xs["tr"], y["tr"], n_bins, task=task)
            edge_memo[key] = base
        return edge_memo[key]

    return dict(task=task, n_out=n_out, y=y, y_mu=y_mu, y_sd=y_sd,
                y_t={"tr": y_tr_t},
                x={"ot_ple": x_rank, "quantile_ple": x_std,
                   "target_ple": x_std, "periodic": x_std,
                   "raw": x_std},
                edges_cache=edges_cache, name=ds.name)


def run(cfg):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    D = _prepare(cfg)
    rows, trial_rows = [], []
    for arm in cfg.arms:
        for backbone in cfg.backbones:
            start = time.perf_counter()

            def objective(trial):
                p = _suggest(trial, arm, backbone)
                val, _ = _train_once(arm, backbone, p, D, cfg,
                                     torch_seed=cfg.seed)
                return val

            sampler = optuna.samplers.TPESampler(seed=cfg.seed)
            study = optuna.create_study(direction="maximize",
                                        sampler=sampler)
            try:
                study.optimize(objective, n_trials=cfg.n_trials)
            except Exception:
                logger.exception("study failed for %s/%s", arm, backbone)
                continue
            best = study.best_trial
            for t in study.trials:
                trial_rows.append(dict(
                    arm=arm, backbone=backbone, number=t.number,
                    value=t.value if t.value is not None else np.nan))
            for es in range(cfg.eval_seeds):
                val, test = _train_once(arm, backbone, best.params, D,
                                        cfg, torch_seed=1000 + es)
                rows.append(dict(
                    arm=arm, backbone=backbone, eval_seed=es,
                    val_score=val, test_score=test,
                    best_val_tuning=float(best.value),
                    best_params=json.dumps(best.params),
                    n_trials=cfg.n_trials,
                    tune_time=time.perf_counter() - start))
            logger.info("%s/%s tuned: best val=%.4f, test=%.4f±%.4f",
                        arm, backbone, best.value,
                        np.mean([r["test_score"] for r in rows
                                 if r["arm"] == arm
                                 and r["backbone"] == backbone]),
                        np.std([r["test_score"] for r in rows
                                if r["arm"] == arm
                                and r["backbone"] == backbone]))

    common = dict(dataset=D["name"], task=D["task"], seed=cfg.seed)
    for r in rows + trial_rows:
        r.update(common)
    out = Path(cfg.out)
    tag = "tune_{}_{}".format(cfg.dataset, cfg.seed)
    paths = [save_results(rows, out / tag),
             save_results(trial_rows, out / (tag + "_trials"))]
    logger.info("TUNE: wrote %s", paths[0])
    return paths


@hydra.main(version_base=None, config_path="../conf", config_name="tune")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
