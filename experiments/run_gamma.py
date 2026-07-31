"""
Experiment — hybrid weight (gamma_wasserstein) sweep on the spike design.

Tests whether the IV + gamma*W1 hybrid can actually move off the pure-IV
binning and reduce the bootstrap fragility of spike features (P1's corrected
conjecture: 47% pure-IV refit flips vs 17% hybrid). Per dataset x feature x
seed the IV baseline (via MIP, so only the objective differs from the hybrid)
and its bootstrap fragility are computed once; then for each gamma the hybrid
is fit, compared to the baseline splits, and its own fragility measured. gamma
is in the same "divergence per (p95-p05)" units as the a1 driver.

Local smoke test:
    python experiments/run_gamma.py dataset=synthetic-spike n_seeds=1 n_boot=6
Real run:
    python experiments/run_gamma.py -m dataset=synthetic-spike n_seeds=10
"""

# Cesar Claros <cesar.claros@outlook.com>
# Copyright (C) 2026

import sys
import time

from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra                                            # noqa: E402
from omegaconf import DictConfig                        # noqa: E402

from experiments import datasets                        # noqa: E402
from experiments.common import (bootstrap_cut_sd, eval_binning,  # noqa: E402
                                expanded_features, feature_array,
                                make_arm, save_results, to_coordinate)


def run(cfg):
    ds = datasets.load(cfg.dataset, n=cfg.get("n", 5000),
                       seed=cfg.get("data_seed", 0)) \
        if str(cfg.dataset).startswith("synthetic") \
        else datasets.load(cfg.dataset)

    features = expanded_features(
        ds, list(cfg.features) if cfg.get("features") else None,
        cfg.get("special_handling", "expand"))
    gammas = [float(g) for g in cfg.gammas]
    rows = []
    for seed in range(cfg.seed_offset, cfg.seed_offset + cfg.n_seeds):
        tr, te = datasets.split_indices(len(ds.y), cfg.test_size, seed)
        for feat in features:
            x = feature_array(ds, feat, cfg.get("special_handling", "expand"))
            mask = np.isfinite(x)
            xtr, ytr = x[tr][mask[tr]], ds.y[tr][mask[tr]]
            xte, yte = x[te][mask[te]], ds.y[te][mask[te]]

            coord = cfg.get("coordinate", "raw")
            xtr, xte = to_coordinate(xtr, xte, ytr, kind=coord)

            scale = np.subtract(*np.nanquantile(xtr, [0.95, 0.05]))
            denom = max(abs(scale), 1e-9)

            # Solver-matched IV baseline: gamma-independent, so fit and
            # bootstrap it once and reuse across the gamma grid.
            def iv_factory():
                return make_arm("iv_mip", monotonic=cfg.monotonic)

            iv = iv_factory().fit(xtr, ytr)
            iv_sig = tuple(np.round(np.asarray(iv.splits, dtype=float), 6))
            iv_eval = eval_binning(iv.splits, xte, yte)
            iv_sd, iv_mism = bootstrap_cut_sd(iv_factory, xtr, ytr,
                                              n_boot=cfg.n_boot, seed=seed)
            iv_norm = iv_sd / abs(scale) if scale else np.nan

            for gamma in gammas:
                def hy_factory(g=gamma / denom):
                    return make_arm("iv_w1", gamma=g, monotonic=cfg.monotonic)

                base = dict(dataset=ds.name, feature=feat, seed=seed,
                            gamma=gamma)
                t0 = time.perf_counter()
                try:
                    hy = hy_factory().fit(xtr, ytr)
                except Exception as err:            # pragma: no cover
                    rows.append(dict(base, fit_time=time.perf_counter() - t0,
                                     status="ERROR:" + type(err).__name__))
                    continue
                hy_sig = tuple(np.round(np.asarray(hy.splits, dtype=float), 6))
                hy_eval = eval_binning(hy.splits, xte, yte)
                hy_sd, hy_mism = bootstrap_cut_sd(hy_factory, xtr, ytr,
                                                  n_boot=cfg.n_boot, seed=seed)
                rows.append(dict(
                    base, status=hy.status,
                    fit_time=time.perf_counter() - t0,
                    differs_from_iv=hy_sig != iv_sig,
                    iv_refit_mismatch=iv_mism, hyb_refit_mismatch=hy_mism,
                    iv_spike_bins=iv_eval["spike_bins"],
                    hyb_spike_bins=hy_eval["spike_bins"],
                    iv_oos_iv=iv_eval["oos_iv"], hyb_oos_iv=hy_eval["oos_iv"],
                    iv_oos_w1=iv_eval["oos_w1"], hyb_oos_w1=hy_eval["oos_w1"],
                    iv_n_bins=iv_eval["n_bins"], hyb_n_bins=hy_eval["n_bins"],
                    iv_cut_sd_norm=iv_norm,
                    hyb_cut_sd_norm=hy_sd / abs(scale) if scale else np.nan))

    out = Path(cfg.out) / "gamma_{}_{}".format(cfg.dataset, cfg.seed_offset)
    path = save_results(rows, out, cfg=cfg)
    print("gamma: wrote {} rows -> {}".format(len(rows), path))
    return path


@hydra.main(version_base=None, config_path="../conf", config_name="gamma")
def main(cfg: DictConfig):
    return run(cfg)


if __name__ == "__main__":
    main()
