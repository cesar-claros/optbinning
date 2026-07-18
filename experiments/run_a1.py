"""
Experiment A1 — objective benchmark on real/synthetic data (Paper A).

Per dataset x feature x arm x seed: fit the binning, evaluate out-of-sample
IV / W1 / bin structure / spike incidence, and bootstrap cut stability.

Local smoke test:
    python experiments/run_a1.py dataset=synthetic-smooth n_seeds=1 n_boot=4
HPC (SLURM arrays via submitit):
    python experiments/run_a1.py -m hydra/launcher=submitit_slurm \
        dataset=german,taiwan,hmeq seed_offset=range(0,10)
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
                                make_arm, save_results)


def run(cfg):
    ds = datasets.load(cfg.dataset, n=cfg.get("n", 5000),
                       seed=cfg.get("data_seed", 0)) \
        if str(cfg.dataset).startswith("synthetic") \
        else datasets.load(cfg.dataset)

    features = expanded_features(
        ds, list(cfg.features) if cfg.get("features") else None,
        cfg.get("special_handling", "expand"))
    rows = []
    for seed in range(cfg.seed_offset, cfg.seed_offset + cfg.n_seeds):
        tr, te = datasets.split_indices(len(ds.y), cfg.test_size, seed)
        for feat in features:
            x = feature_array(ds, feat, cfg.get("special_handling", "expand"))
            mask = np.isfinite(x)
            xtr, ytr = x[tr][mask[tr]], ds.y[tr][mask[tr]]
            xte, yte = x[te][mask[te]], ds.y[te][mask[te]]

            scale = np.subtract(*np.nanquantile(xtr, [0.95, 0.05]))
            lam = float(cfg.lam_frac) * abs(scale) if scale else 1.0

            # FM achieved by the IV solution: guaranteed-feasible anchor for
            # the fm_tau trust threshold (the IV solution is a witness). Only
            # needed when that arm is active.
            fm_ref = None
            if "fm_tau" in cfg.arms:
                from optbinning.binning.metrics import flat_metric_1d
                from experiments.common import binned_stats
                ref = make_arm("iv", monotonic=cfg.monotonic).fit(xtr, ytr)
                ne_r, e_r, w_r = binned_stats(xtr, ytr, ref.splits)
                fm_ref = flat_metric_1d(ne_r / ne_r.sum(), e_r / e_r.sum(),
                                        w_r, lam) if len(w_r) > 1 else 0.0

            for arm in cfg.arms:
                kw = dict(monotonic=cfg.monotonic)
                if arm == "iv_w1":
                    kw["gamma"] = float(cfg.gamma) / max(abs(scale), 1e-9)
                elif arm == "fm_tau":
                    kw["lam"] = lam
                    kw["fm_tau"] = float(cfg.fm_tau_frac) * fm_ref

                def factory(arm=arm, kw=kw):
                    return make_arm(arm, **kw)

                t0 = time.perf_counter()
                try:
                    optb = factory().fit(xtr, ytr)
                    status = optb.status
                    splits = optb.splits
                except Exception as err:            # pragma: no cover
                    rows.append(dict(dataset=ds.name, feature=feat,
                                     arm=arm, seed=seed,
                                     status="ERROR:" + type(err).__name__))
                    continue
                fit_time = time.perf_counter() - t0

                row = dict(dataset=ds.name, feature=feat, arm=arm,
                           seed=seed, status=status, fit_time=fit_time,
                           lam=lam)
                row.update(eval_binning(splits, xte, yte))
                if cfg.n_boot:
                    sd, mism = bootstrap_cut_sd(factory, xtr, ytr,
                                                n_boot=cfg.n_boot,
                                                seed=seed)
                    row.update(cut_sd=sd,
                               cut_sd_norm=sd / abs(scale) if scale else np.nan,
                               refit_mismatch=mism)
                rows.append(row)

    out = Path(cfg.out) / "a1_{}_{}".format(cfg.dataset, cfg.seed_offset)
    path = save_results(rows, out)
    print("A1: wrote {} rows -> {}".format(len(rows), path))
    return path


@hydra.main(version_base=None, config_path="../conf", config_name="a1")
def main(cfg: DictConfig):
    return run(cfg)


if __name__ == "__main__":
    main()
