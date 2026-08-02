"""Experiment W1TAU — the normalized W1 trust-constraint frontier
(Paper A Sec. 3 governance proposal; re-review P0 -- previously untested).

Per dataset x feature x seed, in the DECLARED coordinate:

1. fit pure IV (solver-matched, mip) -> its W1 value = the feasible
   anchor floor; fit pure max-W1 -> the achievable ceiling;
2. sweep rho over fractions of [W1_iv, W1_max] (and beyond the ceiling
   for the infeasibility profile);
3. solve max IV s.t. W1 >= rho exactly (linear constraint, Phi
   coefficients; no new variables);
4. record status, in-sample and OOS IV/W1, bin count, partition hash;
5. record the lambda-path points (iv_w1 arm over a gamma grid) on the
   same feature so unsupported Pareto points -- rho-frontier partitions
   attained by no lambda -- can be identified at analysis time.

Local smoke test:
    python experiments/run_w1tau.py dataset=german n_seeds=1
HPC:
    python experiments/run_w1tau.py -m dataset=german,taiwan,gmsc,hmeq \
        seed_offset=range(0,10) n_seeds=1 coordinate=rank
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
from experiments.common import (eval_binning, expanded_features,  # noqa: E402
                                feature_array, make_arm, save_results,
                                splits_hash, to_coordinate)
from optbinning.binning.metrics import wasserstein_1d   # noqa: E402


def _w1_of(splits, x, y):
    """Binned W1 of a fitted partition on (x, y), pooled-mean atoms."""
    from experiments.common import binned_stats
    ne, e, w = binned_stats(x, y, splits)
    if len(w) < 2:
        return 0.0
    return float(wasserstein_1d(ne / ne.sum(), e / e.sum(), w))


def run(cfg):
    ds = datasets.load(cfg.dataset)
    features = expanded_features(
        ds, list(cfg.features) if cfg.get("features") else None,
        cfg.get("special_handling", "expand"))
    rows = []
    for seed in range(cfg.seed_offset, cfg.seed_offset + cfg.n_seeds):
        tr, te = datasets.split_indices(len(ds.y), cfg.test_size, seed)
        for feat in features:
            x = feature_array(ds, feat,
                              cfg.get("special_handling", "expand"))
            mask = np.isfinite(x)
            xtr, ytr = x[tr][mask[tr]], ds.y[tr][mask[tr]]
            xte, yte = x[te][mask[te]], ds.y[te][mask[te]]
            coord = cfg.get("coordinate", "rank")
            xtr, xte = to_coordinate(xtr, xte, ytr, kind=coord)

            common = dict(dataset=ds.name, feature=feat, seed=seed,
                          coordinate=coord)
            try:
                iv_fit = make_arm("iv_mip",
                                  monotonic=cfg.monotonic).fit(xtr, ytr)
                w1_fit = make_arm("w1",
                                  monotonic=cfg.monotonic).fit(xtr, ytr)
            except Exception as err:                # pragma: no cover
                rows.append(dict(status="ERROR:" + type(err).__name__,
                                 **common))
                continue
            w1_floor = _w1_of(iv_fit.splits, xtr, ytr)
            w1_ceil = _w1_of(w1_fit.splits, xtr, ytr)
            for arm, fit in (("iv_mip", iv_fit), ("w1", w1_fit)):
                row = dict(arm=arm, status=fit.status, frac=np.nan,
                           rho=np.nan, w1_floor=w1_floor, w1_ceil=w1_ceil,
                           splits_hash=splits_hash(fit.splits),
                           n_splits=len(fit.splits),
                           w1_in=_w1_of(fit.splits, xtr, ytr), **common)
                row.update(eval_binning(fit.splits, xte, yte))
                rows.append(row)

            for frac in cfg.fracs:
                rho = w1_floor + float(frac) * (w1_ceil - w1_floor)
                t0 = time.perf_counter()
                try:
                    optb = make_arm("w1_tau", fm_tau=rho,
                                    monotonic=cfg.monotonic).fit(xtr, ytr)
                    status, splits = optb.status, optb.splits
                except Exception as err:            # pragma: no cover
                    rows.append(dict(arm="w1_tau", frac=float(frac),
                                     rho=rho,
                                     status="ERROR:" + type(err).__name__,
                                     **common))
                    continue
                row = dict(arm="w1_tau", status=status, frac=float(frac),
                           rho=rho, w1_floor=w1_floor, w1_ceil=w1_ceil,
                           splits_hash=splits_hash(splits),
                           n_splits=len(splits),
                           w1_in=_w1_of(splits, xtr, ytr),
                           fit_time=time.perf_counter() - t0, **common)
                row.update(eval_binning(splits, xte, yte))
                rows.append(row)

            # lambda-path reference points on the same feature/coordinate
            scale = np.subtract(*np.nanquantile(xtr, [0.95, 0.05]))
            for g in cfg.gammas:
                gam = float(g) / max(abs(scale), 1e-9)
                try:
                    optb = make_arm("iv_w1", gamma=gam,
                                    monotonic=cfg.monotonic).fit(xtr, ytr)
                except Exception as err:            # pragma: no cover
                    rows.append(dict(arm="iv_w1", frac=np.nan, rho=np.nan,
                                     gamma=float(g),
                                     status="ERROR:" + type(err).__name__,
                                     **common))
                    continue
                row = dict(arm="iv_w1", status=optb.status, frac=np.nan,
                           rho=np.nan, gamma=float(g),
                           w1_floor=w1_floor, w1_ceil=w1_ceil,
                           splits_hash=splits_hash(optb.splits),
                           n_splits=len(optb.splits),
                           w1_in=_w1_of(optb.splits, xtr, ytr), **common)
                row.update(eval_binning(optb.splits, xte, yte))
                rows.append(row)

    out = Path(cfg.out) / "w1tau_{}_{}_{}".format(
        cfg.dataset, cfg.get("coordinate", "rank"), cfg.seed_offset)
    path = save_results(rows, out, cfg=cfg)
    print("W1TAU: wrote {} rows -> {}".format(len(rows), path))
    return path


@hydra.main(version_base=None, config_path="../conf", config_name="w1tau")
def main(cfg: DictConfig):
    return run(cfg)


if __name__ == "__main__":
    main()
