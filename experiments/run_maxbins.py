"""
Experiment — hellinger_raw vs iv under a bin-count cap (Paper A).

Both IV and raw-count Hellinger are refinement-monotone objectives, so without
a bin-count cap they saturate to the same finest feasible partition and are
indistinguishable. This driver sweeps max_n_bins and fits the two objectives
pairwise (both under the MIP/CBC solver, so only the objective differs) to
study where and how they diverge: the split-agreement rate as the cap relaxes,
and the out-of-sample IV/W1 each objective retains at every cap.

Local smoke test:
    python experiments/run_maxbins.py dataset=synthetic-spike n_seeds=1
Real data:
    python experiments/run_maxbins.py -m dataset=german,taiwan n_seeds=5
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
                                feature_array, make_arm, save_results)

_ARMS = ("iv_mip", "hellinger_raw")


def _fit_arm(arm, xtr, ytr, xte, yte, max_n_bins, monotonic):
    """Fit one objective at a bin cap; return status, rounded cut signature,
    and out-of-sample metrics."""
    model = make_arm(arm, monotonic=monotonic, max_n_bins=max_n_bins)
    t0 = time.perf_counter()
    try:
        model.fit(xtr, ytr)
        status, splits = model.status, np.asarray(model.splits, dtype=float)
    except Exception as err:                            # pragma: no cover
        return dict(status="ERROR:" + type(err).__name__, sig=(),
                    ev=dict(oos_iv=np.nan, oos_w1=np.nan, n_bins=0),
                    fit_time=time.perf_counter() - t0)
    return dict(status=status, sig=tuple(np.round(splits, 6)),
                ev=eval_binning(splits, xte, yte),
                fit_time=time.perf_counter() - t0)


def run(cfg):
    ds = datasets.load(cfg.dataset, n=cfg.get("n", 5000),
                       seed=cfg.get("data_seed", 0)) \
        if str(cfg.dataset).startswith("synthetic") \
        else datasets.load(cfg.dataset)

    features = expanded_features(
        ds, list(cfg.features) if cfg.get("features") else None,
        cfg.get("special_handling", "expand"))
    grid = [int(m) for m in cfg.max_n_bins_grid]
    # Tag the monotone mode so auto/free runs land in distinct files and stay
    # distinguishable in the pooled table (null trend -> "free").
    mono_tag = str(cfg.monotonic) if cfg.monotonic else "free"
    rows = []
    for seed in range(cfg.seed_offset, cfg.seed_offset + cfg.n_seeds):
        tr, te = datasets.split_indices(len(ds.y), cfg.test_size, seed)
        for feat in features:
            x = feature_array(ds, feat, cfg.get("special_handling", "expand"))
            mask = np.isfinite(x)
            xtr, ytr = x[tr][mask[tr]], ds.y[tr][mask[tr]]
            xte, yte = x[te][mask[te]], ds.y[te][mask[te]]

            for mb in grid:
                iv = _fit_arm("iv_mip", xtr, ytr, xte, yte, mb, cfg.monotonic)
                he = _fit_arm("hellinger_raw", xtr, ytr, xte, yte, mb,
                              cfg.monotonic)
                shared_cuts = len(set(iv["sig"]) & set(he["sig"]))
                rows.append(dict(
                    dataset=ds.name, feature=feat, seed=seed, max_n_bins=mb,
                    monotonic=mono_tag,
                    status_iv=iv["status"], status_hell=he["status"],
                    splits_equal=iv["sig"] == he["sig"],
                    n_shared_cuts=shared_cuts,
                    n_cuts_iv=len(iv["sig"]), n_cuts_hell=len(he["sig"]),
                    iv_oos_iv=iv["ev"]["oos_iv"],
                    hell_oos_iv=he["ev"]["oos_iv"],
                    iv_oos_w1=iv["ev"]["oos_w1"],
                    hell_oos_w1=he["ev"]["oos_w1"],
                    fit_time_iv=iv["fit_time"],
                    fit_time_hell=he["fit_time"]))

    out = Path(cfg.out) / "maxbins_{}_{}_{}".format(
        cfg.dataset, mono_tag, cfg.seed_offset)
    path = save_results(rows, out)
    print("maxbins: wrote {} rows -> {}".format(len(rows), path))
    return path


@hydra.main(version_base=None, config_path="../conf", config_name="maxbins")
def main(cfg: DictConfig):
    return run(cfg)


if __name__ == "__main__":
    main()
