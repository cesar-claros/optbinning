"""
Experiment — fm_tau trust-threshold sweep (Paper B).

Per dataset x feature x seed: fit the IV baseline under the MIP solver (so the
comparison is solver-matched), record its flat metric fm_ref, then sweep the
trust threshold as fractions of fm_ref. Fractions below 1 leave the constraint
slack (the arm recovers the IV solution); fractions above 1 force more
flat-metric separation than IV provides, trading away information value until
the FM <= 2*lambda ceiling makes the program infeasible. Records feasibility,
the achieved FM, and the IV retained relative to the baseline, so the
IV-vs-separation frontier can be traced per feature.

Local smoke test:
    python experiments/run_fmtau.py dataset=synthetic-smooth n_seeds=1
Real data:
    python experiments/run_fmtau.py -m dataset=german,taiwan n_seeds=5
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
from experiments.common import (binned_stats, eval_binning,  # noqa: E402
                                make_arm, save_results)
from optbinning.binning.metrics import flat_metric_1d   # noqa: E402


def _fm_of(x, y, splits, lam):
    """Flat metric FM_lambda of a fixed binning on (x, y)."""
    ne, e, w = binned_stats(x, y, splits)
    if len(w) < 2:
        return 0.0
    return float(flat_metric_1d(ne / ne.sum(), e / e.sum(), w, lam))


def run(cfg):
    ds = datasets.load(cfg.dataset, n=cfg.get("n", 5000),
                       seed=cfg.get("data_seed", 0)) \
        if str(cfg.dataset).startswith("synthetic") \
        else datasets.load(cfg.dataset)

    features = list(cfg.features) if cfg.get("features") else ds.numerical
    fracs = [float(f) for f in cfg.fm_tau_fracs]
    rows = []
    for seed in range(cfg.seed_offset, cfg.seed_offset + cfg.n_seeds):
        tr, te = datasets.split_indices(len(ds.y), cfg.test_size, seed)
        for feat in features:
            x = ds.X[feat].values.astype(float)
            mask = np.isfinite(x)
            xtr, ytr = x[tr][mask[tr]], ds.y[tr][mask[tr]]
            xte, yte = x[te][mask[te]], ds.y[te][mask[te]]

            scale = np.subtract(*np.nanquantile(xtr, [0.95, 0.05]))
            lam = float(cfg.lam_frac) * abs(scale) if scale else 1.0

            # Solver-matched IV baseline and its (achievable) flat metric.
            base = make_arm("iv_mip", monotonic=cfg.monotonic).fit(xtr, ytr)
            base_eval = eval_binning(base.splits, xte, yte)
            fm_ref = _fm_of(xtr, ytr, base.splits, lam)

            for frac in fracs:
                target = frac * fm_ref
                shared = dict(dataset=ds.name, feature=feat, seed=seed,
                              frac=frac, lam=lam, fm_ref=fm_ref,
                              fm_target=target,
                              oos_iv_base=base_eval["oos_iv"],
                              oos_w1_base=base_eval["oos_w1"],
                              n_bins_base=base_eval["n_bins"])
                arm = make_arm("fm_tau", lam=lam, fm_tau=target,
                               monotonic=cfg.monotonic)
                t0 = time.perf_counter()
                try:
                    optb = arm.fit(xtr, ytr)
                except Exception as err:            # pragma: no cover
                    rows.append(dict(shared, fit_time=time.perf_counter() - t0,
                                     status="ERROR:" + type(err).__name__))
                    continue
                fit_time = time.perf_counter() - t0

                row = dict(shared, status=optb.status, fit_time=fit_time,
                           fm_achieved=_fm_of(xtr, ytr, optb.splits, lam))
                row.update(eval_binning(optb.splits, xte, yte))
                rows.append(row)

    out = Path(cfg.out) / "fmtau_{}_{}".format(cfg.dataset, cfg.seed_offset)
    path = save_results(rows, out)
    print("fmtau: wrote {} rows -> {}".format(len(rows), path))
    return path


@hydra.main(version_base=None, config_path="../conf", config_name="fmtau")
def main(cfg: DictConfig):
    return run(cfg)


if __name__ == "__main__":
    main()
