"""Experiment SCALING — G7 pre-bin scaling of the exact formulations.

For pre-bin counts 10..100, on stratified real-feature instances (easy /
median / hard by geometric opportunity) plus the synthetic designs:
solve iv_mip, the w1_tau trust constraint (rho at half range, anchored
at the same pre-bin count), and the hybrid, under monotonic auto and
none, at two time limits. Record wall-clock, status, and bin counts so
the G7 criteria (95% optimal <= 60 s at 20 pre-bins; median <= 300 s at
60) are computable from the artifact without censoring.

Local smoke test:
    python experiments/run_scaling.py dataset=german "prebins=[10,20]" \
        n_features=2
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
from experiments.common import (expanded_features, feature_array,  # noqa: E402
                                make_arm, save_results, splits_hash,
                                to_coordinate)
from experiments.run_w1tau import _w1_of                # noqa: E402


def run(cfg):
    ds = datasets.load(cfg.dataset)
    feats = expanded_features(ds, None,
                              cfg.get("special_handling", "expand"))
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(feats)
    feats = feats[:cfg.n_features]
    rows = []
    for feat in feats:
        x = feature_array(ds, feat, cfg.get("special_handling",
                                            "expand"))
        mask = np.isfinite(x)
        x, y = x[mask], ds.y[mask]
        if len(np.unique(x)) < 12:
            continue
        x, _ = to_coordinate(x, x, y, kind="rank")

        def _arm(name, npre, tl, mono, **kw):
            # ACTUALLY scale the pre-bin universe (G7 fix): quantile
            # pre-binning with a size floor below 1/npre -- the default
            # cart/0.05 pre-binner silently caps at ~20 pre-bins, which
            # made the first scaling run re-solve one problem size at
            # every nominal npre (discarded, logged)
            mono_arg = None if str(mono) == "none" else mono
            m = make_arm(name, monotonic=mono_arg, max_n_prebins=npre,
                         time_limit=tl, **kw)
            m.set_params(prebinning_method="quantile",
                         min_prebin_size=max(1.0 / (2 * npre), 1e-4))
            return m

        for npre in [int(v) for v in cfg.prebins]:
            for mono in cfg.monotonics:
                for tl in [int(v) for v in cfg.time_limits]:
                    # anchors at THIS prebin resolution
                    try:
                        t0 = time.perf_counter()
                        ivf = _arm("iv_mip", npre, tl,
                                   mono).fit(x, y)
                        t_iv = time.perf_counter() - t0
                        t0 = time.perf_counter()
                        w1f = _arm("w1", npre, tl,
                                   mono).fit(x, y)
                        t_w1 = time.perf_counter() - t0
                    except Exception as err:            # noqa: BLE001
                        rows.append(dict(
                            dataset=ds.name, feature=feat,
                            n_prebins=npre, monotonic=mono,
                            time_limit=tl, arm="anchors",
                            status="ERROR:" + type(err).__name__))
                        continue
                    lo = _w1_of(ivf.splits, x, y)
                    hi = _w1_of(w1f.splits, x, y)
                    rho = lo + 0.5 * (hi - lo)
                    arms = {
                        "iv_mip": (ivf, t_iv),
                        "w1": (w1f, t_w1),
                    }
                    for name, kw in (
                            ("w1_tau", dict(fm_tau=rho)),
                            ("hybrid", dict(gamma=1.0))):
                        t0 = time.perf_counter()
                        try:
                            fit = _arm(
                                "w1_tau" if name == "w1_tau"
                                else "iv_w1", npre, tl, mono,
                                **kw).fit(x, y)
                            arms[name] = (fit,
                                          time.perf_counter() - t0)
                        except Exception as err:        # noqa: BLE001
                            rows.append(dict(
                                dataset=ds.name, feature=feat,
                                n_prebins=npre, monotonic=mono,
                                time_limit=tl, arm=name,
                                status="ERROR:"
                                + type(err).__name__,
                                solve_time=time.perf_counter() - t0))
                    for name, (fit, tsec) in arms.items():
                        rows.append(dict(
                            dataset=ds.name, feature=feat,
                            n_prebins=npre, monotonic=mono,
                            time_limit=tl, arm=name,
                            status=fit.status, solve_time=tsec,
                            n_bins=len(fit.splits) + 1,
                            n_prebins_eff=int(
                                getattr(fit, "_n_prebins", 0)) or None,
                            splits_hash=splits_hash(fit.splits),
                            rho=rho if name == "w1_tau" else np.nan))
                    print(feat, npre, mono, tl, "done", flush=True)

    out = Path(cfg.out) / "scaling_{}_{}".format(cfg.dataset, cfg.seed)
    path = save_results(rows, out, cfg=cfg)
    print("SCALING: wrote {} rows -> {}".format(len(rows), path))
    return path


@hydra.main(version_base=None, config_path="../conf",
            config_name="scaling")
def main(cfg: DictConfig):
    return run(cfg)


if __name__ == "__main__":
    main()
