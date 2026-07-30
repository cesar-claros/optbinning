"""
Experiment — spike-selection fragility (P1 Sec. 3.4 reproduction).

Ports the worked example: on the rare-event spike design (synthetic-spike3),
pure IV's 2-bin choice -- isolate the spike (A) or merge it (B) -- is a
bootstrap coin flip, while the IV + gamma*W1 hybrid commits to B and is
reproducible across draws. Reproducing it needs the exact protocol the default
a1/gamma harness does not use: a 2-bin cap, a small min_bin_size so the
~3%-mass spike survives prebinning, and a cut-position (selection), not
cut-count, fragility read. gamma here is the raw gamma_wasserstein weight
(P1's lambda ~ 2.5 maps to ~10 at this feature scale); gamma = 0 is pure IV.

Local smoke test:
    python experiments/run_spike_select.py n_seeds=2 n_boot=50
Real run:
    python experiments/run_spike_select.py dataset=synthetic-spike3 n_seeds=20
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
from experiments.common import (expanded_features,      # noqa: E402
                                feature_array, save_results)
from optbinning import OptimalBinning                   # noqa: E402


def _fit_cut(x, y, gamma_w, ob_kwargs):
    """Fit a 2-bin monotone IV(+gamma*W1) binning; return the single cut
    position (nan if it collapses to one bin or the solve fails)."""
    kw = dict(ob_kwargs)
    if gamma_w:
        kw["gamma_wasserstein"] = gamma_w
    try:
        splits = np.asarray(OptimalBinning(**kw).fit(x, y).splits, dtype=float)
    except Exception:                                   # pragma: no cover
        return np.nan
    return float(splits[0]) if len(splits) >= 1 else np.nan


def _choice(cut, threshold):
    """A = isolate the spike (low cut), B = merge it, 1bin = collapsed."""
    if np.isnan(cut):
        return "1bin"
    return "A" if cut < threshold else "B"


def run(cfg):
    gammas = [float(g) for g in cfg.gammas]
    thr = float(cfg.select_threshold)
    ob_kwargs = dict(dtype="numerical", solver="mip", mip_solver="cbc",
                     divergence="iv", monotonic_trend=cfg.monotonic,
                     max_n_bins=2, max_n_prebins=cfg.max_n_prebins,
                     min_bin_size=cfg.min_bin_size,
                     min_prebin_size=cfg.min_bin_size)
    rows = []
    for seed in range(cfg.seed_offset, cfg.seed_offset + cfg.n_seeds):
        ds = datasets.load(cfg.dataset, n=cfg.get("n", 2000), seed=seed) \
            if str(cfg.dataset).startswith("synthetic") \
            else datasets.load(cfg.dataset)
        feats = expanded_features(
            ds, list(cfg.features) if cfg.get("features") else None,
            cfg.get("special_handling", "expand"))
        for feat in feats:
            x = feature_array(ds, feat, cfg.get("special_handling", "expand"))
            mask = np.isfinite(x)
            x, y = x[mask], ds.y[mask]
            for gamma in gammas:
                t0 = time.perf_counter()
                base = _choice(_fit_cut(x, y, gamma, ob_kwargs), thr)
                rng = np.random.default_rng(seed)
                choices = []
                for _ in range(cfg.n_boot):
                    idx = rng.integers(0, len(x), len(x))
                    choices.append(
                        _choice(_fit_cut(x[idx], y[idx], gamma, ob_kwargs),
                                thr))
                flip = (float(np.mean([c != base for c in choices]))
                        if base != "1bin" else np.nan)
                rows.append(dict(
                    dataset=ds.name, feature=feat, seed=seed, gamma=gamma,
                    baseline=base, flip_rate=flip,
                    p_isolate=float(np.mean([c == "A" for c in choices])),
                    p_infeasible=float(np.mean([c == "1bin" for c in choices])),
                    n_boot=cfg.n_boot, fit_time=time.perf_counter() - t0))

    out = Path(cfg.out) / "spikesel_{}_{}".format(cfg.dataset, cfg.seed_offset)
    path = save_results(rows, out, cfg=cfg)
    print("spikesel: wrote {} rows -> {}".format(len(rows), path))
    return path


@hydra.main(version_base=None, config_path="../conf", config_name="spikesel")
def main(cfg: DictConfig):
    return run(cfg)


if __name__ == "__main__":
    main()
