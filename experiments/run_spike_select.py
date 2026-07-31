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


def _control_kwargs(name, ob_kwargs, cfg):
    """Alternative stabilization mechanisms (re-review P1): the hybrid's
    spike-selection stabilization must beat, not merely exist beside,
    simpler controls. gamma is 0 for every control."""
    kw = dict(ob_kwargs)
    if name == "min_event":
        kw["min_bin_n_event"] = int(cfg.get("ctrl_min_event", 5))
    elif name == "min_size":
        kw["min_bin_size"] = float(cfg.get("ctrl_min_size", 0.08))
    elif name == "hellinger":
        kw["divergence"] = "hellinger"
        kw["solver"] = "cp"
        kw.pop("mip_solver", None)
    else:
        raise ValueError("unknown control: {}".format(name))
    return kw


def _fit_cut_consensus(x, y, cfg, ob_kwargs):
    """Consensus control: barycentric cuts over inner bootstrap folds."""
    from optbinning import ConsensusBinning
    kw = {k: v for k, v in ob_kwargs.items()
          if k not in ("solver", "mip_solver")}
    try:
        cb = ConsensusBinning(
            n_folds=int(cfg.get("ctrl_consensus_folds", 15)),
            random_state=0, **kw).fit(x, y)
        splits = np.asarray(cb.splits_, dtype=float) \
            if hasattr(cb, "splits_") else np.asarray(cb.splits,
                                                     dtype=float)
    except Exception:                                   # pragma: no cover
        return np.nan
    return float(splits[0]) if len(splits) >= 1 else np.nan


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
            arms = [("gamma", g) for g in gammas]
            arms += [("ctrl", c) for c in cfg.get("controls", [])]
            for kind, spec in arms:
                t0 = time.perf_counter()
                if kind == "gamma":
                    def fit(xx, yy, g=spec):
                        return _fit_cut(xx, yy, g, ob_kwargs)
                    tag, gamma = "gamma", float(spec)
                elif spec == "consensus":
                    def fit(xx, yy):
                        return _fit_cut_consensus(xx, yy, cfg, ob_kwargs)
                    tag, gamma = "ctrl:consensus", 0.0
                else:
                    ckw = _control_kwargs(spec, ob_kwargs, cfg)

                    def fit(xx, yy, k=ckw):
                        return _fit_cut(xx, yy, 0.0, k)
                    tag, gamma = "ctrl:" + spec, 0.0
                base = _choice(fit(x, y), thr)
                rng = np.random.default_rng(seed)
                choices = []
                for _ in range(cfg.n_boot):
                    idx = rng.integers(0, len(x), len(x))
                    choices.append(_choice(fit(x[idx], y[idx]), thr))
                flip = (float(np.mean([c != base for c in choices]))
                        if base != "1bin" else np.nan)
                rows.append(dict(
                    dataset=ds.name, feature=feat, seed=seed, gamma=gamma,
                    arm=tag, baseline=base, flip_rate=flip,
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
