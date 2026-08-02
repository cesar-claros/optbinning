"""Experiment EVIDENCE — evidence-response phase diagram (plan E1;
re-review G5: evidence responsiveness must be measured, not asserted).

Ordered three-cluster spike designs over a factorial grid in

    (n, spike mass p_a, spike event ratio q_a/p_a, neighbor contrast).

For every cell and method, record the probability (over independent
samples) of selecting the spike-isolating structure, so the evidence
path P(select spike) vs effective spike event count can be compared:

  - a hard rule should switch only when its feasibility threshold is
    crossed;
  - the hybrid should transition smoothly as evidence accumulates;
  - smoothed IV (add-alpha pseudocounts on the prebin table) is the
    statistical control.

Methods: iv, hybrid at gamma grid, min_event, min_size, smoothed IV.
All share the 2-bin cap and prebin protocol of the spike-selection
driver so results compose with the earlier controls table.

Local smoke test:
    python experiments/run_evidence.py n_rep=5 "ns=[2000]"
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

from experiments.common import save_results             # noqa: E402
from optbinning import OptimalBinning                   # noqa: E402


def _sample(rng, n, p_a, ratio_a, ratio_b):
    """Three-cluster design: spike at 0.05 (mass p_a, event ratio
    ratio_a), body at 0.45 (event ratio ratio_b), tail at 0.80 (high
    rate); returns (x, y)."""
    masses = np.array([p_a, 0.60, 0.40 - p_a])
    rates = np.array([ratio_a / (1 + ratio_a),
                      ratio_b / (1 + ratio_b), 0.58])
    comp = rng.choice(3, size=n, p=masses / masses.sum())
    x = np.array([0.05, 0.45, 0.80])[comp] + rng.normal(0, 0.02, n)
    y = (rng.uniform(0, 1, n) < rates[comp]).astype(int)
    return x, y


def _select(x, y, method, spec, cfg):
    """Fit a 2-bin binning; return True iff the spike is isolated
    (cut below the spike/body midpoint), None on collapse."""
    kw = dict(dtype="numerical", solver="mip", mip_solver="cbc",
              divergence="iv", monotonic_trend=cfg.monotonic,
              max_n_bins=2, max_n_prebins=cfg.max_n_prebins,
              min_bin_size=cfg.min_bin_size,
              min_prebin_size=cfg.min_bin_size)
    if method == "hybrid":
        kw["gamma_wasserstein"] = float(spec)
    elif method == "min_event":
        kw["min_bin_n_event"] = int(spec)
    elif method == "min_size":
        kw["min_bin_size"] = float(spec)
    elif method == "smoothed_iv":
        # add-alpha pseudocounts: augment the sample with alpha events
        # and alpha nonevents inside every prebin-scale stratum, the
        # sample-level analogue of smoothing the prebin count table
        alpha = float(spec)
        qs = np.quantile(x, np.linspace(0, 1, cfg.max_n_prebins + 1))
        mids = (qs[1:] + qs[:-1]) / 2
        reps = int(np.ceil(alpha))
        x = np.concatenate([x, np.repeat(mids, 2 * reps)])
        y = np.concatenate([y, np.tile(
            np.repeat([0, 1], reps), len(mids))])
    try:
        splits = np.asarray(OptimalBinning(**kw).fit(x, y).splits,
                            dtype=float)
    except Exception:                                   # pragma: no cover
        return None
    if len(splits) < 1:
        return None
    return bool(splits[0] < 0.25)


def run(cfg):
    methods = [("iv", 0.0)]
    methods += [("hybrid", g) for g in cfg.gammas]
    methods += [("min_event", k) for k in cfg.min_events]
    methods += [("min_size", s) for s in cfg.min_sizes]
    methods += [("smoothed_iv", a) for a in cfg.alphas]
    rows = []
    for n in [int(v) for v in cfg.ns]:
        for p_a in [float(v) for v in cfg.spike_masses]:
            for ra in [float(v) for v in cfg.spike_ratios]:
                for rb in [float(v) for v in cfg.body_ratios]:
                    t0 = time.perf_counter()
                    for name, spec in methods:
                        picks, fails = [], 0
                        for rep in range(cfg.n_rep):
                            rng = np.random.default_rng(
                                hash((n, p_a, ra, rb, rep)) % 2**32)
                            x, y = _sample(rng, n, p_a, ra, rb)
                            s = _select(x, y, name, spec, cfg)
                            if s is None:
                                fails += 1
                            else:
                                picks.append(s)
                        rows.append(dict(
                            n=n, spike_mass=p_a, spike_ratio=ra,
                            body_ratio=rb,
                            expected_spike_events=n * p_a * ra / (1 + ra),
                            method=name, spec=float(spec),
                            p_isolate=float(np.mean(picks))
                            if picks else np.nan,
                            n_rep=cfg.n_rep, n_fail=fails,
                            cell_time=time.perf_counter() - t0))
    out = Path(cfg.out) / "evidence_{}".format(cfg.get("tag", "grid"))
    path = save_results(rows, out, cfg=cfg)
    print("EVIDENCE: wrote {} rows -> {}".format(len(rows), path))
    return path


@hydra.main(version_base=None, config_path="../conf",
            config_name="evidence")
def main(cfg: DictConfig):
    return run(cfg)


if __name__ == "__main__":
    main()
