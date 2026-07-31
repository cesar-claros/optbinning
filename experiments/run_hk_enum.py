"""Experiment HK-ENUM — exact transport-active HK partition optimization
by enumeration at small pre-bin counts (Paper A Sec. 7; re-review P2:
"implement one transport-active HK partition search or drop the
practical recommendations").

For each instance and each kappa on a grid: enumerate ALL 2^(n-1)
contiguous partitions of the n pre-bins (n <= 14), evaluate HK^2
exactly (bracketed primal-dual; pooled-mean representatives, raw class
counts), and record the optimal partition, its bin count, whether the
optimum is interior (1 < bins < n), the bracket budget, and -- on the
spike instance -- whether the optimizer isolates the spike. The kappa
grid traces the continuation path: Hellinger-regime behavior at small
kappa, activation flips, and coarsening as transport turns on.

Local smoke test:
    python experiments/run_hk_enum.py n_prebins=8 "kappas=[0.3,1.0]"
Full grid:
    python experiments/run_hk_enum.py -m instance=spike,slope,ushape
"""

# Cesar Claros <cesar.claros@outlook.com>
# Copyright (C) 2026

import sys
import time

from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra                                            # noqa: E402
from omegaconf import DictConfig                        # noqa: E402

from experiments.common import save_results             # noqa: E402
from optbinning.binning.uot import hk2                  # noqa: E402


def _instance(name: str, n: int, seed: int = 0):
    """Pre-bin grids with raw class counts (NE, E)."""
    rng = np.random.default_rng(seed)
    u = np.linspace(0.0, 1.0, n)
    if name == "spike":
        ne = np.full(n, 40.0)
        e = np.linspace(2, 30, n)
        ne[1], e[1] = 30.0, 0.5          # rare-event spike cell
    elif name == "slope":
        ne = np.linspace(50, 10, n)
        e = np.linspace(10, 50, n)
    elif name == "ushape":
        c = np.abs(np.linspace(-1, 1, n))
        ne = 40 * (1 - 0.6 * c) + 5
        e = 40 * (0.4 + 0.6 * c) + 5
    else:
        ne = rng.integers(5, 60, n).astype(float)
        e = rng.integers(5, 60, n).astype(float)
    return u, ne, e


def _partitions(n: int):
    """All contiguous partitions as tuples of boundary index sets."""
    for k in range(n):
        for cuts in combinations(range(1, n), k):
            yield (0,) + cuts + (n,)


def run(cfg):
    n = int(cfg.n_prebins)
    if n > 14:
        raise ValueError("enumeration capped at 14 pre-bins "
                         "(2^13 partitions); got {}.".format(n))
    u, ne, e = _instance(cfg.instance, n, cfg.get("seed", 0))
    tot = ne + e
    rows = []
    parts = list(_partitions(n))
    for kappa in [float(k) for k in cfg.kappas]:
        t0 = time.perf_counter()
        # MAXIMIZE HK^2 between the binned class-conditionals (the
        # divergence-maximization convention of optimal binning; the
        # single bin is always the Prop. 7.1 floor)
        best, best_val, best_gap = None, -np.inf, np.nan
        for bounds in parts:
            reps, a, b = [], [], []
            for lo, hi in zip(bounds[:-1], bounds[1:]):
                m = tot[lo:hi].sum()
                reps.append(float((u[lo:hi] * tot[lo:hi]).sum() / m))
                a.append(float(ne[lo:hi].sum()))
                b.append(float(e[lo:hi].sum()))
            val, gap = hk2(np.asarray(reps), np.asarray(a),
                           np.asarray(b), kappa=kappa)
            if val > best_val:
                best, best_val, best_gap = bounds, val, gap
        n_bins = len(best) - 1
        rows.append(dict(
            instance=cfg.instance, n_prebins=n, kappa=kappa,
            best_value=best_val, bracket_gap=best_gap,
            best_n_bins=n_bins,
            interior_optimum=bool(1 < n_bins < n),
            spike_isolated=bool(1 in best and 2 in best)
            if cfg.instance == "spike" else None,
            n_partitions=len(parts),
            enum_time=time.perf_counter() - t0,
            best_bounds=str(best)))
    out = Path(cfg.out) / "hkenum_{}_{}".format(cfg.instance, n)
    path = save_results(rows, out, cfg=cfg)
    print("HK-ENUM: wrote {} rows -> {}".format(len(rows), path))
    return path


@hydra.main(version_base=None, config_path="../conf",
            config_name="hkenum")
def main(cfg: DictConfig):
    return run(cfg)


if __name__ == "__main__":
    main()
