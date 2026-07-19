"""Experiment C1 — torch layer parity and MIP-solution recovery (Paper C).

Trains the autodiff OT-binning layer on raw observations, hardens, applies
the reduced-space polish, and reports the optimality gap to the exhaustive
monotone optimum on a quantile pre-bin grid — side by side with the NumPy
reference implementation (optbinning.binning.soft.SoftBinning).

Local smoke test:
    python experiments/run_c1.py n_seeds=1 n=2500 train_steps=150
HPC:
    python experiments/run_c1.py -m hydra/launcher=submitit_slurm \
        seed_offset=range(0,8) n_seeds=1
"""

# Cesar Claros <cesar.claros@outlook.com>
# Copyright (C) 2026

import logging
import sys
import time

from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra                                            # noqa: E402
import torch                                            # noqa: E402
from omegaconf import DictConfig                        # noqa: E402

from optbinning.binning.soft import SoftBinning         # noqa: E402

from experiments.common import save_results             # noqa: E402
from experiments.paperc.otlayer import (OTBinningLayer,  # noqa: E402
                                        pav_penalty, soft_iv)
from experiments.paperc.reference import (cuts_to_bounds,  # noqa: E402
                                          exact_monotone_optimum,
                                          grid_summary, iv_monotone, polish)

logger = logging.getLogger(__name__)


def _sample(seed: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    rate = 1 / (1 + np.exp(-(4 * (x / 3) + rng.normal(0, 0.25, n))))
    y = (rng.uniform(0, 1, n) < rate).astype(int)
    return x, y


def _train_torch(x: np.ndarray, y: np.ndarray, cfg: DictConfig,
                 floors: bool = True) -> dict:
    device = torch.device(cfg.device)
    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    lo, hi = float(np.quantile(x, 0.005)), float(np.quantile(x, 0.995))
    xt = (xt - lo) / (hi - lo)

    # floors=False ablates the bin-collapse cure (Sec. 5.2): no minimum
    # knot separation, no additive mass floor -- pure softmax masses.
    layer = OTBinningLayer(
        n_bins=cfg.n_bins, sinkhorn_iters=cfg.sinkhorn_iters,
        min_gap=0.35 if floors else 1e-4,
        mass_floor=0.95 if floors else 1.0).to(device)
    layer.set_range(0.0, 1.0)
    optim = torch.optim.Adam(layer.parameters(), lr=cfg.lr)

    start = time.perf_counter()
    for step in range(cfg.train_steps):
        frac = step / max(cfg.train_steps - 1, 1)
        eps = cfg.eps_start * (cfg.eps_end / cfg.eps_start) ** frac
        rho = cfg.rho_max * min(1.0, 2 * frac)
        assign = layer(xt, eps=eps)
        loss = -soft_iv(assign, yt) + rho * pav_penalty(assign, yt)
        optim.zero_grad()
        loss.backward()
        optim.step()
    train_time = time.perf_counter() - start

    hard = layer.harden(xt)
    cuts = lo + (hi - lo) * hard["cuts"]
    return {"cuts": cuts, "contiguous": hard["contiguous"],
            "n_bins_used": int(len(np.unique(hard["assign"]))),
            "train_time": train_time}


def run(cfg: DictConfig) -> Path:
    rows = []
    for seed in range(cfg.seed_offset, cfg.seed_offset + cfg.n_seeds):
        x, y = _sample(seed, cfg.n)
        reps, ne, ev = grid_summary(x, y, cfg.n_prebins)
        exact = exact_monotone_optimum(ne, ev, cfg.n_bins)

        for method, floors in [("torch", True), ("torch_nofloor", False)]:
            if method == "torch_nofloor" and not cfg.get("run_nofloor",
                                                         True):
                continue
            result = _train_torch(x, y, cfg, floors=floors)
            bounds = cuts_to_bounds(result["cuts"], reps)
            iv_hard = iv_monotone(bounds, ne, ev)
            bounds_pol, iv_pol = polish(bounds, ne, ev, cfg.n_bins)
            rows.append(dict(seed=seed, method=method, n=cfg.n,
                             contiguous=result["contiguous"],
                             n_bins_used=result["n_bins_used"],
                             gap_hard=(exact - iv_hard) / exact,
                             gap_polished=(exact - iv_pol) / exact,
                             time=result["train_time"]))

        start = time.perf_counter()
        sb = SoftBinning(n_bins=cfg.n_bins, n_restarts=2,
                         random_state=seed).fit(reps, ne, ev)
        rows.append(dict(seed=seed, method="numpy", n=cfg.n,
                         contiguous=sb.contiguous_,
                         n_bins_used=np.nan, gap_hard=np.nan,
                         gap_polished=(exact - sb.iv_) / exact,
                         time=time.perf_counter() - start))
        logger.info("seed %d done", seed)

    out = Path(cfg.out) / f"c1_{cfg.n}_{cfg.seed_offset}"
    path = save_results(rows, out)
    logger.info("C1: wrote %d rows -> %s", len(rows), path)
    return path


@hydra.main(version_base=None, config_path="../conf", config_name="c1")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
