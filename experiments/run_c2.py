"""Experiment C2 — end-to-end vs two-stage, torch/autodiff (Paper C).

The Sec. 5.3 fusion design under the production layer (replacing the
finite-difference prototype): a fusion weight alpha (true alpha* = 0.75)
is learned JOINTLY with the binning by maximizing soft IV through the
autodiff OT layer; the two-stage arm bins at the default alpha = 0.5.
Evaluation matches the prototype protocol: exhaustive-monotone-optimal
test IV of exact binning at the final alpha, reported as a fraction of
the oracle (exact binning at alpha*). The perturbed-optimizer baseline
is NOT ported (it wraps exact MIP solves by construction) and remains a
prototype-level result, cited as such.

HPC:
    python experiments/run_c2.py -m 'seed_offset=range(0,8)' n_seeds=1 \
        device=cuda out=outputs/c2
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

from experiments.common import save_results             # noqa: E402
from experiments.paperc.otlayer import (OTBinningLayer,  # noqa: E402
                                        pav_penalty, soft_iv)
from experiments.paperc.reference import (exact_monotone_optimum,  # noqa: E402
                                          grid_summary)

logger = logging.getLogger(__name__)

ALPHA_STAR = 0.75


def _sample(seed: int, n: int) -> tuple[np.ndarray, np.ndarray,
                                        np.ndarray]:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    s = ALPHA_STAR * x1 + (1 - ALPHA_STAR) * x2
    rate = 1 / (1 + np.exp(-2.2 * s))
    y = (rng.uniform(0, 1, n) < rate).astype(int)
    return x1, x2, y


def _exact_iv_at_alpha(alpha: float, x1: np.ndarray, x2: np.ndarray,
                       y: np.ndarray, cfg: DictConfig) -> float:
    fused = alpha * x1 + (1 - alpha) * x2
    reps, ne, ev = grid_summary(fused, y, cfg.n_prebins)
    return float(exact_monotone_optimum(ne, ev, cfg.n_bins))


def _end_to_end(x1: np.ndarray, x2: np.ndarray, y: np.ndarray,
                cfg: DictConfig) -> float:
    """Jointly learn alpha and the binning by soft-IV ascent."""
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.get("torch_seed", 0))
    t1 = torch.as_tensor(x1, dtype=torch.float32, device=device)
    t2 = torch.as_tensor(x2, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    alpha = torch.nn.Parameter(torch.tensor(0.5, device=device))
    layer = OTBinningLayer(n_bins=cfg.n_bins,
                           sinkhorn_iters=cfg.sinkhorn_iters).to(device)
    optim = torch.optim.Adam(
        list(layer.parameters()) + [alpha], lr=cfg.lr)
    for step in range(cfg.train_steps):
        frac = step / max(cfg.train_steps - 1, 1)
        eps = cfg.eps_start * (cfg.eps_end / cfg.eps_start) ** frac
        rho = cfg.rho_max * min(1.0, 2 * frac)
        fused = alpha * t1 + (1 - alpha) * t2
        # normalize per step: binning is invariant to monotone
        # rescaling, so only the direction of alpha is identified
        # (prototype identifiability note carries over).
        lo, hi = fused.min().detach(), fused.max().detach()
        fused = (fused - lo) / (hi - lo + 1e-9)
        assign = layer(fused, eps=eps)
        loss = -soft_iv(assign, yt) + rho * pav_penalty(assign, yt)
        optim.zero_grad()
        loss.backward()
        optim.step()
    return float(alpha.detach().cpu())


def run(cfg: DictConfig) -> Path:
    rows = []
    for seed in range(cfg.seed_offset, cfg.seed_offset + cfg.n_seeds):
        x1, x2, y = _sample(seed, cfg.n)
        xt1, xt2, yt = _sample(seed + 10_000, cfg.n_test)

        oracle = _exact_iv_at_alpha(ALPHA_STAR, xt1, xt2, yt, cfg)
        start = time.perf_counter()
        a_hat = _end_to_end(x1, x2, y, cfg)
        t_e2e = time.perf_counter() - start
        for method, alpha in [("end_to_end", a_hat),
                              ("two_stage", 0.5),
                              ("oracle", ALPHA_STAR)]:
            iv = _exact_iv_at_alpha(alpha, xt1, xt2, yt, cfg)
            rows.append(dict(
                seed=seed, method=method, alpha=float(alpha),
                test_iv=iv, frac_of_oracle=iv / oracle,
                time=t_e2e if method == "end_to_end" else 0.0))
        logger.info("seed %d: alpha_hat=%.3f frac=%.3f", seed, a_hat,
                    rows[-3]["frac_of_oracle"])

    out = Path(cfg.out) / f"c2_{cfg.seed_offset}"
    path = save_results(rows, out, cfg=cfg)
    logger.info("C2: wrote %d rows -> %s", len(rows), path)
    return path


@hydra.main(version_base=None, config_path="../conf", config_name="c2")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
