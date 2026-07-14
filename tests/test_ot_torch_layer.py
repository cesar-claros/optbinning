"""
Torch OT-binning layer testing (Paper C / P6). Skipped when torch is not
installed (the layer lives in the experiments stack, not the library).
"""

# Cesar Claros <cesar.claros@outlook.com>
# Copyright (C) 2026

import sys

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.paperc.otlayer import (OTBinningLayer,     # noqa: E402
                                        pav_penalty, soft_iv)
from experiments.paperc.reference import (cuts_to_bounds,   # noqa: E402
                                          exact_monotone_optimum,
                                          grid_summary, iv_monotone, polish)


def _data(seed=0, n=1500):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    rate = 1 / (1 + np.exp(-1.5 * x))
    y = (rng.uniform(0, 1, n) < rate).astype(int)
    return (torch.as_tensor((x - x.min()) / (x.max() - x.min()),
                            dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.float32), x, y)


def test_forward_shape_and_marginals():
    xt, _, _, _ = _data()
    layer = OTBinningLayer(n_bins=6)
    assign = layer(xt, eps=0.1)
    assert assign.shape == (len(xt), 6)
    assert torch.allclose(assign.sum(dim=1),
                          torch.ones(len(xt)), atol=1e-5)
    w = layer.bin_positions()
    assert torch.all(torch.diff(w) > 0)
    beta = layer.bin_masses()
    assert torch.all(beta > 0) and abs(float(beta.sum()) - 1) < 1e-6


def test_gradients_flow():
    xt, yt, _, _ = _data()
    layer = OTBinningLayer(n_bins=5)
    assign = layer(xt, eps=0.1)
    loss = -soft_iv(assign, yt) + pav_penalty(assign, yt)
    loss.backward()
    assert layer.theta_w.grad is not None
    assert float(layer.theta_w.grad.abs().sum()) > 0
    assert layer.theta_b.grad is not None


def test_annealed_recovery_small():
    # short-budget version of C1: contiguity (P6 Thm. 3.1) and a modest
    # polished gap to the exhaustive monotone optimum.
    xt, yt, x, y = _data(seed=1)
    layer = OTBinningLayer(n_bins=5, sinkhorn_iters=30)
    optim = torch.optim.Adam(layer.parameters(), lr=0.05)
    steps = 120
    for step in range(steps):
        frac = step / (steps - 1)
        eps = 0.25 * (0.01 / 0.25) ** frac
        assign = layer(xt, eps=eps)
        loss = -soft_iv(assign, yt) + 40 * frac * pav_penalty(assign, yt)
        optim.zero_grad()
        loss.backward()
        optim.step()

    hard = layer.harden(xt)
    assert hard["contiguous"]

    reps, ne, ev = grid_summary(x, y, 25)
    cuts = x.min() + (x.max() - x.min()) * hard["cuts"]
    bounds = cuts_to_bounds(cuts, reps)
    _, iv_pol = polish(bounds, ne, ev, 5)
    exact = exact_monotone_optimum(ne, ev, 5)
    assert (exact - iv_pol) / exact <= 0.15
    assert iv_monotone(bounds, ne, ev) > 0
