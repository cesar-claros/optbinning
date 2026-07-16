"""
P9 application drivers, torch-side testing (E-CAL / E-TOK / E-SURV).
Skipped when torch (or hydra, for the tokenization net) is missing.
"""

# Cesar Claros <cesar.claros@outlook.com>
# Copyright (C) 2026

import sys

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _scores(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    s = rng.beta(2, 3, n)
    y = (rng.uniform(0, 1, n) < np.clip(s ** 1.5 + 0.05, 0, 1)) \
        .astype(float)
    return s, y


def test_ot_calibrator_monotone_and_bounded():
    pytest.importorskip("hydra")
    from omegaconf import OmegaConf
    from experiments.run_cal import _GRID, _fit_ot

    s, y = _scores()
    cfg = OmegaConf.create(dict(n_cal_bins=8, sinkhorn_iters=15,
                                steps=120, lr=0.05, eps_start=0.3,
                                eps_end=0.03, device="cpu", seed=0))
    cal, edges = _fit_ot(s, y, cfg)
    p = cal(_GRID)
    assert np.all(np.diff(p) >= -1e-12)      # monotone map (PAV-pooled)
    assert np.all((p > 0) & (p < 1))
    assert len(edges) == 7
    assert np.all(np.diff(edges) > 0)


def test_vocabnet_soft_hard_and_utilization():
    pytest.importorskip("hydra")
    from omegaconf import OmegaConf
    from experiments.run_tok import VocabNet

    torch.manual_seed(0)
    cfg = OmegaConf.create(dict(sinkhorn_iters=10, hidden=32,
                                ft_layers=1, ft_heads=4))
    net = VocabNet("ot", n_features=3, n_bins=6, cfg=cfg)
    x = torch.rand(128, 3)
    net.train()
    tok = net.tokens(x, eps=0.1)
    assert tok.shape == (128, 3, 6)
    assert torch.allclose(tok.sum(-1), torch.ones(128, 3), atol=1e-4)
    out = net(x, eps=0.1)
    assert out.shape == (128,)
    out.sum().backward()
    assert net.ot.theta_w.grad is not None

    net.eval()
    hard = net.tokens(x)
    assert torch.all(hard.sum(-1) == 1)      # one-hot lookup
    assert torch.all((hard == 0) | (hard == 1))


def test_surv_ot_grid_ordered_and_floored():
    pytest.importorskip("hydra")
    from omegaconf import OmegaConf
    from experiments.run_surv import _grid_ot, _min_events, _synthetic

    x, t, delta = _synthetic(3000, seed=0, censor_scale=3.0)
    cfg = OmegaConf.create(dict(n_intervals=6, steps=200, lr=0.05,
                                min_events=15, floor_weight=0.001,
                                device="cpu", seed=0))
    edges = _grid_ot(x, t, delta, cfg)
    assert len(edges) == 5
    assert np.all(np.diff(edges) > 0)
    assert 0 < edges[0] and edges[-1] < t.max() * 1.01
    assert _min_events(edges, t, delta) >= 5   # floor keeps events spread
