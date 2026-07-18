"""
E-Q1 quantizer arm testing (P8 note). Skipped when torch is missing.
"""

# Cesar Claros <cesar.claros@outlook.com>
# Copyright (C) 2026

import sys

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.paperq.quantizers import (FSQ, OTQuantizer,  # noqa: E402
                                           VQEMA)


def test_vqema_shapes_and_ema():
    torch.manual_seed(0)
    q = VQEMA(n_codes=32, dim=4)
    z = torch.randn(256, 4, requires_grad=True)     # encoder output
    before = q.codebook.clone()
    q.train()
    zq, loss = q(z)
    assert zq.shape == z.shape and loss.item() >= 0
    assert not torch.allclose(q.codebook, before)   # EMA moved codes
    zq.sum().backward()                             # straight-through
    assert z.grad is not None
    assert float(z.grad.abs().sum()) > 0
    codes = q.codes(z)
    assert codes.shape == (256,) and codes.max() < 32


def test_fsq_levels_and_codes():
    torch.manual_seed(0)
    q = FSQ(dim=3, n_levels=4)
    z = torch.randn(128, 3, requires_grad=True)
    zq, _ = q(z)
    # quantized values on the fixed uniform grid in [-1, 1]
    grid = torch.linspace(-1, 1, 4)
    d = (zq[..., None] - grid).abs().min(-1).values
    assert float(d.max()) < 1e-6
    zq.sum().backward()
    assert z.grad is not None                       # STE passes gradients
    codes = q.codes(z)
    assert codes.max() < 4 ** 3


def test_ot_quantizer_soft_hard_and_grads():
    torch.manual_seed(0)
    q = OTQuantizer(dim=4, n_levels=8)
    z = torch.randn(512, 4, requires_grad=True)
    q.train()
    zq, commit = q(z, eps=0.1)
    assert zq.shape == z.shape
    assert float(commit) > 0                    # commitment active
    commit.backward(retain_graph=True)
    assert z.grad is not None                   # collapse visible to enc
    z.grad = None
    zq.sum().backward()
    assert z.grad is not None and float(z.grad.abs().sum()) > 0
    assert q.layer.theta_w.grad is not None         # knots trainable

    q.eval()
    with torch.no_grad():
        zq_hard, _ = q(z)
    w = q.layer.bin_positions()
    d = (zq_hard[:, :, None] - w[None]).abs().min(-1).values
    assert float(d.max()) < 1e-6                    # values are knots


def test_ot_quantizer_batch_independence():
    # the deployed (eval) quantizer must be a static per-sample map:
    # codes of a sample cannot depend on what else is in the batch.
    torch.manual_seed(1)
    q = OTQuantizer(dim=3, n_levels=5)
    q.eval()
    z = torch.randn(64, 3)
    with torch.no_grad():
        full = q.codes(z)
        single = torch.cat([q.codes(z[i:i + 1]) for i in range(64)])
    assert torch.equal(full, single)


def test_ot_quantizer_structural_utilization():
    # per-channel: every level receives mass under a spread input
    # (floored marginals + interval cells); audit table is ordered.
    torch.manual_seed(0)
    q = OTQuantizer(dim=2, n_levels=4)
    q.eval()
    z = torch.atanh(torch.rand(4000, 2) * 1.9 - 0.95)
    idx = q._interval_index(torch.tanh(z))
    for d in range(2):
        assert len(torch.unique(idx[:, d])) == 4
    edges = q.audit()
    assert torch.all(torch.diff(edges, dim=1) > 0)
    assert torch.all(edges > -1) and torch.all(edges < 1)
