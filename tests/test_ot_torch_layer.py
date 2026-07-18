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

from experiments.paperc.backbones import FeatureTokenTransformer  # noqa: E402
from experiments.paperc.otlayer import (MultiOTBinningLayer,  # noqa: E402
                                        OTBinningLayer, pav_penalty,
                                        pav_penalty_multi, soft_iv,
                                        soft_iv_multi)
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


def test_multi_layer_parity_with_single():
    # the vectorized layer must reproduce the single-feature layer
    # exactly when parameters coincide (same recursion, batched).
    xt, yt, _, _ = _data()
    single = OTBinningLayer(n_bins=6, sinkhorn_iters=25)
    multi = MultiOTBinningLayer(1, n_bins=6, sinkhorn_iters=25)
    with torch.no_grad():
        single.theta_w.copy_(torch.randn(6) * 0.3)
        single.theta_b.copy_(torch.randn(6) * 0.3)
        multi.theta_w.copy_(single.theta_w[None, :])
        multi.theta_b.copy_(single.theta_b[None, :])

    a_single = single(xt, eps=0.08)
    a_multi = multi(xt[:, None], eps=0.08)[:, 0, :]
    assert torch.allclose(a_single, a_multi, atol=1e-5)

    iv_s = soft_iv(a_single, yt)
    iv_m = soft_iv_multi(a_multi[:, None, :], yt)
    assert torch.allclose(iv_s, iv_m, atol=1e-6)

    pen_s = pav_penalty(a_single, yt)
    pen_m = pav_penalty_multi(a_multi[:, None, :], yt)
    assert torch.allclose(pen_s, pen_m, atol=1e-6)


def test_bin_edges_and_interp_tokens():
    # learned-knot PLE (token_mode=ple_interp): edges strictly inside
    # the range and increasing; tokens in [0, 1], monotone in x, and
    # saturating at the range endpoints (0 at lo, 1 at hi).
    torch.manual_seed(0)
    multi = MultiOTBinningLayer(3, n_bins=6)
    with torch.no_grad():
        multi.theta_w.copy_(torch.randn(3, 6) * 0.4)
    lo = torch.tensor([0.0, -1.0, 2.0])
    hi = torch.tensor([1.0, 1.0, 5.0])
    multi.set_range(lo, hi)

    e = multi.bin_edges()
    assert e.shape == (3, 5)
    assert torch.all(torch.diff(e, dim=1) > 0)
    assert torch.all(e > lo[:, None]) and torch.all(e < hi[:, None])

    grid = torch.linspace(0, 1, 64)[:, None]
    x = lo[None, :] + (hi - lo)[None, :] * grid
    tok = multi.interp_tokens(x)
    assert tok.shape == (64, 3, 6)
    assert torch.all(tok >= 0) and torch.all(tok <= 1)
    assert torch.all(torch.diff(tok, dim=0) >= -1e-6)
    assert torch.allclose(tok[0], torch.zeros(3, 6), atol=1e-6)
    assert torch.allclose(tok[-1], torch.ones(3, 6), atol=1e-6)


def test_interp_tokens_gradients_reach_knots():
    # the knot positions must be trainable through the spline tokens
    # (differentiable a.e. in the edges).
    torch.manual_seed(1)
    multi = MultiOTBinningLayer(2, n_bins=5)
    with torch.no_grad():
        multi.theta_w.copy_(torch.randn(2, 5) * 0.3)
    x = torch.rand(128, 2)
    tok = multi.interp_tokens(x)
    (tok * torch.randn_like(tok)).sum().backward()
    assert multi.theta_w.grad is not None
    assert float(multi.theta_w.grad.abs().sum()) > 0


def test_feature_token_transformer():
    torch.manual_seed(0)
    net = FeatureTokenTransformer(n_features=7, token_dim=4, d_model=32,
                                  n_layers=1, n_heads=4)
    tok = torch.randn(16, 7, 4)
    out = net(tok)
    assert out.shape == (16,)
    out.sum().backward()
    assert net.weight.grad is not None
    assert float(net.weight.grad.abs().sum()) > 0
    with pytest.raises(ValueError):
        FeatureTokenTransformer(3, 2, d_model=30, n_heads=4)


@pytest.mark.filterwarnings(
    "ignore:Type google._upb:DeprecationWarning")
def test_tokenized_net_ple_interp_under_ft():
    # end-to-end: ple_interp tokens through the FT backbone train the
    # knots; need_assign=False skips Sinkhorn (assign is None).
    # (the protobuf DeprecationWarning is ortools', imported via
    # optbinning when run_c3 loads -- upstream, not ours.)
    pytest.importorskip("hydra")
    from experiments.run_c3 import TokenizedNet

    torch.manual_seed(0)
    edges = [np.linspace(0, 1, 9) for _ in range(4)]
    net = TokenizedNet("ot_ple", edges, n_bins=8, backbone="ft",
                       hidden=32, token_mode="ple_interp")
    x = torch.rand(64, 4)
    logits, assign = net(x, eps=0.1, need_assign=False)
    assert logits.shape == (64,)
    assert assign is None
    logits.sum().backward()
    assert net.ot.theta_w.grad is not None
    assert float(net.ot.theta_w.grad.abs().sum()) > 0

    logits, assign = net(x, eps=0.1, need_assign=True)
    assert assign is not None and assign.shape == (64, 4, 8)


def test_tokenized_net_ot_frozen_arm():
    # frozen-edge PLE arm: consumes full edge arrays, trains head only.
    pytest.importorskip("hydra")
    from experiments.run_c3 import TokenizedNet

    torch.manual_seed(0)
    edges = [np.concatenate(([0.0], np.sort(np.random.rand(7)), [1.0]))
             for _ in range(3)]
    net = TokenizedNet("ot_frozen", edges, n_bins=8, backbone="linear",
                       hidden=16)
    x = torch.rand(64, 3)
    logits, assign = net(x, need_assign=False)
    assert logits.shape == (64,) and assign is None
    assert not hasattr(net, "ot")            # no layer in stage 2
    tok, _ = net.tokens(x, eps=0.1)
    assert tok.shape == (64, 3, 8)
    assert torch.all(tok >= 0) and torch.all(tok <= 1)


def test_tokenized_net_sentinel_token_routing():
    # special_handling='token': sentinel entries get a one-hot in the
    # reserved trailing channels, their base encoding is zeroed, clean
    # entries are unaffected, and the aux assignment is masked.
    pytest.importorskip("hydra")
    from experiments.run_c3 import TokenizedNet

    torch.manual_seed(0)
    edges = [np.linspace(0, 1, 9)] * 2
    net = TokenizedNet("ot_ple", edges, n_bins=8, backbone="linear",
                       hidden=16, token_mode="ple_interp", n_special=3)
    assert net.token_dim == 8 + 3
    x = torch.rand(32, 2)
    codes = torch.zeros(32, 2, dtype=torch.long)
    codes[0, 0] = 2                      # one sentinel entry, code #2
    tok, assign = net.tokens(x, eps=0.1, need_assign=True, codes=codes)
    assert tok.shape == (32, 2, 11)
    assert torch.all(tok[0, 0, :8] == 0)           # base zeroed
    assert tok[0, 0, 8 + 1] == 1 and tok[0, 0, 8] == 0  # one-hot @ code
    assert torch.all(tok[1:, :, 8:] == 0)          # clean rows: no spec
    assert float(assign[0, 0].sum()) == 0          # aux masked
    assert abs(float(assign[1, 0].sum()) - 1) < 1e-4
    with pytest.raises(ValueError):
        net.tokens(x, eps=0.1, codes=None)


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
