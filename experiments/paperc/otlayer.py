"""Differentiable OT-binning layer, PyTorch implementation (Paper C / P6).

Entropic-OT soft assignment of a feature batch to ``n_bins`` learnable bins
with learnable mass marginals, floored parametrizations (the bin-collapse
cure of P6 Sec. 4.1), an isotonic (PAV) monotonicity penalty, and annealed
hardening whose limit is a.s. a contiguous monotone partition (P6,
Thm. 3.1). Gradients flow through unrolled log-domain Sinkhorn iterations.
"""

# Cesar Claros <cesar.claros@outlook.com>
# Copyright (C) 2026

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from experiments.paperc.pav import _pav_blocks

_MIN_GAP = 0.35
_MASS_FLOOR = 0.95


class OTBinningLayer(nn.Module):
    """Soft binning of a scalar feature via entropic optimal transport.

    Parameters
    ----------
    n_bins:
        Number of bins (transport targets).
    sinkhorn_iters:
        Unrolled log-domain Sinkhorn iterations per forward pass.
    learn_masses:
        If False, bin masses are fixed uniform (SwAV-style equipartition).

    Notes
    -----
    ``forward`` expects the feature scaled to roughly [0, 1] (use
    :meth:`set_range` with training statistics). The entropic temperature
    ``eps`` is a forward argument so the trainer controls annealing.
    """

    def __init__(self, n_bins: int = 8, sinkhorn_iters: int = 40,
                 learn_masses: bool = True) -> None:
        super().__init__()
        if n_bins < 2:
            raise ValueError(f"n_bins must be >= 2; got {n_bins}.")
        self.n_bins = n_bins
        self.sinkhorn_iters = sinkhorn_iters
        self.theta_w = nn.Parameter(torch.zeros(n_bins))
        self.theta_b = nn.Parameter(torch.zeros(n_bins),
                                    requires_grad=learn_masses)
        self.register_buffer("x_lo", torch.tensor(0.0))
        self.register_buffer("x_hi", torch.tensor(1.0))

    def set_range(self, lo: float, hi: float) -> None:
        """Fix the feature range used to place bin representatives."""
        self.x_lo.fill_(lo)
        self.x_hi.fill_(hi)

    def bin_positions(self) -> Tensor:
        """Ordered bin representatives with a floored minimum separation."""
        inc = _MIN_GAP + nn.functional.softplus(self.theta_w)
        cum = torch.cumsum(inc, dim=0)
        unit = (cum - inc / 2) / cum[-1]
        return self.x_lo + (self.x_hi - self.x_lo) * (0.02 + 0.96 * unit)

    def bin_masses(self) -> Tensor:
        """Bin mass marginals with an additive floor (no empty bins)."""
        beta = torch.softmax(self.theta_b, dim=0)
        return (1 - _MASS_FLOOR) / self.n_bins + _MASS_FLOOR * beta

    def forward(self, x: Tensor, eps: float = 0.1) -> Tensor:
        """Soft assignment matrix of shape ``(batch, n_bins)``.

        Rows are the conditional transport shares of each observation
        (each row sums to one).
        """
        w = self.bin_positions()
        beta = self.bin_masses()
        cost = (x[:, None] - w[None, :]) ** 2
        log_k = -cost / eps
        log_a = -torch.log(torch.tensor(float(len(x)), device=x.device))
        log_b = torch.log(beta)

        f = torch.zeros(len(x), device=x.device)
        g = torch.zeros(self.n_bins, device=x.device)
        for _ in range(self.sinkhorn_iters):
            g = -eps * torch.logsumexp(
                log_k + (f / eps + log_a)[:, None], dim=0) + eps * log_b
            f = -eps * torch.logsumexp(
                log_k + (g / eps)[None, :], dim=1)
        plan = torch.exp(log_k + (f / eps + log_a)[:, None]
                         + (g / eps)[None, :])
        return plan / plan.sum(dim=1, keepdim=True).clamp_min(1e-30)

    @torch.no_grad()
    def harden(self, x: Tensor, eps: float = 0.003) -> dict:
        """Hard bin assignment at low temperature.

        Returns a dict with per-point bin indices, cut values (midpoints
        between adjacent occupied bins along the sorted feature), and a
        contiguity flag (P6 Thm. 3.1 predicts True).
        """
        assign = self.forward(x, eps=eps).argmax(dim=1)
        order = torch.argsort(x)
        sorted_assign = assign[order]
        contiguous = bool(torch.all(torch.diff(sorted_assign) >= 0))
        xs = x[order]
        change = torch.nonzero(torch.diff(sorted_assign) != 0).flatten()
        cuts = ((xs[change] + xs[change + 1]) / 2).cpu().numpy()
        return {"assign": assign.cpu().numpy(),
                "cuts": np.sort(np.unique(cuts)),
                "contiguous": contiguous}


class MultiOTBinningLayer(nn.Module):
    """Vectorized OT-binning of ``n_features`` scalar features at once.

    Functionally identical to ``n_features`` independent
    :class:`OTBinningLayer` instances, but runs a single batched Sinkhorn
    on ``(batch, n_features, n_bins)`` tensors — one fused kernel sequence
    per iteration instead of a per-feature Python loop, which is the
    difference between idle and saturated GPU on wide tabular data.
    """

    def __init__(self, n_features: int, n_bins: int = 8,
                 sinkhorn_iters: int = 15,
                 learn_masses: bool = True) -> None:
        super().__init__()
        if n_bins < 2:
            raise ValueError(f"n_bins must be >= 2; got {n_bins}.")
        self.n_features = n_features
        self.n_bins = n_bins
        self.sinkhorn_iters = sinkhorn_iters
        self.theta_w = nn.Parameter(torch.zeros(n_features, n_bins))
        self.theta_b = nn.Parameter(torch.zeros(n_features, n_bins),
                                    requires_grad=learn_masses)
        self.register_buffer("x_lo", torch.zeros(n_features))
        self.register_buffer("x_hi", torch.ones(n_features))

    def set_range(self, lo: Tensor, hi: Tensor) -> None:
        """Fix per-feature ranges used to place bin representatives."""
        self.x_lo.copy_(lo)
        self.x_hi.copy_(hi)

    def bin_positions(self) -> Tensor:
        """Ordered representatives, shape ``(n_features, n_bins)``."""
        inc = _MIN_GAP + nn.functional.softplus(self.theta_w)
        cum = torch.cumsum(inc, dim=1)
        unit = (cum - inc / 2) / cum[:, -1:]
        span = (self.x_hi - self.x_lo)[:, None]
        return self.x_lo[:, None] + span * (0.02 + 0.96 * unit)

    def bin_masses(self) -> Tensor:
        """Mass marginals, shape ``(n_features, n_bins)``."""
        beta = torch.softmax(self.theta_b, dim=1)
        return (1 - _MASS_FLOOR) / self.n_bins + _MASS_FLOOR * beta

    def forward(self, x: Tensor, eps: float = 0.1) -> Tensor:
        """Soft assignments of shape ``(batch, n_features, n_bins)``;
        each ``(b, f)`` row sums to one."""
        w = self.bin_positions()
        beta = self.bin_masses()
        cost = (x[:, :, None] - w[None, :, :]) ** 2
        log_k = -cost / eps
        log_a = -torch.log(torch.tensor(float(len(x)), device=x.device))
        log_b = torch.log(beta)

        f = torch.zeros(x.shape, device=x.device)
        for _ in range(self.sinkhorn_iters):
            g = -eps * torch.logsumexp(
                log_k + (f / eps + log_a)[:, :, None], dim=0) \
                + eps * log_b
            f = -eps * torch.logsumexp(
                log_k + (g / eps)[None, :, :], dim=2)
        plan = torch.exp(log_k + (f / eps + log_a)[:, :, None]
                         + (g / eps)[None, :, :])
        return plan / plan.sum(dim=2, keepdim=True).clamp_min(1e-30)

    def bin_edges(self) -> Tensor:
        """Interior bin boundaries (midpoints of consecutive learned
        representatives), shape ``(n_features, n_bins - 1)``. Strictly
        increasing by the min-gap floor; differentiable in ``theta_w``."""
        w = self.bin_positions()
        return (w[:, 1:] + w[:, :-1]) / 2

    def interp_tokens(self, x: Tensor) -> Tensor:
        """Piecewise-linear (PLE) encoding with the LEARNED bin edges as
        knots, shape ``(batch, n_features, n_bins)``.

        This is the lossless, spline-basis token family of Gorishniy et
        al. with learnable knot positions: differentiable a.e. in both
        the input and the edges, and the bins remain contiguous intervals
        by construction (the audit table is the edge vector itself)."""
        edges = torch.cat([self.x_lo[:, None], self.bin_edges(),
                           self.x_hi[:, None]], dim=1)
        width = (edges[:, 1:] - edges[:, :-1]).clamp_min(1e-9)
        return ((x[:, :, None] - edges[None, :, :-1])
                / width[None]).clamp(0.0, 1.0)

    @torch.no_grad()
    def harden(self, x: Tensor, eps: float = 0.003,
               iters: int = 60) -> list[dict]:
        """Per-feature hard assignment at low temperature (full-precision
        Sinkhorn only here, where exactness matters)."""
        saved = self.sinkhorn_iters
        self.sinkhorn_iters = iters
        assign = self.forward(x, eps=eps).argmax(dim=2)
        self.sinkhorn_iters = saved
        out = []
        for i in range(self.n_features):
            order = torch.argsort(x[:, i])
            sorted_assign = assign[order, i]
            xs = x[order, i]
            change = torch.nonzero(torch.diff(sorted_assign) != 0).flatten()
            cuts = ((xs[change] + xs[change + 1]) / 2).cpu().numpy()
            out.append({
                "contiguous": bool(torch.all(
                    torch.diff(sorted_assign) >= 0)),
                "cuts": np.sort(np.unique(cuts))})
        return out


def soft_iv_multi(assign: Tensor, y: Tensor) -> Tensor:
    """Sum of per-feature IVs of a ``(batch, n_features, n_bins)`` soft
    assignment (single einsum pass; no per-feature loop)."""
    y0 = (y == 0).float()
    y1 = 1.0 - y0
    p = torch.einsum("bfm,b->fm", assign, y0) / y0.sum().clamp_min(1.0)
    q = torch.einsum("bfm,b->fm", assign, y1) / y1.sum().clamp_min(1.0)
    p = p.clamp_min(1e-8)
    q = q.clamp_min(1e-8)
    return ((p - q) * torch.log(p / q)).sum()


def pav_penalty_multi(assign: Tensor, y: Tensor) -> Tensor:
    """Sum of per-feature PAV monotonicity penalties (rates computed in
    one einsum; only the tiny block search loops in Python)."""
    y1 = (y == 1).float()
    events = torch.einsum("bfm,b->fm", assign, y1)
    total = assign.sum(dim=0).clamp_min(1e-8)
    rate = events / total
    mass = total / total.sum(dim=1, keepdim=True)

    penalty = rate.new_zeros(())
    rate_np = rate.detach().cpu().numpy()
    mass_np = mass.detach().cpu().numpy()
    for i in range(assign.shape[1]):
        for block in _pav_blocks(rate_np[i], mass_np[i]):
            idx = torch.as_tensor(block, device=rate.device)
            w = mass[i, idx]
            mean = (rate[i, idx] * w).sum() / w.sum()
            penalty = penalty + (w * (rate[i, idx] - mean) ** 2).sum()
    return penalty


def soft_iv(assign: Tensor, y: Tensor) -> Tensor:
    """Differentiable IV of a soft assignment against binary targets."""
    y0 = (y == 0).float()
    y1 = 1.0 - y0
    p = (assign * y0[:, None]).sum(dim=0) / y0.sum().clamp_min(1.0)
    q = (assign * y1[:, None]).sum(dim=0) / y1.sum().clamp_min(1.0)
    p = p.clamp_min(1e-8)
    q = q.clamp_min(1e-8)
    return ((p - q) * torch.log(p / q)).sum()


def pav_penalty(assign: Tensor, y: Tensor) -> Tensor:
    """Mass-weighted squared distance of soft event rates to their
    isotonic projection (blocks detached; gradients flow through the
    block means, matching the a.e. PAV Jacobian)."""
    y1 = (y == 1).float()
    events = (assign * y1[:, None]).sum(dim=0)
    total = assign.sum(dim=0).clamp_min(1e-8)
    rate = events / total
    mass = total / total.sum()

    blocks = _pav_blocks(rate.detach().cpu().numpy(),
                         mass.detach().cpu().numpy())
    penalty = rate.new_zeros(())
    for block in blocks:
        idx = torch.as_tensor(block, device=rate.device)
        w = mass[idx]
        mean = (rate[idx] * w).sum() / w.sum()
        penalty = penalty + (w * (rate[idx] - mean) ** 2).sum()
    return penalty


# _pav_blocks lives in experiments.paperc.pav (torch-free; re-exported
# above for backward compatibility of imports from this module).
