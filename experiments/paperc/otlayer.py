"""Differentiable OT-binning layer, PyTorch implementation (Paper C / P6).

Entropic-OT soft assignment of a feature batch to ``n_bins`` learnable bins
with learnable mass marginals, floored parametrizations (the bin-collapse
cure of P6 Sec. 4.1), an isotonic (PAV) monotonicity penalty, and two
hardening paths: batchwise argmax (diagnostic; contiguous at every
temperature by the single-crossing property) and mass-coordinate static
hardening (deployment; tie-aware cuts from the learned cumulative masses
with hard-mass error <= 2*a_max, exact on the atom grid, plus explicit
PAV block merging for exactly monotone hard rates -- the corrected
Theorem 4.1). Gradients flow through unrolled log-domain Sinkhorn
iterations.
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
                 learn_masses: bool = True, min_gap: float = _MIN_GAP,
                 mass_floor: float = _MASS_FLOOR) -> None:
        super().__init__()
        if n_bins < 2:
            raise ValueError(f"n_bins must be >= 2; got {n_bins}.")
        self.n_bins = n_bins
        self.sinkhorn_iters = sinkhorn_iters
        self.min_gap = min_gap          # collapse cure: ablatable (C1)
        self.mass_floor = mass_floor
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
        inc = self.min_gap + nn.functional.softplus(self.theta_w)
        cum = torch.cumsum(inc, dim=0)
        unit = (cum - inc / 2) / cum[-1]
        return self.x_lo + (self.x_hi - self.x_lo) * (0.02 + 0.96 * unit)

    def bin_masses(self) -> Tensor:
        """Bin mass marginals with an additive floor (no empty bins)."""
        beta = torch.softmax(self.theta_b, dim=0)
        return ((1 - self.mass_floor) / self.n_bins
                + self.mass_floor * beta)

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

    @torch.no_grad()
    def harden_static(self, x: Tensor, y: Tensor | None = None,
                      trend: "str | list[str]" = "auto") -> list[dict]:
        """Mass-coordinate static hardening (deployment path).

        Derives the deployed cuts from the LEARNED CUMULATIVE MASSES
        rather than from representative midpoints: ties are aggregated
        into distinct atoms, each cumulative bin mass C_k is projected
        onto the nearest cumulative atom boundary (a cut can never fall
        inside a tied atom), and, when binary targets are supplied,
        adjacent bins are merged into their weighted-PAV blocks so the
        hard event rates are EXACTLY monotone. The result is a static,
        batch-invariant interval partition with checkable postconditions
        (see :func:`mass_coordinate_partition` for the returned audit
        fields).
        """
        beta = self.bin_masses().detach().cpu().numpy()
        xs = x.detach().cpu().numpy()
        ys = None if y is None else y.detach().cpu().numpy()
        trends = ([trend] * self.n_features if isinstance(trend, str)
                  else list(trend))
        if len(trends) != self.n_features:
            raise ValueError("per-feature trend list must have length "
                             f"{self.n_features}; got {len(trends)}.")
        return [mass_coordinate_partition(xs[:, i], beta[i], ys,
                                          trend=trends[i])
                for i in range(self.n_features)]

    def bin_edges(self) -> Tensor:
        """Interior bin boundaries (midpoints of consecutive learned
        representatives), shape ``(n_features, n_bins - 1)``. Strictly
        increasing by the min-gap floor; differentiable in ``theta_w``."""
        w = self.bin_positions()
        return (w[:, 1:] + w[:, :-1]) / 2

    def mass_edges(self) -> Tensor:
        """Interior knots at the learned CUMULATIVE MASSES,
        shape ``(n_features, n_bins - 1)``: ``e_k = x_lo + span * C_k``
        with ``C_k = sum_{l<=k} beta_l``. Under the rank
        reparametrization the quantile function is the identity, so
        these knots ARE the regularized quantiles of the training
        measure at the learned masses -- the unified mass-coordinate
        model (reviewer Option A): the benchmark PLE knots and the
        static-deployment coordinates (``harden_static``) are the SAME
        parameters ``beta``. Strictly increasing by the mass floor;
        differentiable in ``theta_b``."""
        c = torch.cumsum(self.bin_masses(), dim=1)[:, :-1]
        span = (self.x_hi - self.x_lo)[:, None]
        return self.x_lo[:, None] + span * c

    def interp_tokens(self, x: Tensor,
                      edges: Tensor | None = None) -> Tensor:
        """Piecewise-linear (PLE) encoding with learned knots, shape
        ``(batch, n_features, n_bins)``.

        ``edges=None`` uses the representative-midpoint knots
        (:meth:`bin_edges`, the legacy learned-knot arm); pass
        :meth:`mass_edges` for the unified mass-coordinate arm. This is
        the lossless, spline-basis token family of Gorishniy et al.
        with learnable knot positions: differentiable a.e. in both the
        input and the edges, and the bins remain contiguous intervals
        by construction (the audit table is the edge vector itself)."""
        if edges is None:
            edges = self.bin_edges()
        edges = torch.cat([self.x_lo[:, None], edges,
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


def mass_coordinate_partition(x: np.ndarray, beta: np.ndarray,
                              y: np.ndarray | None = None,
                              trend: str = "auto") -> dict:
    """Tie-aware static partition from cumulative bin masses (numpy).

    Implements the mass-coordinate hardening of the corrected theorem:

    1. aggregate equal values into distinct atoms ``u_1 < ... < u_T``
       with masses ``a_t`` and cumulative boundaries
       ``G = (0, A_1, ..., A_T = 1)``;
    2. project each learned cumulative mass ``C_k`` onto the nearest
       ``G`` entry (never inside a tied atom); duplicate or endpoint
       projections drop the corresponding boundary (the bin is absorbed
       -- counted, not hidden);
    3. place each interior cut at the midpoint of the adjacent distinct
       atoms;
    4. with binary ``y``: compute hard-bin event rates, pool them into
       weighted-PAV blocks, MERGE each block into one bin, and
       recompute -- the merged rates exactly satisfy the declared
       trend. Trends: ``"ascending"`` / ``"descending"``; ``"auto"``
       picks the direction with the smaller weighted SSE; ``"peak"`` /
       ``"valley"`` allow one direction change (changepoint chosen by
       weighted SSE over all positions; two directional PAV merges);
       ``"none"`` skips merging (the unconstrained hard partition).

    Returns a dict with the audit fields: ``cuts_raw``, ``n_bins_raw``,
    ``hard_mass_raw``, ``mass_err_max`` (max ``|hard - soft|`` bin mass,
    theorem bound ``2 * a_max``), ``a_max``, ``n_absorbed``, and -- when
    ``y`` is given -- ``event_rate_raw``, ``n_violations_raw``,
    ``max_violation_raw``, ``cuts``, ``hard_mass``, ``event_rate``,
    ``trend``, ``n_merges``, ``monotone`` (exact check).
    """
    if trend not in ("auto", "ascending", "descending", "peak",
                     "valley", "none"):
        raise ValueError(f"invalid trend '{trend}'.")
    u, counts = np.unique(np.asarray(x, dtype=float), return_counts=True)
    n = counts.sum()
    grid = np.concatenate(([0.0], np.cumsum(counts) / n))   # G, len T+1
    c_int = np.cumsum(np.asarray(beta, dtype=float))[:-1]   # C_1..C_{M-1}
    # nearest cumulative atom boundary (ties resolve leftward); interior
    # C_k lie strictly in (0, 1) by the mass floor, so pos is in [1, T]
    pos = np.searchsorted(grid, c_int)
    pos = np.where(c_int - grid[pos - 1] <= grid[pos] - c_int,
                   pos - 1, pos)
    bounds = np.concatenate(([0], pos, [len(grid) - 1]))
    hard_mass_all = np.diff(grid[bounds])                   # aligned to beta
    mass_err_max = float(np.max(np.abs(hard_mass_all - beta)))
    keep = np.unique(pos[(pos > 0) & (pos < len(grid) - 1)])
    n_absorbed = len(beta) - 1 - len(keep)
    cuts_raw = ((u[keep - 1] + u[keep]) / 2 if len(keep)
                else np.empty(0))
    out: dict = {
        "cuts_raw": np.asarray(cuts_raw, dtype=float),
        "n_bins_raw": len(keep) + 1,
        "hard_mass_raw": np.diff(
            np.concatenate(([0.0], grid[keep], [1.0]))),
        "mass_err_max": mass_err_max,
        "a_max": float(counts.max() / n),
        "n_absorbed": int(n_absorbed),
        "contiguous": True,
    }
    if y is None:
        return out
    yb = np.asarray(y, dtype=float)
    idx = np.searchsorted(out["cuts_raw"], np.asarray(x, dtype=float),
                          side="right")
    m = len(keep) + 1
    tot = np.bincount(idx, minlength=m).astype(float)
    ev = np.bincount(idx, weights=yb, minlength=m)
    rate = ev / np.maximum(tot, 1.0)
    diffs = np.diff(rate)

    def dir_blocks(r, t, sign):
        """PAV blocks in one direction plus their weighted SSE."""
        blocks = _pav_blocks(sign * r, t)
        e = 0.0
        for b in blocks:
            mean = r[b].dot(t[b]) / t[b].sum()
            e += float(((r[b] - mean) ** 2 * t[b]).sum())
        return blocks, e

    if trend == "auto":
        sse = {name: dir_blocks(rate, tot, s)[1]
               for name, s in (("ascending", 1.0), ("descending", -1.0))}
        trend = min(sse, key=lambda k: sse[k])
    sign = -1.0 if trend in ("descending", "valley") else 1.0
    viol = np.maximum(-sign * diffs, 0.0)
    out["event_rate_raw"] = rate
    out["n_violations_raw"] = int((viol > 1e-15).sum())
    out["max_violation_raw"] = float(viol.max()) if len(viol) else 0.0

    changepoint = None
    if trend == "none":
        blocks = [[k] for k in range(m)]
    elif trend in ("peak", "valley"):
        s1, s2 = (1.0, -1.0) if trend == "peak" else (-1.0, 1.0)
        best = None
        for c in range(1, m):
            bl, e1 = dir_blocks(rate[:c], tot[:c], s1)
            br, e2 = dir_blocks(rate[c:], tot[c:], s2)
            if best is None or e1 + e2 < best[0]:
                best = (e1 + e2, bl,
                        [[i + c for i in b] for b in br], c)
        _, bl, br, changepoint = best
        blocks = bl + br
        n_left = len(bl)
    else:
        blocks = dir_blocks(rate, tot, sign)[0]
    keep_cut = sorted(b[-1] for b in
                      (sorted(bl) for bl in blocks))[:-1]   # block right edges
    cuts = out["cuts_raw"][np.asarray(keep_cut, dtype=int)] \
        if keep_cut else np.empty(0)
    idx2 = np.searchsorted(cuts, np.asarray(x, dtype=float), side="right")
    m2 = len(cuts) + 1
    tot2 = np.bincount(idx2, minlength=m2).astype(float)
    ev2 = np.bincount(idx2, weights=yb, minlength=m2)
    rate2 = ev2 / np.maximum(tot2, 1.0)
    if trend in ("peak", "valley"):
        s1 = 1.0 if trend == "peak" else -1.0
        mono = bool(
            np.all(s1 * np.diff(rate2[:n_left]) >= -1e-15)
            and np.all(-s1 * np.diff(rate2[n_left:]) >= -1e-15))
    elif trend == "none":
        mono = bool(np.all(np.diff(rate2) >= -1e-15)
                    or np.all(np.diff(rate2) <= 1e-15))
    else:
        mono = bool(np.all(sign * np.diff(rate2) >= -1e-15))
    out.update(cuts=np.asarray(cuts, dtype=float),
               hard_mass=tot2 / n, event_rate=rate2, trend=trend,
               n_merges=out["n_bins_raw"] - m2, monotone=mono,
               changepoint=changepoint)
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


def pav_penalty_multi(assign: Tensor, y: Tensor,
                      sign: "np.ndarray | None" = None) -> Tensor:
    """Sum of per-feature PAV monotonicity penalties (rates computed in
    one einsum; only the tiny block search loops in Python).

    ``sign`` (per-feature +1/-1, default all +1) declares each
    feature's trend direction so the auxiliary and the deployment-time
    merge penalize the SAME direction (reviewer alignment fix): blocks
    are found on ``sign * rate``. Gradient note, stated exactly: block
    membership is DETACHED; gradients flow through both the rates and
    the mass weights inside each block (the locally-fixed-block
    derivative, not the constant-weight approximation)."""
    y1 = (y == 1).float()
    events = torch.einsum("bfm,b->fm", assign, y1)
    total = assign.sum(dim=0).clamp_min(1e-8)
    rate = events / total
    mass = total / total.sum(dim=1, keepdim=True)

    penalty = rate.new_zeros(())
    rate_np = rate.detach().cpu().numpy()
    mass_np = mass.detach().cpu().numpy()
    signs = (np.ones(assign.shape[1]) if sign is None
             else np.asarray(sign, dtype=float))
    for i in range(assign.shape[1]):
        for block in _pav_blocks(signs[i] * rate_np[i], mass_np[i]):
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
