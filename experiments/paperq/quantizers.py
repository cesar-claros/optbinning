"""Quantizer arms for the E-Q1 feasibility experiment (P8 note).

Three discrete bottlenecks at a matched code budget, applied to the
per-position latent vectors of a small autoencoder:

  VQEMA       : vanilla VQ-VAE quantizer with EMA codebook updates (the
                TiTok/VQGAN baseline; deliberately NO dead-code restarts
                -- utilization pathologies are part of what is measured).
  FSQ         : finite scalar quantization (Mentzer et al. 2023) --
                fixed uniform levels per channel on tanh-bounded
                latents, round with a straight-through estimator.
  OTQuantizer : learned-knot scalar quantizer -- per-channel OT binning
                (the P6 layer) on tanh-bounded latents. Training uses
                annealed soft barycentric dequantization (smooth, no
                straight-through); evaluation uses static interval
                quantization on the learned edges, which is
                batch-independent and auditable (the code map is the
                edge table).
"""

# Cesar Claros <cesar.claros@outlook.com>
# Copyright (C) 2026

from __future__ import annotations

import torch
from torch import Tensor, nn

from experiments.paperc.otlayer import MultiOTBinningLayer


class VQEMA(nn.Module):
    """Vanilla vector quantizer with EMA codebook updates.

    Parameters
    ----------
    n_codes:
        Codebook size N.
    dim:
        Latent channel dimension D.
    decay:
        EMA decay for cluster sizes and code sums.
    beta:
        Commitment loss weight.
    eps:
        Laplace smoothing for EMA cluster sizes.
    """

    def __init__(self, n_codes: int, dim: int, decay: float = 0.99,
                 beta: float = 0.25, eps: float = 1e-5) -> None:
        super().__init__()
        self.n_codes = n_codes
        self.decay = decay
        self.beta = beta
        self.eps = eps
        codebook = torch.randn(n_codes, dim) * 0.5
        self.register_buffer("codebook", codebook)
        self.register_buffer("cluster_size", torch.zeros(n_codes))
        self.register_buffer("embed_avg", codebook.clone())

    def codes(self, z: Tensor) -> Tensor:
        """Nearest-code indices of shape ``(batch,)``."""
        d = (z.pow(2).sum(1, keepdim=True)
             - 2 * z @ self.codebook.t()
             + self.codebook.pow(2).sum(1)[None, :])
        return d.argmin(dim=1)

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """Straight-through quantized latents and the commitment loss."""
        idx = self.codes(z)
        z_q = self.codebook[idx]
        if self.training:
            with torch.no_grad():
                onehot = nn.functional.one_hot(
                    idx, self.n_codes).type_as(z)
                self.cluster_size.mul_(self.decay).add_(
                    onehot.sum(0), alpha=1 - self.decay)
                self.embed_avg.mul_(self.decay).add_(
                    onehot.t() @ z, alpha=1 - self.decay)
                n = self.cluster_size.sum()
                size = ((self.cluster_size + self.eps)
                        / (n + self.n_codes * self.eps) * n)
                self.codebook.copy_(self.embed_avg / size[:, None])
        loss = self.beta * nn.functional.mse_loss(z, z_q.detach())
        return z + (z_q - z).detach(), loss


class FSQ(nn.Module):
    """Finite scalar quantization: fixed uniform levels per channel.

    Parameters
    ----------
    dim:
        Latent channel dimension D.
    n_levels:
        Levels per channel M (code budget M ** D).
    """

    def __init__(self, dim: int, n_levels: int) -> None:
        super().__init__()
        self.dim = dim
        self.n_levels = n_levels
        self.n_codes = n_levels ** dim

    def _round(self, zb: Tensor) -> Tensor:
        half = (self.n_levels - 1) / 2
        zq = torch.round(zb * half) / half
        return zb + (zq - zb).detach()          # straight-through

    def codes(self, z: Tensor) -> Tensor:
        """Mixed-radix product code of shape ``(batch,)``."""
        half = (self.n_levels - 1) / 2
        idx = torch.round((torch.tanh(z) + 1) * half).long()
        idx = idx.clamp(0, self.n_levels - 1)
        base = self.n_levels ** torch.arange(self.dim, device=z.device)
        return (idx * base[None, :]).sum(dim=1)

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        zq = self._round(torch.tanh(z))
        return zq, z.new_zeros(())


class OTQuantizer(nn.Module):
    """Learned-knot scalar quantizer built on the P6 OT-binning layer.

    Training path (``self.training``): annealed entropic-OT soft
    assignment with floored mass marginals (structural utilization) and
    barycentric dequantization -- smooth end to end, no straight-through
    estimator. Evaluation path: static interval quantization on the
    learned edges (midpoints of the ordered representatives), followed
    by representative lookup. The deployed code map is therefore a table
    of ``dim * (n_levels - 1)`` thresholds.

    Parameters
    ----------
    dim:
        Latent channel dimension D.
    n_levels:
        Learned levels per channel M (code budget M ** D).
    sinkhorn_iters:
        Unrolled Sinkhorn iterations in the training path.
    """

    def __init__(self, dim: int, n_levels: int,
                 sinkhorn_iters: int = 15) -> None:
        super().__init__()
        self.dim = dim
        self.n_levels = n_levels
        self.n_codes = n_levels ** dim
        self.layer = MultiOTBinningLayer(dim, n_bins=n_levels,
                                         sinkhorn_iters=sinkhorn_iters)
        self.layer.set_range(-torch.ones(dim), torch.ones(dim))

    def _interval_index(self, zb: Tensor) -> Tensor:
        edges = self.layer.bin_edges().detach()          # (D, M-1)
        return torch.searchsorted(
            edges, zb.t().contiguous()).t()              # (B, D)

    def codes(self, z: Tensor) -> Tensor:
        """Mixed-radix product code from the interval quantizer."""
        idx = self._interval_index(torch.tanh(z))
        base = self.n_levels ** torch.arange(self.dim, device=z.device)
        return (idx * base[None, :]).sum(dim=1)

    def audit(self) -> Tensor:
        """The deployed code map: per-channel edges ``(D, M-1)``."""
        return self.layer.bin_edges().detach()

    def forward(self, z: Tensor,
                eps: float = 0.1) -> tuple[Tensor, Tensor]:
        zb = torch.tanh(z)
        w = self.layer.bin_positions()                   # (D, M)
        if self.training:
            assign = self.layer(zb, eps=eps)             # (B, D, M)
            zq = (assign * w[None]).sum(dim=2)
        else:
            idx = self._interval_index(zb)               # (B, D)
            zq = w.detach().t().gather(0, idx)
        return zq, z.new_zeros(())
