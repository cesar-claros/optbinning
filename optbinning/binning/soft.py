"""
Soft (differentiable) OT-binning layer, NumPy reference implementation
(OT-WoE extension; project note P6 / Paper C).

Entropic-OT soft assignment of pre-bins to learnable bins with learnable
bin-mass marginals, isotonic (PAV) monotonicity penalty, and temperature
annealing. The annealed hard limit is a.s. a contiguous partition with
monotone event rates — a feasible point of the exact binning MIP (P6,
Thm. 3.1: strict submodularity => unique northwest-corner coupling;
argmax-hardening is contiguous; PAV fixed points are monotone). The
optional polish performs reduced-space exact refinement (boundary moves +
greedy cut insertion), the warm-start role of the layer.

This is the verification-grade reference; the autodiff (torch) variant for
end-to-end training lives in the Paper C experiment stack.
"""

# Cesar Claros <cesar.claros@outlook.com>
# Copyright (C) 2026

import numbers

import numpy as np

from scipy.special import logsumexp


def _softplus(z):
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0)


def _pav_isotonic(y, w):
    """Weighted increasing isotonic regression (pool adjacent violators).
    Returns fitted values and the pooled blocks."""
    vals = []
    wts = []
    idx = []
    for i, (yy, ww) in enumerate(zip(map(float, y), map(float, w))):
        vals.append(yy)
        wts.append(ww)
        idx.append([i])
        while len(vals) > 1 and vals[-2] > vals[-1] + 1e-15:
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / (wts[-2]
                                                             + wts[-1])
            wts[-2] += wts[-1]
            vals[-2] = v
            idx[-2] += idx[-1]
            vals.pop()
            wts.pop()
            idx.pop()
    out = np.empty(len(y))
    for v, ii in zip(vals, idx):
        for i in ii:
            out[i] = v
    return out, idx


def _sinkhorn_batch(C, a, beta, eps, iters):
    """Log-domain Sinkhorn, batched over the leading dimension
    (stabilized with logsumexp; naive exp-sum-log under/overflows at
    small eps)."""
    la = np.log(a)[None, :, None]
    lb = np.log(beta)[:, None, :]
    K = -C / eps
    f = np.zeros(C.shape[:2])[:, :, None]
    g = np.zeros((C.shape[0], 1, C.shape[2]))
    for _ in range(iters):
        g = -eps * logsumexp(K + f / eps + la, axis=1, keepdims=True)
        f = -eps * logsumexp(K + g / eps + lb, axis=2, keepdims=True)
    return np.exp(K + (f + g) / eps + la + lb)


class SoftBinning:
    """Soft OT-binning of pre-binned data with annealed hardening.

    Parameters
    ----------
    n_bins : int, optional (default=5)

    n_steps : int, optional (default=200)
        Annealing/optimization steps per restart.

    n_restarts : int, optional (default=3)

    eps_start, eps_end : float, optional (defaults 0.25, 0.012)
        Geometric entropic-temperature schedule.

    rho_max : float, optional (default=60.0)
        Final weight of the PAV monotonicity penalty (ramped).

    learning_rate : float, optional (default=0.1)

    sinkhorn_iters : int, optional (default=90)

    polish : bool, optional (default=True)
        Reduced-space exact refinement of the hardened partition.

    random_state : int or None, optional (default=None)

    Attributes
    ----------
    bounds_ : numpy.ndarray
        Pre-bin boundary indices of the hardened partition
        (bins = [bounds_[k], bounds_[k+1]) in pre-bin index space).

    iv_ : float
        IV of the hardened (monotone-pooled) partition.

    contiguous_ : bool
        Whether the hard assignment was contiguous (Thm. 3.1 predicts
        True a.s.; asserted in the verification suite).
    """

    def __init__(self, n_bins=5, n_steps=200, n_restarts=3, eps_start=0.25,
                 eps_end=0.012, rho_max=60.0, learning_rate=0.1,
                 sinkhorn_iters=90, polish=True, random_state=None):
        if not isinstance(n_bins, numbers.Integral) or n_bins < 2:
            raise ValueError("n_bins must be an integer >= 2; got {}."
                             .format(n_bins))
        self.n_bins = n_bins
        self.n_steps = n_steps
        self.n_restarts = n_restarts
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.rho_max = rho_max
        self.learning_rate = learning_rate
        self.sinkhorn_iters = sinkhorn_iters
        self.polish = polish
        self.random_state = random_state

        self.bounds_ = None
        self.iv_ = None
        self.contiguous_ = None

    # ------------------------------------------------------------------ #

    def _params(self, theta):
        m = self.n_bins
        inc = 0.35 + _softplus(theta[..., :m])
        c = np.cumsum(inc, axis=-1)
        w = 0.02 + 0.96 * (c - inc / 2) / c[..., -1:]
        tb = theta[..., m:]
        beta = np.exp(tb - tb.max(axis=-1, keepdims=True))
        beta = beta / beta.sum(axis=-1, keepdims=True)
        return w, 0.05 / m + 0.95 * beta

    def _loss_batch(self, theta, u, pne, pe, a, n0, n1, eps, rho):
        w, beta = self._params(theta)
        C = (u[None, :, None] - w[:, None, :]) ** 2
        pi = _sinkhorn_batch(C, a, beta, eps, self.sinkhorn_iters)
        phi = pi / np.maximum(pi.sum(axis=2, keepdims=True), 1e-30)
        P = np.maximum(np.einsum("bnm,n->bm", phi, pne), 1e-12)
        Q = np.maximum(np.einsum("bnm,n->bm", phi, pe), 1e-12)
        iv = ((P - Q) * np.log(P / Q)).sum(axis=1)
        rate = Q * n1 / np.maximum(Q * n1 + P * n0, 1e-12)
        mass = (P * n0 + Q * n1) / (n0 + n1)
        pen = np.array([np.sum(mass[i] * (rate[i] -
                        _pav_isotonic(rate[i], mass[i])[0]) ** 2)
                        for i in range(len(theta))])
        return -iv + rho * pen

    def _iv_mono(self, bounds, n_nonevent, n_event):
        ne = np.add.reduceat(n_nonevent, bounds[:-1])
        e = np.add.reduceat(n_event, bounds[:-1])
        rate = e / np.maximum(e + ne, 1e-12)
        _, blocks = _pav_isotonic(rate, e + ne)
        pm = np.array([ne[bl].sum() for bl in blocks]) / n_nonevent.sum()
        qm = np.array([e[bl].sum() for bl in blocks]) / n_event.sum()
        keep = (pm > 0) & (qm > 0)
        pm, qm = pm[keep], qm[keep]
        if len(pm) == 0:
            return -np.inf
        return float(np.sum((pm - qm) * np.log(pm / qm)))

    def _polish(self, bounds, n_nonevent, n_event):
        n = len(n_nonevent)
        bounds = bounds.copy()
        best = self._iv_mono(bounds, n_nonevent, n_event)
        improved = True
        while improved:
            improved = False
            for j in range(1, len(bounds) - 1):
                for d in (-1, 1):
                    nb = bounds.copy()
                    nb[j] += d
                    if nb[j] <= nb[j - 1] or nb[j] >= nb[j + 1]:
                        continue
                    v = self._iv_mono(nb, n_nonevent, n_event)
                    if v > best + 1e-12:
                        bounds, best, improved = nb, v, True
        while len(bounds) - 1 < self.n_bins:
            cand = None
            for c in range(1, n):
                if c in bounds:
                    continue
                nb = np.sort(np.concatenate((bounds, [c])))
                v = self._iv_mono(nb, n_nonevent, n_event)
                if cand is None or v > cand[1]:
                    cand = (nb, v)
            if cand is None or cand[1] <= best + 1e-12:
                break
            bounds, best = cand
        return bounds, best

    # ------------------------------------------------------------------ #

    def fit(self, u, n_nonevent, n_event):
        """Fit on pre-binned data: positions u and per-pre-bin class
        counts."""
        u = np.asarray(u, dtype=float)
        n_nonevent = np.asarray(n_nonevent, dtype=float)
        n_event = np.asarray(n_event, dtype=float)

        n0, n1 = n_nonevent.sum(), n_event.sum()
        pne = n_nonevent / n0
        pe = n_event / n1
        a = (n_nonevent + n_event) / (n0 + n1)

        m = self.n_bins
        p_dim = 2 * m
        eye = np.eye(p_dim)
        d = 1e-4

        best = None
        rng = np.random.default_rng(self.random_state)
        for _ in range(self.n_restarts):
            theta = rng.normal(0, 0.3, p_dim)
            mom = np.zeros(p_dim)
            vel = np.zeros(p_dim)
            for s in range(self.n_steps):
                t = s / max(self.n_steps - 1, 1)
                eps = self.eps_start * (self.eps_end / self.eps_start) ** t
                rho = self.rho_max * min(1.0, 2 * t)
                batch = np.vstack([theta] + [theta + d * e for e in eye]
                                  + [theta - d * e for e in eye])
                loss = self._loss_batch(batch, u, pne, pe, a, n0, n1,
                                        eps, rho)
                grad = (loss[1:1 + p_dim] - loss[1 + p_dim:]) / (2 * d)
                mom = 0.9 * mom + 0.1 * grad
                vel = 0.99 * vel + 0.01 * grad ** 2
                theta -= self.learning_rate * mom / (np.sqrt(vel) + 1e-8)

            final = self._loss_batch(theta[None], u, pne, pe, a, n0, n1,
                                     self.eps_end, self.rho_max)[0]
            if best is None or final < best[0]:
                best = (final, theta.copy())

        theta = best[1]
        w, beta = self._params(theta[None])
        C = (u[None, :, None] - w[:, None, :]) ** 2
        pi = _sinkhorn_batch(C, a, beta, 0.003, 160)[0]
        assign = np.argmax(pi, axis=1)
        self.contiguous_ = bool(np.all(np.diff(assign) >= 0))

        cuts = np.where(np.diff(assign) > 0)[0] + 1
        bounds = np.concatenate(([0], np.sort(cuts), [len(u)])).astype(int)

        if self.polish:
            bounds, iv = self._polish(bounds, n_nonevent, n_event)
        else:
            iv = self._iv_mono(bounds, n_nonevent, n_event)

        self.bounds_ = bounds
        self.iv_ = iv
        return self
