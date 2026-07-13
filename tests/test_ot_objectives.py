"""
Optimal-transport objective matrices testing (OT-WoE extension).

Pinned regression tests porting the certified checks of project notes P1
(Prop. 7.1, Prop. 1.4, Thm. 2.3, Prop. 2.4, Prop. 2.5) and P2 (Thm. 2.4).
"""

# Cesar Claros <cesar.claros@outlook.com>
# Copyright (C) 2026

import numpy as np

from pytest import approx

from optbinning.binning.metrics import cramer_1d
from optbinning.binning.metrics import hellinger_raw
from optbinning.binning.metrics import wasserstein_1d
from optbinning.binning.model_data import transport_model_data


def _random_instance(rng, n=None, mlr=False):
    n = n or rng.integers(5, 14)
    u = np.sort(rng.uniform(0, 1, n))
    while np.min(np.diff(u)) < 1e-4:
        u = np.sort(rng.uniform(0, 1, n))
    n_nonevent = rng.integers(1, 60, n).astype(float)
    if mlr:
        ratio = np.sort(rng.uniform(0.05, 5.0, n))
        n_event = np.maximum(np.round(n_nonevent * ratio), 1.0)
    else:
        n_event = rng.integers(1, 60, n).astype(float)
    return u, n_nonevent, n_event


def _random_partition(rng, n):
    m = rng.integers(1, n + 1)
    if m == 1:
        cuts = np.array([], dtype=int)
    else:
        cuts = np.sort(rng.choice(np.arange(1, n), size=m - 1, replace=False))
    return np.concatenate(([0], cuts, [n]))


def _binned(u, n_nonevent, n_event, bounds):
    idx = bounds[:-1]
    ne = np.add.reduceat(n_nonevent, idx)
    e = np.add.reduceat(n_event, idx)
    xs = np.add.reduceat(u * (n_nonevent + n_event), idx)
    w = xs / (ne + e)
    return ne / n_nonevent.sum(), e / n_event.sum(), w, ne, e


def _phi_total(PHI, bounds):
    return sum(PHI[bounds[k + 1] - 1][bounds[k]]
               for k in range(len(bounds) - 1))


def test_phi_w1_extent_additivity():
    # P1 Prop. 7.1: sum of PHI contributions equals the exact W1 between
    # the binned class-conditionals with pooled-mean atoms.
    rng = np.random.default_rng(42)
    for _ in range(30):
        u, ne, e = _random_instance(rng)
        x_sum = u * (ne + e)
        PHI, _, _ = transport_model_data(ne, e, x_sum, cramer_p=1)
        for _ in range(5):
            bounds = _random_partition(rng, len(u))
            p, q, w, _, _ = _binned(u, ne, e, bounds)
            direct = wasserstein_1d(p, q, w)
            assert _phi_total(PHI, bounds) == approx(direct, abs=1e-10)


def test_phi_cramer_extent_additivity():
    # P1 Prop. 1.4: extent-additivity for every Cramer exponent p >= 1.
    rng = np.random.default_rng(7)
    for p_exp in (2, 3):
        for _ in range(15):
            u, ne, e = _random_instance(rng)
            x_sum = u * (ne + e)
            PHI, _, _ = transport_model_data(ne, e, x_sum, cramer_p=p_exp)
            bounds = _random_partition(rng, len(u))
            p, q, w, _, _ = _binned(u, ne, e, bounds)
            direct = cramer_1d(p, q, w, p=p_exp)
            assert _phi_total(PHI, bounds) == approx(direct, abs=1e-10)


def test_theta_hellinger_raw_and_bhattacharyya():
    # P2 Thm. 2.4: THETA sums to the raw-count Hellinger objective, which
    # equals N0 + N1 - 2 sqrt(N0 N1) BC(p, q).
    rng = np.random.default_rng(3)
    for _ in range(20):
        u, ne, e = _random_instance(rng)
        _, THETA, _ = transport_model_data(ne, e, u * (ne + e))
        bounds = _random_partition(rng, len(u))
        _, _, _, ne_b, e_b = _binned(u, ne, e, bounds)
        total = sum(THETA[bounds[k + 1] - 1][bounds[k]]
                    for k in range(len(bounds) - 1))
        assert total == approx(hellinger_raw(ne_b, e_b), abs=1e-9)

        n0, n1 = ne.sum(), e.sum()
        bc = np.sum(np.sqrt(ne_b / n0) * np.sqrt(e_b / n1))
        assert total == approx(n0 + n1 - 2 * np.sqrt(n0 * n1) * bc, abs=1e-8)


def test_w1_collapse_under_monotone_rates():
    # P1 Thm. 2.3: monotone binned event rates => W1 equals the absolute
    # binned mean gap.
    rng = np.random.default_rng(11)
    checked = 0
    while checked < 25:
        u, ne, e = _random_instance(rng, mlr=True)
        bounds = _random_partition(rng, len(u))
        p, q, w, ne_b, e_b = _binned(u, ne, e, bounds)
        rate = e_b / (ne_b + e_b)
        if np.any(np.diff(rate) < -1e-12):
            continue
        checked += 1
        gap = abs(np.dot(q, w) - np.dot(p, w))
        assert wasserstein_1d(p, q, w) == approx(gap, abs=1e-10)


def test_sigma_decomposition():
    # P1 Prop. 2.4: binned mean gap = fine mean gap - sum of SIGMA.
    rng = np.random.default_rng(19)
    for _ in range(20):
        u, ne, e = _random_instance(rng)
        _, _, SIGMA = transport_model_data(
            ne, e, u * (ne + e),
            x_sum_nonevent=u * ne, x_sum_event=u * e)
        bounds = _random_partition(rng, len(u))
        p, q, w, _, _ = _binned(u, ne, e, bounds)
        gap_fine = np.dot(e / e.sum(), u) - np.dot(ne / ne.sum(), u)
        gap_binned = np.dot(q, w) - np.dot(p, w)
        s_total = sum(SIGMA[bounds[k + 1] - 1][bounds[k]]
                      for k in range(len(bounds) - 1))
        assert gap_binned == approx(gap_fine - s_total, abs=1e-10)


def test_split_delta_closed_form():
    # P1 Prop. 2.5: split value = spread * N/(n1+n2) * (p1 q2 - p2 q1)
    # under first-order stochastic dominance.
    rng = np.random.default_rng(23)
    checked = 0
    while checked < 15:
        u, ne, e = _random_instance(rng, mlr=True)
        n = len(u)
        cum = np.cumsum(ne / ne.sum() - e / e.sum())[:-1]
        if not np.all(cum >= -1e-12):
            continue
        bounds = _random_partition(rng, n)
        cands = [c for c in range(1, n) if c not in bounds]
        if not cands:
            continue
        checked += 1
        c = int(rng.choice(cands))
        nb = np.sort(np.concatenate((bounds, [c])))

        p0, q0, w0, _, _ = _binned(u, ne, e, bounds)
        p1, q1, w1, _, _ = _binned(u, ne, e, nb)
        delta = (wasserstein_1d(p1, q1, w1) - wasserstein_1d(p0, q0, w0))

        k = np.searchsorted(bounds, c) - 1
        a, b = bounds[k], bounds[k + 1]
        pl, ql = (ne[a:c] / ne.sum()).sum(), (e[a:c] / e.sum()).sum()
        pr, qr = (ne[c:b] / ne.sum()).sum(), (e[c:b] / e.sum()).sum()
        nl = (ne + e)[a:c].sum()
        nr = (ne + e)[c:b].sum()
        wl = np.average(u[a:c], weights=(ne + e)[a:c])
        wr = np.average(u[c:b], weights=(ne + e)[c:b])
        total = ne.sum() + e.sum()
        closed = (wr - wl) * total / (nl + nr) * (pl * qr - pr * ql)
        assert delta == approx(closed, abs=1e-10)
