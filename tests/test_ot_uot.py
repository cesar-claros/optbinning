"""
Certified Hellinger-Kantorovich solver testing (OT-WoE extension; P2).
"""

# Cesar Claros <cesar.claros@outlook.com>
# Copyright (C) 2026

import numpy as np

from pytest import approx

from optbinning.binning.uot import cluster_blocks
from optbinning.binning.uot import hk2
from optbinning.binning.uot import hk_activation_margin


def test_two_dirac_closed_form():
    # HK^2(a delta_x, b delta_y) = a + b - 2 sqrt(ab) cos(d/k ^ pi/2).
    rng = np.random.default_rng(0)
    for _ in range(15):
        a, b = rng.uniform(0.2, 5, 2)
        d = rng.uniform(0.01, 2.2)
        v, g = hk2(np.array([0., d]), np.array([a, 0.]), np.array([0., b]),
                   kappa=1.0)
        ref = a + b - 2 * np.sqrt(a * b) * np.cos(min(d, np.pi / 2))
        assert v == approx(ref, abs=max(2 * g, 1e-7))


def test_colocated_and_hellinger_regime():
    rng = np.random.default_rng(1)
    # colocated atoms: exact squared Hellinger, zero gap.
    a, b = 2.3, 0.7
    v, g = hk2(np.array([0.5]), np.array([a]), np.array([b]))
    assert g == 0.0
    assert v == approx((np.sqrt(a) - np.sqrt(b)) ** 2, abs=1e-12)

    # all gaps beyond the horizon: raw-count Hellinger, zero gap
    # (P2, Thm. 2.4), including zero-count cells (finite).
    w = np.cumsum(rng.uniform(np.pi / 2 * 1.05, 3, 6))
    a = rng.uniform(0, 4, 6)
    b = rng.uniform(0, 4, 6)
    b[2] = 0.0
    v, g = hk2(w, a, b, kappa=1.0)
    assert g == 0.0
    assert v == approx(np.sum((np.sqrt(a) - np.sqrt(b)) ** 2), abs=1e-12)


def test_bounds_floor_and_ceiling():
    # (sqrt(Ma) - sqrt(Mb))^2 <= HK^2 <= sum (sqrt(a)-sqrt(b))^2.
    rng = np.random.default_rng(2)
    for _ in range(8):
        k = rng.integers(2, 5)
        w = np.sort(rng.uniform(0, 1.5, k))
        while k > 1 and np.min(np.diff(w)) < 1e-3:
            w = np.sort(rng.uniform(0, 1.5, k))
        a = rng.uniform(0.1, 3, k)
        b = rng.uniform(0.1, 3, k)
        v, g = hk2(w, a, b, kappa=rng.uniform(0.4, 2.0))
        ceil = np.sum((np.sqrt(a) - np.sqrt(b)) ** 2)
        floor = (np.sqrt(a.sum()) - np.sqrt(b.sum())) ** 2
        assert v <= ceil + g + 1e-9
        assert v >= floor - g - 1e-9


def test_rectangle_identity_violation_pinned():
    # P2 Thm. 3.1 (impossibility): on the pinned instance the rectangle
    # defect at kappa=1 is +0.1099, far beyond the certified gap budget.
    u = np.array([0.0, 0.3, 0.6, 0.9])
    ne = np.array([4.0, 3.0, 2.0, 1.0])
    e = np.array([1.0, 2.0, 3.0, 4.0])
    nrec = ne + e

    def agg(bounds):
        a = np.add.reduceat(ne, bounds[:-1])
        b = np.add.reduceat(e, bounds[:-1])
        w = np.array([np.average(u[s:t], weights=nrec[s:t])
                      for s, t in zip(bounds[:-1], bounds[1:])])
        return w, a, b

    parts = {"12|34": np.array([0, 2, 4]),
             "12|3|4": np.array([0, 2, 3, 4]),
             "1|2|34": np.array([0, 1, 2, 4]),
             "1|2|3|4": np.array([0, 1, 2, 3, 4])}
    vals = {}
    budget = 0.0
    for key, bounds in parts.items():
        v, g = hk2(*agg(bounds), kappa=1.0)
        vals[key] = v
        budget += g

    defect = (vals["12|34"] - vals["12|3|4"]
              - vals["1|2|34"] + vals["1|2|3|4"])
    assert budget < 1e-3
    assert defect == approx(0.109898, abs=max(2e-3, 2 * budget))


def test_activation_criterion_pinned():
    # P2 Prop. 2.5 on the spike example's binning B: inactive at
    # kappa=0.5 (0.63 < 1), active at kappa=0.75 (1.35 > 1) — and the
    # solver value leaves the Hellinger ceiling exactly when active.
    d, orr = 0.381, 2.32
    assert hk_activation_margin(d, 0.5, orr) < 0
    assert hk_activation_margin(d, 0.75, orr) > 0

    w = np.array([0.4195, 0.8005])
    a = np.array([5.00, 5.00])
    b = np.array([3.01, 6.99])
    ceil = np.sum((np.sqrt(a) - np.sqrt(b)) ** 2)

    v_in, g_in = hk2(w, a, b, kappa=0.5)
    assert v_in == approx(ceil, abs=max(2 * g_in, 1e-5))

    v_act, g_act = hk2(w, a, b, kappa=0.75)
    assert v_act < ceil - max(2 * g_act, 1e-5)


def test_cluster_blocks():
    w = np.array([0.0, 0.1, 3.0, 3.1, 9.0])
    blocks = cluster_blocks(w, kappa=1.0)
    assert [list(b) for b in blocks] == [[0, 1], [2, 3], [4]]
