"""
SequentialMonitor testing (OT-WoE extension; project note P5).
"""

# Cesar Claros <cesar.claros@outlook.com>
# Copyright (C) 2026

import numpy as np

from pytest import raises

from optbinning import SequentialMonitor

LAM = 8.0  # score-point scale for a synthetic score in [300, 850]


def _reference(seed=0, n=20000):
    rng = np.random.default_rng(seed)
    return 300 + 550 * rng.beta(5, 4, n)


def test_h0_validity():
    # streams drawn from the reference itself: false-alarm-ever rate must
    # respect the lifetime budget (alpha = 0.05, 25 streams => expect ~0-1
    # alarms; assert a loose upper bound).
    ref = _reference()
    rng = np.random.default_rng(1)
    alarms = 0
    for _ in range(25):
        mon = SequentialMonitor(ref, lam=LAM, alpha=0.05,
                                restart_every=1500)
        for _ in range(20):
            batch = rng.choice(ref, size=200, replace=True)
            mon.update(batch)
        alarms += mon.alarm_
    assert alarms <= 3


def test_drift_detection():
    # a clear location drift must be caught, with wealth crossing 1/alpha.
    ref = _reference()
    rng = np.random.default_rng(2)
    mon = SequentialMonitor(ref, lam=LAM, alpha=0.05, restart_every=1500)
    detected_at = None
    for b in range(40):
        drift = 25.0 if b >= 10 else 0.0     # drift starts at obs 2000
        batch = rng.choice(ref, size=200, replace=True) + drift
        out = mon.update(batch)
        if out["alarm"] and detected_at is None:
            detected_at = out["n_seen"]
    assert detected_at is not None
    assert detected_at > 2000                # no alarm before the change
    assert mon.wealth_ >= 1 / 0.05
    assert mon.attribution_ is not None


def test_tolerance_budget_validity():
    # sub-budget drift (FM <= W1 = shift < tolerance): no alarm.
    ref = _reference()
    rng = np.random.default_rng(3)
    shift = 2.0
    mon = SequentialMonitor(ref, lam=LAM, alpha=0.05, tolerance=3 * shift,
                            restart_every=1500)
    for _ in range(40):
        batch = rng.choice(ref, size=200, replace=True) + shift
        mon.update(batch)
    assert not mon.alarm_


def test_wealth_is_evidence_ordered():
    # stronger drift => at least as much wealth at matched sample counts.
    ref = _reference()
    wealth = []
    for drift in (0.0, 12.0, 30.0):
        rng = np.random.default_rng(4)
        mon = SequentialMonitor(ref, lam=LAM, alpha=0.05)
        for _ in range(15):
            batch = rng.choice(ref, size=200, replace=True) + drift
            mon.update(batch)
        wealth.append(mon.wealth_)
    assert wealth[2] >= wealth[1] >= wealth[0] - 1e-9


def test_validation():
    ref = _reference(n=1000)
    with raises(ValueError):
        SequentialMonitor(ref, lam=-1.0)
    with raises(ValueError):
        SequentialMonitor(ref, lam=LAM, alpha=1.5)
    with raises(ValueError):
        SequentialMonitor(ref, lam=LAM, tolerance=-0.1)
    with raises(ValueError):
        SequentialMonitor(ref, lam=LAM, restart_every=0)
