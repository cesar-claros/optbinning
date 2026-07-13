"""
ConsensusBinning testing (OT-WoE extension; project note P7).
"""

# Cesar Claros <cesar.claros@outlook.com>
# Copyright (C) 2026

import numpy as np

from pytest import approx, raises

from optbinning import ConsensusBinning


def _data(seed=0, n=3000):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    prob = 1 / (1 + np.exp(-(1.6 * x - 0.4)))
    y = (rng.uniform(0, 1, n) < prob).astype(int)
    return x, y


def _fit(**kwargs):
    params = dict(n_folds=8, random_state=7, name="x", dtype="numerical",
                  solver="cp", monotonic_trend="ascending",
                  min_n_bins=4, max_n_bins=4)
    params.update(kwargs)
    x, y = _data()
    return ConsensusBinning(**params).fit(x, y), x, y


def test_prop11_sorted_coordinate_mean():
    # P7 Prop. 1.1: with equal fold cut counts, the barycenter is the
    # sorted-coordinate mean.
    cb, _, _ = _fit()
    counts = {len(s) for s in cb.fold_splits_}
    assert counts == {len(cb.splits_)}
    arr = np.vstack(cb.fold_splits_)
    assert cb.splits_ == approx(arr.mean(axis=0))


def test_attributes_and_monotonicity():
    cb, _, _ = _fit()
    assert np.all(np.diff(cb.splits_) > 0)
    assert cb.stability_ >= 0
    assert cb.stability_index_ >= 0
    assert len(cb.cut_dispersion_) == len(cb.splits_)
    assert len(cb.base_splits_) >= 1


def test_reproducibility():
    cb1, _, _ = _fit()
    cb2, _, _ = _fit()
    assert cb1.splits_ == approx(cb2.splits_)


def test_barycenter_optimality():
    # the consensus minimizes total squared sorted-matched disagreement
    # among nearby alternatives.
    cb, _, _ = _fit()
    arr = np.vstack(cb.fold_splits_)

    def obj(c):
        return np.sum((arr - np.sort(c)[None, :]) ** 2)

    base = obj(cb.splits_)
    rng = np.random.default_rng(1)
    for _ in range(30):
        pert = cb.splits_ + rng.normal(0, 0.05, len(cb.splits_))
        assert obj(pert) >= base - 1e-12


def test_heterogeneous_reduction():
    # quantile-function averaging path: reduce to fewer cuts than modal.
    cb, _, _ = _fit(target_n_cuts=2)
    assert len(cb.splits_) == 2
    assert np.all(np.diff(cb.splits_) > 0)
    lo = min(s.min() for s in cb.fold_splits_)
    hi = max(s.max() for s in cb.fold_splits_)
    assert lo <= cb.splits_[0] and cb.splits_[-1] <= hi


def test_validation():
    with raises(ValueError):
        ConsensusBinning(n_folds=1)
    with raises(ValueError):
        ConsensusBinning(resampling="jackknife")
    with raises(ValueError):
        ConsensusBinning(subsample_fraction=1.5)
    with raises(ValueError):
        ConsensusBinning(target_n_cuts=0)
