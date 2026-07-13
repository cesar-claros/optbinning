"""
Flat-metric drift monitoring testing (OT-WoE extension; project note P4).
"""

# Cesar Claros <cesar.claros@outlook.com>
# Copyright (C) 2026

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

from optbinning import BinningProcess, Scorecard, ScorecardMonitoring


def _make_data(seed, n, shift=0.0):
    rng = np.random.default_rng(seed)
    f0 = rng.normal(0, 1, n) + shift
    f1 = rng.uniform(0, 1, n)
    z = 1.4 * f0 + 0.8 * f1 - 0.6
    y = (rng.uniform(0, 1, n) < 1 / (1 + np.exp(-z))).astype(int)
    X = pd.DataFrame({"f0": f0, "f1": f1})
    return X, y


def _monitoring(shift):
    X_e, y_e = _make_data(0, 5000)
    bp = BinningProcess(variable_names=["f0", "f1"])
    sc = Scorecard(binning_process=bp,
                   estimator=LogisticRegression(),
                   scaling_method="min_max",
                   scaling_method_params={"min": 300, "max": 850})
    sc.fit(X_e, y_e)

    X_a, y_a = _make_data(1, 5000, shift=shift)
    m = ScorecardMonitoring(scorecard=sc, psi_method="cart", psi_n_bins=10)
    m.fit(X_a, y_a, X_e, y_e)
    return m


def test_fm_table_null():
    # same population: small FM, no significance at fixed seed.
    m = _monitoring(shift=0.0)
    tab = m.fm_table(n_permutations=60, random_state=0)
    assert isinstance(tab, pd.DataFrame)
    assert tab["FM"].iloc[0] >= 0
    assert tab["p-value"].iloc[0] > 0.05
    # certificate must dominate the realized PD change (P4 Prop. 3.1)
    assert (tab["PD impact bound"].iloc[0]
            >= tab["realized |dPD|"].iloc[0] - 1e-12)


def test_fm_table_drift():
    # shifted feature: larger FM, significant, certificate still valid.
    m0 = _monitoring(shift=0.0)
    m1 = _monitoring(shift=0.6)
    t0 = m0.fm_table(n_permutations=60, random_state=0)
    t1 = m1.fm_table(n_permutations=60, random_state=0)
    assert t1["FM"].iloc[0] > t0["FM"].iloc[0]
    assert t1["p-value"].iloc[0] <= 0.05
    assert (t1["PD impact bound"].iloc[0]
            >= t1["realized |dPD|"].iloc[0] - 1e-12)


def test_fm_lambda_profile_monotone():
    # FM is nondecreasing in lambda (destroy/create only gets pricier).
    m = _monitoring(shift=0.3)
    lam0 = 0.05 * (m._score_expected.max() - m._score_expected.min())
    tab = m.fm_table(lam_grid=[lam0, 2 * lam0, 4 * lam0, 8 * lam0])
    fm = tab["FM"].values
    assert np.all(np.diff(fm) >= -1e-10)
