"""
Dataset registry for the OT-WoE experiments (E0.3).

Real datasets are fetched once and cached under data/ (override with the
OTWOE_DATA environment variable). Datasets requiring manual download
(license/agreement) raise with instructions. The synthetic generator
reproduces the paper designs with dials for spikes, near-ties, and the
four drift protocols.
"""

# Cesar Claros <cesar.claros@outlook.com>
# Copyright (C) 2026

import os

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(os.environ.get(
    "OTWOE_DATA", Path(__file__).resolve().parents[1] / "data"))


@dataclass
class Dataset:
    name: str
    X: pd.DataFrame
    y: np.ndarray
    numerical: list = field(default_factory=list)
    special_codes: list = field(default_factory=list)
    time_column: str = None


def _cache_path(name):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / "{}.csv".format(name)


def _from_cache_or(name, fetch):
    path = _cache_path(name)
    if path.exists():
        return pd.read_csv(path)
    df = fetch()
    df.to_csv(path, index=False)
    return df


# --------------------------------------------------------------------- #
# Real datasets
# --------------------------------------------------------------------- #

def load_german():
    """Statlog German credit (UCI id=144; n=1000). y = 1 for bad."""
    def fetch():
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=144)
        df = ds.data.features.copy()
        df["__target__"] = ds.data.targets.iloc[:, 0].values
        return df

    df = _from_cache_or("german", fetch)
    y = (df["__target__"].values == 2).astype(int)
    X = df.drop(columns="__target__")
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    return Dataset("german", X, y, numerical=num)


def load_taiwan():
    """Default of credit card clients (UCI id=350; n=30000)."""
    def fetch():
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=350)
        df = ds.data.features.copy()
        df["__target__"] = ds.data.targets.iloc[:, 0].values
        return df

    df = _from_cache_or("taiwan", fetch)
    y = df["__target__"].values.astype(int)
    X = df.drop(columns="__target__")
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    return Dataset("taiwan", X, y, numerical=num)


def load_hmeq():
    """HMEQ home-equity loans (n=5960). y = BAD."""
    def fetch():
        url = ("http://www.creditriskanalytics.net/uploads/1/9/5/1/"
               "19511601/hmeq.csv")
        return pd.read_csv(url)

    df = _from_cache_or("hmeq", fetch)
    y = df["BAD"].values.astype(int)
    X = df.drop(columns="BAD")
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    return Dataset("hmeq", X, y, numerical=num)


def load_gmsc():
    """Give Me Some Credit (Kaggle; manual download).
    Place cs-training.csv at data/gmsc/cs-training.csv."""
    path = DATA_DIR / "gmsc" / "cs-training.csv"
    if not path.exists():
        raise FileNotFoundError(
            "GMSC requires a manual Kaggle download "
            "(competition GiveMeSomeCredit). Place cs-training.csv at {}."
            .format(path))
    df = pd.read_csv(path, index_col=0)
    y = df["SeriousDlqin2yrs"].values.astype(int)
    X = df.drop(columns="SeriousDlqin2yrs")
    return Dataset("gmsc", X, y, numerical=list(X.columns))


def load_heloc():
    """FICO HELOC explainability dataset (manual; license agreement).
    Place heloc_dataset_v1.csv at data/heloc/heloc_dataset_v1.csv.
    Special codes -7, -8, -9 encode missing/invalid conditions."""
    path = DATA_DIR / "heloc" / "heloc_dataset_v1.csv"
    if not path.exists():
        raise FileNotFoundError(
            "HELOC requires accepting the FICO license. Place "
            "heloc_dataset_v1.csv at {}.".format(path))
    df = pd.read_csv(path)
    y = (df["RiskPerformance"] == "Bad").astype(int).values
    X = df.drop(columns="RiskPerformance")
    return Dataset("heloc", X, y, numerical=list(X.columns),
                   special_codes=[-7, -8, -9])


# --------------------------------------------------------------------- #
# Synthetic generator (paper designs)
# --------------------------------------------------------------------- #

def make_synthetic(design="smooth", n=5000, seed=0, drift=None):
    """Two-feature synthetic credit frame.

    design : "smooth" (logistic rate in a fused direction), "spike"
        (adds a low-count extreme-rate cluster; P1 Sec. 3.4), "near_tie"
        (two near-equivalent cut configurations; P7 Sec. 3), "ushape"
        (non-monotone U-shaped risk in f0; separates refinement-monotone
        objectives such as iv vs hellinger_raw once a bin cap binds).

    drift : None or dict(kind=..., magnitude=...), kinds:
        "location" (shift f0), "tail" (move mass to the upper tail of f0),
        "support" (mass beyond the previous f0 maximum),
        "volume" (resample n * magnitude rows).
    """
    rng = np.random.default_rng(seed)
    f0 = rng.normal(0, 1, n)
    f1 = rng.uniform(0, 1, n)

    z = 1.5 * f0 + 0.8 * f1 - 0.5
    if design == "spike":
        idx = rng.choice(n, size=max(5, n // 200), replace=False)
        f0[idx] = rng.normal(-3.6, 0.05, len(idx))
        z = 1.5 * f0 + 0.8 * f1 - 0.5
        z[idx] = -6.0                     # extreme-rate low-count cluster
    elif design == "near_tie":
        z = np.where(f0 < -0.4, -1.2, np.where(f0 < 0.4, 0.05, 1.2))
        z = z + 0.8 * (f1 - 0.5)
    elif design == "ushape":
        z = 2.0 * f0 ** 2 - 1.5 + 0.8 * (f1 - 0.5)

    y = (rng.uniform(0, 1, n) < 1 / (1 + np.exp(-z))).astype(int)
    X = pd.DataFrame({"f0": f0, "f1": f1})

    if drift is not None:
        kind = drift["kind"]
        m = float(drift.get("magnitude", 0.5))
        if kind == "location":
            X["f0"] = X["f0"] + m
        elif kind == "tail":
            k = int(m * n)
            idx = rng.choice(n, size=k, replace=False)
            X.loc[X.index[idx], "f0"] = np.quantile(f0, 0.90) + \
                np.abs(rng.normal(0, 0.3, k))
        elif kind == "support":
            k = int(m * n)
            idx = rng.choice(n, size=k, replace=False)
            X.loc[X.index[idx], "f0"] = f0.max() + rng.uniform(0.5, 1.5, k)
        elif kind == "volume":
            k = int(m * n)
            idx = rng.choice(n, size=k, replace=True)
            X = X.iloc[idx].reset_index(drop=True)
            y = y[idx]
        else:
            raise ValueError("Unknown drift kind: {}".format(kind))

    return Dataset("synthetic-" + design, X, y, numerical=["f0", "f1"])


REGISTRY = {"german": load_german, "taiwan": load_taiwan,
            "hmeq": load_hmeq, "gmsc": load_gmsc, "heloc": load_heloc}


def load(name, **synthetic_kwargs):
    if name.startswith("synthetic"):
        design = name.split("-", 1)[1] if "-" in name else "smooth"
        return make_synthetic(design=design, **synthetic_kwargs)
    return REGISTRY[name]()


def split_indices(n, test_size=0.4, seed=0):
    """Deterministic train/test split indices."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = int(test_size * n)
    return perm[n_test:], perm[:n_test]
