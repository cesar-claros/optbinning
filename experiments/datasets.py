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


def load_adult():
    """Adult census income (UCI id=2; n=48842). y = 1 for income >50K.

    Standard binary task of the Gorishniy et al. tabular-DL benchmarks.
    Six numerical features; capital-gain/loss are zero-inflated and
    heavy-tailed (the rank-geometry stress case)."""
    def fetch():
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=2)
        df = ds.data.features.copy()
        df["__target__"] = ds.data.targets.iloc[:, 0].astype(str).values
        return df

    df = _from_cache_or("adult", fetch)
    y = df["__target__"].str.contains(">50K").astype(int).values
    X = df.drop(columns="__target__")
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    return Dataset("adult", X, y, numerical=num)


_DIABETES_ID_CODES = ["encounter_id", "patient_nbr", "admission_type_id",
                      "discharge_disposition_id", "admission_source_id"]


def load_diabetes():
    """Diabetes 130-US hospitals 1999-2008 (UCI id=296; n=101766).

    y = 1 for readmission within 30 days ('<30'), the standard task.
    Numerical features are visit/lab/medication counts (skewed,
    zero-inflated); integer-coded nominal IDs are excluded from
    `numerical` (categorical despite the dtype)."""
    def fetch():
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=296)
        df = ds.data.features.copy()
        df["__target__"] = ds.data.targets.iloc[:, 0].astype(str).values
        return df

    df = _from_cache_or("diabetes", fetch)
    y = (df["__target__"] == "<30").astype(int).values
    X = df.drop(columns="__target__")
    num = [c for c in X.columns
           if pd.api.types.is_numeric_dtype(X[c])
           and c not in _DIABETES_ID_CODES]
    return Dataset("diabetes", X, y, numerical=num)


def load_baf(variant="Base"):
    """Bank Account Fraud suite (NeurIPS 2022; manual Kaggle download).

    Place Base.csv (and optionally Variant I..V csvs) at data/baf/.
    1M rows, target fraud_bool; `month` (0-7) is kept in X as the time
    column but excluded from `numerical` (temporal drift analyses,
    Paper B). Missing values are sentinel-coded (-1; negative for
    intended_balcon_amount) -- the sentinel + heavy-tail regime."""
    path = DATA_DIR / "baf" / "{}.csv".format(variant)
    if not path.exists():
        raise FileNotFoundError(
            "BAF requires a manual Kaggle download (dataset "
            "sgpjesus/bank-account-fraud-dataset-neurips-2022). "
            "Place {}.csv at {}.".format(variant, path))
    df = pd.read_csv(path)
    y = df["fraud_bool"].values.astype(int)
    X = df.drop(columns="fraud_bool")
    num = [c for c in X.columns
           if pd.api.types.is_numeric_dtype(X[c]) and c != "month"]
    return Dataset("baf", X, y, numerical=num,
                   special_codes=[-1], time_column="month")


_HIGGS_URLS = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/"
    "HIGGS.csv.gz",
    "https://archive.ics.uci.edu/static/public/280/higgs.zip",
)
_HIGGS_COLUMNS = [
    "lep_pt", "lep_eta", "lep_phi", "miss_e_mag", "miss_e_phi",
    "jet1_pt", "jet1_eta", "jet1_phi", "jet1_btag",
    "jet2_pt", "jet2_eta", "jet2_phi", "jet2_btag",
    "jet3_pt", "jet3_eta", "jet3_phi", "jet3_btag",
    "jet4_pt", "jet4_eta", "jet4_phi", "jet4_btag",
    "m_jj", "m_jjj", "m_lv", "m_jlv", "m_bb", "m_wbb", "m_wwbb"]
_HIGGS_N_ROWS = 11000000
_HIGGS_SMALL_N = 98049


def _fetch_higgs_raw():
    """Download the HIGGS source archive once (~2.6 GB)."""
    raw = DATA_DIR / "higgs" / "HIGGS.csv.gz"
    if raw.exists():
        return raw
    raw.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request
    try:
        urllib.request.urlretrieve(_HIGGS_URLS[0], raw)
        return raw
    except Exception:                                    # noqa: BLE001
        pass
    import shutil
    import zipfile
    zpath = raw.parent / "higgs.zip"
    urllib.request.urlretrieve(_HIGGS_URLS[1], zpath)
    with zipfile.ZipFile(zpath) as zf:
        member = next(m for m in zf.namelist()
                      if m.lower().endswith(".csv.gz"))
        with zf.open(member) as src, open(raw, "wb") as dst:
            shutil.copyfileobj(src, dst)
    zpath.unlink()
    return raw


def _sample_csv_rows(path, keep, chunksize=1000000):
    """Exact deterministic row subsample of a headerless csv, chunked
    so the 11M-row source never sits in memory."""
    keep = np.sort(np.asarray(keep))
    parts = []
    start = 0
    for chunk in pd.read_csv(path, header=None, dtype=np.float32,
                             chunksize=chunksize):
        idx = keep[(keep >= start) & (keep < start + len(chunk))]
        if len(idx):
            parts.append(chunk.iloc[idx - start])
        start += len(chunk)
    return pd.concat(parts, ignore_index=True)


def load_higgs_small():
    """HIGGS boson detection (UCI id=280), 98049-row seeded subsample.

    Matches the "higgs-small" size of the Gorishniy et al. tabular-DL
    benchmarks; 28 kinematic features, y = 1 for signal. The first call
    downloads the 2.6 GB source once and caches the subsample (~25 MB);
    subsequent loads read the cache only."""
    def fetch():
        raw = _fetch_higgs_raw()
        rng = np.random.default_rng(0)
        keep = rng.choice(_HIGGS_N_ROWS, _HIGGS_SMALL_N, replace=False)
        df = _sample_csv_rows(raw, keep)
        df.columns = ["__target__"] + _HIGGS_COLUMNS
        return df

    df = _from_cache_or("higgs-small", fetch)
    y = df["__target__"].values.astype(int)
    X = df.drop(columns="__target__")
    return Dataset("higgs-small", X, y, numerical=list(X.columns))


# --------------------------------------------------------------------- #
# Synthetic generator (paper designs)
# --------------------------------------------------------------------- #

def make_synthetic(design="smooth", n=5000, seed=0, drift=None):
    """Two-feature synthetic credit frame.

    design : "smooth" (logistic rate in a fused direction), "spike"
        (adds a low-count extreme-rate cluster at a location extreme; P1
        Sec. 3.4), "near_tie" / "spike2" (competing discrete cut
        configurations with a near-neutral middle step: IV is nearly
        indifferent to how that middle is split -- a near-tie that flips pure
        IV under bootstrap -- while the location-aware W1 term breaks the tie
        toward a coarser binning; empirically this, not a localized spike, is
        the geometry where the IV+W1 hybrid diverges from IV; P7 Sec. 3),
        "ushape" (non-monotone U-shaped risk in f0; separates
        refinement-monotone objectives such as iv vs hellinger_raw once a bin
        cap binds), "spike3" (faithful port of the P1 Sec. 3.4 example: three
        clusters at u = (0.05, 0.45, 0.80) with ascending rates
        (0.016, 0.405, 0.583) whose low cluster is a rare-event spike, ~1
        event in ~60 records, WoE ~ +4; isolating vs merging it is a bootstrap
        coin flip, so pure IV's 2-bin choice flips ~47% of the time while the
        hybrid's is stable. Reproduces the effect only under max_n_bins=2 and
        a cut-position, not cut-count, fragility read).

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
    elif design in ("near_tie", "spike2"):
        # Near-neutral middle step (WoE ~ 0) between two decisive outer steps:
        # IV is nearly indifferent to how the middle is split (the near-tie
        # that flips pure IV under bootstrap) while the W1 term breaks it
        # toward a coarser binning. This, not a localized spike, is where the
        # IV+W1 hybrid diverges from IV.
        z = np.where(f0 < -0.4, -1.2, np.where(f0 < 0.4, 0.05, 1.2))
        z = z + 0.8 * (f1 - 0.5)
    elif design == "ushape":
        z = 2.0 * f0 ** 2 - 1.5 + 0.8 * (f1 - 0.5)
    elif design == "spike3":
        # Three clusters at u = (0.05, 0.45, 0.80), ascending rates
        # (0.016, 0.405, 0.583). The low cluster is a fixed ~60-record
        # rare-event spike (~1 event): under bootstrap the lone event vanishes
        # with probability ~e^{-1}, so isolating vs merging it is a coin flip.
        k = 60
        n_mid = int(0.37 * (n - k))
        n_hi = n - k - n_mid
        f0 = np.concatenate([rng.normal(0.05, 0.01, k),
                             rng.normal(0.45, 0.04, n_mid),
                             rng.normal(0.80, 0.04, n_hi)])
        z = np.concatenate([np.full(k, -4.12), np.full(n_mid, -0.385),
                            np.full(n_hi, 0.335)])

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
            "hmeq": load_hmeq, "gmsc": load_gmsc, "heloc": load_heloc,
            "adult": load_adult, "diabetes": load_diabetes,
            "baf": load_baf, "higgs-small": load_higgs_small}


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
