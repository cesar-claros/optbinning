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
from functools import partial
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
    task: str = "binary"          # binary | multiclass | regression


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


def load_baf(variant="Base", name="baf"):
    """Bank Account Fraud suite (NeurIPS 2022; manual Kaggle download).

    Place Base.csv (and optionally Variant I..V csvs) at data/baf/.
    1M rows each, target fraud_bool; `month` (0-7) is kept in X as the
    time column but excluded from `numerical` (temporal drift analyses,
    Paper B). The five variants carry documented, distinct bias/shift
    patterns -- the L1 replication set. Missing values are
    sentinel-coded (-1; negative for intended_balcon_amount)."""
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
    return Dataset(name, X, y, numerical=num,
                   special_codes=[-1], time_column="month")


BAF_VARIANTS = {"baf": "Base", "baf-v1": "Variant I",
                "baf-v2": "Variant II", "baf-v3": "Variant III",
                "baf-v4": "Variant IV", "baf-v5": "Variant V"}


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
# Gorishniy et al. suite completion (tasks: binary/multiclass/regression)
# --------------------------------------------------------------------- #

def _from_openml(cache_name, data_id, target_to_str=False):
    def fetch():
        from sklearn.datasets import fetch_openml
        d = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
        df = d.data.copy()
        t = d.target
        df["__target__"] = (t.astype(str) if target_to_str
                            else t).values
        return df

    return _from_cache_or(cache_name, fetch)


def load_gesture():
    """Gesture Phase Segmentation (OpenML 4538; n=9873, 5 classes)."""
    df = _from_openml("gesture", 4538, target_to_str=True)
    y = pd.Categorical(df["__target__"]).codes.astype(int)
    X = df.drop(columns="__target__")
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    return Dataset("gesture", X, y, numerical=num, task="multiclass")


def load_churn():
    """Churn Modelling (Kaggle; manual). Place Churn_Modelling.csv at
    data/churn/. y = Exited (binary)."""
    path = DATA_DIR / "churn" / "Churn_Modelling.csv"
    if not path.exists():
        raise FileNotFoundError(
            "Churn Modelling requires a manual Kaggle download "
            "(shrutimechlearn/churn-modelling). Place "
            "Churn_Modelling.csv at {}.".format(path))
    df = pd.read_csv(path).drop(
        columns=["RowNumber", "CustomerId", "Surname"], errors="ignore")
    y = df["Exited"].values.astype(int)
    X = df.drop(columns="Exited")
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    return Dataset("churn", X, y, numerical=num)


def load_california():
    """California Housing (sklearn; regression on MedHouseVal)."""
    def fetch():
        from sklearn.datasets import fetch_california_housing
        d = fetch_california_housing(as_frame=True)
        df = d.data.copy()
        df["__target__"] = d.target.values
        return df

    df = _from_cache_or("california", fetch)
    y = df["__target__"].values.astype(float)
    X = df.drop(columns="__target__")
    return Dataset("california", X, y, numerical=list(X.columns),
                   task="regression")


def load_house16h():
    """House 16H (OpenML 574; regression on price)."""
    df = _from_openml("house16h", 574)
    y = df["__target__"].values.astype(float)
    X = df.drop(columns="__target__")
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    return Dataset("house16h", X, y, numerical=num, task="regression")


def load_otto():
    """Otto Group Products (Kaggle; manual). Place train.csv at
    data/otto/. 9 classes; integer count features (tie-heavy -- the
    regime map predicts this is hard territory for learned knots)."""
    path = DATA_DIR / "otto" / "train.csv"
    if not path.exists():
        raise FileNotFoundError(
            "Otto requires a manual Kaggle download (competition "
            "otto-group-product-classification-challenge). Place "
            "train.csv at {}.".format(path))
    df = pd.read_csv(path).drop(columns="id", errors="ignore")
    y = pd.Categorical(df["target"]).codes.astype(int)
    X = df.drop(columns="target")
    return Dataset("otto", X, y, numerical=list(X.columns),
                   task="multiclass")


def load_facebook():
    """Facebook Comment Volume (UCI id=363; regression). Auto-fetches
    the zip and uses Features_Variant_5. NOTE: the Gorishniy reference
    construction additionally clips ALL splits at the train target's
    99th percentile and drops two degenerate columns (verified in their
    bin/datasets.py), so published RMSEs are ~3x smaller than raw-scale
    values; this loader is UNCLIPPED -- within-table comparisons only,
    never cross-paper."""
    def fetch():
        import io
        import urllib.request
        import zipfile
        url = ("https://archive.ics.uci.edu/static/public/363/"
               "facebook+comment+volume+dataset.zip")
        raw = urllib.request.urlopen(url).read()
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            member = next(m for m in zf.namelist()
                          if "Variant_5.csv" in m and "Train" in m)
            df = pd.read_csv(zf.open(member), header=None)
        df.columns = [f"f{i}" for i in range(df.shape[1] - 1)] \
            + ["__target__"]
        return df

    df = _from_cache_or("facebook", fetch)
    y = df["__target__"].values.astype(float)
    X = df.drop(columns="__target__")
    return Dataset("facebook", X, y, numerical=list(X.columns),
                   task="regression")


def load_santander():
    """Santander Customer Transactions (Kaggle; manual). Place
    train.csv at data/santander/. Binary; 200 numeric features."""
    path = DATA_DIR / "santander" / "train.csv"
    if not path.exists():
        raise FileNotFoundError(
            "Santander requires a manual Kaggle download (competition "
            "santander-customer-transaction-prediction). Place "
            "train.csv at {}.".format(path))
    df = pd.read_csv(path).drop(columns="ID_code", errors="ignore")
    y = df["target"].values.astype(int)
    X = df.drop(columns="target")
    return Dataset("santander", X, y, numerical=list(X.columns))


def load_covertype():
    """Covertype (sklearn; n=581012, 7 classes; all 54 columns treated
    as numerical, following the Gorishniy protocol)."""
    from sklearn.datasets import fetch_covtype
    d = fetch_covtype()                    # sklearn caches internally
    X = pd.DataFrame(d.data, columns=[f"f{i}" for i in range(54)])
    y = d.target.astype(int) - 1
    return Dataset("covertype", X, y, numerical=list(X.columns),
                   task="multiclass")


def load_mslr():
    """MSLR-WEB10K Fold 1 (manual; regression on relevance 0-4).
    Download from the Microsoft LETOR page and place Fold1/train.txt at
    data/mslr/Fold1/train.txt (svmlight format; we use the train file
    and our own splits -- a deviation from the official folds, noted)."""
    path = DATA_DIR / "mslr" / "Fold1" / "train.txt"
    if not path.exists():
        raise FileNotFoundError(
            "MSLR-WEB10K requires a manual download (Microsoft LETOR). "
            "Place Fold1/train.txt at {}.".format(path))
    from sklearn.datasets import load_svmlight_file
    # LETOR files carry qid: tokens; query_id=True is required to parse
    # them. Query structure is then discarded: we treat relevance as a
    # plain regression target with row-level splits (a deviation from
    # the LTR protocol, noted -- consistent with the Gorishniy usage).
    xs, y, _qid = load_svmlight_file(str(path), query_id=True)
    X = pd.DataFrame(np.asarray(xs.todense(), dtype=np.float32),
                     columns=[f"f{i}" for i in range(xs.shape[1])])
    return Dataset("mslr", X, y.astype(float),
                   numerical=list(X.columns), task="regression")


def load_compas():
    """ProPublica COMPAS two-year recidivism (auto-fetch).

    Audit CASE STUDY for Paper C Sec. 6 -- deliberately not a benchmark
    row. Standard ProPublica/Dressel-Farid filtering (screening window
    [-30, 30] days, is_recid != -1, ordinary-traffic charges excluded,
    valid score_text). Predictors EXCLUDE protected attributes; race and
    sex are kept as plain columns solely for the group-disparity
    certificate. Provenance and contested aspects discussed in-draft."""
    def fetch():
        url = ("https://raw.githubusercontent.com/propublica/"
               "compas-analysis/master/compas-scores-two-years.csv")
        return pd.read_csv(url)

    df = _from_cache_or("compas", fetch)
    mask = (df["days_b_screening_arrest"].between(-30, 30)
            & (df["is_recid"] != -1)
            & (df["c_charge_degree"] != "O")
            & (df["score_text"] != "N/A"))
    df = df[mask].copy()
    df["charge_degree_F"] = (df["c_charge_degree"] == "F").astype(float)
    y = df["two_year_recid"].values.astype(int)
    num = ["age", "priors_count", "juv_fel_count", "juv_misd_count",
           "juv_other_count", "charge_degree_F"]
    X = df[num + ["race", "sex"]].reset_index(drop=True)
    return Dataset("compas", X, y, numerical=num)


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


def load_aps():
    """APS Failure at Scania Trucks (UCI id=421; plan dataset panel).

    Rare component failure (~1.7% positive), heavy missingness encoded
    as 'na' strings, anonymized numeric sensor aggregates -- rank
    geometry is the primary coordinate (plan Sec. 5.2). Not importable
    via the ucimlrepo API (server refuses import for this id), so the
    loader downloads the official archive zip directly. The official
    train/test division is preserved in the ``__official_test__``
    column (0 = official train, 1 = official test) for the plan's
    official-split study; the pilot's repeated-split protocol ignores
    it. Each csv carries a license preamble before the header, located
    by content rather than by a fixed line count.
    """
    def fetch():
        import io
        import urllib.request
        import zipfile

        url = ("https://archive.ics.uci.edu/static/public/421/"
               "aps+failure+at+scania+trucks.zip")
        with urllib.request.urlopen(url, timeout=120) as r:
            zf = zipfile.ZipFile(io.BytesIO(r.read()))
        # some UCI archives nest a second zip; flatten if needed
        names = zf.namelist()
        inner = [n for n in names if n.lower().endswith(".zip")]
        if inner and not any(n.lower().endswith(".csv") for n in names):
            zf = zipfile.ZipFile(io.BytesIO(zf.open(inner[0]).read()))
            names = zf.namelist()

        def read_member(key, flag):
            cand = [n for n in names if key in n.lower()
                    and n.lower().endswith(".csv")]
            if not cand:
                raise FileNotFoundError(
                    "no member matching '{}' in {}".format(key, names))
            raw = zf.open(cand[0]).read().decode("utf-8", "replace")
            lines = raw.split("\n")
            hdr = next(i for i, ln in enumerate(lines)
                       if ln.startswith("class"))
            df = pd.read_csv(io.StringIO("\n".join(lines[hdr:])),
                             na_values="na")
            df["__official_test__"] = flag
            return df

        return pd.concat([read_member("training_set", 0),
                          read_member("test_set", 1)],
                         ignore_index=True)

    df = _from_cache_or("aps", fetch)
    y = (df["class"].astype(str).str.strip() == "pos").astype(int).values
    X = df.drop(columns="class")
    for c in X.columns:
        if c != "__official_test__":
            X[c] = pd.to_numeric(X[c], errors="coerce")
    num = [c for c in X.columns
           if pd.api.types.is_numeric_dtype(X[c])
           and c != "__official_test__"]
    return Dataset("aps", X, y, numerical=num)


def load_bank():
    """Bank Marketing (UCI id=222; plan dataset panel).

    Term-deposit subscription; the raw file is date-ordered, so row
    order is preserved and a synthetic time index is exposed for the
    chronological split (plan Sec. 5.2). 'duration' is EXCLUDED from
    the deployable numerical set: it is known only after the call
    (target leakage; plan requirement).
    """
    def fetch():
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=222)
        df = ds.data.features.copy()
        df["__target__"] = ds.data.targets.iloc[:, 0].values
        return df

    df = _from_cache_or("bank", fetch)
    y = (df["__target__"].astype(str).str.strip() == "yes").astype(
        int).values
    X = df.drop(columns="__target__")
    X["__row_order__"] = np.arange(len(X))
    num = [c for c in X.columns
           if pd.api.types.is_numeric_dtype(X[c])
           and c not in ("duration", "__row_order__")]
    return Dataset("bank", X, y, numerical=num,
                   time_column="__row_order__")


REGISTRY = {"german": load_german, "taiwan": load_taiwan,
            "aps": load_aps, "bank": load_bank,
            "hmeq": load_hmeq, "gmsc": load_gmsc, "heloc": load_heloc,
            "adult": load_adult, "diabetes": load_diabetes,
            "higgs-small": load_higgs_small,
            "gesture": load_gesture, "churn": load_churn,
            "california": load_california, "house16h": load_house16h,
            "otto": load_otto, "facebook": load_facebook,
            "santander": load_santander, "covertype": load_covertype,
            "mslr": load_mslr, "compas": load_compas}
for _key, _variant in BAF_VARIANTS.items():
    REGISTRY[_key] = partial(load_baf, _variant, _key)


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
