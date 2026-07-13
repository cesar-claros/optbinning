"""
Aggregate experiment outputs into the paper tables (E0.3).

Usage:
    python experiments/tables.py a1 "outputs/a1/*.parquet"
    python experiments/tables.py b2 "outputs/b2/*.parquet"
"""

# Cesar Claros <cesar.claros@outlook.com>
# Copyright (C) 2026

import glob
import sys

from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import load_results             # noqa: E402


def collect(pattern):
    frames = [load_results(p) for p in sorted(glob.glob(pattern))]
    if not frames:
        raise SystemExit("No result files match: {}".format(pattern))
    return pd.concat(frames, ignore_index=True)


def table_a1(pattern):
    df = collect(pattern)
    df = df[~df["status"].astype(str).str.startswith("ERROR")]
    agg = df.groupby(["dataset", "feature", "arm"]).agg(
        oos_iv=("oos_iv", "mean"), oos_iv_sd=("oos_iv", "std"),
        oos_w1=("oos_w1", "mean"), n_bins=("n_bins", "mean"),
        spike_bins=("spike_bins", "mean"), cut_sd=("cut_sd", "mean"),
        refit_mismatch=("refit_mismatch", "mean"),
        fit_time=("fit_time", "mean")).round(4).reset_index()
    return agg


def table_b2(pattern):
    df = collect(pattern)
    agg = df.groupby(["design", "drift", "magnitude"]).agg(
        n=("rep", "count"),
        fm_power=("fm_reject", "mean"), psi_power=("psi_reject", "mean"),
        ks_power=("ks_reject", "mean"), fm=("fm", "mean"),
        pd_bound=("pd_bound", "mean"),
        pd_realized=("pd_realized", "mean")).round(4).reset_index()
    cert_ok = (df["pd_bound"] >= df["pd_realized"] - 1e-12).mean()
    print("certificate valid on {:.1%} of runs".format(cert_ok))
    return agg


if __name__ == "__main__":
    kind, pattern = sys.argv[1], sys.argv[2]
    table = {"a1": table_a1, "b2": table_b2}[kind](pattern)
    print(table.to_markdown(index=False))
