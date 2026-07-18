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

import numpy as np
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
    if "cut_sd_norm" not in df.columns:      # tolerate pre-normalization files
        df["cut_sd_norm"] = float("nan")
    agg = df.groupby(["dataset", "feature", "arm"]).agg(
        oos_iv=("oos_iv", "mean"), oos_iv_sd=("oos_iv", "std"),
        oos_w1=("oos_w1", "mean"), n_bins=("n_bins", "mean"),
        spike_bins=("spike_bins", "mean"), cut_sd=("cut_sd", "mean"),
        cut_sd_norm=("cut_sd_norm", "mean"),
        refit_mismatch=("refit_mismatch", "mean"),
        fit_time=("fit_time", "mean")).round(4).reset_index()
    return agg


_FMTAU_AGG = dict(
    n=("frac", "count"), feasible=("feasible", "mean"),
    binds=("binds", "mean"), oos_iv=("oos_iv", "mean"),
    oos_iv_base=("oos_iv_base", "mean"),
    iv_retention=("iv_retention", "mean"), fm_gain=("fm_gain", "mean"),
    oos_w1=("oos_w1", "mean"), n_bins=("n_bins", "mean"),
    fit_time=("fit_time", "mean"))

_MAXBINS_AGG = dict(
    n=("max_n_bins", "count"), agree_rate=("splits_equal", "mean"),
    iv_oos_iv=("iv_oos_iv", "mean"), hell_oos_iv=("hell_oos_iv", "mean"),
    d_oos_iv=("d_oos_iv", "mean"), iv_oos_w1=("iv_oos_w1", "mean"),
    hell_oos_w1=("hell_oos_w1", "mean"), d_oos_w1=("d_oos_w1", "mean"))


def _fmtau_prep(pattern):
    """Collect fm_tau rows and add the per-row derived columns.

    The solution-quality columns (oos_iv, oos_w1, n_bins, fm_achieved,
    iv_retention, fm_gain) are only meaningful for a feasible solution, so
    they are NaN on infeasible rows: otherwise the group means conflate the
    feasibility rate with solution quality when feasible. Read ``feasible``
    (the rate) alongside them.
    """
    df = collect(pattern)
    if "lam_frac" not in df.columns:         # tolerate pre-sweep files
        df["lam_frac"] = np.nan
    df["feasible"] = df["status"].astype(str).eq("OPTIMAL")
    df["iv_retention"] = df["oos_iv"] / df["oos_iv_base"].replace(0, np.nan)
    df["fm_gain"] = df["fm_achieved"] / df["fm_ref"].replace(0, np.nan)
    df["binds"] = df["fm_achieved"] > df["fm_ref"] + 1e-9
    df.loc[~df["feasible"], ["oos_iv", "oos_iv_base", "oos_w1", "n_bins",
                             "fm_achieved", "iv_retention", "fm_gain"]] = np.nan
    return df


def _maxbins_prep(pattern):
    """Collect maxbins rows and add the signed iv-vs-hellinger deltas."""
    df = collect(pattern)
    if "monotonic" not in df.columns:        # tolerate pre-tag files
        df["monotonic"] = "unknown"
    df["d_oos_iv"] = df["hell_oos_iv"] - df["iv_oos_iv"]
    df["d_oos_w1"] = df["hell_oos_w1"] - df["iv_oos_w1"]
    return df


def table_a1_spike(pattern):
    """Hybrid (iv_w1) vs pure iv bootstrap-fragility comparison, per feature.

    Built for the synthetic-spike a1 run (P1 corrected conjecture: the hybrid
    buys a lower spike-refit flip rate over pure IV). Spike datasets are
    isolated automatically when present, so pointing this at the full a1 glob
    still gives just the spike comparison; otherwise all rows are kept so the
    same view works on any a1 output. refit_reduction = iv - hybrid, so a
    positive value means the hybrid is more bootstrap-stable.
    """
    df = collect(pattern)
    df = df[~df["status"].astype(str).str.startswith("ERROR")]
    spike = df[df["dataset"].astype(str).str.contains("spike")]
    if len(spike):
        df = spike
    df = df[df["arm"].isin(["iv", "iv_w1"])]
    cols = ["refit_mismatch", "spike_bins", "cut_sd_norm", "n_bins"]
    g = df.groupby(["dataset", "feature", "arm"])[cols].mean().reset_index()
    seeds = (df[df["arm"] == "iv"].groupby(["dataset", "feature"])
             .size().rename("n").reset_index())
    iv = g[g["arm"] == "iv"].drop(columns="arm")
    hy = g[g["arm"] == "iv_w1"].drop(columns="arm")
    m = iv.merge(hy, on=["dataset", "feature"], suffixes=("_iv", "_hyb"))
    m = m.merge(seeds, on=["dataset", "feature"])
    m["refit_reduction"] = m["refit_mismatch_iv"] - m["refit_mismatch_hyb"]
    order = ["dataset", "feature", "n",
             "refit_mismatch_iv", "refit_mismatch_hyb", "refit_reduction",
             "spike_bins_iv", "spike_bins_hyb",
             "cut_sd_norm_iv", "cut_sd_norm_hyb", "n_bins_iv", "n_bins_hyb"]
    return m[order].round(4)


def table_gamma(pattern):
    """Hybrid-weight sweep per (dataset, feature): does a larger gamma move the
    hybrid off the pure-IV binning and cut its bootstrap fragility?

    differs_from_iv is the fraction of seeds whose hybrid splits differ from
    the IV baseline; refit_reduction = iv - hybrid, so positive means the
    hybrid is more bootstrap-stable than pure IV at that gamma.
    """
    df = collect(pattern)
    df = df[~df["status"].astype(str).str.startswith("ERROR")]
    df["refit_reduction"] = (df["iv_refit_mismatch"]
                             - df["hyb_refit_mismatch"])
    df["cut_sd_reduction"] = (df["iv_cut_sd_norm"]
                              - df["hyb_cut_sd_norm"])
    return (df.groupby(["dataset", "feature", "gamma"]).agg(
        n=("gamma", "count"),
        differs_from_iv=("differs_from_iv", "mean"),
        iv_refit=("iv_refit_mismatch", "mean"),
        hyb_refit=("hyb_refit_mismatch", "mean"),
        refit_reduction=("refit_reduction", "mean"),
        cut_sd_reduction=("cut_sd_reduction", "mean"),
        hyb_oos_iv=("hyb_oos_iv", "mean"),
        iv_oos_iv=("iv_oos_iv", "mean")).round(4).reset_index())


def table_spikesel(pattern):
    """Spike-selection fragility by gamma (P1 Sec. 3.4). gamma=0 is pure IV.

    flip_rate is the fraction of bootstrap refits whose 2-bin choice differs
    from the full-data choice; it should fall as gamma grows (pure IV ~0.47,
    hybrid ~0.17 in the note). baseline_isolate is how often the full-data
    choice is to isolate the spike; p_isolate is the bootstrap rate of the
    isolate choice.
    """
    df = collect(pattern)
    df["is_A"] = (df["baseline"].astype(str) == "A").astype(float)
    return (df.groupby(["dataset", "feature", "gamma"]).agg(
        n=("gamma", "count"),
        baseline_isolate=("is_A", "mean"),
        flip_rate=("flip_rate", "mean"),
        p_isolate=("p_isolate", "mean"),
        p_infeasible=("p_infeasible", "mean")).round(4).reset_index())


def table_fmtau(pattern):
    """fm_tau frontier pooled across features, by trust radius and threshold."""
    return (_fmtau_prep(pattern).groupby(["lam_frac", "frac"])
            .agg(**_FMTAU_AGG).round(4).reset_index())


def table_fmtau_feat(pattern):
    """fm_tau frontier per (dataset, feature), by trust radius and threshold."""
    return (_fmtau_prep(pattern)
            .groupby(["dataset", "feature", "lam_frac", "frac"])
            .agg(**_FMTAU_AGG).round(4).reset_index())


def table_maxbins(pattern):
    """iv-vs-hellinger cap sweep pooled across features, by monotone mode
    and bin cap."""
    return (_maxbins_prep(pattern).groupby(["monotonic", "max_n_bins"])
            .agg(**_MAXBINS_AGG).round(4).reset_index())


def table_maxbins_feat(pattern):
    """iv-vs-hellinger cap sweep per (dataset, feature), by monotone mode and
    bin cap."""
    return (_maxbins_prep(pattern)
            .groupby(["dataset", "feature", "monotonic", "max_n_bins"])
            .agg(**_MAXBINS_AGG).round(4).reset_index())


def table_fusion(pattern):
    """End-to-end feature fusion upstream of binning, per (dataset, group, arm).

    All arms are scored by the same certified metric (out-of-sample IV of the
    exact binning of the fused score), so pct_oracle = test_iv / oracle test_iv
    reads how close each fusion gets to the ceiling: the story is
    endtoend >> equal (learning the upstream transform helps) and
    endtoend -> oracle (it approaches the best attainable fusion).
    """
    df = collect(pattern)
    g = (df.groupby(["dataset", "group", "arm"]).agg(
        n=("seed", "count"),
        test_iv=("test_iv", "mean"), test_iv_sd=("test_iv", "std"),
        test_auc=("test_auc", "mean"), train_iv=("train_iv", "mean"),
        n_bins=("n_bins", "mean")).round(4).reset_index())
    orc = (g[g["arm"] == "oracle"].set_index(["dataset", "group"])["test_iv"]
           .rename("oracle_iv"))
    g = g.merge(orc, on=["dataset", "group"], how="left")
    g["pct_oracle"] = (g["test_iv"] / g["oracle_iv"].replace(0, np.nan)).round(3)
    order = {"equal": 0, "pca1": 1, "best_single": 2, "endtoend": 3,
             "oracle": 4}
    g["_o"] = g["arm"].map(order).fillna(9)
    return (g.sort_values(["dataset", "group", "_o"])
            .drop(columns=["oracle_iv", "_o"]).reset_index(drop=True))


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
    table = {"a1": table_a1, "a1_spike": table_a1_spike, "b2": table_b2,
             "gamma": table_gamma, "spikesel": table_spikesel,
             "fmtau": table_fmtau, "fmtau_feat": table_fmtau_feat,
             "maxbins": table_maxbins, "maxbins_feat": table_maxbins_feat,
             "fusion": table_fusion}[kind](pattern)
    print(table.to_markdown(index=False))
