"""Paper C: consolidated Gorishniy-suite table (11 datasets, 6 arms).

Merges the campaign folders (dedup rule: c3_suite wins collisions),
coalesces the legacy `auc` column into `score`, picks each arm's best
backbone per dataset (direction-aware: RMSE is lower-better), ranks the
five neural tokenizers, and reports the seedwise paired comparison of
the OT layer against the periodic (Fourier) baseline at their own best
heads.

Usage:
    python experiments/table_suite.py outputs
"""

# Cesar Claros <cesar.claros@outlook.com>
# Copyright (C) 2026

import glob
import sys

import numpy as np
import pandas as pd

SUITE = ["gesture", "churn", "california", "house16h", "adult", "otto",
         "higgs-small", "facebook", "santander", "covertype", "mslr"]
ARMS = ["raw", "quantile_ple", "target_ple", "ot_ple", "periodic",
        "lightgbm"]
FOLDERS = ["c3_suite", "c3_suite_periodic", "c3_expansion"]


def load(out_root):
    frames = []
    for folder in FOLDERS:
        for p in glob.glob(f"{out_root}/{folder}/*.parquet"):
            frames.append(pd.read_parquet(p))
    full = pd.concat(frames, ignore_index=True)
    full = full[full.dataset.isin(SUITE)].copy()
    full.loc[full.get("score").isna(), "score"] = full.auc
    full["task"] = full.groupby("dataset").task.transform(
        lambda s: s.dropna().iloc[0] if s.notna().any() else "binary")
    return full.drop_duplicates(["dataset", "arm", "backbone", "seed"],
                                keep="first")


def main(out_root):
    full = load(out_root)
    m = full.groupby(["dataset", "task", "arm", "backbone"]).score.mean()
    rows = []
    for ds in SUITE:
        sub = m.xs(ds, level=0)
        task = sub.index[0][0]
        sign = -1 if task == "regression" else 1
        line = {"dataset": ds, "task": task}
        for arm in ARMS:
            s = sub.xs(arm, level=1)
            best_bb = (sign * s.droplevel(0)).idxmax()
            line[arm] = float(s.xs(best_bb, level=1).iloc[0])
            line[arm + "_bb"] = best_bb
        neural = ARMS[:-1]
        rank = sorted(neural, key=lambda a: -sign * line[a])
        line["ot_rank"] = rank.index("ot_ple") + 1
        line["best_neural"] = rank[0]

        a = full[(full.dataset == ds) & (full.arm == "ot_ple")
                 & (full.backbone == line["ot_ple_bb"])
                 ].set_index("seed").score
        b = full[(full.dataset == ds) & (full.arm == "periodic")
                 & (full.backbone == line["periodic_bb"])
                 ].set_index("seed").score
        d = (sign * (a - b)).dropna()
        line["ot_vs_periodic"] = float(d.mean())
        line["ot_wins"] = int((d > 0).sum())
        rows.append(line)
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(t[["dataset", "task"] + ARMS
            + ["ot_rank", "best_neural", "ot_vs_periodic",
               "ot_wins"]].round(4).to_string(index=False))
    print("\not rank histogram:",
          t.ot_rank.value_counts().sort_index().to_dict())
    print("ot better than periodic on",
          int((np.sign(t.ot_vs_periodic) > 0).sum()), "of", len(t))
    return t


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "outputs")
