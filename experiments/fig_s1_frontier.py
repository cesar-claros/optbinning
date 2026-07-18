"""Paper C flagship figure: stability-performance panels (experiment S1).

Small multiples, one panel per dataset, nothing normalized: x = cut
movement across refits (mean pairwise Hausdorff on the reference rank
scale — the fraction of population mass a bin boundary crosses between
refits; linear axis), y = test AUC of the identical WoE-logistic
downstream. Faint dots are individual seeds; large dots are means.
Quantile edges: stable but target-blind. IV-optimal MILP: informed but
an order of magnitude less stable. The OT layer: near-quantile
stability at MILP-level (or better) AUC, with the taiwan/german
trade-off visible rather than hidden.

Usage:
    python experiments/fig_s1_frontier.py \
        "outputs/s1_fixseed/*_cuts.parquet" documentation/figures
"""

# Cesar Claros <cesar.claros@outlook.com>
# Copyright (C) 2026

import glob
import sys

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

ARM_STYLE = {
    "optbinning": ("#c44e52", "IV-optimal MILP (two-stage)"),
    "ot_ple": ("#4c72b0", "OT layer (this paper)"),
    "quantile": ("#8c8c8c", "quantile edges"),
}


def load(cuts_glob):
    cuts = pd.concat([pd.read_parquet(p)
                      for p in sorted(glob.glob(cuts_glob))])
    perf = pd.concat([pd.read_parquet(p) for p in sorted(
        glob.glob(cuts_glob.replace("_cuts", "_perf")))])
    move = cuts.groupby(["dataset", "arm", "seed"]).hausdorff.mean()
    auc = perf.groupby(["dataset", "arm", "seed"]).auc.mean()
    return pd.concat([move, auc], axis=1).reset_index()


def main(cuts_glob, out_dir):
    df = load(cuts_glob)
    order = ["german", "taiwan", "gmsc", "adult"]
    fig, axes = plt.subplots(2, 2, figsize=(6.2, 4.6), sharex=True)
    for ax, ds in zip(axes.ravel(), order):
        sub = df[df.dataset == ds]
        for arm, (color, label) in ARM_STYLE.items():
            s = sub[sub.arm == arm]
            ax.scatter(s.hausdorff, s.auc, s=14, color=color,
                       alpha=0.35, lw=0, zorder=2)   # one dot per seed
            ax.scatter(s.hausdorff.mean(), s.auc.mean(), s=52,
                       color=color, edgecolor="white", lw=0.7,
                       zorder=3, label=label)
        ax.set_title(ds, fontsize=9.5, loc="left")
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(fontsize=7.6, frameon=False, loc="lower left")
    for ax in axes[1]:
        ax.set_xlabel("cut movement across refits\n"
                      "(fraction of population mass)", fontsize=8.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("test AUC", fontsize=9)
    axes[0, 0].set_xlim(-0.004, 0.125)
    fig.tight_layout()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig-s1-frontier.{ext}", dpi=220)
    print("wrote", out / "fig-s1-frontier.pdf")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "outputs/s1_fixseed/*_cuts.parquet",
         sys.argv[2] if len(sys.argv) > 2 else "../documentation/figures")
