"""
Barycentric consensus binning across resampling folds (OT-WoE extension).

Aggregates the cut points of per-fold optimal binnings by the exact 1-D
Wasserstein barycenter (Agueh-Carlier quantile averaging): for equal fold
bin counts this is the sorted-coordinate mean (project note P7, Prop. 1.1);
heterogeneous counts are handled by quantile-function averaging with an
optimal reduction to the target count (Prop. 1.2). Reports the fold
disagreement S and the cube-root-calibrated stability index n^(2/3) * S
(P7, Sec. 3; cut estimators are Kim-Pollard cube-root, see Paper D).
"""

# Cesar Claros <cesar.claros@outlook.com>
# Copyright (C) 2026

import numbers

import numpy as np

from .binning import OptimalBinning


class ConsensusBinning:
    """Consensus optimal binning via 1-D Wasserstein barycenters of fold
    cut points.

    Parameters
    ----------
    n_folds : int, optional (default=25)
        Number of resampling folds.

    resampling : str, optional (default="bootstrap")
        "bootstrap" (n-out-of-n with replacement) or "subsample"
        (m-out-of-n without replacement; see ``subsample_fraction``).
        Subsampling is the inference-friendly choice (Paper D).

    subsample_fraction : float, optional (default=0.7)
        Subsample size as a fraction of n when ``resampling="subsample"``.

    target_n_cuts : int or None, optional (default=None)
        Number of consensus cuts. If None, the modal fold cut count.

    random_state : int or None, optional (default=None)

    **optb_params : parameters forwarded to ``OptimalBinning``.

    Attributes
    ----------
    splits_ : numpy.ndarray
        Consensus (barycentric) cut points; continuous values.

    base_splits_ : numpy.ndarray
        Cut points of the fit on the full sample.

    fold_splits_ : list of numpy.ndarray
        Per-fold cut points.

    stability_ : float
        S = average squared sorted-matched distance between fold cuts and
        the consensus (equal-count folds only), i.e. the mean W2^2
        disagreement up to the 1/(M-1) normalization.

    stability_index_ : float
        n^(2/3) * S, comparable across sample sizes in the cube-root
        regime (P7, Sec. 3).

    cut_dispersion_ : numpy.ndarray
        Per-cut standard deviation across equal-count folds.
    """

    def __init__(self, n_folds=25, resampling="bootstrap",
                 subsample_fraction=0.7, target_n_cuts=None,
                 random_state=None, **optb_params):
        if not isinstance(n_folds, numbers.Integral) or n_folds < 2:
            raise ValueError("n_folds must be an integer >= 2; got {}."
                             .format(n_folds))
        if resampling not in ("bootstrap", "subsample"):
            raise ValueError('resampling must be "bootstrap" or '
                             '"subsample".')
        if (not isinstance(subsample_fraction, numbers.Number) or
                not 0 < subsample_fraction <= 1):
            raise ValueError("subsample_fraction must be in (0, 1].")
        if target_n_cuts is not None and (
                not isinstance(target_n_cuts, numbers.Integral) or
                target_n_cuts < 1):
            raise ValueError("target_n_cuts must be a positive integer "
                             "or None.")

        self.n_folds = n_folds
        self.resampling = resampling
        self.subsample_fraction = subsample_fraction
        self.target_n_cuts = target_n_cuts
        self.random_state = random_state
        self.optb_params = optb_params

        self.splits_ = None
        self.base_splits_ = None
        self.fold_splits_ = None
        self.stability_ = None
        self.stability_index_ = None
        self.cut_dispersion_ = None
        self._is_fitted = False

    def fit(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        n = len(x)
        rng = np.random.default_rng(self.random_state)

        base = OptimalBinning(**self.optb_params).fit(x, y)
        self.base_splits_ = np.asarray(base.splits, dtype=float)

        fold_splits = []
        m = (n if self.resampling == "bootstrap"
             else int(np.ceil(self.subsample_fraction * n)))
        for _ in range(self.n_folds):
            if self.resampling == "bootstrap":
                idx = rng.integers(0, n, n)
            else:
                idx = rng.choice(n, size=m, replace=False)
            optb = OptimalBinning(**self.optb_params).fit(x[idx], y[idx])
            s = np.asarray(optb.splits, dtype=float)
            if len(s):
                fold_splits.append(np.sort(s))

        if not fold_splits:
            raise RuntimeError("No fold produced any split.")

        self.fold_splits_ = fold_splits

        counts = np.array([len(s) for s in fold_splits])
        if self.target_n_cuts is not None:
            target = self.target_n_cuts
        else:
            values, freq = np.unique(counts, return_counts=True)
            target = int(values[np.argmax(freq)])

        equal = [s for s in fold_splits if len(s) == target]
        if len(equal) == len(fold_splits):
            # Prop. 1.1: barycenter = sorted-coordinate mean.
            arr = np.vstack(equal)
            self.splits_ = arr.mean(axis=0)
        else:
            # Prop. 1.2: quantile-function averaging on a fine level grid,
            # then optimal reduction to `target` atoms (block means).
            tgrid = np.linspace(1e-6, 1 - 1e-6, 2001)
            finv = np.zeros_like(tgrid)
            for s in fold_splits:
                mb = len(s)
                pos = np.minimum((tgrid * mb).astype(int), mb - 1)
                finv += s[pos]
            finv /= len(fold_splits)
            self.splits_ = np.array(
                [finv[(tgrid > (j - 1) / target) &
                      (tgrid <= j / target)].mean()
                 for j in range(1, target + 1)])

        if len(equal) >= 2:
            arr = np.vstack(equal)
            center = arr.mean(axis=0)
            self.stability_ = float(
                np.mean(np.sum((arr - center) ** 2, axis=1)))
            self.stability_index_ = float(n ** (2. / 3.) * self.stability_)
            self.cut_dispersion_ = arr.std(axis=0)

        self._is_fitted = True
        return self
