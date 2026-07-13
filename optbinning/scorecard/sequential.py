"""
Anytime-valid sequential drift monitoring via betting e-processes
(OT-WoE extension; project note P5 / Paper B, Sec. 5).

Tests the composite drift-budget null "flat metric between the incoming
score distribution and the reference is at most `tolerance`" with
time-uniform type-I control over an unlimited horizon: wealth processes
bet on bounded-Lipschitz witness functions (payoff means are at most
FM - tolerance <= 0 under the null, by the flat-metric dual), mixed over a
witness dictionary and over restart times (Prop. 5.1: a weighted mixture
of restarted supermartingales is a supermartingale, so Ville's inequality
gives P(ever false alarm) <= alpha even with restarts, which restore fast
detection of late change points).

Scores are compressed to a reference quantile grid; the monitor tests the
discretized distributions exactly.
"""

# Cesar Claros <cesar.claros@outlook.com>
# Copyright (C) 2026

import numbers

import numpy as np


class SequentialMonitor:
    """Anytime-valid sequential monitor for score drift.

    Parameters
    ----------
    expected_scores : array-like
        Reference (development) score sample.

    lam : float
        Bounded-Lipschitz radius in score units (witnesses satisfy
        |f| <= lam, Lip(f) <= 1); 2*lam is the flat-metric horizon.

    alpha : float, optional (default=0.05)
        Lifetime false-alarm budget: P(ever alarm) <= alpha under the
        drift-budget null.

    tolerance : float, optional (default=0.0)
        Drift budget epsilon in score units. tolerance=0 tests equality
        (of grid-discretized distributions); tolerance>0 alarms only on
        drift whose flat metric provably exceeds epsilon.

    n_centers : int, optional (default=7)
        Witness dictionary size: ramps and tents (and negatives) centered
        at this many reference quantiles; K = 4 * n_centers witnesses.

    restart_every : int or None, optional (default=None)
        Start a fresh e-process every this many observations (change
        detection; P5 Sec. 5). Restart j receives prior weight
        1 / (j * (j + 1)). None: single process from the start.

    bet_fraction : float, optional (default=0.5)
        Safety cap: bets are clipped to bet_fraction / (2*lam + tolerance).

    n_grid : int, optional (default=200)
        Reference quantile-grid resolution.

    Attributes
    ----------
    alarm_ : bool
        Sticky alarm flag (wealth ever reached 1/alpha).

    wealth_ : float
        Current mixture wealth (evidence against the drift budget; e-value).

    n_seen_ : int

    attribution_ : str
        Description of the witness carrying maximal wealth (drift type
        diagnosis: ramps = displacement, tents = local mass anomaly).
    """

    def __init__(self, expected_scores, lam, alpha=0.05, tolerance=0.0,
                 n_centers=7, restart_every=None, bet_fraction=0.5,
                 n_grid=200):
        expected_scores = np.asarray(expected_scores, dtype=float)
        if not isinstance(lam, numbers.Number) or lam <= 0:
            raise ValueError("lam must be > 0; got {}.".format(lam))
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1); got {}.".format(alpha))
        if tolerance < 0:
            raise ValueError("tolerance must be >= 0.")
        if not 0 < bet_fraction < 1:
            raise ValueError("bet_fraction must be in (0, 1).")
        if restart_every is not None and (
                not isinstance(restart_every, numbers.Integral) or
                restart_every < 1):
            raise ValueError("restart_every must be a positive integer "
                             "or None.")

        self.lam = float(lam)
        self.alpha = float(alpha)
        self.tolerance = float(tolerance)
        self.restart_every = restart_every
        self.bet_fraction = bet_fraction

        # Reference quantile grid and pmf.
        edges = np.unique(np.quantile(expected_scores,
                                      np.linspace(0, 1, n_grid + 1)))[1:-1]
        self._edges = edges
        cells = np.searchsorted(edges, expected_scores)
        n_cells = len(edges) + 1
        cnt = np.bincount(cells, minlength=n_cells).astype(float)
        self._atoms = np.where(
            cnt > 0,
            np.bincount(cells, weights=expected_scores,
                        minlength=n_cells) / np.maximum(cnt, 1),
            0.5 * (np.concatenate([[expected_scores.min()], edges]) +
                   np.concatenate([edges, [expected_scores.max()]])))
        self._pi = cnt / cnt.sum()

        # Witness dictionary on the grid atoms.
        centers = np.quantile(expected_scores,
                              (np.arange(n_centers) + 0.5) / n_centers)
        F = []
        self._labels = []
        for c in centers:
            F.append(np.clip(self._atoms - c, -self.lam, self.lam))
            self._labels.append("ramp@{:.4g}".format(c))
            F.append(np.maximum(0.0, self.lam - np.abs(self._atoms - c)))
            self._labels.append("tent@{:.4g}".format(c))
        F = np.array(F)
        self._F = np.vstack([F, -F])
        self._labels += ["-" + lb for lb in self._labels]
        self._K = len(self._F)
        self._mu_f = self._F @ self._pi

        self._bmax = bet_fraction / (2 * self.lam + self.tolerance)
        self._g2_prior = (2 * self.lam + self.tolerance) ** 2

        # Restart states: weight, logW (K,), sum_g (K,), sum_g2 (K,).
        self._states = []
        self._add_restart()
        self.n_seen_ = 0
        self.alarm_ = False
        self.wealth_ = 1.0
        self.attribution_ = None

    def _add_restart(self):
        j = len(self._states) + 1
        weight = 1.0 / (j * (j + 1)) if self.restart_every else 1.0
        self._states.append({
            "weight": weight,
            "logw": np.zeros(self._K),
            "sum_g": np.zeros(self._K),
            "sum_g2": np.full(self._K, self._g2_prior)})

    def update(self, scores_batch):
        """Process a batch of incoming scores; returns a status dict.

        Bets are predictable per batch (fixed within the batch from prior
        state), so validity is unaffected by batching.
        """
        scores_batch = np.asarray(scores_batch, dtype=float)
        cells = np.searchsorted(self._edges, scores_batch)

        # payoffs (K, B)
        g = (self._F[:, cells] - self._mu_f[:, None]) - self.tolerance

        for st in self._states:
            bet = np.clip(st["sum_g"] / np.maximum(st["sum_g2"], 1e-12),
                          0.0, self._bmax)
            st["logw"] += np.log1p(bet[:, None] * g).sum(axis=1)
            st["sum_g"] += g.sum(axis=1)
            st["sum_g2"] += (g * g).sum(axis=1)

        self.n_seen_ += len(scores_batch)

        if self.restart_every:
            while len(self._states) < 1 + self.n_seen_ // self.restart_every:
                self._add_restart()

        wealth = 0.0
        best = (-np.inf, None)
        for st in self._states:
            w_k = np.exp(np.minimum(st["logw"], 700.0))
            wealth += st["weight"] * w_k.mean()
            k = int(np.argmax(st["logw"]))
            if st["logw"][k] > best[0]:
                best = (st["logw"][k], self._labels[k])

        self.wealth_ = float(wealth)
        self.attribution_ = best[1]
        if wealth >= 1.0 / self.alpha:
            self.alarm_ = True

        return {"n_seen": self.n_seen_, "wealth": self.wealth_,
                "alarm": self.alarm_, "attribution": self.attribution_}
