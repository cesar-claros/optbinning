"""
Model data for optimal binning formulations.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019


import numpy as np

from scipy import stats

from .metrics import jeffrey
from .metrics import jensen_shannon
from .metrics import hellinger
from .metrics import triangular
from .metrics import brier
from .metrics import neg_brier
from .metrics import log_score


def test_proportions(e1, ne1, e2, ne2, zscore):
    n1 = e1 + ne1
    n2 = e2 + ne2
    p1 = e1 / n1
    p2 = e2 / n2
    p = (e1 + e2) / (n1 + n2)

    z = (p1 - p2) / np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return abs(z) < zscore


def find_pvalue_violation_indices(n, E, NE, max_pvalue, max_pvalue_policy):
    pvalue_violation_indices = []
    zscore = stats.norm.ppf(1.0 - max_pvalue / 2)

    if max_pvalue_policy == "all":
        for i in range(n - 1):
            for r in range(i + 1):
                ev = E[i][r]
                nev = NE[i][r]
                for j in range(i + 1, n):
                    for k in range(i + 1, j + 1):
                        ev2 = E[j][k]
                        nev2 = NE[j][k]
                        if test_proportions(ev, nev, ev2, nev2, zscore):
                            pvalue_violation_indices.append(([i, r], [j, k]))

    elif max_pvalue_policy == "consecutive":
        for i in range(n - 1):
            for r in range(i + 1):
                ev = E[i][r]
                nev = NE[i][r]
                for j in range(i + 1, n):
                    ev2 = E[j][i + 1]
                    nev2 = NE[j][i + 1]
                    if test_proportions(ev, nev, ev2, nev2, zscore):
                        pvalue_violation_indices.append(([i, r], [j, i+1]))

    return pvalue_violation_indices


def find_pvalue_violation_indices_continuous(n, U, S, R, max_pvalue,
                                             max_pvalue_policy):
    pvalue_violation_indices = []

    if max_pvalue_policy == "all":
        for i in range(n - 1):
            for t in range(i + 1):
                u = U[i][t]
                s = S[i][t]
                r = R[i][t]
                for j in range(i + 1, n):
                    for k in range(i + 1, j + 1):
                        u2 = U[j][k]
                        s2 = S[j][k]
                        r2 = R[j][k]
                        if stats.ttest_ind_from_stats(
                                u, s, r, u2, s2, r2, False)[1] > max_pvalue:
                            pvalue_violation_indices.append(([i, t], [j, k]))

    elif max_pvalue_policy == "consecutive":
        for i in range(n - 1):
            for k in range(i + 1):
                u = U[i][k]
                s = S[i][k]
                r = R[i][k]
                for j in range(i + 1, n):
                    u2 = U[j][i + 1]
                    s2 = S[j][i + 1]
                    r2 = R[j][i + 1]
                    if stats.ttest_ind_from_stats(
                            u, s, r, u2, s2, r2, False)[1] > max_pvalue:
                        pvalue_violation_indices.append(([i, k], [j, i+1]))

    return pvalue_violation_indices


def find_min_diff_violation_indices(n, X, min_diff):
    min_diff_violation_indices = []

    for i in range(n - 1):
        for k in range(i + 1):
            x = X[i][k]
            for j in range(i + 1, n):
                x2 = X[j][i + 1]
                if abs(x - x2) < min_diff:
                    min_diff_violation_indices.append(([i, k], [j, i+1]))

    return min_diff_violation_indices


def model_data(divergence, n_nonevent, n_event, max_pvalue, max_pvalue_policy,
               min_event_rate_diff, scale=None, return_nonevent_event=False):
    n = len(n_nonevent)

    t_n_event = n_event.sum()
    t_n_nonevent = n_nonevent.sum()

    D = []
    V = []

    E = []
    NE = []

    for i in range(1, n + 1):
        s_event = n_event[:i][::-1].cumsum()[::-1]
        s_nonevent = n_nonevent[:i][::-1].cumsum()[::-1]
        rate = s_event / (s_nonevent + s_event)

        p = s_event / t_n_event
        q = s_nonevent / t_n_nonevent

        if divergence == "iv":
            iv = jeffrey(p, q)
        elif divergence == "js":
            iv = jensen_shannon(p, q)
        elif divergence == "hellinger":
            iv = hellinger(p, q)
        elif divergence == "triangular":
            iv = triangular(p, q)
        elif divergence == "brier":
            iv = brier(p, q)
        elif divergence == "neg_brier":
            iv = neg_brier(p, q)
        elif divergence == "log_score":
            iv = log_score(p, q)

        if scale is not None:
            rate *= scale
            iv *= scale

            D.append(rate.astype(np.int64))
            V.append(iv.astype(np.int64))
        else:
            D.append(rate)
            V.append(iv)

        if max_pvalue is not None or return_nonevent_event:
            E.append(s_event)
            NE.append(s_nonevent)

    if max_pvalue is not None:
        pvalue_violation_indices = find_pvalue_violation_indices(
            n, E, NE, max_pvalue, max_pvalue_policy)
    else:
        pvalue_violation_indices = []

    if min_event_rate_diff > 0:
        if scale is not None:
            min_diff = int(min_event_rate_diff * scale)
        else:
            min_diff = min_event_rate_diff

        min_diff_violation_indices = find_min_diff_violation_indices(
            n, D, min_diff)
    else:
        min_diff_violation_indices = []

    if return_nonevent_event:
        return D, V, NE, E, pvalue_violation_indices

    return D, V, pvalue_violation_indices, min_diff_violation_indices


def multiclass_model_data(n_nonevent, n_event, max_pvalue, max_pvalue_policy,
                          min_event_rate_diff, scale=None):

    n, n_classes = n_nonevent.shape

    DD = []
    VV = []
    PV = []
    MD = []

    for c in range(n_classes):
        t_n_event = n_event[:, c].sum()
        t_n_nonevent = n_nonevent[:, c].sum()

        D = []
        V = []

        E = []
        NE = []

        for i in range(1, n + 1):
            s_event = n_event[:i, c][::-1].cumsum()[::-1]
            s_nonevent = n_nonevent[:i, c][::-1].cumsum()[::-1]
            rate = s_event / (s_nonevent + s_event)

            p = s_event / t_n_event
            q = s_nonevent / t_n_nonevent
            iv = jeffrey(p, q)

            if scale is not None:
                rate *= scale
                iv *= scale

                rate = rate.astype(np.int64)
                iv = iv.astype(np.int64)

            D.append(rate)
            V.append(iv)

            if max_pvalue is not None:
                E.append(s_event)
                NE.append(s_nonevent)

        if max_pvalue is not None:
            pvalue_violation_indices = find_pvalue_violation_indices(
                n, E, NE, max_pvalue, max_pvalue_policy)
        else:
            pvalue_violation_indices = []

        if min_event_rate_diff > 0:
            if scale is not None:
                min_diff = int(min_event_rate_diff * scale)
            else:
                min_diff = min_event_rate_diff

            min_diff_violation_indices = find_min_diff_violation_indices(
                n, D, min_diff)
        else:
            min_diff_violation_indices = []

        DD.append(D)
        VV.append(V)
        PV.append(pvalue_violation_indices)
        MD.append(min_diff_violation_indices)

    return DD, VV, PV, MD


def continuous_model_data(n_records, sums, ssums, max_pvalue,
                          max_pvalue_policy, min_mean_diff, scale=None):

    n = len(n_records)

    U = []
    UP = []
    S = []
    R = []
    V = []

    t_mean = sums.sum() / n_records.sum()

    for i in range(1, n + 1):
        s_n_records = n_records[:i][::-1].cumsum()[::-1]
        s_sums = sums[:i][::-1].cumsum()[::-1]
        s_ssums = ssums[:i][::-1].cumsum()[::-1]

        mean = s_sums / s_n_records
        std = np.sqrt(s_ssums / s_n_records - mean ** 2)
        norm = np.absolute(mean - t_mean)

        if scale is not None:
            mean_scaled = mean * scale
            norm_scaled = norm * scale

            mean_scaled = mean_scaled.astype(np.int64)
            norm_scaled = norm_scaled.astype(np.int64)

            U.append(mean_scaled)
            V.append(norm_scaled)
        else:
            U.append(mean)
            V.append(norm)

        if max_pvalue is not None or min_mean_diff > 0:
            UP.append(mean)

        if max_pvalue is not None:
            R.append(s_n_records)
            S.append(std)

    if max_pvalue is not None:
        pvalue_violation_indices = find_pvalue_violation_indices_continuous(
            n, UP, S, R, max_pvalue, max_pvalue_policy)
    else:
        pvalue_violation_indices = []

    if min_mean_diff > 0:
        min_diff_violation_indices = find_min_diff_violation_indices(
            n, UP, min_mean_diff)
    else:
        min_diff_violation_indices = []

    return U, V, pvalue_violation_indices, min_diff_violation_indices


# ---------------------------------------------------------------------------
# Aggregated matrices for optimal-transport objectives (OT-WoE extension).
#
# Cesar Claros <cesar.claros@outlook.com>
# Copyright (C) 2026. Theory: project notes P1 (Prop. 7.1 / Prop. 1.4:
# extent-additivity of W1 and Cramer-p with pooled-mean representatives;
# Prop. 2.4: sigma-penalty), P2 (Thm. 2.4: raw-count Hellinger regime).
# ---------------------------------------------------------------------------

TRANSPORT_DIVERGENCES = ("w1", "cramer2", "hellinger_raw")


def pooled_means(n_records, x_sum):
    """Pooled within-bin means U[i][k] (candidate bin spanning pre-bins
    k..i), in the same row layout as the ``model_data`` matrices. Used by
    the flat-metric margin expressions (project note P3, Theorem B).

    Parameters
    ----------
    n_records : numpy.ndarray, shape = (n_prebins,)
        Pooled record counts per pre-bin.

    x_sum : numpy.ndarray, shape = (n_prebins,)
        Pooled feature-value sums per pre-bin.

    Returns
    -------
    U : list of numpy.ndarray
    """
    n_records = np.asarray(n_records, dtype=float)
    x_sum = np.asarray(x_sum, dtype=float)
    n = len(n_records)

    U = []
    for i in range(1, n + 1):
        s_rec = n_records[:i][::-1].cumsum()[::-1]
        s_x = x_sum[:i][::-1].cumsum()[::-1]
        U.append(np.divide(s_x, s_rec,
                           out=np.zeros_like(s_x), where=s_rec > 0))
    return U


def transport_model_data(n_nonevent, n_event, x_sum, cramer_p=1,
                         x_sum_nonevent=None, x_sum_event=None, scale=None):
    """Aggregated lower-triangular matrices for transport-type objectives,
    in the same row layout as ``model_data``: row i, entry k = contribution
    of the candidate bin spanning pre-bins k..i (k = 0..i).

    PHI[i][k] : W1 (cramer_p=1) or Cramer-p contribution
                U_[k,i] * (G_{k-1}^p - G_i^p), where U is the pooled
                within-bin mean and G_t = |F_ne(t) - F_e(t)| the cumulative
                pooled-proportion gap at the boundary after pre-bin t
                (G_{-1} = 0). Summing PHI over the bins of any feasible
                contiguous partition gives exactly the Cramer-p discrepancy
                between the binned class-conditional distributions with
                pooled-mean atoms (extent-additive; MILP-linear).

    THETA[i][k]: raw-count squared-Hellinger contribution
                (sqrt(NE_[k,i]) - sqrt(E_[k,i]))^2. Finite on zero-event
                bins; equals the transport-active HK objective below its
                scale threshold.

    SIGMA[i][k]: (optional; requires class-wise x-sums) within-bin residual
                class displacement c_[k,i] * (mu_e - mu_ne); under a
                monotone event-rate trend, maximizing W1 is equivalent to
                minimizing the SIGMA total.

    Parameters
    ----------
    n_nonevent : numpy.ndarray, shape = (n_prebins,)
        Non-event counts per pre-bin.

    n_event : numpy.ndarray, shape = (n_prebins,)
        Event counts per pre-bin.

    x_sum : numpy.ndarray, shape = (n_prebins,)
        Sum of the feature values per pre-bin (pooled), for the pooled-mean
        representatives U.

    cramer_p : float (default=1)
        Cramer exponent; 1 gives the 1-Wasserstein objective.

    x_sum_nonevent : numpy.ndarray or None
        Class-wise x-sums; together with ``x_sum_event`` enables SIGMA.

    x_sum_event : numpy.ndarray or None

    scale : int or None
        Integer scaling for CP formulations, as in ``model_data``.

    Returns
    -------
    PHI, THETA, SIGMA : lists of numpy.ndarray (SIGMA is None if class-wise
        sums are not provided).
    """
    n_nonevent = np.asarray(n_nonevent, dtype=float)
    n_event = np.asarray(n_event, dtype=float)
    x_sum = np.asarray(x_sum, dtype=float)

    n = len(n_nonevent)
    t_n_nonevent = n_nonevent.sum()
    t_n_event = n_event.sum()

    # Cumulative pooled-proportion gaps at pre-bin boundaries; data
    # constants, invariant under any merge (the Prop. 7.1 key fact).
    gap = np.abs(np.cumsum(n_nonevent) / t_n_nonevent
                 - np.cumsum(n_event) / t_n_event)
    gap_p = gap ** cramer_p
    gap_left = np.concatenate(([0.], gap_p[:-1]))   # G_{k-1}^p, G_{-1} = 0

    with_sigma = x_sum_nonevent is not None and x_sum_event is not None
    if with_sigma:
        x_sum_nonevent = np.asarray(x_sum_nonevent, dtype=float)
        x_sum_event = np.asarray(x_sum_event, dtype=float)

    PHI = []
    THETA = []
    SIGMA = [] if with_sigma else None

    for i in range(1, n + 1):
        s_nonevent = n_nonevent[:i][::-1].cumsum()[::-1]
        s_event = n_event[:i][::-1].cumsum()[::-1]
        s_records = s_nonevent + s_event
        s_xsum = x_sum[:i][::-1].cumsum()[::-1]

        u = np.divide(s_xsum, s_records,
                      out=np.zeros_like(s_xsum), where=s_records > 0)

        phi = u * (gap_left[:i] - gap_p[i - 1])
        theta = (np.sqrt(s_nonevent) - np.sqrt(s_event)) ** 2

        if with_sigma:
            s_xne = x_sum_nonevent[:i][::-1].cumsum()[::-1]
            s_xe = x_sum_event[:i][::-1].cumsum()[::-1]
            mu_ne = np.divide(s_xne, s_nonevent,
                              out=np.zeros_like(s_xne), where=s_nonevent > 0)
            mu_e = np.divide(s_xe, s_event,
                             out=np.zeros_like(s_xe), where=s_event > 0)
            c = np.divide(s_nonevent * s_event, s_records,
                          out=np.zeros_like(s_records), where=s_records > 0)
            c *= (1. / t_n_nonevent + 1. / t_n_event)
            sigma = c * (mu_e - mu_ne)

        if scale is not None:
            PHI.append((phi * scale).astype(np.int64))
            THETA.append((theta * scale).astype(np.int64))
            if with_sigma:
                SIGMA.append((sigma * scale).astype(np.int64))
        else:
            PHI.append(phi)
            THETA.append(theta)
            if with_sigma:
                SIGMA.append(sigma)

    return PHI, THETA, SIGMA
