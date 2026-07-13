"""
Optimal binning metrics.
"""

# Guillermo Navas-Palencia <g.navas.palencia@gmail.com>
# Copyright (C) 2019

import numpy as np

from scipy import special
from scipy import stats
from sklearn.utils import check_array
from sklearn.utils import check_consistent_length


def _check_x_y(x, y):
    x = check_array(x, ensure_2d=False, ensure_all_finite=True)
    y = check_array(y, ensure_2d=False, ensure_all_finite=True)

    check_consistent_length(x, y)

    return x, y


def entropy(x):
    """Calculate the entropy of a discrete probability distribution.

    Parameters
    ----------
    x : array-like
        Discrete probability distribution.

    Returns
    -------
    entropy : float
    """
    x = np.asarray(x)
    return -special.xlogy(x, x).sum()


def gini(event, nonevent):
    """Calculate the Gini coefficient given the number of events and
    non-events.

    Parameters
    ----------
    event : array-like
        Number of events.

    nonevent : array-like
        Number of non-events.

    Returns
    -------
    gini : float
    """
    event, nonevent = _check_x_y(event, nonevent)

    mask = (event + nonevent) > 0
    event = event[mask].astype(np.float64)
    nonevent = nonevent[mask].astype(np.float64)

    n = len(event)
    if n <= 1:
        return 0
    else:
        te = event.sum()
        tne = nonevent.sum()

        ner = nonevent / (event + nonevent)
        idx = np.argsort(ner)
        ev = event[idx]
        ne = nonevent[idx]

        s = np.zeros(n)
        s[1:] = 2.0 * ne[:-1].cumsum()

        return 1.0 - np.dot(ev, ne + s) / (te * tne)


def kullback_leibler(x, y, return_sum=False):
    """Calculate the Kullback-Leibler divergence between two distributions.

    Parameters
    ----------
    x : array-like
        Discrete probability distribution.

    y : array-like
        Discrete probability distribution.

    return_sum : bool
        Return sum of kullback-leibler values.

    Returns
    -------
    kullback_leibler : float or numpy.ndarray
    """
    x, y = _check_x_y(x, y)

    if return_sum:
        return special.xlogy(x, x / y).sum()
    else:
        return special.xlogy(x, x / y)


def jeffrey(x, y, return_sum=False):
    """Calculate the Jeffrey's divergence between two distributions.

    Parameters
    ----------
    x : array-like
        Discrete probability distribution.

    y : array-like
        Discrete probability distribution.

    return_sum : bool
        Return sum of jeffrey values.

    Returns
    -------
    jeffrey : float or numpy.ndarray
    """
    x, y = _check_x_y(x, y)

    j = special.xlogy(x - y, x / y)

    if return_sum:
        return j.sum()
    else:
        return j


def jensen_shannon(x, y, return_sum=False):
    """Calculate the Jensen-Shannon divergence between two distributions.

    Parameters
    ----------
    x : array-like
        Discrete probability distribution.

    y : array-like
        Discrete probability distribution.

    return_sum : bool
        Return sum of jensen shannon values.

    Returns
    -------
    jensen_shannon : float or numpy.ndarray
    """
    x, y = _check_x_y(x, y)

    m = 0.5 * (x + y)
    return 0.5 * (kullback_leibler(x, m, return_sum) +
                  kullback_leibler(y, m, return_sum))


def jensen_shannon_multivariate(X, weights=None):
    """Calculate Jensen-Shannon divergence between several distributions.

    Parameters
    ----------
    X : array-like, shape = (n_samples, n_distributions)
        Discrete probability distributions.

    weights : array-like, shape = (n_distributions)
        Array of weights associated with the distributions. If None all
        distributions are assumed to have equal weight.

    Returns
    -------
    jensen_shannon : float
    """
    if X.ndim < 2:
        raise ValueError("X must be a 2-D array")

    X = np.asarray(X)
    n = X.shape[1]

    if weights is not None:
        weights = np.asarray(weights)

        if len(weights) != n:
            raise ValueError("Shapes of X and weights differ.")

        if weights.sum() != 1.0:
            raise ValueError("weights must sum 1.")
    else:
        weights = np.ones(n) / n

    js = entropy(np.sum(weights * X, axis=1))
    js -= np.dot(weights, [entropy(X[:, i]) for i in range(n)])

    return js


def hellinger(x, y, return_sum=False):
    """Calculate the Hellinger discrimination between two distributions.

    Parameters
    ----------
    x : array-like
        Discrete probability distribution.

    y : array-like
        Discrete probability distribution.

    return_sum : bool
        Return sum of jensen shannon values.

    Returns
    -------
    hellinger : float or numpy.ndarray
    """
    x, y = _check_x_y(x, y)

    h = 0.5 * (np.sqrt(x) - np.sqrt(y)) ** 2

    if return_sum:
        return h.sum()
    else:
        return h


def triangular(x, y, return_sum=False):
    """Calculate the LeCam or triangular discrimination between two
    distributions.

    Parameters
    ----------
    x : array-like
        Discrete probability distribution.

    y : array-like
        Discrete probability distribution.

    return_sum : bool
        Return sum of jensen shannon values.

    Returns
    -------
    triangular : float or numpy.ndarray
    """
    x, y = _check_x_y(x, y)

    t = (x - y) ** 2 / (x + y)

    if return_sum:
        return t.sum()
    else:
        return t


def brier(x, y, return_sum=False):
    """Calculate the Brier divergence between two distributions.

    Parameters
    ----------
    x : array-like
        Discrete probability distribution.

    y : array-like
        Discrete probability distribution.

    return_sum : bool
        Return sum of brier values.

    Returns
    -------
    brier : float or numpy.ndarray
    """
    x, y = _check_x_y(x, y)

    t = (x - y) ** 2

    if return_sum:
        return t.sum()
    else:
        return t

def neg_brier(x, y, return_sum=False):
    """Calculate the negative Brier divergence between two distributions.

    Parameters
    ----------
    x : array-like
        Discrete probability distribution.

    y : array-like
        Discrete probability distribution.

    return_sum : bool
        Return sum of brier values.

    Returns
    -------
    brier : float or numpy.ndarray
    """
    x, y = _check_x_y(x, y)

    t = -(x - y) ** 2

    if return_sum:
        return t.sum()
    else:
        return t

def log_score(x, y, return_sum=False, eps=1e-8):
    """Calculate the log likelihood between two distributions.

    Parameters
    ----------
    x : array-like
        Discrete probability distribution.

    y : array-like
        Discrete probability distribution.

    return_sum : bool
        Return sum of log likelihood values.

    Returns
    -------
    log_likelihood : float or numpy.ndarray
    """
    x, y = _check_x_y(x, y)

    t = x * np.log(y + eps) + (1 - x) * np.log(1 - y + eps)

    if return_sum:
        return t.sum()
    else:
        return t


def test_proportions(e1, ne1, e2, ne2):
    n1 = e1 + ne1
    n2 = e2 + ne2
    p1 = e1 / n1
    p2 = e2 / n2
    p = (e1 + e2) / (n1 + n2)

    z = (p1 - p2) / np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))

    statistic = abs(z)
    pvalue = stats.norm.sf(statistic)

    return statistic, 2 * pvalue


def frequentist_pvalue(obs, pvalue_method):
    if pvalue_method == "chi2":
        t, p, _, _ = stats.chi2_contingency(obs, correction=False)
        return t, p
    else:
        oddratio, p = stats.fisher_exact(obs)
        return oddratio, p


def chi2_cramer_v(n_nev, n_ev):
    obs = np.array([n_nev, n_ev])
    t, p, _, _ = stats.chi2_contingency(obs, correction=False)
    cramer_v = (t / (n_nev.sum() + n_ev.sum())) ** 0.5

    return t, cramer_v


def chi2_cramer_v_multi(n_ev):
    r, k = n_ev.shape
    t, p, _, _ = stats.chi2_contingency(n_ev, correction=False)
    cramer_v = (t / n_ev.sum() / min(k - 1, r - 1)) ** 0.5

    return t, cramer_v


def bayesian_probability(obs, n_samples):
    aA, aB, bA, bB = obs.ravel()

    r = np.arange(1, n_samples + 1)
    np.random.shuffle(r)
    v = (r - 0.5) / n_samples

    p = special.betainc(aA, bA, stats.beta(aB, bB).ppf(v)).mean()
    return p, 1 - p


def hhi(s, normalized=False):
    """Compute the Herfindahl–Hirschman Index (HHI).

    Parameters
    ----------
    s : array-like
        Fractions (exposure)

    normalized : bool (default=False)
        Whether to compute the normalized HHI.
    """
    s = np.asarray(s)
    h = np.sum(s ** 2)

    if normalized:
        n = len(s)
        if n == 1:
            return 1
        else:
            n1 = 1. / n
            return (h - n1) / (1 - n1)

    return h


def binning_quality_score(iv, p_values, hhi_norm):
    # Score 1: Information value
    c = 0.39573882184806863
    score_1 = iv * np.exp(1/2 * (1 - (iv / c) ** 2)) / c

    # Score 2: statistical significance (pairwise p-values)
    p_values = np.asarray(p_values)
    score_2 = np.prod(1 - p_values)

    # Score 3: homogeneity
    score_3 = 1. - hhi_norm

    return score_1 * score_2 * score_3


def multiclass_binning_quality_score(js, n_classes, p_values, hhi_norm):
    js_norm = js / np.log(n_classes)

    return binning_quality_score(js_norm, p_values, hhi_norm)


def continuous_binning_quality_score(rwoe, p_values, hhi_norm):
    # Score 1: ratio sum absolute WoEs / mean
    if rwoe == 0:
        score_1 = 0
    else:
        score_1 = max(1 - 1 / rwoe, 0)

    # Score 2: statistical significance (pairwise p-values)
    p_values = np.asarray(p_values)
    score_2 = np.prod(1 - p_values)

    # Score 2: homogeneity
    score_3 = 1. - hhi_norm

    return score_1 * score_2 * score_3


# ---------------------------------------------------------------------------
# Optimal-transport metrics (OT-WoE extension).
#
# Cesar Claros <cesar.claros@outlook.com>
# Copyright (C) 2026. Theory and numerical certification in the project
# documentation notes P1 (W1/Cramer aggregation), P3 (flat metric), P4
# (drift monitoring).
# ---------------------------------------------------------------------------

def wasserstein_1d(x, y, atoms):
    """Calculate the 1-Wasserstein distance between two atomic distributions
    supported on a common ordered grid.

    W_1 = sum_k |cumsum(x - y)_k| * (atoms_{k+1} - atoms_k).

    Parameters
    ----------
    x : array-like, shape = (n_atoms,)
        First distribution (masses; nonnegative, need not be normalized if
        both totals coincide).

    y : array-like, shape = (n_atoms,)
        Second distribution.

    atoms : array-like, shape = (n_atoms,)
        Increasing atom positions.

    Returns
    -------
    wasserstein_1d : float
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    atoms = np.asarray(atoms, dtype=float)

    if len(atoms) < 2:
        return 0.0

    cum = np.cumsum(x - y)[:-1]
    return float(np.sum(np.abs(cum) * np.diff(atoms)))


def cramer_1d(x, y, atoms, p=2):
    """Calculate the Cramer-p discrepancy int |F_x - F_y|^p between two
    atomic distributions on a common ordered grid.

    For p=1 this equals the 1-Wasserstein distance; p=2 is (half) the 1D
    energy distance. Extent-additive over contiguous bins for every p >= 1
    (project note P1, Prop. 1.4).

    Parameters
    ----------
    x : array-like, shape = (n_atoms,)
        First distribution.

    y : array-like, shape = (n_atoms,)
        Second distribution.

    atoms : array-like, shape = (n_atoms,)
        Increasing atom positions.

    p : float (default=2)
        Exponent, p >= 1.

    Returns
    -------
    cramer_1d : float
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    atoms = np.asarray(atoms, dtype=float)

    if len(atoms) < 2:
        return 0.0

    cum = np.cumsum(x - y)[:-1]
    return float(np.sum(np.abs(cum) ** p * np.diff(atoms)))


def hellinger_raw(nx, ny, return_sum=True):
    """Calculate the unnormalized (raw-count) squared Hellinger distance
    sum_k (sqrt(nx_k) - sqrt(ny_k))^2.

    Identity: sum (sqrt(nx)-sqrt(ny))^2 = N0 + N1 - 2*sqrt(N0*N1)*BC(p, q),
    with BC the Bhattacharyya coefficient of the normalized profiles, so
    maximizing the raw-count version is equivalent to maximizing the
    normalized Hellinger divergence. Finite on zero-count cells (project
    note P2, Thm. 2.4).

    Parameters
    ----------
    nx : array-like, shape = (n_bins,)
        First count vector (nonnegative).

    ny : array-like, shape = (n_bins,)
        Second count vector (nonnegative).

    return_sum : bool (default=True)
        Return sum of bin contributions.

    Returns
    -------
    hellinger_raw : float or numpy.ndarray
    """
    nx = np.asarray(nx, dtype=float)
    ny = np.asarray(ny, dtype=float)

    h = (np.sqrt(nx) - np.sqrt(ny)) ** 2

    if return_sum:
        return float(h.sum())
    return h


def flat_metric_1d(x, y, atoms, lam, validate=False):
    """Calculate the flat metric (bounded-Lipschitz distance) between two
    nonnegative atomic measures on a common ordered grid.

    FM_lam(x, y) = min over transport + creation/destruction of
    [transport cost (distance) + lam * (mass created + destroyed)].
    Equal totals are NOT required. Computed exactly via the level-set
    dynamic program on the TV-L1 representation (project note P3, Thm. A
    and Sec. 5.3); certified against LP ground truth to ~1e-15.

    Parameters
    ----------
    x : array-like, shape = (n_atoms,) or (n_batch, n_atoms)
        First measure(s). A 2D array evaluates a batch on the shared grid.

    y : array-like, shape = (n_atoms,) or (n_batch, n_atoms)
        Second measure(s).

    atoms : array-like, shape = (n_atoms,)
        Increasing atom positions.

    lam : float
        Mass creation/destruction price; 2*lam is the transport horizon.

    validate : bool (default=False)
        Basic input validation.

    Returns
    -------
    flat_metric_1d : float or numpy.ndarray of shape (n_batch,)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    atoms = np.asarray(atoms, dtype=float)

    single = (x.ndim == 1)
    A = np.atleast_2d(x)
    B = np.atleast_2d(y)

    if validate:
        if np.any(A < 0) or np.any(B < 0):
            raise ValueError("Measures must be nonnegative.")
        if np.any(np.diff(atoms) <= 0):
            raise ValueError("Atoms must be strictly increasing.")
        if A.shape != B.shape or A.shape[-1] != len(atoms):
            raise ValueError("Shape mismatch.")
        if lam < 0:
            raise ValueError("lam must be nonnegative; got {}.".format(lam))

    P, M = A.shape

    if M == 1:
        out = lam * np.abs(A[:, 0] - B[:, 0])
        return float(out[0]) if single else out

    H = np.cumsum(A - B, axis=1)              # (P, M); H[:, -1] = mass gap
    rho = np.diff(atoms)                      # (M - 1,)

    # Level-set property: an optimal adjustment profile takes values in
    # {0} U {H_1, ..., H_M} per instance.
    levels = np.concatenate([np.zeros((P, 1)), H], axis=1)
    levels = np.sort(levels, axis=1)          # (P, M + 1)
    K = M + 1
    gaps = np.diff(levels, axis=1)

    def _minplus(v):
        # inf-convolution with lam * |.| over the sorted level grid
        t = v.copy()
        for i in range(1, K):
            t[:, i] = np.minimum(t[:, i], t[:, i - 1] + lam * gaps[:, i - 1])
        for i in range(K - 2, -1, -1):
            t[:, i] = np.minimum(t[:, i], t[:, i + 1] + lam * gaps[:, i])
        return t

    value = lam * np.abs(levels) + rho[0] * np.abs(H[:, 0:1] - levels)
    for k in range(1, M - 1):
        value = _minplus(value) + rho[k] * np.abs(H[:, k:k + 1] - levels)
    value = value + lam * np.abs(H[:, -1:] - levels)

    out = value.min(axis=1)
    return float(out[0]) if single else out
