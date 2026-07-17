"""
Experiment B2 — drift monitoring power study (Paper B).

Per drift protocol x replication: fit a scorecard on the expected sample,
inject drift into the actual sample, and compare FM (permutation p-value +
PD certificate) against PSI (Yurdakul chi2) and KS at matched size.

Local smoke test:
    python experiments/run_b2.py n_reps=3 n_permutations=30
HPC:
    python experiments/run_b2.py -m hydra/launcher=submitit_slurm \
        drift_kind=none,location,tail,support rep_offset=range(0,20)
"""

# Cesar Claros <cesar.claros@outlook.com>
# Copyright (C) 2026

import sys

from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra                                            # noqa: E402
from omegaconf import DictConfig                        # noqa: E402
from scipy import stats                                 # noqa: E402
from sklearn.linear_model import LogisticRegression     # noqa: E402

from optbinning import BinningProcess, Scorecard        # noqa: E402
from optbinning import ScorecardMonitoring              # noqa: E402

from experiments import datasets                        # noqa: E402
from experiments.common import save_results             # noqa: E402


def _psi_chi2_pvalue(monitoring, n_a, n_e, n_bins):
    """Yurdakul: PSI ~ (1/n + 1/m) chi2_{B-1} under the null."""
    psi = monitoring.psi_table()["PSI"].sum()
    scale = 1.0 / n_a + 1.0 / n_e
    return psi, float(1 - stats.chi2.cdf(psi / scale, df=n_bins - 1))


def run(cfg):
    rows = []
    for rep in range(cfg.rep_offset, cfg.rep_offset + cfg.n_reps):
        expected = datasets.make_synthetic(design=cfg.design, n=cfg.n,
                                           seed=1000 + rep)
        drift = (None if cfg.drift_kind == "none"
                 else dict(kind=cfg.drift_kind,
                           magnitude=cfg.drift_magnitude))
        actual = datasets.make_synthetic(design=cfg.design, n=cfg.n,
                                         seed=2000 + rep, drift=drift)

        # Fit/reference split of the expected sample: the scorecard is fit
        # on the first half so that BOTH the reference and the actual
        # samples are out-of-sample — otherwise the in-sample optimism of
        # the reference scores shows up as spurious "drift" (and breaks
        # the exchangeability underlying the permutation null).
        h = len(expected.y) // 2
        sc_fit_X, sc_fit_y = expected.X.iloc[:h], expected.y[:h]
        ref_X, ref_y = expected.X.iloc[h:], expected.y[h:]

        bp = BinningProcess(variable_names=list(expected.numerical))
        sc = Scorecard(binning_process=bp,
                       estimator=LogisticRegression(),
                       scaling_method="min_max",
                       scaling_method_params={"min": 300, "max": 850})
        sc.fit(sc_fit_X, sc_fit_y)

        mon = ScorecardMonitoring(scorecard=sc, psi_method=cfg.psi_method,
                                  psi_n_bins=cfg.psi_n_bins)
        mon.fit(actual.X, actual.y, ref_X, ref_y)

        # FM with permutation p-value and PD certificate
        fm_tab = mon.fm_table(n_permutations=cfg.n_permutations,
                              random_state=rep)
        fm, fm_p = fm_tab["FM"].iloc[0], fm_tab["p-value"].iloc[0]

        # PSI (Yurdakul chi2)
        psi, psi_p = _psi_chi2_pvalue(mon, len(actual.y), len(ref_y),
                                      cfg.psi_n_bins)

        # KS on scores
        ks = stats.ks_2samp(mon._score_actual, mon._score_expected)

        rows.append(dict(
            design=cfg.design, drift=cfg.drift_kind,
            magnitude=cfg.drift_magnitude, rep=rep,
            fm=fm, fm_p=fm_p, fm_reject=fm_p <= cfg.alpha,
            psi=psi, psi_p=psi_p, psi_reject=psi_p <= cfg.alpha,
            ks=ks.statistic, ks_p=ks.pvalue, ks_reject=ks.pvalue <= cfg.alpha,
            pd_bound=fm_tab["PD impact bound"].iloc[0],
            pd_realized=fm_tab["realized |dPD|"].iloc[0]))

    out = Path(cfg.out) / "b2_{}_{}_{}".format(cfg.design, cfg.drift_kind,
                                               cfg.rep_offset)
    path = save_results(rows, out)
    print("B2: wrote {} rows -> {}".format(len(rows), path))
    return path


@hydra.main(version_base=None, config_path="../conf", config_name="b2")
def main(cfg: DictConfig):
    return run(cfg)


if __name__ == "__main__":
    main()
