#!/usr/bin/env bash
#
# OT-WoE experiment suite: A1 (real + synthetic-spike), the fm_tau trust
# frontier swept over the trust radius lam_frac, and maxbins (free + auto),
# across SEEDS seeds. This is a long run; launch it in the background so it
# survives logout:
#
#   nohup bash run_all.sh > run_all.nohup.log 2>&1 &
#   tail -f run_all.nohup.log            # watch progress
#
# Tunables via environment (defaults in brackets):
#   PY [python]            interpreter, e.g. PY="uv run python"
#   SEEDS [10]             seeds per dataset (run internally, seed_offset=0)
#   DATASETS [german,taiwan,gmsc,hmeq]
#   LAM_FRACS [0.05,0.1,0.2,0.4]   fm_tau trust radii to sweep
#   NBOOT [50]             a1 bootstrap refits for cut stability
#   ARCHIVE [1]            move prior outputs aside before running (1/0)
# Example quick trial:  SEEDS=3 NBOOT=20 bash run_all.sh

set -uo pipefail
cd "$(dirname "$0")"                       # code/ root

PY="${PY:-python}"
SEEDS="${SEEDS:-10}"
DATASETS="${DATASETS:-german,taiwan,gmsc,hmeq}"
LAM_FRACS="${LAM_FRACS:-0.05,0.1,0.2,0.4}"
NBOOT="${NBOOT:-50}"
ARCHIVE="${ARCHIVE:-1}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="logs/${STAMP}"
mkdir -p "$LOGDIR"

log() { echo "[$(date '+%F %T')] $*"; }

# Preserve any prior outputs so the fresh multi-seed run is unambiguous: a1 and
# maxbins filenames are seed-offset only and would be overwritten, and old
# un-tagged fmtau files would double-count lam=0.1 against the new lam-tagged
# ones.
if [ "$ARCHIVE" = "1" ]; then
    arch="outputs/_archive_${STAMP}"
    for d in a1 a1_fmtau a1_maxbins a1_gamma a1_spikesel; do
        if [ -d "outputs/$d" ] && [ -n "$(ls -A "outputs/$d" 2>/dev/null)" ]; then
            mkdir -p "$arch"
            mv "outputs/$d" "$arch/"
            log "archived outputs/$d -> $arch/$d"
        fi
    done
fi

run_step() {                              # run_step <name> <cmd...>
    local name="$1"
    shift
    log ">>> $name"
    if "$@" > "$LOGDIR/${name}.log" 2>&1; then
        log "<<< $name OK"
    else
        log "!!! $name FAILED (exit $?); see $LOGDIR/${name}.log"
    fi
}

log "START  seeds=$SEEDS  datasets=$DATASETS  lam_fracs=$LAM_FRACS  nboot=$NBOOT"

# (2) iv_mip retired in conf/a1.yaml; A1 objective benchmark on real data.
run_step a1_real $PY experiments/run_a1.py -m dataset="$DATASETS" \
    n_seeds="$SEEDS" seed_offset=0 n_boot="$NBOOT"

# (3) A1 on the spike designs (location-extreme "spike" and interior near-tie
#     "spike2"): the hybrid vs iv bootstrap-fragility story real credit
#     features cannot exercise.
run_step a1_spike $PY experiments/run_a1.py -m \
    dataset=synthetic-spike,synthetic-spike2 \
    n_seeds="$SEEDS" seed_offset=0 n_boot="$NBOOT"

# Hybrid-weight sweep on both spike designs: does a larger gamma move the
# hybrid off pure IV and cut the spike-refit fragility? spike2's interior
# near-tie is the geometry where it is conjectured to pay off.
run_step gamma $PY experiments/run_gamma.py -m \
    dataset=synthetic-spike,synthetic-spike2 \
    n_seeds="$SEEDS" seed_offset=0 n_boot="$NBOOT"

# Spike-selection fragility: reproduce P1 Sec. 3.4's 47%/17% coin flip in the
# harness under the required protocol (2-bin cap, small min-size, cut-position
# read) with the faithful spike3 design.
run_step spikesel $PY experiments/run_spike_select.py \
    dataset=synthetic-spike3 n_seeds="$SEEDS" seed_offset=0

# (1) fm_tau trust-threshold frontier swept over the trust radius lam_frac.
run_step fmtau $PY experiments/run_fmtau.py -m dataset="$DATASETS" \
    lam_frac="$LAM_FRACS" n_seeds="$SEEDS" seed_offset=0

# maxbins: iv vs hellinger_raw under bin caps, both monotone modes.
run_step maxbins_free $PY experiments/run_maxbins.py -m dataset="$DATASETS" \
    n_seeds="$SEEDS" seed_offset=0

run_step maxbins_auto $PY experiments/run_maxbins.py -m dataset="$DATASETS" \
    n_seeds="$SEEDS" seed_offset=0 monotonic=auto

log "DONE. Aggregate with:"
log "  $PY experiments/tables.py a1       \"outputs/a1/*.parquet\""
log "  $PY experiments/tables.py a1_spike \"outputs/a1/*.parquet\""
log "  $PY experiments/tables.py gamma    \"outputs/a1_gamma/*.parquet\""
log "  $PY experiments/tables.py spikesel \"outputs/a1_spikesel/*.parquet\""
log "  $PY experiments/tables.py fmtau    \"outputs/a1_fmtau/*.parquet\""
log "  $PY experiments/tables.py maxbins  \"outputs/a1_maxbins/*.parquet\""
