#!/usr/bin/env bash
# A1 interface-cross-attention grid (docs/history/PLAN_A1_INTERFACE_XATTN.md §5) on ONE base PLM,
# BALANCED splitter. Each arm runs the full 10-fold balanced CV via run_sweep.sh with the A1 head
# (config_a1_ankh3_large.sh -> lightatt_a1_ankh3_large.json) and a distinct interface mask, into a
# distinct results subdir. The three mask arms ARE the §5 controls that separate the interface
# PRIOR from generic added cross-attention capacity:
#
#   real     — the true 8A interface mask (scratch/interface_masks). The A1 hypothesis.
#   dense    — all-pairs mask (interface_masks_dense): cross-attention with NO interface prior.
#   shuffled — column-permuted mask (interface_masks_shuffled): same per-row density, wrong pairing.
#
# Decision rule (§5): A1 is a real lever only if real > dense AND real > shuffled on the worst-40
# tail. If real ~ shuffled, the gain is just capacity. The base (no-xattn) comparison is the
# existing balanced ankh3_large arm (0.835); it is NOT re-run here.
#
#   MAXPAR=2 bash experiments/embedding_sweep/run_a1_grid.sh                 # ankh3_large, all 3 arms
#   GRID_ONLY="real" bash experiments/embedding_sweep/run_a1_grid.sh        # just the pilot arm
#
# Prereq: masks built for all three variants:
#   ./.venv/bin/python experiments/interface_xattn/build_masks.py                    # real
#   ./.venv/bin/python experiments/interface_xattn/build_masks.py --variant dense
#   ./.venv/bin/python experiments/interface_xattn/build_masks.py --variant shuffle
#
# Results:  scratch/results/a1_<TAG>/<arm>/<TAG>/fold_*/training_run/  (gitignored)
# Aggregate each arm with experiments/embedding_sweep/aggregate.py pointed at its dir; judge on the
# worst-40 tail (experiments/mint/diagnose_shrinkage.py), not bulk PCC.
#
# A1xC1 (§5 control 4, "the key cell"): once C1's best setting is known, add an arm, e.g.
#   "real_c1:--interface_mask_dir scratch/interface_masks --tail_loss_mode linear --tail_loss_alpha 2"
# (both flags coexist; no code change needed).
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../.." || exit 1
TAG=${TAG:-ankh3_large}
CFG=${CFG:-experiments/embedding_sweep/config_a1_ankh3_large.sh}
ROOT=scratch/results/a1_${TAG}
log(){ echo "[$(date +%H:%M:%S)] $*"; }

# arm label -> mask dir (via --interface_mask_dir extra train arg)
declare -a GRID=(
  "real:--interface_mask_dir scratch/interface_masks"
  "dense:--interface_mask_dir scratch/interface_masks_dense"
  "shuffled:--interface_mask_dir scratch/interface_masks_shuffled"
)

for point in "${GRID[@]}"; do
  label=${point%%:*}; args=${point#*:}
  if [ -n "${GRID_ONLY:-}" ] && [[ " $GRID_ONLY " != *" $label "* ]]; then
    log "skip $label (not in GRID_ONLY)"; continue
  fi
  log "===== A1 grid arm '$label'  base=$TAG  [$args] ====="
  A1_RESULTS_DIR="$ROOT/$label" A1_EXTRA="$args" ES_CONFIG="$CFG" \
    bash experiments/embedding_sweep/run_sweep.sh "$TAG"
done
log "A1 GRID DONE for base=$TAG"
