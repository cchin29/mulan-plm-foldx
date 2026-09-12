#!/usr/bin/env bash
# C1 tail-weighted-loss grid (S1102_CROSSFOLD_MISPREDICT.md §C1) on ONE base PLM, BALANCED splitter.
#
# The A2/MINT diagnostic concluded the tail bottleneck is the head + MSE loss (extremes regressed to
# the mean), not embedding partner-blindness — promoting C1 to the top of the reranked list. This
# driver tests whether reweighting the loss de-shrinks the high-|ΔΔG| tail. For each grid point it
# runs the full 10-fold balanced CV via run_sweep.sh with a distinct loss setting and a distinct
# results subdir, so the plain-MSE point (mse_a0) is an identically-trained in-run control for the
# reweighted points. Reuses the PLM's existing embedding cache; no new embeddings.
#
#   MAXPAR=2 bash experiments/embedding_sweep/run_c1_grid.sh              # ankh3_large, full grid
#   TAG=esm2 MAXPAR=2 bash experiments/embedding_sweep/run_c1_grid.sh    # a different base PLM
#   GRID_ONLY="lin_a1 lin_a2" bash experiments/embedding_sweep/run_c1_grid.sh   # subset of points
#
# Results:  scratch/results/c1_<TAG>/<point>/<TAG>/fold_*/training_run/  (gitignored)
# Aggregate each point with experiments/embedding_sweep/aggregate.py pointed at its dir.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../.." || exit 1
TAG=${TAG:-ankh3_large}
CFG=${CFG:-experiments/embedding_sweep/config_c1_ankh3_large.sh}
ROOT=scratch/results/c1_${TAG}
log(){ echo "[$(date +%H:%M:%S)] $*"; }

# point label -> extra train args. mse_a0 = plain MSE control (must reproduce the balanced base).
declare -a GRID=(
  "mse_a0:--tail_loss_mode none"
  "lin_a1:--tail_loss_mode linear --tail_loss_alpha 1"
  "lin_a2:--tail_loss_mode linear --tail_loss_alpha 2"
  "lin_a4:--tail_loss_mode linear --tail_loss_alpha 4"
  "focal_g1:--tail_loss_mode focal --tail_loss_gamma 1"
  "focal_g2:--tail_loss_mode focal --tail_loss_gamma 2"
)

for point in "${GRID[@]}"; do
  label=${point%%:*}; args=${point#*:}
  if [ -n "${GRID_ONLY:-}" ] && [[ " $GRID_ONLY " != *" $label "* ]]; then
    log "skip $label (not in GRID_ONLY)"; continue
  fi
  log "===== C1 grid point '$label'  base=$TAG  [$args] ====="
  C1_RESULTS_DIR="$ROOT/$label" C1_EXTRA="$args" ES_CONFIG="$CFG" \
    bash experiments/embedding_sweep/run_sweep.sh "$TAG"
done
log "C1 GRID DONE for base=$TAG"
