# C1 (tail-weighted loss, S1102_CROSSFOLD_MISPREDICT.md §C1) on ankh3_large, BALANCED splitter.
#
# Same dataset / 300ep-patience30 budget / balanced folds as the balanced base ankh3_large arm
# (scratch/results/embedding_sweep_balanced/ankh3_large, PCC 0.835) — the ONLY change is the
# training loss (--tail_loss_*). So the grid's plain-MSE point reproduces that base as an
# identically-trained in-run control, and every reweighted point differs from it only by the loss.
# Reuses the existing ankh3_large embedding cache; no new embeddings are generated.
#
# This config only pins the shared balanced settings + the C1 metrics root. The per-grid-point
# results dir and loss args are supplied by run_c1_grid.sh via C1_RESULTS_DIR / C1_EXTRA (read
# with defaults below), because run_sweep.sh sources this file and would otherwise clobber any
# ES_RESULTS_DIR / ES_EXTRA_TRAIN_ARGS the caller exported.
#
# Usage: driven by experiments/embedding_sweep/run_c1_grid.sh (do not call run_sweep directly).
source experiments/embedding_sweep/config_balanced.sh
export ES_RESULTS_DIR=${C1_RESULTS_DIR:-scratch/results/c1_ankh3_large/_scratch}
export ES_METRICS_DIR=experiments/embedding_sweep/results_c1_ankh3_large
export ES_EXTRA_TRAIN_ARGS=${C1_EXTRA:-}
