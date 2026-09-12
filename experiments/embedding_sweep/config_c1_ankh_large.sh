# C1 (tail-weighted loss, S1102_CROSSFOLD_MISPREDICT.md §C1) on **Ankh v1 Large**
# (ElnaggarLab/ankh-large, tag `ankh`) — the MuLAN-paper REFERENCE model. BALANCED splitter.
#
# v1 sibling of config_c1_ankh3_large.sh (the ankh3 grid config, renamed from the formerly-misleading
# config_c1_ankh.sh — see
# experiments/ANKH_PROVENANCE_AUDIT.md). C1 changes only the training loss (--tail_loss_*); it reuses
# the default balanced head (config_balanced.sh) and the existing Ankh-v1 embedding cache
# (scratch/embeddings, tag `ankh`). So the grid's plain-MSE point reproduces the Ankh-v1 balanced base
# (embedding_sweep_balanced/ankh) as an identically-trained in-run control.
#
# Reads the per-grid-point results dir + loss args from the caller (C1_RESULTS_DIR / C1_EXTRA).
#
# Usage (pass CFG explicitly until fix_ankh_labels.sh makes CFG overridable in run_c1_grid.sh):
#   TAG=ankh CFG=experiments/embedding_sweep/config_c1_ankh_large.sh MAXPAR=2 \
#     bash experiments/embedding_sweep/run_c1_grid.sh          # -> scratch/results/c1_ankh
source experiments/embedding_sweep/config_balanced.sh
export ES_RESULTS_DIR=${C1_RESULTS_DIR:-scratch/results/c1_ankh/_scratch}
export ES_METRICS_DIR=experiments/embedding_sweep/results_c1_ankh
export ES_EXTRA_TRAIN_ARGS=${C1_EXTRA:-}
