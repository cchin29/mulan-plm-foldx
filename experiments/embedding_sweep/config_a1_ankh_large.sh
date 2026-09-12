# A1 (interface cross-attention, docs/history/PLAN_A1_INTERFACE_XATTN.md) on **Ankh v1 Large**
# (ElnaggarLab/ankh-large, tag `ankh`) — the MuLAN-paper REFERENCE model. BALANCED splitter.
#
# This is the v1 sibling of config_a1_ankh3_large.sh (the ankh3 grid config, renamed from the
# formerly-misleading config_a1_ankh.sh — see
# experiments/ANKH_PROVENANCE_AUDIT.md). It reuses the existing Ankh-v1 embedding cache
# (scratch/embeddings, tag `ankh`); no new embeddings. Same 300ep/patience-30 balanced budget and the
# same A1 head geometry (xattn_dim 1536 — ankh-large and ankh3-large share that width), just an
# honestly-named lightatt config so the reference arm is not filed under an "ankh3" name.
#
# Baseline for comparison: the Ankh-v1 balanced base arm (embedding_sweep_balanced/ankh) and the
# paper-reference cv10_ankh_converge, NOT the ankh3_large 0.835 arm.
#
# Like config_a1_ankh3_large.sh, this reads the per-arm results dir + extra train args from the caller
# (A1_RESULTS_DIR / A1_EXTRA) so run_sweep.sh sourcing it does not clobber them; the per-arm
# --interface_mask_dir (real / dense / shuffled) rides in A1_EXTRA.
#
# Usage (until fix_ankh_labels.sh makes CFG overridable in run_a1_grid.sh, pass CFG explicitly):
#   TAG=ankh CFG=experiments/embedding_sweep/config_a1_ankh_large.sh MAXPAR=2 \
#     bash experiments/embedding_sweep/run_a1_grid.sh          # -> scratch/results/a1_ankh
source experiments/embedding_sweep/config_balanced.sh
export ES_MODEL_CONFIG=${A1_MODEL_CONFIG:-models/config/lightatt_a1_ankh_large.json}
export ES_RESULTS_DIR=${A1_RESULTS_DIR:-scratch/results/a1_ankh/_scratch}
export ES_METRICS_DIR=experiments/embedding_sweep/results_a1_ankh
export ES_EXTRA_TRAIN_ARGS=${A1_EXTRA:-}
