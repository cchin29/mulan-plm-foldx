# A1 (interface cross-attention, docs/history/PLAN_A1_INTERFACE_XATTN.md) on ankh3_large,
# BALANCED splitter.
#
# Same dataset / 300ep-patience30 budget / balanced folds as the balanced base ankh3_large arm
# (scratch/results/embedding_sweep_balanced/ankh3_large, PCC 0.835) — the changes are (a) the
# model config (interface_xattn head, lightatt_a1_ankh3_large.json, xattn_dim 1536) and (b) an
# interface mask dir fed at train time (--interface_mask_dir, supplied per grid arm). Reuses the
# existing ankh3_large embedding cache; no new embeddings.
#
# Like config_c1_ankh3_large.sh, this pins the shared balanced settings + the A1 model config + the A1
# metrics root, and reads the per-arm results dir and extra train args from the caller
# (A1_RESULTS_DIR / A1_EXTRA) so run_sweep.sh sourcing this file does not clobber them. The
# per-arm --interface_mask_dir (real / dense / shuffled) rides in A1_EXTRA.
#
# Usage: driven by experiments/embedding_sweep/run_a1_grid.sh (do not call run_sweep directly).
source experiments/embedding_sweep/config_balanced.sh
export ES_MODEL_CONFIG=${A1_MODEL_CONFIG:-models/config/lightatt_a1_ankh3_large.json}
export ES_RESULTS_DIR=${A1_RESULTS_DIR:-scratch/results/a1_ankh3_large/_scratch}
export ES_METRICS_DIR=experiments/embedding_sweep/results_a1_ankh3_large
export ES_EXTRA_TRAIN_ARGS=${A1_EXTRA:-}
