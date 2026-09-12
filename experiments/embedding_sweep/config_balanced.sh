# Balanced-splitter variant of the embedding-sweep config (2026-07-12).
#
# SAME dataset (S1102_filtered, 1100) and SAME 300ep/patience30 budget as the paper
# config, but the fork's BALANCED splitter (equal-size folds via block-assign + shuffle)
# instead of the paper's `random` rng.integers partition. Kept in separate split /
# results / metrics / aug dirs so the paper-faithful rng.integers runs — especially the
# Ankh-large paper-reproduction anchor in scratch/results/embedding_sweep/ankh — stay
# intact for comparison against the paper's 0.868.
#
# Usage (drivers + gen_splits honor ES_CONFIG; aug driver honors ES_AUG_DIR):
#   ES_CONFIG=experiments/embedding_sweep/config_balanced.sh \
#       ./.venv/bin/python experiments/embedding_sweep/gen_splits.py
#   ES_CONFIG=experiments/embedding_sweep/config_balanced.sh bash scratch/embsweep_base_driver.sh ankh
source experiments/embedding_sweep/config.sh
export ES_SPLIT_METHOD=balanced
export ES_SPLIT_DIR=experiments/embedding_sweep/splits/balanced_seed42
export ES_RESULTS_DIR=scratch/results/embedding_sweep_balanced
export ES_METRICS_DIR=experiments/embedding_sweep/results_balanced
export ES_AUG_DIR=scratch/embsweep_aug_splits_balanced
