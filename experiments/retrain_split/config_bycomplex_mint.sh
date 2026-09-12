# MINT arm for the by-complex (Phase-1) sweep.
#
# MINT is base-arm only (partner-context embeddings; no FoldX add_scores arms). It reuses the
# SAME regular ./.venv/bin/mulan-train (the .venv-mint is only for GENERATING embeddings) but
# reads the row-keyed MINT bundle cache via --mint_pair True. The row-keyed cache is complete
# for all 1100 muts (bundles keyed by s1__s2__mut, not fold index), so the by-complex re-fold
# reuses it read-only with no regeneration.
#
# Usage:
#   MAXPAR=2 bash experiments/retrain_split/run_bycomplex_mint.sh
source experiments/retrain_split/config_bycomplex.sh
export ES_EXTRA_TRAIN_ARGS="--mint_pair True"
