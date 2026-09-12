# MINT arm for the homology-clustered (Phase-2) sweep.
#
# Same as config_bycomplex_mint.sh but layered on the CLUSTERED split: sources
# config_clustered.sh (mmseqs <=60%-id family-GroupKFold(3), retrain_clustered results dir)
# and adds --mint_pair True. MINT is base-arm only (partner-context embeddings; no FoldX
# add_scores arms) and reuses the SAME regular ./.venv/bin/mulan-train. The row-keyed MINT
# bundle cache (bundles keyed by s1__s2__mut, not fold index) is complete for all 1100 muts,
# so the clustered re-fold reuses it read-only with no regeneration.
#
# Usage:
#   ES_CONFIG=experiments/retrain_split/config_clustered_mint.sh \
#       MAXPAR=2 bash experiments/retrain_split/run_bycomplex_mint.sh
source experiments/retrain_split/config_clustered.sh
export ES_EXTRA_TRAIN_ARGS="--mint_pair True"
