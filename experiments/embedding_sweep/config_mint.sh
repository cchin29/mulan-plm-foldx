# MINT (A2) arm — partner-context embeddings on the BALANCED splitter.
#
# SAME dataset (S1102_filtered, 1100), SAME balanced splits, SAME 300ep/patience30 budget as the
# balanced base/aug/FoldX arms in scripts_plots/S1102_CROSSFOLD_MISPREDICT.md — so the MINT pooled
# OOF PCC is directly comparable. Differs only in: embeddings are MINT partner-context bundles read
# via --mint_pair, and results go to an ISOLATED dir (scratch/results/mint_a2) so nothing in
# embedding_sweep_balanced is touched.
#
# Usage (from repo root; MINT cache must be generated first — experiments/mint/gen_mint_emb.py):
#   ES_CONFIG=experiments/embedding_sweep/config_mint.sh MAXPAR=2 \
#       bash experiments/embedding_sweep/run_sweep.sh mint
source experiments/embedding_sweep/config_balanced.sh
export ES_RESULTS_DIR=scratch/results/mint_a2
export ES_METRICS_DIR=experiments/embedding_sweep/results_mint
export ES_EXTRA_TRAIN_ARGS="--mint_pair True"
