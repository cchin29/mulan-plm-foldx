# MINT-mono (A2 §8a control) — monomer MINT embeddings on the BALANCED splitter.
#
# Same MINT-650M backbone as the complex arm (config_mint.sh) but each chain is embedded ALONE
# (gen_mint_emb.py --single-chain into scratch/embeddings_mint_mono). Compared to the complex arm this
# isolates PARTNER CONTEXT; compared to the ESM2 base it isolates the MINT-vs-ESM2 backbone — so the
# 0.866-vs-0.878 pooled gap can be attributed rather than conflated (docs/history/PLAN_MINT_A2.md §5 caveat, §8a).
# Reads the mono cache via the same --mint_pair loader; results ISOLATED in scratch/results/mint_mono.
#
# Usage (from repo root; mono cache must be generated first):
#   env MINT_EMB=scratch/embeddings_mint_mono ./.venv-mint/bin/python experiments/mint/gen_mint_emb.py --single-chain
#   ES_CONFIG=experiments/embedding_sweep/config_mint_mono.sh MAXPAR=2 \
#       bash experiments/embedding_sweep/run_sweep.sh mint_mono
source experiments/embedding_sweep/config_balanced.sh
export ES_RESULTS_DIR=scratch/results/mint_mono
export ES_METRICS_DIR=experiments/embedding_sweep/results_mint_mono
export ES_EXTRA_TRAIN_ARGS="--mint_pair True"
