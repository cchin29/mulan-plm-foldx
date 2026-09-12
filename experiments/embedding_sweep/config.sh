# Single source of truth for the paper-faithful embedding sweep.
# Sourced by run_sweep.sh; parsed by gen_splits.py / aggregate.py (simple `export K=V` lines).
# All paths are relative to the repo root; run everything from the repo root.

# --- paper-faithful CV setup (MuLAN paper, Methods "Implementation details" + Table 1) ---
export ES_SEED=42                 # dataset-split + init seed (paper)
export ES_SPLIT_METHOD=random     # data.split_data method: upstream rng.integers (= paper)
export ES_NUM_FOLDS=10            # 10-fold CV for S1102 single-point mutations (paper)
export ES_NUM_EPOCHS=300          # "train until convergence" — no 50-epoch cap
export ES_PATIENCE=30             # early-stop patience (LR-plateau patience 5 is built into trainer)
export ES_BATCH=32                # paper
export ES_LR=5e-4                 # paper
export ES_MODEL_CONFIG=models/config/lightatt_default_config.json   # add_scores=false (= paper head)

# --- data (1100 = S1102 minus 2I9B; see data/PROVENANCE.md) ---
export ES_DATASET=experiments/embedding_sweep/data/S1102_filtered.tsv
export ES_WT_FASTA=experiments/embedding_sweep/data/wt_sequences.fasta
export ES_BASENAME=S1102_filtered

# --- tracked splits (small text; pins the exact partition) ---
export ES_SPLIT_DIR=experiments/embedding_sweep/splits/paper_seed42

# --- model registry (tag -> plm / embeddings dir) lives in models.tsv, read by bash + python ---
export ES_MODELS_TSV=experiments/embedding_sweep/models.tsv

# --- heavy run outputs stay OUT of git (gitignored scratch/) ---
export ES_RESULTS_DIR=scratch/results/embedding_sweep

# --- tracked aggregated metrics ---
export ES_METRICS_DIR=experiments/embedding_sweep/results
