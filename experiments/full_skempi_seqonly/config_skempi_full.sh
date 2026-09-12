# Full-SKEMPI single-point, sequence-only (step 1) training config.
# docs/history/PLAN_FULL_SKEMPI.md §6. Reuses run_bycomplex.sh as an arm-generic driver:
#   ES_CONFIG=experiments/full_skempi_seqonly/config_skempi_full.sh \
#       ARMS=base MAXPAR=1 bash experiments/retrain_split/run_bycomplex.sh ankh esm2 saprot
#
# SAME paper training budget (300ep/p30, bs32, lr5e-4, lightatt_default head) as the S1102
# sweeps — the ONLY changes are the dataset (full-SKEMPI single-point, 4,922 muts), the
# homology-clustered 3-fold split, and ISOLATED embedding/results dirs (nothing in the S1102
# work is touched). Base arm only for now; the FoldX arms are step 2 (gated on §8b).
source experiments/embedding_sweep/config.sh          # 300ep/p30/bs32/lr5e-4 + lightatt_default
export ES_MODELS_TSV=experiments/full_skempi_seqonly/models_skempi_full.tsv   # isolated emb dirs
export ES_NUM_FOLDS=3
export ES_BASENAME=skempi_sp
export ES_WT_FASTA=scratch/skempi_full/wt_sequences.fasta
export ES_SPLIT_DIR=scratch/splits_skempi_full_clustered_id60_kfold           # base 4-col
export ES_RESULTS_DIR=scratch/results/full_skempi
export ES_METRICS_DIR=experiments/full_skempi_seqonly/results

# --- FoldX arms (step 2, §8b). Splits built by merge_foldx_full_skempi.py from
#     scratch/foldx_skempi_full/results (g1->A/g2->B chain remap). Coverage **99.2% measured
#     2026-08-01** (12393/12495 non-zero scalars). The "28% row coverage" this header used to
#     state was the pre-fix __cov_pre lineage and is why arms trained before 2026-07-29 carry a
#     __cov_pre archive. Only consumed when ARMS includes foldx / foldx_scalar. ---
export ES_FOLDXSCALAR_DIR=scratch/foldx_skempi_full/splits_skempi_full_foldx        # 5-col scalar
export ES_FOLDXDEC_DIR=scratch/foldx_skempi_full/splits_skempi_full_foldxdec         # 16-col (12 terms)
export ES_SCALAR_CONFIG=models/config/lightatt_addscores_config.json                 # foldx-scalar head (phase1)
export ES_MLP_CONFIG=models/config/lightatt_addscores_mlp_config.json                # foldx-MLP head (phase2)
