# Full-SKEMPI single-point, sequence-only, BY-COMPLEX honest split.
# By-complex analog of config_skempi_full.sh (which uses the homology-clustered split). Same
# paper training budget (300ep/p30, bs32, lr5e-4, lightatt_default head) and same skempi_sp
# dataset/basename — the ONLY change is the split (by-complex seed42 instead of clustered id60)
# and the ISOLATED results/foldx dirs, so nothing in the clustered-SP work is touched.
#   ES_CONFIG=experiments/full_skempi_seqonly/config_skempi_full_bycomplex.sh \
#       ARMS="base foldx foldx_scalar" MAXPAR=1 bash experiments/retrain_split/run_bycomplex.sh esmc6b
source experiments/embedding_sweep/config.sh          # 300ep/p30/bs32/lr5e-4 + lightatt_default
export ES_MODELS_TSV=experiments/full_skempi_seqonly/models_skempi_full.tsv   # isolated emb dirs
export ES_NUM_FOLDS=3
export ES_BASENAME=skempi_sp
export ES_WT_FASTA=scratch/skempi_full/wt_sequences.fasta
export ES_SPLIT_DIR=scratch/splits_skempi_full_bycomplex_seed42               # base 4-col
export ES_RESULTS_DIR=scratch/results/full_skempi_bycomplex
export ES_METRICS_DIR=experiments/full_skempi_seqonly/results_bycomplex

# --- FoldX arms: built by merge_foldx_full_skempi.py --src <bycomplex split> into a DISTINCT
#     outdir so the clustered-SP foldx splits are not clobbered. Only consumed when ARMS
#     includes foldx / foldx_scalar. ---
export ES_FOLDXSCALAR_DIR=scratch/foldx_skempi_full/bycomplex/splits_skempi_full_foldx     # 5-col scalar
export ES_FOLDXDEC_DIR=scratch/foldx_skempi_full/bycomplex/splits_skempi_full_foldxdec      # 16-col (12 terms)
export ES_SCALAR_CONFIG=models/config/lightatt_addscores_config.json                        # foldx-scalar head
export ES_MLP_CONFIG=models/config/lightatt_addscores_mlp_config.json                       # foldx-MLP head
