# Full-SKEMPI MULTI-POINT, by-complex (whole-#Pdb-grouping) 3-fold split.
# Inherits the single-point config (paper budget 300ep/p30/bs32/lr5e-4, lightatt heads,
# models registry) and overrides ONLY the dataset paths -> multi-point. Drives
# run_bycomplex.sh (arms: base / foldx / foldx_scalar).
#   ES_CONFIG=experiments/full_skempi_seqonly/config_skempi_full_mp_bycomplex.sh \
#       ARMS="base foldx" MAXPAR=1 bash experiments/retrain_split/run_bycomplex.sh ankh saprot prostt5
source experiments/full_skempi_seqonly/config_skempi_full.sh   # budget + heads + models.tsv

export ES_BASENAME=skempi_mp
export ES_WT_FASTA=scratch/skempi_full/wt_sequences_multipoint.fasta
export ES_SPLIT_DIR=scratch/splits_skempi_full_mp_bycomplex_seed42                      # base 4-col
export ES_FOLDXDEC_DIR=scratch/foldx_skempi_full/splits_mp_bycomplex_mp_foldxdec        # 16-col (12 terms)
export ES_FOLDXSCALAR_DIR=scratch/foldx_skempi_full/splits_mp_bycomplex_mp_foldx        # 5-col scalar
export ES_RESULTS_DIR=scratch/results/full_skempi_mp_bycomplex
export ES_METRICS_DIR=experiments/full_skempi_seqonly/results_mp_bycomplex
