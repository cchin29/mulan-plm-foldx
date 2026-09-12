# Full-SKEMPI COMBINED single+multi, homology-clustered (mmseqs <=60%) 3-fold. The clustered analog
# of config_skempi_full_bycomplex_all.sh / the CATH skempi_all tier. Split: build_split_clustered_all.py.
# FoldX: merge_foldx_cath.py -> clustered_all/splits_cath_foldx{,dec}.
#   Coverage **99.1% measured 2026-08-01** (17238/17403 non-zero scalars across all three folds;
#   5746/5801 per fold, matching the bycomplex_all figure exactly since both tiers partition the
#   same 5801 combined rows). The previous "90.8% (dual-key)" in this header was the RETRACTED
#   figure -- that merge's second key landed on other real rows and handed them a different
#   mutation's ddG. config_skempi_full_bycomplex_all.sh says "do NOT cite the middle figure";
#   this file was citing it. Uncovered rows -> standardized 0 = neutral.
source experiments/full_skempi_seqonly/config_skempi_full.sh
export ES_NUM_FOLDS=3
export ES_BASENAME=skempi_all
export ES_WT_FASTA=scratch/skempi_full/wt_sequences_cath_all.fasta
export ES_SPLIT_DIR=scratch/splits_skempi_full_clustered_all_id60_kfold
export ES_FOLDXSCALAR_DIR=scratch/foldx_skempi_full/clustered_all/splits_cath_foldx
export ES_FOLDXDEC_DIR=scratch/foldx_skempi_full/clustered_all/splits_cath_foldxdec
export ES_RESULTS_DIR=scratch/results/full_skempi_clustered_all
export ES_METRICS_DIR=experiments/full_skempi_seqonly/results_clustered_all
export ES_MODEL_CONFIG=models/config/lightatt_default_config.json
export ES_SCALAR_CONFIG=models/config/lightatt_addscores_config.json
export ES_MLP_CONFIG=models/config/lightatt_addscores_mlp_config.json
