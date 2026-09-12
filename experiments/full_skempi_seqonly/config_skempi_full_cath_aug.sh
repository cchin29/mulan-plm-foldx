# CATH FoldX x Tier-1-aug combo. Reuses the foldx/foldx_scalar arm machinery but points at the
# augmented splits (reverse-mutation + identity, FoldX taken as antisymmetric) + aug fasta. Arms foldx/
# foldx_scalar here == foldx_mlp_aug / foldx_scalar_aug. Built by merge_foldx_aug.py.
#
# Splits rebuilt 2026-08-05: the synthesized rows' FoldX channel was negated after standardisation
# instead of before, a ~1-sd constant offset on half of train. Any result from an augmented FoldX arm
# trained before that date is uninterpretable — see
# experiments/beyond_foldx/FOLDX_ANTISYMMETRY_RESULT.md §1.
source experiments/full_skempi_seqonly/config_skempi_full_cath.sh
export ES_WT_FASTA=scratch/skempi_full/wt_sequences_cath_all_aug.fasta
export ES_FOLDXSCALAR_DIR=scratch/foldx_skempi_full/splits_cath_foldx_aug
export ES_FOLDXDEC_DIR=scratch/foldx_skempi_full/splits_cath_foldxdec_aug
export ES_RESULTS_DIR=scratch/results/full_skempi_cath_aug
export ES_SCALAR_CONFIG=models/config/lightatt_addscores_config.json
export ES_MLP_CONFIG=models/config/lightatt_addscores_mlp_config.json

# The base arm's 4-col split. WITHOUT this line `ARMS=base` inherits ES_SPLIT_DIR from the
# unaugmented parent (splits_skempi_full_cath_kfold) and trains the PLAIN base into the _aug results
# dir, exiting 0 — a wrong answer that reports success, not a failure. Same trap the clustered-ALL
# aug config carries a note about; this tier had no override at all until 2026-08-05.
#
# Build it by dropping the FoldX scalar column from splits_cath_foldx_aug (`cut -f1-4`). That is
# exact on this tier, verified the same way as clustered-ALL: the unaugmented 4-col split is
# byte-identical to its 5-col FoldX sibling minus that column on train, val and test (4213 / 878 /
# 687 rows). Augmentation is train-only here too — the aug val and test files are byte-identical to
# the unaugmented ones, and only train grows, 4213 -> 8683.
#
#   for s in train val test; do
#     mkdir -p scratch/splits_skempi_full_cath_aug/fold_0
#     cut -f1-4 scratch/foldx_skempi_full/splits_cath_foldx_aug/fold_0/skempi_all_$s.tsv \
#       > scratch/splits_skempi_full_cath_aug/fold_0/skempi_all_$s.tsv
#   done
#
# CATH is a single hold-out, so fold_0 only — not the three folds the other aug tiers build.
export ES_SPLIT_DIR=scratch/splits_skempi_full_cath_aug
