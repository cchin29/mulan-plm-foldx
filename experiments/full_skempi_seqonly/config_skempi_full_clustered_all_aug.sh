# Tier-1 aug (reverse-mutation + identity rows, FoldX taken as antisymmetric) on the COMBINED single+multi
# homology-clustered tier. Arms foldx/foldx_scalar here == foldx_mlp_aug / foldx_scalar_aug.
# Augmentation is TRAIN-ONLY: the aug splits' val/test TSVs are byte-identical to the unaugmented
# tier's on every fold, which is what makes an aug-vs-plain delta a paired comparison on one test set.
#
# Splits rebuilt 2026-08-05: the synthesized rows' FoldX channel was negated after standardisation
# instead of before, a ~1-sd constant offset on half of train. Any result from an augmented FoldX arm
# trained before that date is uninterpretable — see
# experiments/beyond_foldx/FOLDX_ANTISYMMETRY_RESULT.md §1.
source experiments/full_skempi_seqonly/config_skempi_full_clustered_all.sh
export ES_WT_FASTA=scratch/skempi_full/wt_sequences_clustered_all_aug.fasta
export ES_FOLDXSCALAR_DIR=scratch/foldx_skempi_full/clustered_all/splits_cath_foldx_aug
export ES_FOLDXDEC_DIR=scratch/foldx_skempi_full/clustered_all/splits_cath_foldxdec_aug
export ES_RESULTS_DIR=scratch/results/full_skempi_clustered_all_aug

# The base arm's 4-col split. WITHOUT this line `ARMS=base` inherits ES_SPLIT_DIR from the
# unaugmented parent and trains the PLAIN base into the _aug results dir, exiting 0 — a wrong
# answer that reports success, not a failure. Built by dropping the FoldX scalar column from
# splits_cath_foldx_aug (`cut -f1-4`), which is exact: the unaugmented 4-col split is byte-identical
# to its 5-col FoldX sibling minus that column, on train, val and test across every fold.
export ES_SPLIT_DIR=scratch/splits_skempi_full_clustered_all_aug
