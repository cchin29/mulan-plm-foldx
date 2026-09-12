# Full-SKEMPI CATH-superfamily hold-out (USP-ddG / CATH-ddG's EXACT 813-mut test set) config.
#
# THE frontier-comparable split: USP-ddG ships its CATH-superfamily partition as a static
# `cath_fold` column (train/val) in data/SKEMPI2/skempi_v2.csv. build_split_cath.py joins it to
# our full-SKEMPI rows BY COMPLEX (cath_fold is constant per complex — verified 0 splits), giving
# the literal 53 held-out test superfamilies. Single fold (num_folds=1), train on all cath train
# complexes, test on the held-out superfamilies — matches USP-ddG Table 1 exactly.
#
# TEST: 687 rows (421 single / 266 multiple) over 50/53 val complexes (3 not embedded:
# 1S0W,1XXM,2NOJ). TRAIN: 5091 rows -> 4213 train / 878 early-stop val (10% of complexes, seed 2024).
#
# Splits built by:  .venv/bin/python experiments/full_skempi_seqonly/build_split_cath.py
# Report + single/multiple test subsets: scratch/splits_skempi_full_cath_kfold/{JOIN_REPORT.txt,test_*.tsv}
#
# Usage (base arm; Mac-resident PLMs):
#   ES_CONFIG=experiments/full_skempi_seqonly/config_skempi_full_cath.sh \
#       ARMS=base MAXPAR=1 bash experiments/retrain_split/run_bycomplex.sh ankh esm2 saprot prostt5
# GPU-box PLMs (embeddings only there): same command with esmc6b / esmc600m / ankh3_large / ankh3_xl
source experiments/full_skempi_seqonly/config_skempi_full.sh   # budget + heads + models.tsv

export ES_NUM_FOLDS=1
export ES_BASENAME=skempi_all
export ES_WT_FASTA=scratch/skempi_full/wt_sequences_cath_all.fasta
export ES_SPLIT_DIR=scratch/splits_skempi_full_cath_kfold                       # base 4-col
export ES_RESULTS_DIR=scratch/results/full_skempi_cath
export ES_METRICS_DIR=experiments/full_skempi_seqonly/results_cath

# --- FoldX arms: CATH FoldX splits already built by merge_foldx_cath.py (mixed single+multi, one
#     shared train-fit z-standardization). Coverage **99.1% measured 2026-08-01** (5746/5801 per
#     fold). The "single 28.1% / multi 98.7% / combined 48.1%" triple this header used to state was
#     the pre-fix __cov_pre lineage, corrected here on 2026-08-07 for the same reason as
#     config_skempi_full.sh. Include foldx/foldx_scalar in ARMS to use them; base arm needs none
#     of this. ---
export ES_FOLDXSCALAR_DIR=scratch/foldx_skempi_full/splits_cath_foldx           # 5-col scalar
export ES_FOLDXDEC_DIR=scratch/foldx_skempi_full/splits_cath_foldxdec           # 16-col 12-term
