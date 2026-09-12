# Full-SKEMPI COMBINED single+multi, sequence-only, BY-COMPLEX honest split.
# The frontier-matched analog of config_skempi_full_bycomplex.sh (single-point only): RDE-Network /
# DiffAffinity / Prompt-DDG / BA-DDG all evaluate on the COMBINED single+multi SKEMPI v2 set with a
# whole-PDB (by-complex) hold-out, so a like-for-like per-structure-Spearman comparison needs one
# model trained on single+multi together — NOT the SP-only and MP-only models pooled (two separately
# trained models -> a Frankenstein ranking). This mirrors how the CATH tier (config_skempi_full_cath.sh,
# ES_BASENAME=skempi_all) already trains one combined model, from which CATH-single/CATH-multiple are
# free re-scores.
#
# DATA: scratch/skempi_full/all_point.tsv = single_point.tsv (4165) + multi_point.tsv (1636) = 5801 rows
#       / 337 complexes. Split built fresh (build_splits_bycomplex.py, seed42, 3 folds, val-frac 0.15)
#       so the greedy by-complex bin-packing is over the COMBINED per-complex sizes -> ~1934 muts &
#       ~40 complexes>=10muts per test fold. Integrity verified: each complex is a test complex in
#       exactly one fold; train/val/test complex-disjoint within every fold.
#         .venv/bin/python -c "import sys;sys.path.insert(0,'experiments/retrain_split');\
#           import build_splits_bycomplex as B;B.BASENAME='skempi_all';\
#           B.build('scratch/skempi_full/all_point.tsv','scratch/splits_skempi_full_bycomplex_all_seed42',3,0.15,42)"
#
# Embeddings are per-WT-sequence and already generated (the SP and MP by-complex runs both read the
# same scratch/embeddings_skempi_full/<tag> caches, which cover the union) — so this is a TRAINING
# round only, no re-embedding. WT fasta = wt_sequences_cath_all.fasta (674 headers / 337 complexes,
# verified to cover every single+multi complex).
#
# Usage (base arm; Mac-resident PLMs — ankh v1 Large first, then fastest-first):
#   ES_CONFIG=experiments/full_skempi_seqonly/config_skempi_full_bycomplex_all.sh \
#       ARMS=base MAXPAR=1 bash experiments/retrain_split/run_bycomplex.sh ankh
#   ...esmc600m prostt5 saprot esm2
# GPU-box PLMs (embeddings live there): same command with esmc6b / saprot13b / ankh3_large / ankh3_xl
# (set MULAN_ROOT / MULAN_VENV_BIN for the Linux layout — see docs/history/runbooks/RUN_BYCOMPLEX_ALL_LINUX.md).
source experiments/embedding_sweep/config.sh          # 300ep/p30/bs32/lr5e-4 + lightatt_default
export ES_MODELS_TSV=experiments/full_skempi_seqonly/models_skempi_full.tsv   # isolated emb dirs
export ES_NUM_FOLDS=3
export ES_BASENAME=skempi_all
export ES_WT_FASTA=scratch/skempi_full/wt_sequences_cath_all.fasta
export ES_SPLIT_DIR=scratch/splits_skempi_full_bycomplex_all_seed42           # base 4-col
export ES_RESULTS_DIR=scratch/results/full_skempi_bycomplex_all
export ES_METRICS_DIR=experiments/full_skempi_seqonly/results_bycomplex_all

# --- FoldX arms (combined single+multi). Built on the GPU box with merge_foldx_cath.py (the mixed
#     single+multi merger — NOT merge_foldx_full_skempi.py, which is single-point only and would zero
#     the 1636 multi rows): --src $ES_SPLIT_DIR --outdir scratch/foldx_skempi_full/bycomplex_all,
#     dispatching per row (comma in col2 -> multi) with ONE joint train-only z-standardization.
#     Pulled in with the 2026-07-25 handoff, then rebuilt twice. Coverage history — do NOT cite
#     the middle figure: ~48% (single-point-only merge) -> 90.8% (the post-6319caf DUAL-KEY merge,
#     since RETRACTED: the second key landed on other real rows and handed them a different mutation's
#     ddG) -> **99.1% measured 2026-07-30 (5746/5801 in fold_0)** on the corrected single-key merger.
#     Uncovered rows -> standardized 0 = neutral. Only consumed when ARMS includes foldx / foldx_scalar. ---
export ES_FOLDXSCALAR_DIR=scratch/foldx_skempi_full/bycomplex_all/splits_cath_foldx     # 5-col scalar
export ES_FOLDXDEC_DIR=scratch/foldx_skempi_full/bycomplex_all/splits_cath_foldxdec      # 16-col (12 terms)
export ES_SCALAR_CONFIG=models/config/lightatt_addscores_config.json                     # foldx-scalar head
export ES_MLP_CONFIG=models/config/lightatt_addscores_mlp_config.json                    # foldx-MLP head
