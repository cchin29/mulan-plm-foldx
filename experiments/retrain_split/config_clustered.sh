# Phase 2 (docs/history/PLAN_RETRAIN_BYCOMPLEX.md §7): homology-clustered split config.
#
# The honest, frontier-comparable generalization split: whole SEQUENCE FAMILIES quarantined
# per fold (mmseqs easy-cluster <=60% identity on the 220 interface chains -> single-linkage
# 110 complexes into 35 families -> family-GroupKFold(3)). Same dataset / PLMs / embedding
# caches / 300ep-p30 budget / 3 arms as the by-complex run — ONLY the grouping changes
# (complex -> family), so leaky -> by-complex -> clustered is an apples-to-apples ladder on the
# same per-structure complexes. NOTE the mega-family imbalance: fold_0 test = the protease-
# inhibitor family (593 muts / 37 complexes, 54%), folds 1/2 ~254 each — report per-fold.
#
# Splits built by: experiments/retrain_split/cluster_split.py --min-seq-id 0.6 --emit kfold
# FoldX variants by: merge_foldx_decomposed.py / merge_foldx.py --src <kfold dir> --out clustered_id60
#
# Usage:  ES_CONFIG=experiments/retrain_split/config_clustered.sh \
#             MAXPAR=2 bash experiments/retrain_split/run_bycomplex.sh ankh
source experiments/embedding_sweep/config.sh
export ES_SPLIT_METHOD=clustered_id60
export ES_NUM_FOLDS=3
export ES_SPLIT_DIR=scratch/splits_clustered_id60_kfold                       # base 4-col
export ES_FOLDXDEC_DIR=scratch/foldx_s1102/splits_clustered_id60_foldxdec     # 16-col (12 FoldX terms, phase2 MLP)
export ES_FOLDXSCALAR_DIR=scratch/foldx_s1102/splits_clustered_id60_foldx     # 5-col (single Interaction Energy, phase1 scalar)
export ES_RESULTS_DIR=scratch/results/retrain_clustered
export ES_METRICS_DIR=experiments/retrain_split/results_clustered
export ES_MLP_CONFIG=models/config/lightatt_addscores_mlp_config.json         # foldx-MLP head (phase2)
export ES_SCALAR_CONFIG=models/config/lightatt_addscores_config.json          # foldx-scalar head (phase1)
