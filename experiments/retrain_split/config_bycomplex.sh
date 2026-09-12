# By-complex (leakage-controlled) variant of the embedding-sweep config.
# See docs/history/PLAN_RETRAIN_BYCOMPLEX.md.
#
# SAME dataset (S1102_filtered, 1100), SAME PLMs / embedding caches, SAME 300ep/patience30
# budget as the balanced series — the ONLY thing that changes is the split membership:
# 3 folds partitioned by WHOLE COMPLEX (train/val/test never share a PDB). The drop vs the
# balanced numbers is the leakage; the FoldX-base Delta under this split is the result.
#
# Isolated split / results / metrics dirs so nothing in embedding_sweep_balanced or the
# paper anchors is touched. scratch/ is gitignored.
#
# Usage:
#   MAXPAR=2 bash experiments/retrain_split/run_bycomplex.sh ankh
source experiments/embedding_sweep/config.sh
export ES_SPLIT_METHOD=bycomplex
export ES_NUM_FOLDS=3
export ES_SPLIT_DIR=scratch/splits_bycomplex_seed42                       # base 4-col splits
export ES_FOLDXDEC_DIR=scratch/foldx_s1102/splits_bycomplex_foldxdec      # 16-col (12 FoldX terms, phase2 MLP)
export ES_FOLDXSCALAR_DIR=scratch/foldx_s1102/splits_bycomplex_foldx      # 5-col (single Interaction Energy, phase1 scalar)
export ES_RESULTS_DIR=scratch/results/retrain_bycomplex
export ES_METRICS_DIR=experiments/retrain_split/results
export ES_MLP_CONFIG=models/config/lightatt_addscores_mlp_config.json     # foldx-MLP head (phase2)
export ES_SCALAR_CONFIG=models/config/lightatt_addscores_config.json      # foldx-scalar head (phase1)
