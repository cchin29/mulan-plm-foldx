#!/usr/bin/env bash
# A2 §9 — MINT (partner-context) + FoldX add_scores, BALANCED splitter, S1102.
#
# Layers the orthogonal FoldX channel onto the MINT-complex arm. FoldX enters at the HEAD via
# add_scores (outside the pooled enc(1)*enc(2) product where MINT's cross-chain term washed out,
# §5), so it reaches the linear head directly. Two FoldX flavours + optional C1 tail-loss:
#
#   scalar     FoldX single ddG scalar   (splits_balanced_foldx    + addscores_config)
#   mlp        FoldX 12-term decomposed  (splits_balanced_foldxdec + addscores_mlp_config)
#   scalar_c1  scalar    + C1 tail-weighted loss (C1_EXTRA)
#   mlp_c1     mlp       + C1 tail-weighted loss (C1_EXTRA)
#
# Everything else is held IDENTICAL to the MINT-complex arm (mint row-keyed cache via --mint_pair,
# balanced folds, 300ep/patience30, config_balanced budget) so PCC is directly comparable to the
# 0.856-pooled MINT-complex baseline and to the FoldX arms on the other PLMs. FoldX is
# model-INDEPENDENT physics; the pre-built balanced FoldX splits carry the ddG columns (cols 4+)
# while cols 0-2 (the complex + mutation key) are untouched, so --mint_pair still resolves the
# per-row bundles exactly as in config_mint.sh. Results are ISOLATED under scratch/results/mint_*
# so nothing in embedding_sweep_balanced / mint_a2 is touched. Resumable, MPS.
#
# Prereqs: MINT cache (experiments/mint/gen_mint_emb.py) and the balanced FoldX splits
# (scratch/foldx_s1102/merge_foldx{,_decomposed}.py --out balanced) must both exist.
#
#   Usage:  bash experiments/mint/run_mint_foldx.sh <arm>
#           MAXPAR=2 bash experiments/mint/run_mint_foldx.sh scalar
#           MAXPAR=2 bash experiments/mint/run_mint_foldx.sh mlp
#     C1:   C1_EXTRA="--tail_loss_mode linear --tail_loss_alpha 2" \
#               bash experiments/mint/run_mint_foldx.sh scalar_c1   # after the c1_ankh winner is known
set -uo pipefail
# Repo root resolved from this script's own location (experiments/<sub>/); override with MULAN_ROOT.
cd "${MULAN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ARM=${1:?usage: $0 <scalar|mlp|scalar_c1|mlp_c1>}
source experiments/embedding_sweep/config_balanced.sh   # budget + WT fasta + basename + folds
PY=./.venv/bin
MAXPAR=${MAXPAR:-2}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_ENABLE_MPS_FALLBACK=1
log(){ echo "[$(date +%H:%M:%S)] $*"; }

EMB=scratch/embeddings_mint
PLM=mint
SCALAR_SPLITS=scratch/foldx_s1102/splits_balanced_foldx
DEC_SPLITS=scratch/foldx_s1102/splits_balanced_foldxdec
SCALAR_CFG=models/config/lightatt_addscores_config.json
MLP_CFG=models/config/lightatt_addscores_mlp_config.json

# arm -> (splits, cfg, results dir, tail-loss extra). C1_EXTRA is only consumed by the *_c1 arms.
case "$ARM" in
  scalar)     SPLITS=$SCALAR_SPLITS; CFG=$SCALAR_CFG; OUT=scratch/results/mint_foldxscalar;    TAIL="" ;;
  mlp)        SPLITS=$DEC_SPLITS;    CFG=$MLP_CFG;    OUT=scratch/results/mint_foldxmlp;       TAIL="" ;;
  scalar_c1)  SPLITS=$SCALAR_SPLITS; CFG=$SCALAR_CFG; OUT=scratch/results/mint_foldxscalar_c1; TAIL=${C1_EXTRA:?scalar_c1 needs C1_EXTRA=\"--tail_loss_mode ...\"} ;;
  mlp_c1)     SPLITS=$DEC_SPLITS;    CFG=$MLP_CFG;    OUT=scratch/results/mint_foldxmlp_c1;    TAIL=${C1_EXTRA:?mlp_c1 needs C1_EXTRA=\"--tail_loss_mode ...\"} ;;
  *) log "!! unknown arm '$ARM' (want scalar|mlp|scalar_c1|mlp_c1)"; exit 1 ;;
esac

[ -d "$EMB/mut" ] || { log "!! MINT cache missing ($EMB) -> experiments/mint/gen_mint_emb.py"; exit 1; }
[ -f "$SPLITS/fold_$((ES_NUM_FOLDS-1))/${ES_BASENAME}_test.tsv" ] || {
  log "!! FoldX splits missing ($SPLITS) -> scratch/foldx_s1102/merge_foldx*.py --out balanced"; exit 1; }

log "=== MINT+FoldX  arm=$ARM  cfg=$(basename "$CFG")  splits=$(basename "$SPLITS")  ${ES_NUM_EPOCHS}ep/p${ES_PATIENCE} MAXPAR=$MAXPAR ${TAIL:+tail=[$TAIL]} ==="
for i in $(seq 0 $((ES_NUM_FOLDS-1))); do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXPAR" ]; do wait -n; done
  ( R=$OUT/fold_${i}; F=$SPLITS/fold_${i}; mkdir -p "$R"
    if [ -f "$R/training_run/all_results.json" ]; then log "  skip $ARM fold $i (done)"; exit 0; fi
    $PY/mulan-train --train_data "$F/${ES_BASENAME}_train.tsv" \
      --eval_data "$F/${ES_BASENAME}_val.tsv" --test_data "$F/${ES_BASENAME}_test.tsv" \
      --train_fasta_file "$ES_WT_FASTA" --test_fasta_file "$ES_WT_FASTA" \
      --embeddings_dir "$EMB" --plm_model_name "$PLM" --mint_pair True \
      --model_name_or_config_path "$CFG" --save_model True --add_zs_scores True $TAIL \
      --num_epochs "$ES_NUM_EPOCHS" --early_stopping_patience "$ES_PATIENCE" \
      --batch_size "$ES_BATCH" --learning_rate "$ES_LR" --report_to none --disable_tqdm True \
      --output_dir "$R/training_run" > "$R/training_run.log" 2>&1
    pcc=$(grep -o '"test_pcc":[0-9.-]*' "$R/training_run/all_results.json" 2>/dev/null)
    log "  DONE $ARM fold $i -> $pcc" ) &
done
wait
log "=== MINT+FoldX arm=$ARM complete -> $OUT ==="
