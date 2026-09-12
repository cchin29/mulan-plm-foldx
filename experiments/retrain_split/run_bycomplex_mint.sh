#!/usr/bin/env bash
# MINT base arm on the 3 by-complex folds (Phase-1). Dedicated driver (not run_bycomplex.sh)
# because MINT has no FoldX arms and needs --mint_pair; output is named `mint_base` so
# evaluate.py finds it at scratch/results/retrain_bycomplex/mint_base/fold_{0,1,2}/. Reuses the
# row-keyed MINT bundle cache read-only (no regen). Resumable. Base mulan-train, MPS.
#
# Usage:  MAXPAR=2 bash experiments/retrain_split/run_bycomplex_mint.sh
set -u
# Repo root resolved from this script's own location (experiments/<sub>/); override with MULAN_ROOT.
cd "${MULAN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "${ES_CONFIG:-experiments/retrain_split/config_bycomplex_mint.sh}"
PY=./.venv/bin
MAXPAR=${MAXPAR:-2}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_ENABLE_MPS_FALLBACK=1
log(){ echo "[$(date +%H:%M:%S)] $*"; }

EMB=scratch/embeddings_mint       # row-keyed bundle cache (wt/ + mut/)
PLM=mint
RES=$ES_RESULTS_DIR/mint_base
[ -d "$EMB/mut" ] || { log "!! MINT cache $EMB/mut missing"; exit 1; }
log "=== MINT base (mint_pair)  emb=$EMB  ${ES_NUM_EPOCHS}ep/p${ES_PATIENCE} ==="
for i in $(seq 0 $((ES_NUM_FOLDS-1))); do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXPAR" ]; do wait -n; done
  ( R=$RES/fold_${i}; F=$ES_SPLIT_DIR/fold_${i}
    if [ -f "$R/training_run/all_results.json" ]; then log "  skip mint fold $i (done)"; exit 0; fi
    mkdir -p "$R"
    $PY/mulan-train --train_data "$F/${ES_BASENAME}_train.tsv" \
      --eval_data "$F/${ES_BASENAME}_val.tsv" --test_data "$F/${ES_BASENAME}_test.tsv" \
      --train_fasta_file "$ES_WT_FASTA" --test_fasta_file "$ES_WT_FASTA" \
      --embeddings_dir "$EMB" --plm_model_name "$PLM" \
      --model_name_or_config_path "$ES_MODEL_CONFIG" --save_model True \
      --num_epochs "$ES_NUM_EPOCHS" --early_stopping_patience "$ES_PATIENCE" \
      --batch_size "$ES_BATCH" --learning_rate "$ES_LR" --report_to none --disable_tqdm True \
      ${ES_EXTRA_TRAIN_ARGS:-} \
      --output_dir "$R/training_run" > "$R/training_run.log" 2>&1
    log "  DONE mint fold $i -> $(grep -o '\"test_pcc\":[0-9.-]*' "$R/training_run/all_results.json" 2>/dev/null)" ) &
done
wait
log "=== MINT base complete ==="
