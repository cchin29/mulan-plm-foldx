#!/usr/bin/env bash
# Paper-faithful embedding sweep on S1102 (Mac/MPS). Trains MuLAN LightAttModel for one or
# more embedding models using the tracked paper-faithful CV setup (see config.sh):
#   splitter=random (rng.integers, seed 42) | budget=train-to-convergence | add_scores=false.
# Heavy outputs (checkpoints/logs/predictions) go to the gitignored ES_RESULTS_DIR; only the
# small splits and aggregated metrics are tracked. Existing runs are never touched.
# Reuses each model's existing embedding cache (same 1100 muts, re-folded).
#
#   bash experiments/embedding_sweep/run_sweep.sh ankh prostt5 esm2 ...
#   MAXPAR=2 bash experiments/embedding_sweep/run_sweep.sh ankh
#   ES_CONFIG=experiments/embedding_sweep/config_balanced.sh bash run_sweep.sh ankh   # balanced set
set -u
# Repo root is two levels up from this script (portable — no hardcoded absolute path).
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../.." || exit 1
# Honor ES_CONFIG like gen_splits.py / aggregate.py do (default: the paper-faithful
# config.sh). Sourcing the wrong config would write into the wrong results/splits dir
# — e.g. hardcoding config.sh here silently polluted the paper anchor when the caller
# meant the balanced set.
source "${ES_CONFIG:-experiments/embedding_sweep/config.sh}"
PY=./.venv/bin
MAXPAR=${MAXPAR:-2}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_ENABLE_MPS_FALLBACK=1
log(){ echo "[$(date +%H:%M:%S)] $*"; }

# tag -> "<plm>\t<emb_dir>" from models.tsv (skip comment/blank lines)
model_spec(){ awk -F'\t' -v t="$1" '$1==t{print $2"\t"$3; found=1} END{exit !found}' \
  <(grep -v '^#' "$ES_MODELS_TSV" | grep -v '^[[:space:]]*$'); }

run_model(){
  local TAG=$1 SPEC PLM EMB
  if ! SPEC=$(model_spec "$TAG"); then log "!! unknown tag '$TAG' (not in $ES_MODELS_TSV)"; return 1; fi
  PLM=$(printf '%s' "$SPEC" | cut -f1); EMB=$(printf '%s' "$SPEC" | cut -f2)
  if [ ! -d "$EMB" ]; then log "!! embeddings dir '$EMB' missing for '$TAG' (skipping)"; return 1; fi
  log "=== MODEL $TAG  plm=$PLM emb=$EMB  epochs=$ES_NUM_EPOCHS patience=$ES_PATIENCE ==="
  for i in $(seq 0 $((ES_NUM_FOLDS-1))); do
    while [ "$(jobs -rp | wc -l)" -ge "$MAXPAR" ]; do wait -n; done
    ( R=$ES_RESULTS_DIR/${TAG}/fold_${i}; F=$ES_SPLIT_DIR/fold_${i}; mkdir -p "$R"
      $PY/mulan-train --train_data "$F/${ES_BASENAME}_train.tsv" \
        --eval_data "$F/${ES_BASENAME}_val.tsv" --test_data "$F/${ES_BASENAME}_test.tsv" \
        --train_fasta_file "$ES_WT_FASTA" --test_fasta_file "$ES_WT_FASTA" \
        --embeddings_dir "$EMB" --plm_model_name "$PLM" \
        --model_name_or_config_path "$ES_MODEL_CONFIG" --save_model True \
        --num_epochs "$ES_NUM_EPOCHS" --early_stopping_patience "$ES_PATIENCE" \
        --batch_size "$ES_BATCH" --learning_rate "$ES_LR" --report_to none --disable_tqdm True \
        ${ES_EXTRA_TRAIN_ARGS:-} \
        --output_dir "$R/training_run" > "$R/training_run.log" 2>&1
      pcc=$(grep -o '"test_pcc":[0-9.]*' "$R/training_run/all_results.json" 2>/dev/null)
      log "  DONE $TAG fold $i -> $pcc" ) &
  done
  wait
  log "=== MODEL $TAG complete ==="
}

[ "$#" -eq 0 ] && { echo "usage: $0 <tag> [<tag> ...]"; echo "tags:"; grep -v '^#' "$ES_MODELS_TSV" | awk '{print "  "$1}'; exit 1; }
for m in "$@"; do run_model "$m"; done
log "SWEEP DONE for: $*   (aggregate: $PY/python experiments/embedding_sweep/aggregate.py)"
