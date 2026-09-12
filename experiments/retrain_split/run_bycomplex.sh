#!/usr/bin/env bash
# Leakage-controlled by-complex retrain (docs/history/PLAN_RETRAIN_BYCOMPLEX.md §3-4). For each PLM tag
# it trains TWO arms on the 3 by-complex folds, reusing each model's EXISTING embedding
# cache read-only (no PLM load, no embedding regen — only the split membership changed):
#
#   base          -> lightatt_default (add_scores=false)              splits = ES_SPLIT_DIR (4-col)
#   foldx         -> lightatt_addscores_mlp + --add_zs_scores (12 terms, phase2)  splits = ES_FOLDXDEC_DIR (16-col)
#   foldx_scalar  -> lightatt_addscores + --add_zs_scores (1 Interaction-Energy scalar, phase1)  splits = ES_FOLDXSCALAR_DIR (5-col)
#
# Output: $ES_RESULTS_DIR/<tag>_{base,foldx}/fold_{0,1,2}/training_run/. Resumable (skips a
# fold whose all_results.json exists). Runs on the BASE mulan-train (no interface_xattn), so
# it is compatible with the current committed code. Uses ankh = Ankh v1 Large (paper
# reference), NOT ankh3 — the tag->plm/emb mapping comes from models.tsv.
#
# Usage:  MAXPAR=2 bash experiments/retrain_split/run_bycomplex.sh ankh [<tag> ...]
#         ARMS="base foldx" (default both) to run a subset.
set -u
# Repo root + venv are env-overridable so this runs on the GPU box too (default = Mac layout).
cd "${MULAN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "${ES_CONFIG:-experiments/retrain_split/config_bycomplex.sh}"
PY="${MULAN_VENV_BIN:-./.venv/bin}"
MAXPAR=${MAXPAR:-2}
ARMS=${ARMS:-"base foldx foldx_scalar"}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_ENABLE_MPS_FALLBACK=1
log(){ echo "[$(date +%H:%M:%S)] $*"; }

# tag -> (plm_model_name, embeddings_dir) from the shared registry (skip comments/blank).
lookup(){ awk -F'\t' -v t="$1" '$1==t{print $2"\t"$3; found=1} END{exit !found}' "$ES_MODELS_TSV"; }

train_fold(){  # $1=arm $2=splits_dir $3=cfg $4=extra_flags $5=plm $6=emb $7=results $8=fold
  local arm=$1 sp=$2 cfg=$3 extra=$4 plm=$5 emb=$6 res=$7 i=$8
  local R=$res/fold_${i} F=$sp/fold_${i}
  if [ -f "$R/training_run/all_results.json" ]; then log "  skip $arm fold $i (done)"; return 0; fi
  mkdir -p "$R"
  $PY/mulan-train --train_data "$F/${ES_BASENAME}_train.tsv" \
    --eval_data "$F/${ES_BASENAME}_val.tsv" --test_data "$F/${ES_BASENAME}_test.tsv" \
    --train_fasta_file "$ES_WT_FASTA" --test_fasta_file "$ES_WT_FASTA" \
    --embeddings_dir "$emb" --plm_model_name "$plm" \
    --model_name_or_config_path "$cfg" --save_model True $extra \
    --num_epochs "$ES_NUM_EPOCHS" --early_stopping_patience "$ES_PATIENCE" \
    --batch_size "$ES_BATCH" --learning_rate "$ES_LR" --report_to none --disable_tqdm True \
    --output_dir "$R/training_run" > "$R/training_run.log" 2>&1
  local rc=$?
  # A fold that crashed used to be logged as DONE with an empty test_pcc, because the log line
  # never checked anything. That is how esmc600m_base in full_skempi_bycomplex_all became three
  # empty directories under a queue task reporting Success: mulan-train died on an SDK-only
  # encoder missing from the legacy map, and the driver said DONE three times in six seconds.
  if [ ! -f "$R/training_run/all_results.json" ]; then
    log "  !! FAILED $arm fold $i (rc=$rc) -- no all_results.json; last log line:"
    log "     $(tail -1 "$R/training_run.log" 2>/dev/null)"
    : > "$R/.FAILED"
    return 1
  fi
  rm -f "$R/.FAILED"
  # The pattern allows the space and the exponent forms: the JSON is written as
  # '"test_pcc": 0.16...', so the old '"test_pcc":[0-9.-]*' matched empty and every healthy fold
  # logged a bare '"test_pcc":' -- indistinguishable at a glance from a crashed one logging nothing.
  log "  DONE $arm fold $i -> $(grep -o '"test_pcc": *[0-9.eE+-]*' "$R/training_run/all_results.json" 2>/dev/null | head -1)"
}

run_arm(){  # $1=tag $2=arm $3=plm $4=emb
  local tag=$1 arm=$2 plm=$3 emb=$4 sp cfg extra res
  case $arm in
    base)         sp=$ES_SPLIT_DIR;         cfg=$ES_MODEL_CONFIG;  extra="" ;;
    foldx)        sp=$ES_FOLDXDEC_DIR;      cfg=$ES_MLP_CONFIG;    extra="--add_zs_scores True" ;;
    foldx_scalar) sp=$ES_FOLDXSCALAR_DIR;   cfg=$ES_SCALAR_CONFIG; extra="--add_zs_scores True" ;;
    *) log "!! unknown arm $arm"; return 1 ;;
  esac
  [ -d "$sp/fold_0" ] || { log "!! splits dir $sp missing"; return 1; }
  res=$ES_RESULTS_DIR/${tag}_${arm}
  log "=== $tag $arm  plm=$plm emb=$emb  splits=$sp  ${ES_NUM_EPOCHS}ep/p${ES_PATIENCE} ==="
  for i in $(seq 0 $((ES_NUM_FOLDS-1))); do
    while [ "$(jobs -rp | wc -l)" -ge "$MAXPAR" ]; do wait -n; done
    train_fold "$arm" "$sp" "$cfg" "$extra" "$plm" "$emb" "$res" "$i" &
  done
  wait
  log "=== $tag $arm complete ==="
}

run_model(){
  local tag=$1 row plm emb
  row=$(lookup "$tag") || { log "!! tag $tag not in $ES_MODELS_TSV"; return 1; }
  plm=$(echo "$row" | cut -f1); emb=$(echo "$row" | cut -f2)
  [ -d "$emb" ] || { log "!! emb dir $emb missing for $tag"; return 1; }
  for arm in $ARMS; do run_arm "$tag" "$arm" "$plm" "$emb"; done
}

[ "$#" -eq 0 ] && { echo "usage: $0 <tag> [<tag> ...]  (ARMS='base foldx')"; exit 1; }
for m in "$@"; do run_model "$m"; done
failed=$(find "$ES_RESULTS_DIR" -name .FAILED 2>/dev/null | wc -l | tr -d ' ')
if [ "$failed" -gt 0 ]; then
  log "!! BY-COMPLEX RETRAIN INCOMPLETE: $failed fold(s) failed -- markers under $ES_RESULTS_DIR"
  find "$ES_RESULTS_DIR" -name .FAILED 2>/dev/null | sed 's|/.FAILED$||;s|^|     |'
  exit 1
fi
log "BY-COMPLEX RETRAIN DONE: $*  (arms: $ARMS)"
