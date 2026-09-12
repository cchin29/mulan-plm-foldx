#!/bin/bash
# Scaffold driver: full-SKEMPI attention -> interface AUROC head-to-head, the leakage-
# controlled scale-up of the S1102 analysis (attention→interface question). Uses the by-complex retrain
# heads (evaluated on held-out complexes -> honest generalization) over the 102 full-SKEMPI
# interface masks built by build_masks_skempi_full.py.
#
# Prereq: ./.venv/bin/python experiments/attention_interface/build_masks_skempi_full.py
# Then:   bash experiments/attention_interface/run_skempi_full.sh   (queue it; MPS-bound)
# Aggregate the s1102_skempi_full_*.csv afterward the same way plot_s1102_interface.py does.
#
# SPLIT selects the head family: bycomplex (default, leakage-controlled) or clustered.
# FOLD selects the encoder fold (attention is a weakly fold-sensitive encoder property;
# fold-0 mirrors the S1102 choice). esmc6b omitted (head GPU-only, not local).
set -u
# Repo root resolved from this script's own location (experiments/<sub>/); override with MULAN_ROOT.
cd "${MULAN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PY=.venv/bin/python
SPLIT="${SPLIT:-bycomplex}"; FOLD="${FOLD:-0}"
MASKS=scratch/interface_masks_skempi_full
FASTA=scratch/skempi_full/wt_sequences.fasta
CKPT() { echo "scratch/results/retrain_${SPLIT}/$1_base/fold_${FOLD}/training_run/model.ckpt"; }
S=experiments/attention_interface/score_s1102.py

run() {  # $1=tag  $2=(optional) emb-dir for cached structure/SDK PLMs
  local tag="$1" emb="${2:-}"
  local ck; ck=$(CKPT "$tag")
  [ -f "$ck" ] || { echo "SKIP $tag (no head: $ck)"; return; }
  echo "########## $tag ($SPLIT fold $FOLD) ##########"
  if [ -n "$emb" ]; then
    $PY $S --tag "$tag" --suffix _skempi_full --ckpt "$ck" --mask-dir $MASKS --wt-fasta $FASTA --emb-dir "$emb"
  else
    $PY $S --tag "$tag" --suffix _skempi_full --ckpt "$ck" --mask-dir $MASKS --wt-fasta $FASTA
  fi 2>&1 | grep -E "macro AUROC|missing-emb|chains scored"
}

# live-embed (pure-AA): reproduces the head's training embeddings
run ankh
run esm2
run prostt5
run ankh3_large
run ankh3_xl
# cached full-SKEMPI embeddings (structure/SDK PLMs) — keys match the WT fasta (PDB.g1.g2_role)
run saprot     scratch/embeddings_skempi_full/saprot
run esmc600m   scratch/embeddings_skempi_full/esmc600m
# TODO: esm3 / saprot13b need their full-SKEMPI WT caches verified/generated before enabling.
echo "ALL DONE"
