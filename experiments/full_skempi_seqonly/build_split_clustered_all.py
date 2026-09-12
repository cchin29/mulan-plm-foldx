#!/usr/bin/env python3
"""Homology-clustered (mmseqs <=60% id) split for the full-SKEMPI COMBINED single+multi set.

The SP+MP analog of build_split_clustered.py (which is single-point only). Same clustering pipeline
(UF single-linkage of complexes sharing a chain-cluster -> family-GroupKFold), pointed at the
combined table so the greedy family->fold packing is over the COMBINED per-complex sizes — the
clustered counterpart of splits_skempi_full_bycomplex_all_seed42 and the CATH `skempi_all` tier.

    data  = scratch/skempi_full/all_point.tsv            (single_point + multi_point, 5801 rows)
    fasta = scratch/skempi_full/wt_sequences_cath_all.fasta   (covers every single+multi complex)

--emit kfold writes scratch/splits_skempi_full_clustered_all_id{pct}_kfold/fold_{0,1,2}/
skempi_all_{train,val,test}.tsv.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments/retrain_split"))
import cluster_split as cs   # noqa: E402

cs.DATA = os.path.join(ROOT, "scratch/skempi_full/all_point.tsv")
cs.FASTA = os.path.join(ROOT, "scratch/skempi_full/wt_sequences_cath_all.fasta")
cs.B.BASENAME = "skempi_all"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-seq-id", type=float, default=0.6)
    ap.add_argument("--cov", type=float, default=0.8)
    ap.add_argument("--emit", choices=["kfold", "lofo"], default=None)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    pct = int(round(a.min_seq_id * 100))
    workdir = os.path.join(ROOT, f"scratch/clust_mmseqs_skempi_full_all_id{pct}")
    rows = cs.B.read_rows(cs.DATA)
    groups = cs.B.group_by_complex(rows)
    chain2rep = cs.run_mmseqs(a.min_seq_id, a.cov, workdir)
    print(f"[mmseqs] {len(chain2rep)} partner-labels -> {len(set(chain2rep.values()))} clusters "
          f"@ id>={a.min_seq_id} cov>={a.cov}")
    fam = cs.build_families(groups, chain2rep)
    cs.report(groups, fam)
    if a.emit == "kfold":
        out = os.path.join(ROOT, f"scratch/splits_skempi_full_clustered_all_id{pct}_kfold")
        cs.emit_kfold(groups, fam, out, a.folds, a.val_frac, a.seed)
        print(f"[emit] kfold -> {out}")


if __name__ == "__main__":
    main()
