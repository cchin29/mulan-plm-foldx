#!/usr/bin/env python3
"""Homology-clustered (mmseqs <=60% id) split for the full-SKEMPI single-point set.

Thin wrapper that reuses the retrain-split clustering pipeline verbatim (UF single-linkage of
complexes that share a chain-cluster -> family-GroupKFold, docs/history/PLAN_FULL_SKEMPI.md §3c), pointed at
the full-SKEMPI inputs instead of S1102:

    data  = scratch/skempi_full/single_point.tsv        (built by build_skempi_full.py)
    fasta = scratch/skempi_full/wt_sequences.fasta       (partner-group labels = tsv columns)

Complex id = PDB code (col0.split('_')[0]); clustering unit = the partner group-label (mirrors
retrain, where the partner was a single chain). Default = ASSESS only (no files); --emit kfold
writes fold_{0,1,2}/skempi_sp_{train,val,test}.tsv.

    .venv/bin/python experiments/full_skempi_seqonly/build_split_clustered.py                # assess
    .venv/bin/python experiments/full_skempi_seqonly/build_split_clustered.py --emit kfold
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments/retrain_split"))
import cluster_split as cs   # noqa: E402  (reuse run_mmseqs/build_families/report/emit_kfold)

# Re-point the pipeline at the full-SKEMPI inputs + a distinct output basename.
cs.DATA = os.path.join(ROOT, "scratch/skempi_full/single_point.tsv")
cs.FASTA = os.path.join(ROOT, "scratch/skempi_full/wt_sequences.fasta")
cs.B.BASENAME = "skempi_sp"


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
    workdir = os.path.join(ROOT, f"scratch/clust_mmseqs_skempi_full_id{pct}")
    rows = cs.B.read_rows(cs.DATA)
    groups = cs.B.group_by_complex(rows)
    chain2rep = cs.run_mmseqs(a.min_seq_id, a.cov, workdir)
    nclust = len(set(chain2rep.values()))
    print(f"[mmseqs] {len(chain2rep)} partner-labels -> {nclust} clusters "
          f"@ id>={a.min_seq_id} cov>={a.cov}")
    fam = cs.build_families(groups, chain2rep)
    cs.report(groups, fam)

    if a.emit:
        out = os.path.join(ROOT, f"scratch/splits_skempi_full_clustered_id{pct}_{a.emit}")
        if a.emit == "kfold":
            cs.emit_kfold(groups, fam, out, a.folds, a.val_frac, a.seed)
        else:
            cs.emit_lofo(groups, fam, out, a.val_frac, a.seed)


if __name__ == "__main__":
    main()
