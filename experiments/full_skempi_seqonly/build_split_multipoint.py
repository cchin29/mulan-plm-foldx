#!/usr/bin/env python3
"""Multi-point CV splits for full-SKEMPI: homology-clustered (mmseqs <=60% id) AND by-complex.

The multi-point analog of build_split_clustered.py (single-point clustered) and
experiments/retrain_split/build_splits_bycomplex.py (by-complex). Reuses BOTH pipelines
verbatim, pointed at the multi-point inputs and DISTINCT output/workdir paths so the
single-point clustered split (scratch/splits_skempi_full_clustered_id60_kfold, consumed by
the live sweep via config_skempi_full.sh ES_SPLIT_DIR) is never touched.

    data  = scratch/skempi_full/multi_point.tsv               (build_multipoint.py)
    fasta = scratch/skempi_full/wt_sequences_multipoint.fasta  (partner-group labels = tsv cols)

Complex id = the dot-joined #Pdb grouping (col0.split('_')[0], e.g. 2C5D.AB.CD) — the same
authoritative key build_multipoint.py / merge_foldx_multipoint.py use (PLAN §A.2/A.4), so a
complex never straddles train/test in either split. Basename `skempi_mp`.

Outputs (both gitignored under scratch/):
    scratch/splits_skempi_full_mp_clustered_id60_kfold/fold_{0,1,2}/skempi_mp_{train,val,test}.tsv
    scratch/splits_skempi_full_mp_bycomplex_seed42/fold_{0,1,2}/skempi_mp_{train,val,test}.tsv

Then the +FoldX fold tsvs are produced by merge_foldx_multipoint.py --split-dir <one of the
above> (train-only standardized, honours the audit quarantine).

    .venv/bin/python experiments/full_skempi_seqonly/build_split_multipoint.py            # both
    .venv/bin/python experiments/full_skempi_seqonly/build_split_multipoint.py --only clustered
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments/retrain_split"))
import cluster_split as cs          # noqa: E402  (run_mmseqs/build_families/report/emit_kfold)
import build_splits_bycomplex as B  # noqa: E402  (read_rows/group_by_complex/build)

DATA = os.path.join(ROOT, "scratch/skempi_full/multi_point.tsv")
FASTA = os.path.join(ROOT, "scratch/skempi_full/wt_sequences_multipoint.fasta")

# Re-point the shared pipeline at the multi-point inputs + a distinct basename. cluster_split
# emits via B.write_split(f"{B.BASENAME}_..."), so setting B.BASENAME renames both builders'
# fold files in lockstep. cs.DATA/cs.FASTA feed run_mmseqs' fasta + the group table.
cs.DATA = DATA
cs.FASTA = FASTA
B.BASENAME = "skempi_mp"


def build_clustered(min_seq_id, cov, folds, val_frac, seed):
    pct = int(round(min_seq_id * 100))
    workdir = os.path.join(ROOT, f"scratch/clust_mmseqs_skempi_full_mp_id{pct}")
    rows = B.read_rows(DATA)
    groups = B.group_by_complex(rows)
    chain2rep = cs.run_mmseqs(min_seq_id, cov, workdir)
    nclust = len(set(chain2rep.values()))
    print(f"[mmseqs] {len(chain2rep)} partner-labels -> {nclust} clusters "
          f"@ id>={min_seq_id} cov>={cov}")
    fam = cs.build_families(groups, chain2rep)
    cs.report(groups, fam)
    out = os.path.join(ROOT, f"scratch/splits_skempi_full_mp_clustered_id{pct}_kfold")
    cs.emit_kfold(groups, fam, out, folds, val_frac, seed)
    print(f"[clustered] -> {out}")


def build_bycomplex(folds, val_frac, seed):
    out = os.path.join(ROOT, "scratch/splits_skempi_full_mp_bycomplex_seed42")
    B.build(DATA, out, folds, val_frac, seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["clustered", "bycomplex"], default=None,
                    help="build just one (default: both)")
    ap.add_argument("--min-seq-id", type=float, default=0.6)
    ap.add_argument("--cov", type=float, default=0.8)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    if a.only != "bycomplex":
        build_clustered(a.min_seq_id, a.cov, a.folds, a.val_frac, a.seed)
    if a.only != "clustered":
        build_bycomplex(a.folds, a.val_frac, a.seed)


if __name__ == "__main__":
    main()
