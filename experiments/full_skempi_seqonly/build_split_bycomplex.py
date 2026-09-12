#!/usr/bin/env python3
"""By-complex CV split for the full-SKEMPI SINGLE-point set.

The by-complex analog of build_split_clustered.py (which does the homology-clustered
single-point split). Reuses experiments/retrain_split/build_splits_bycomplex.py verbatim,
pointed at the full-SKEMPI single-point inputs with a distinct output dir + basename, so the
existing clustered single-point split (scratch/splits_skempi_full_clustered_id60_kfold,
consumed by config_skempi_full.sh) is never touched.

    data = scratch/skempi_full/single_point.tsv   (build_skempi_full.py)

Complex id = col0.split("_")[0] = the dot-joined partner grouping (e.g. 1AHW.AB.C) — the same
authoritative key build_split_multipoint.py's by-complex builder uses, so a complex never
straddles train/test. Basename `skempi_sp` (matches config_skempi_full.sh ES_BASENAME), 3 folds,
seed 42 — mirrors splits_skempi_full_mp_bycomplex_seed42 for the multi-point set.

Then the +FoldX fold tsvs are produced by merge_foldx_full_skempi.py --src <this out dir>
--outdir scratch/foldx_skempi_full/bycomplex (a DISTINCT outdir so the clustered-SP foldx
splits are not clobbered).

    python experiments/full_skempi_seqonly/build_split_bycomplex.py
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments/retrain_split"))
import build_splits_bycomplex as B  # noqa: E402  (read_rows/group_by_complex/build)

DATA = os.path.join(ROOT, "scratch/skempi_full/single_point.tsv")
B.BASENAME = "skempi_sp"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default=os.path.join(ROOT, "scratch/splits_skempi_full_bycomplex_seed42"))
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    B.build(a.data, a.out, a.folds, a.val_frac, a.seed)


if __name__ == "__main__":
    main()
