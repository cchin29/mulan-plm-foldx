#!/usr/bin/env python3
"""Merge FoldX binding ddG into the S1102 CV splits as the add_scores channel.

Stage 1 (this script): appends the *total* FoldX binding ddG as a 5th column
(zero-shot score) to each fold's train/val/test tsv. The score is standardized
per fold using train-only mean/std (leakage-free) and clipped to +/-4; mutations
with no FoldX value get 0 (the standardized mean). Train on the output with
``models/config/lightatt_addscores_config.json`` and ``--add_zs_scores True``.

Reads one JSON per complex in skempi-foldx's results format -- ``{"muts":
{"<mutation>": {"Interaction Energy": ..., <11 more terms>}}, "meta": ...}`` --
which is what ``data/foldx/results_sp/`` ships. Writes
<out-dir>/splits_<tag>_foldx/fold_i/ and prints per-fold coverage.

Usage: python experiments/foldx_s1102/merge_foldx.py
           [--src <split_dir>] [--results <json_dir>] [--out <tag>] [--out-dir <dir>]

Defaults: --src experiments/embedding_sweep/splits/paper_seed42 (the S1102 paper
split), --results data/foldx/results_sp, --out ankh, --out-dir . (the current
directory). Any fold directory laid out as fold_i/<base>_{train,val,test}.tsv
works; the basename is detected from *_train.tsv, so the S1131/S4169/S2003
benchmark splits merge the same way.

The S1102 runs in docs/RESULTS.md section 18 were built with this script from an
earlier, S1102-only FoldX campaign whose energies are not the ones in
data/foldx/ (a different mutation list per complex gives different values --
see docs/FOLDX.md on determinism). Against the shipped store this script
produces a valid channel, not that exact one.
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(ROOT, "data", "foldx", "results_sp")
SRC_DEFAULT = os.path.join(ROOT, "experiments", "embedding_sweep", "splits", "paper_seed42")
TERM = "Interaction Energy"
CLIP = 4.0


def load_foldx(results_dir):
    """(pdb, role_mut) -> total binding ddG."""
    d = {}
    for f in glob.glob(os.path.join(results_dir, "*.json")):
        pdb = os.path.splitext(os.path.basename(f))[0]
        r = json.load(open(f))
        for role_mut, terms in r.get("muts", {}).items():
            if TERM in terms:
                d[(pdb, role_mut)] = terms[TERM]
    return d


def read_tsv(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                rows.append(p)
    return rows


def foldx_for(row, fx):
    pdb = row[0].split("_")[0]
    return fx.get((pdb, row[2]))


def detect_base(fold_dir):
    """Split-file prefix (e.g. 'S1102_filtered' or 'S1131') from *_train.tsv."""
    cands = sorted(glob.glob(os.path.join(fold_dir, "*_train.tsv")))
    if not cands:
        raise SystemExit(f"no *_train.tsv in {fold_dir}")
    return os.path.basename(cands[0])[: -len("_train.tsv")]


def main():
    args = sys.argv[1:]
    src = SRC_DEFAULT
    tag = "ankh"
    results_dir = RESULTS
    if "--src" in args:
        src = args[args.index("--src") + 1]
    if "--out" in args:
        tag = args[args.index("--out") + 1]
    if "--results" in args:
        results_dir = args[args.index("--results") + 1]
    out_dir = os.getcwd()
    if "--out-dir" in args:
        out_dir = args[args.index("--out-dir") + 1]
    out_root = os.path.join(out_dir, f"splits_{tag}_foldx")

    fx = load_foldx(results_dir)
    print(f"[merge] FoldX values for {len(fx)} (pdb,mut) pairs  (from {results_dir})")

    n_fold = len([d for d in os.listdir(src) if d.startswith("fold_")])
    tot_rows = tot_cov = 0
    for i in range(n_fold):
        fd = os.path.join(src, f"fold_{i}")
        base_name = detect_base(fd)
        train = read_tsv(os.path.join(fd, f"{base_name}_train.tsv"))
        # fit standardizer on train FoldX values
        tv = [foldx_for(r, fx) for r in train]
        present = [v for v in tv if v is not None]
        mean = sum(present) / len(present)
        var = sum((v - mean) ** 2 for v in present) / max(1, len(present) - 1)
        std = var ** 0.5 or 1.0

        def z(v):
            if v is None:
                return 0.0
            zz = (v - mean) / std
            return max(-CLIP, min(CLIP, zz))

        od = os.path.join(out_root, f"fold_{i}")
        os.makedirs(od, exist_ok=True)
        cov_fold = rows_fold = 0
        for split in ("train", "val", "test"):
            rows = read_tsv(os.path.join(fd, f"{base_name}_{split}.tsv"))
            with open(os.path.join(od, f"{base_name}_{split}.tsv"), "w") as fh:
                for r in rows:
                    v = foldx_for(r, fx)
                    cov_fold += v is not None
                    rows_fold += 1
                    # keep first 4 cols (s1,s2,mut,label), append standardized zs
                    base = r[:4] if len(r) >= 4 else r
                    fh.write("\t".join(base + [f"{z(v):.5f}"]) + "\n")
        tot_rows += rows_fold
        tot_cov += cov_fold
        print(f"[merge] fold {i}: coverage {cov_fold}/{rows_fold} "
              f"({100*cov_fold/rows_fold:.1f}%)  mean={mean:.3f} std={std:.3f}")
    print(f"[merge] TOTAL coverage {tot_cov}/{tot_rows} "
          f"({100*tot_cov/tot_rows:.1f}%)  ->  {out_root}")


if __name__ == "__main__":
    main()
