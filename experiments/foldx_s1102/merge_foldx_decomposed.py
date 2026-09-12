#!/usr/bin/env python3
"""Stage 2: merge the 12 decomposed FoldX terms into the S1102 CV splits.

Like merge_foldx.py, but instead of the single Interaction Energy scalar it appends
all 12 decomposed AnalyseComplex terms (mut-WT) as columns 5..16. Each term is
standardized independently using train-only mean/std (leakage-free) and clipped to
+/-4; rows with no FoldX value get an all-zero vector (the standardized mean).
Train on the output with ``models/config/lightatt_addscores_mlp_config.json``
(``zs_mlp: true, zs_input_dim: 12``) and ``--add_zs_scores True``.

Format written per row: s1  s2  mut  label  z0 z1 ... z11
Reads one JSON per complex in skempi-foldx's results format (what
``data/foldx/results_sp/`` ships). Writes <out-dir>/splits_<tag>_foldxdec/fold_i/.

Usage: python experiments/foldx_s1102/merge_foldx_decomposed.py
           [--src <split_dir>] [--results <json_dir>] [--out <tag>] [--out-dir <dir>]

Defaults, and the note on which FoldX campaign the section 18 runs used: see
merge_foldx.py.
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(ROOT, "data", "foldx", "results_sp")
SRC_DEFAULT = os.path.join(ROOT, "experiments", "embedding_sweep", "splits", "paper_seed42")
TERMS = [
    "Interaction Energy", "Backbone Hbond", "Sidechain Hbond", "Van der Waals",
    "Electrostatics", "Solvation Polar", "Solvation Hydrophobic",
    "Van der Waals clashes", "entropy sidechain", "entropy mainchain",
    "torsional clash", "backbone clash",
]
CLIP = 4.0


def load_foldx(results_dir):
    """(pdb, role_mut) -> [12 term values] in TERMS order."""
    d = {}
    for f in glob.glob(os.path.join(results_dir, "*.json")):
        pdb = os.path.splitext(os.path.basename(f))[0]
        r = json.load(open(f))
        for role_mut, terms in r.get("muts", {}).items():
            if all(t in terms for t in TERMS):
                d[(pdb, role_mut)] = [terms[t] for t in TERMS]
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
    return fx.get((row[0].split("_")[0], row[2]))


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
    out_root = os.path.join(out_dir, f"splits_{tag}_foldxdec")

    fx = load_foldx(results_dir)
    print(f"[merge-dec] FoldX 12-term vectors for {len(fx)} (pdb,mut) pairs  (from {results_dir})")

    n_fold = len([d for d in os.listdir(src) if d.startswith("fold_")])
    tot_rows = tot_cov = 0
    for i in range(n_fold):
        fd = os.path.join(src, f"fold_{i}")
        base_name = detect_base(fd)
        train = read_tsv(os.path.join(fd, f"{base_name}_train.tsv"))
        # per-term train-only mean/std
        present = [foldx_for(r, fx) for r in train]
        present = [v for v in present if v is not None]
        means, stds = [], []
        for j in range(len(TERMS)):
            col = [v[j] for v in present]
            m = sum(col) / len(col)
            var = sum((x - m) ** 2 for x in col) / max(1, len(col) - 1)
            means.append(m)
            stds.append(var ** 0.5 or 1.0)

        def z(vec):
            if vec is None:
                return [0.0] * len(TERMS)
            out = []
            for j, x in enumerate(vec):
                zz = (x - means[j]) / stds[j]
                out.append(max(-CLIP, min(CLIP, zz)))
            return out

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
                    base = r[:4] if len(r) >= 4 else r
                    zs = [f"{x:.5f}" for x in z(v)]
                    fh.write("\t".join(base + zs) + "\n")
        tot_rows += rows_fold
        tot_cov += cov_fold
        print(f"[merge-dec] fold {i}: coverage {cov_fold}/{rows_fold} "
              f"({100*cov_fold/rows_fold:.1f}%)")
    print(f"[merge-dec] TOTAL coverage {tot_cov}/{tot_rows} "
          f"({100*tot_cov/tot_rows:.1f}%)  ->  {out_root}")


if __name__ == "__main__":
    main()
