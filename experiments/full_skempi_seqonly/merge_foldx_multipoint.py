#!/usr/bin/env python3
"""Merge multi-point FoldX ΔΔG into the MuLAN add_scores channel — the multi-point
analog of scratch/foldx_s1102/merge_foldx.py / merge_foldx_decomposed.py.

The join problem (why this isn't the single-point merge): our multi-point mutation
strings use MuLAN's A/B-partner + contiguous-position convention (`build_multipoint.py`),
while the FoldX `results_multipoint/<code>.json` `variants` are keyed by the RAW SKEMPI
`Mutation(s)_cleaned` string (real chain + cleaned resnum). They do NOT match on text.
`build_multipoint.py` therefore emits a SIDECAR — `multi_point_foldx_key.tsv` mapping each
row `(wt1,wt2,mulan_muts) -> (code, raw_variant)` — and this merge joins through it.

Trust: `audit_multipoint_foldx.py` writes `multipoint_foldx_exclude.tsv`, the `code.g1.g2`
complex-groupings FoldX scored under the wrong interface (2C5D.AB.CD, 3SE3.B.C). Rows in
those groupings are treated as unmapped (standardized 0), never given a bad FoldX value.

Two modes:
  (default) ASSEMBLE — read `multi_point.tsv`, attach the RAW 12-term FoldX vector (+ the
    Interaction-Energy scalar) to every row; unmapped/quarantined -> zeros. Writes
    `scratch/skempi_full/multi_point_foldx.tsv` and prints coverage + a FoldX-vs-ΔΔG
    sanity correlation. Standardization is deferred to split time (below).
  --split-dir DIR --out TAG [--decomposed]  — per-fold, train-only z-standardized merge
    exactly like merge_foldx{,_decomposed}.py (clip ±4, unmapped -> 0), for when the
    multi-point clustered split exists. Scalar (5th col) by default; 12 cols with
    --decomposed. Writes splits_<TAG>_mp_foldx{,dec}/.

Run:
  ./.venv/bin/python experiments/full_skempi_seqonly/merge_foldx_multipoint.py
  ./.venv/bin/python experiments/full_skempi_seqonly/merge_foldx_multipoint.py \
      --split-dir scratch/splits_skempi_full_mp_clustered/... --out ankh --decomposed
"""
from __future__ import annotations

import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SKEMPI_FULL = os.path.join(ROOT, "scratch", "skempi_full")
RESULTS = os.path.join(ROOT, "scratch", "foldx_skempi_full", "results_multipoint")
KEY = os.path.join(SKEMPI_FULL, "multi_point_foldx_key.tsv")
MULTI_TSV = os.path.join(SKEMPI_FULL, "multi_point.tsv")
EXCL = os.path.join(HERE, "multipoint_foldx_exclude.tsv")
TERMS = [
    "Interaction Energy", "Backbone Hbond", "Sidechain Hbond", "Van der Waals",
    "Electrostatics", "Solvation Polar", "Solvation Hydrophobic",
    "Van der Waals clashes", "entropy sidechain", "entropy mainchain",
    "torsional clash", "backbone clash",
]
CLIP = 4.0


def load_foldx(results_dir):
    """(code, raw_variant) -> [12 term values] in TERMS order (needs all terms)."""
    d = {}
    for f in glob.glob(os.path.join(results_dir, "*.json")):
        code = os.path.splitext(os.path.basename(f))[0]
        r = json.load(open(f))
        for var, terms in r.get("variants", {}).items():
            if all(t in terms for t in TERMS):
                d[(code, var)] = [terms[t] for t in TERMS]
    return d


def load_join(key_path):
    """(wt1,wt2,mulan_muts) -> (code, raw_variant)."""
    j = {}
    with open(key_path) as fh:
        for ln in fh:
            if ln.startswith("#"):
                continue
            p = ln.rstrip("\n").split("\t")
            if len(p) >= 5:
                j[(p[0], p[1], p[2])] = (p[3], p[4])
    return j


def load_exclude(excl_path):
    """set of untrustworthy code.g1.g2 complex-grouping labels."""
    s = set()
    if os.path.exists(excl_path):
        with open(excl_path) as fh:
            for ln in fh:
                if ln and not ln.startswith("#"):
                    s.add(ln.split("\t", 1)[0])
    return s


def read_tsv(path):
    rows = []
    with open(path) as fh:
        for ln in fh:
            p = ln.rstrip("\n").split("\t")
            if len(p) >= 3:
                rows.append(p)
    return rows


def cid_of(row):
    return row[0].rsplit("_", 1)[0]          # code.g1.g2  from  code.g1.g2_g1


def foldx_for(row, join, fx, exclude):
    """12-term FoldX vector for a multi_point row, or None (unmapped / quarantined)."""
    if cid_of(row) in exclude:
        return None
    cr = join.get((row[0], row[1], row[2]))
    if cr is None:
        return None
    return fx.get(cr)


def _pearson(x, y):
    n = len(x)
    if n < 3:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x) ** 0.5
    syy = sum((b - my) ** 2 for b in y) ** 0.5
    return sxy / (sxx * syy) if sxx and syy else float("nan")


def assemble(results_dir):
    fx = load_foldx(results_dir)
    join = load_join(KEY)
    exclude = load_exclude(EXCL)
    rows = read_tsv(MULTI_TSV)
    out_path = os.path.join(SKEMPI_FULL, "multi_point_foldx.tsv")
    cov = quar = 0
    ie, ddg = [], []
    with open(out_path, "w") as fh:
        fh.write("#wt1\twt2\tmulan_muts\tddG\t" + "\t".join(t.replace(" ", "_") for t in TERMS) + "\n")
        for r in rows:
            quar += cid_of(r) in exclude
            v = foldx_for(r, join, fx, exclude)
            cov += v is not None
            vec = v if v is not None else [0.0] * len(TERMS)
            if v is not None and len(r) >= 4:
                ie.append(v[0]); ddg.append(float(r[3]))
            fh.write("\t".join(r[:4]) + "\t" + "\t".join(f"{x:.4f}" for x in vec) + "\n")
    print(f"[assemble] rows {len(rows)}  FoldX-covered {cov} ({100*cov/len(rows):.1f}%)  "
          f"quarantined {quar}  unmapped {len(rows) - cov - quar}")
    print(f"[assemble] FoldX Interaction-Energy vs experimental ΔΔG (covered rows): "
          f"Pearson r = {_pearson(ie, ddg):.3f}  (n={len(ie)})")
    print(f"[assemble] wrote {out_path}")


def detect_base(fold_dir):
    cands = sorted(glob.glob(os.path.join(fold_dir, "*_train.tsv")))
    if not cands:
        raise SystemExit(f"no *_train.tsv in {fold_dir}")
    return os.path.basename(cands[0])[: -len("_train.tsv")]


def merge_split(src, tag, results_dir, decomposed):
    fx = load_foldx(results_dir)
    join = load_join(KEY)
    exclude = load_exclude(EXCL)
    ncol = len(TERMS) if decomposed else 1
    suffix = "foldxdec" if decomposed else "foldx"
    out_root = os.path.join(ROOT, "scratch", "foldx_skempi_full", f"splits_{tag}_mp_{suffix}")
    print(f"[merge{'-dec' if decomposed else ''}] FoldX for {len(fx)} (code,variant) pairs; "
          f"join {len(join)} rows; exclude {len(exclude)} groupings -> {out_root}")

    def vec_for(row):
        v = foldx_for(row, join, fx, exclude)
        if v is None:
            return None
        return v if decomposed else [v[0]]

    n_fold = len([d for d in os.listdir(src) if d.startswith("fold_")])
    tot_rows = tot_cov = 0
    for i in range(n_fold):
        fd = os.path.join(src, f"fold_{i}")
        base = detect_base(fd)
        train = read_tsv(os.path.join(fd, f"{base}_train.tsv"))
        present = [v for v in (vec_for(r) for r in train) if v is not None]
        means, stds = [], []
        for j in range(ncol):
            col = [v[j] for v in present]
            m = sum(col) / len(col) if col else 0.0
            var = sum((x - m) ** 2 for x in col) / max(1, len(col) - 1) if col else 1.0
            means.append(m); stds.append(var ** 0.5 or 1.0)

        def z(vec):
            if vec is None:
                return [0.0] * ncol
            return [max(-CLIP, min(CLIP, (vec[j] - means[j]) / stds[j])) for j in range(ncol)]

        od = os.path.join(out_root, f"fold_{i}")
        os.makedirs(od, exist_ok=True)
        cov = rf = 0
        for split in ("train", "val", "test"):
            rows = read_tsv(os.path.join(fd, f"{base}_{split}.tsv"))
            with open(os.path.join(od, f"{base}_{split}.tsv"), "w") as fh:
                for r in rows:
                    v = vec_for(r)
                    cov += v is not None; rf += 1
                    fh.write("\t".join(r[:4] + [f"{x:.5f}" for x in z(v)]) + "\n")
        tot_rows += rf; tot_cov += cov
        print(f"[merge] fold {i}: coverage {cov}/{rf} ({100*cov/rf:.1f}%)")
    print(f"[merge] TOTAL coverage {tot_cov}/{tot_rows} ({100*tot_cov/tot_rows:.1f}%) -> {out_root}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=RESULTS, help="results_multipoint dir")
    ap.add_argument("--split-dir", default=None, help="multi-point split dir (per-fold standardized mode)")
    ap.add_argument("--out", default="ankh", help="tag for splits_<tag>_mp_foldx{,dec}/ (split mode)")
    ap.add_argument("--decomposed", action="store_true", help="12 terms instead of the scalar")
    a = ap.parse_args()
    if a.split_dir:
        merge_split(a.split_dir, a.out, a.results, a.decomposed)
    else:
        assemble(a.results)


if __name__ == "__main__":
    main()
