#!/usr/bin/env python3
"""Merge FoldX into the CATH "all" split (mixed single- AND multi-point rows) as the add_scores
channel — the FoldX-arm enabler for the CATH-superfamily retrain.

The CATH split (`skempi_all_*.tsv`) interleaves single-point rows (col2 = one mutation) and
multi-point rows (col2 = comma-joined mutations). The existing mergers each handle only ONE:
  * single: merge_foldx_full_skempi.py  — lookup (pdb, remapped_mut) in results/*.json
  * multi : merge_foldx_multipoint.py   — lookup via join key -> (code,variant) in results_multipoint/*.json
This script imports both loaders, dispatches per row on whether col2 has a comma, and fits ONE
train-only z-standardization over the combined 12-term FoldX vectors (leakage-free, clipped to
+/-CLIP; uncovered rows -> 0 = standardized mean). Matches the single/multi scripts term-for-term.

Emits (base_name auto-detected = skempi_all):
  <outdir>/splits_cath_foldx/fold_0/skempi_all_{train,val,test}.tsv     5-col scalar (Interaction Energy)
  <outdir>/splits_cath_foldxdec/fold_0/skempi_all_{train,val,test}.tsv  16-col 12-term decomposed

Run:  .venv/bin/python experiments/full_skempi_seqonly/merge_foldx_cath.py
"""
import argparse, glob, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import merge_foldx_full_skempi as spm   # single-point loaders
import merge_foldx_multipoint as mpm    # multi-point loaders

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
TERMS = spm.TERMS
CLIP = spm.CLIP


def build_lookup(src, base_name, sp_results, mp_results):
    """Return vec_for(row) -> 12-term list or None, covering both single and multi rows."""
    sp_groups = spm.parse_groups(src, base_name)
    sp_fx = spm.load_foldx(sp_results, sp_groups)          # (pdb, remapped_mut) -> (scalar,[12])
    sp_trusted = spm.load_trusted_groupings()              # grouping guard for the SP path
    mp_fx = mpm.load_foldx(mp_results)                     # (code, variant) -> [12]
    mp_join = mpm.load_join(mpm.KEY)
    mp_excl = mpm.load_exclude(mpm.EXCL)
    print(f"[cath-merge] single-pt FoldX pairs: {len(sp_fx)} | multi-pt pairs: {len(mp_fx)} | "
          f"join {len(mp_join)} | exclude {len(mp_excl)} | sp-guard {len(sp_trusted)} codes")

    def vec_for(row):
        if "," in row[2]:                                  # multi-point row
            return mpm.foldx_for(row, mp_join, mp_fx, mp_excl)
        # single-point row -- go through spm.foldx_for so the grouping guard applies here too
        # (a bare sp_fx.get() keys on the PDB code alone and would hand 3SE4.B.A rows the
        # ddG of 3SE4.B.C, the only grouping FoldX actually scored).
        v = spm.foldx_for(row, sp_fx, sp_trusted)
        return v[1] if v is not None else None

    return vec_for


def fit_std(vecs):
    present = [v for v in vecs if v is not None]
    means, stds = [], []
    for j in range(len(TERMS)):
        col = [v[j] for v in present]
        m = sum(col) / len(col) if col else 0.0
        var = sum((x - m) ** 2 for x in col) / max(1, len(col) - 1) if col else 0.0
        means.append(m)
        stds.append(var ** 0.5 or 1.0)
    return means, stds


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="scratch/splits_skempi_full_cath_kfold")
    ap.add_argument("--sp-results", default="scratch/foldx_skempi_full/results_all",
                    help="unified SP FoldX dir (build_results_all.py); SP coverage 28%%->58%%")
    ap.add_argument("--mp-results", default="scratch/foldx_skempi_full/results_multipoint")
    ap.add_argument("--outdir", default="scratch/foldx_skempi_full")
    a = ap.parse_args()

    folds = sorted(glob.glob(os.path.join(a.src, "fold_*")))
    if not folds:
        raise SystemExit(f"no fold_* under {a.src}")
    base = spm.detect_base(folds[0])
    vec_for = build_lookup(a.src, base, a.sp_results, a.mp_results)

    out_scalar = os.path.join(a.outdir, "splits_cath_foldx")
    out_dec = os.path.join(a.outdir, "splits_cath_foldxdec")

    tot_rows = tot_cov = cov_sp = cov_mp = n_sp = n_mp = 0
    for i in range(len(folds)):
        fd = os.path.join(a.src, f"fold_{i}")
        train = spm.read_tsv(os.path.join(fd, f"{base}_train.tsv"))
        means, stds = fit_std([vec_for(r) for r in train])

        def z(v, j):
            return max(-CLIP, min(CLIP, (v[j] - means[j]) / stds[j]))

        os.makedirs(os.path.join(out_scalar, f"fold_{i}"), exist_ok=True)
        os.makedirs(os.path.join(out_dec, f"fold_{i}"), exist_ok=True)
        for split in ("train", "val", "test"):
            rows = spm.read_tsv(os.path.join(fd, f"{base}_{split}.tsv"))
            fs = open(os.path.join(out_scalar, f"fold_{i}", f"{base}_{split}.tsv"), "w")
            fdz = open(os.path.join(out_dec, f"fold_{i}", f"{base}_{split}.tsv"), "w")
            for r in rows:
                is_multi = "," in r[2]
                v = vec_for(r)
                tot_rows += 1
                n_mp += is_multi; n_sp += not is_multi
                if v is not None:
                    tot_cov += 1; cov_mp += is_multi; cov_sp += not is_multi
                head = r[:4] if len(r) >= 4 else r
                sc = z(v, 0) if v is not None else 0.0
                dec = [z(v, j) for j in range(len(TERMS))] if v is not None else [0.0] * len(TERMS)
                fs.write("\t".join(head + [f"{sc:.5f}"]) + "\n")
                fdz.write("\t".join(head + [f"{x:.5f}" for x in dec]) + "\n")
            fs.close(); fdz.close()

    print(f"[cath-merge] rows {tot_rows}  covered {tot_cov} ({100*tot_cov/tot_rows:.1f}%)")
    print(f"[cath-merge]   single {cov_sp}/{n_sp} ({100*cov_sp/max(1,n_sp):.1f}%)  "
          f"multi {cov_mp}/{n_mp} ({100*cov_mp/max(1,n_mp):.1f}%)")
    print(f"[cath-merge]   scalar -> {out_scalar}")
    print(f"[cath-merge]   decomp -> {out_dec}")


if __name__ == "__main__":
    main()
