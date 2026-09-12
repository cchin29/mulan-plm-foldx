#!/usr/bin/env python3
"""Merge full-SKEMPI FoldX binding ddG into the clustered CV splits as the add_scores channel.

Full-SKEMPI analog of scratch/foldx_s1102/merge_foldx{,_decomposed}.py. Emits BOTH arms in
one pass:
  * scalar  (5-col):  s1 s2 mut label  z(Interaction Energy)          -> <out>_foldx/
  * decomp (16-col):  s1 s2 mut label  z0 z1 ... z11 (12 FoldX terms)  -> <out>_foldxdec/

Two full-SKEMPI-specific differences vs the S1102 scripts:
  1. Complex id uses DOTS (`1AHW.AB.C_AB`) not underscores, so pdb = col0.split(".")[0].
  2. single_point.tsv concatenates each interface GROUP into one pseudo-chain (g1->A, g2->B)
     and renumbers residues contiguously, while the FoldX JSONs are keyed by the ORIGINAL PDB
     chain + that chain's own per-chain residue index. Joining therefore needs BOTH the chain
     letter and the residue NUMBER remapped (remap_mut_resnum); remapping only the letter is
     what held single-point coverage at 87.8% until 2026-07-29.

Standardization matches the S1102 scripts exactly: per-fold, per-term train-only mean/std
(leakage-free), clipped to +/-CLIP; rows with no FoldX value get 0 (the standardized mean).

Usage:
  python experiments/full_skempi_seqonly/merge_foldx_full_skempi.py \
      --src   scratch/splits_skempi_full_clustered_id60_kfold \
      --results scratch/foldx_skempi_full/results \
      --outdir  scratch/foldx_skempi_full
"""
import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_skempi_full as bsf  # noqa: E402  (load_mapping/group_seq = authoritative numbering)

SKEMPI_CSV = "scratch/skempi_v2.csv"

TERMS = [
    "Interaction Energy", "Backbone Hbond", "Sidechain Hbond", "Van der Waals",
    "Electrostatics", "Solvation Polar", "Solvation Hydrophobic",
    "Van der Waals clashes", "entropy sidechain", "entropy mainchain",
    "torsional clash", "backbone clash",
]
CLIP = 4.0


def read_tsv(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                rows.append(p)
    return rows


def detect_base(fold_dir):
    cands = sorted(glob.glob(os.path.join(fold_dir, "*_train.tsv")))
    if not cands:
        raise SystemExit(f"no *_train.tsv in {fold_dir}")
    return os.path.basename(cands[0])[: -len("_train.tsv")]


def parse_groups(src, base_name):
    """pdb code -> (g1, g2) chain-group strings, from any split row's col0 (code.g1.g2_...)."""
    g = {}
    for fd in sorted(glob.glob(os.path.join(src, "fold_*"))):
        for split in ("train", "val", "test"):
            p = os.path.join(fd, f"{base_name}_{split}.tsv")
            if not os.path.exists(p):
                continue
            for r in read_tsv(p):
                code_grp = r[0].split("_")[0].split(".")   # [code, g1, g2]
                if len(code_grp) >= 3:
                    g[code_grp[0]] = (code_grp[1], code_grp[2])
    return g


def remap_mut(mut, groups_for_pdb):
    """CHAIN-ONLY remap (legacy). Kept as a fallback key -- see remap_mut_resnum for the
    authoritative form. Correct only when the mutated chain is the FIRST member of its group
    AND that chain's author resseq happens to equal its 1-based sequence index.

    mut = WT + origchain + pos... + newAA. origchain in g1 -> 'A', in g2 -> 'B'.
    """
    if groups_for_pdb is None or len(mut) < 3:
        return None
    g1, g2 = groups_for_pdb
    ch = mut[1]
    if ch in g1:
        nch = "A"
    elif ch in g2:
        nch = "B"
    else:
        return None
    return mut[0] + nch + mut[2:]


_MAP_CACHE = {}
_OFF_CACHE = {}


def _chains_for(code):
    if code not in _MAP_CACHE:
        try:
            _MAP_CACHE[code] = bsf.load_mapping(code)
        except Exception:
            _MAP_CACHE[code] = None
    return _MAP_CACHE[code]


def _offsets_for(code, group):
    """{chain: offset} for a concatenated interface group, or None if unmappable."""
    key = (code, group)
    if key not in _OFF_CACHE:
        chains = _chains_for(code)
        if chains is None or any(c not in chains for c in group):
            _OFF_CACHE[key] = None
        else:
            _, off = bsf.group_seq(chains, group)
            _OFF_CACHE[key] = off
    return _OFF_CACHE[key]


def remap_mut_resnum(mut, groups_for_pdb, code):
    """AUTHORITATIVE remap: chain letter AND residue number, inverting build_skempi_full.

    build_skempi_full.py numbers a mutation as  pos = off[chain] + <per-chain seq index>,
    i.e. each interface group's member chains are concatenated into one pseudo-chain and
    renumbered contiguously from 1. The FoldX JSONs are keyed by original PDB chain +
    per-chain index, so remapping only the chain letter (remap_mut) leaves the residue
    number in the wrong coordinate system and the key silently fails to join. That was the
    whole 87.8% -> 99.2% single-point gap (see SP_FOLDX_COVERAGE_AUDIT.md).

    The FoldX JSON residue number is the per-chain 1-based SEQUENCE INDEX, not the author
    resseq -- measured, not assumed: interpreting it as author resseq (resseq2idx lookup)
    recovers ZERO rows that the sequence-index reading does not already recover, on every
    tier. Emitting the author-resseq reading as an extra candidate key is therefore pure
    downside: it lands on OTHER real split rows and hands them a different mutation's ddG
    (15 rows in the SP fold_0 alone, e.g. 1DAN KB152A getting 1.8033 instead of -0.0578).
    That mistake shipped briefly on 2026-07-29 and cost the first CATH rerun; do not
    reintroduce a second convention without re-measuring its unique-recovery count first.

    Returns the split-convention key <wt><A|B><pos><mt>, or None if unmappable.
    """
    if groups_for_pdb is None or len(mut) < 4:
        return None
    g1, g2 = groups_for_pdb
    ch = mut[1]
    if ch in g1:
        nch, group = "A", g1
    elif ch in g2:
        nch, group = "B", g2
    else:
        return None
    resseq = mut[2:-1]          # STRING -- may carry an insertion code ('116A'); not an index
    if not resseq.isdigit():
        return None
    off = _offsets_for(code, group)
    if off is None:
        return None
    n = int(resseq)
    if not (1 <= n <= len(_chains_for(code)[ch]["seq"])):
        return None
    return f"{mut[0]}{nch}{off[ch] + n}{mut[-1]}"


def load_trusted_groupings(csv_path=SKEMPI_CSV):
    """code -> (g1, g2) that FoldX actually scored: SKEMPI's FIRST-SEEN #Pdb grouping.

    build_ddg collapses each PDB to its first-seen grouping, so a complex appearing under two
    groupings has FoldX numbers for only one of them -- rows built under the other grouping
    would silently receive the WRONG interface's ddG. Only 2C5D / 3SE3 / 3SE4 are affected in
    SKEMPI v2, but deriving the rule beats hand-listing it.
    """
    first = {}
    if not os.path.exists(csv_path):
        return first
    with open(csv_path, newline="") as fh:
        rd = csv.reader(fh, delimiter=";")
        hdr = next(rd, None)
        i = hdr.index("#Pdb") if hdr and "#Pdb" in hdr else 0
        for row in rd:
            if not row:
                continue
            parts = row[i].split("_")
            if len(parts) >= 3:
                first.setdefault(parts[0], (parts[1], parts[2]))
    return first


def load_foldx(results_dir, groups):
    """(pdb, mut) -> (scalar Interaction Energy, [12-term vector]).

    Dual-keyed: each FoldX mutation is indexed under BOTH its g1->A/g2->B remapped key
    AND its raw original key. Standard complexes' split rows use the remapped form (their
    FoldX JSONs are keyed by original PDB chain, so remap_mut aligns them); but the
    benchmark-sourced complexes (S4169 scans: 3BT1/1PPF/1R0R/3SGB/...) carry the split's
    NATIVE keys already (e.g. json 'AB18S' == split col2 'AB18S') and their chain isn't in
    g1/g2, so remap returns None and would silently drop them. Indexing the raw key too
    recovers those (~58%->~88% single-point coverage). Additive: remapped and raw forms of
    the same mut differ, so no collisions; setdefault keeps the remapped value authoritative.
    """
    d = {}
    for f in glob.glob(os.path.join(results_dir, "*.json")):
        pdb = os.path.splitext(os.path.basename(f))[0]
        gp = groups.get(pdb)
        r = json.load(open(f))
        muts = r.get("muts", {})
        vals = {}
        for orig_mut, terms in muts.items():
            if not all(t in terms for t in TERMS):
                continue
            vals[orig_mut] = (terms["Interaction Energy"], [terms[t] for t in TERMS])
        # Pass 1 -- AUTHORITATIVE derived key (written, so it wins any collision).
        for orig_mut, val in vals.items():
            rn = remap_mut_resnum(orig_mut, gp, pdb)
            if rn is not None:
                d[(pdb, rn)] = val
        # Pass 2 -- raw native key, never overwrites an authoritative key. Needed for the
        # benchmark-sourced complexes (S4169 scans: 3BT1/1PPF/1R0R/3SGB/...) whose JSONs
        # already carry the split's own key form; 1213 of 4165 SP rows rely on this.
        # NB the old chain-only remap_mut() key is deliberately NOT indexed here: measured
        # across every tier it covers ZERO rows the two keys above don't already cover, and
        # for multi-chain groups it lands on the wrong row (same failure mode as the
        # author-resseq candidate removed above).
        for orig_mut, val in vals.items():
            d.setdefault((pdb, orig_mut), val)
    return d


def foldx_for(row, fx, trusted=None):
    """FoldX value for a split row, or None.

    `trusted` (code -> first-seen grouping) gates the lookup: a row whose complex-grouping is
    NOT the one FoldX scored gets None rather than another interface's ddG.
    """
    cid = row[0].split("_")[0].split(".")
    code = cid[0]
    if trusted and len(cid) >= 3:
        want = trusted.get(code)
        if want is not None and (cid[1], cid[2]) != want:
            return None
    return fx.get((code, row[2]))


def standardize_fit(values, ndim):
    """Return (means, stds) fit on the present (non-None) train values. ndim=1 -> scalar."""
    present = [v for v in values if v is not None]
    means, stds = [], []
    for j in range(ndim):
        col = [(v[1][j] if ndim > 1 else v[0]) for v in present]
        m = sum(col) / len(col) if col else 0.0
        var = sum((x - m) ** 2 for x in col) / max(1, len(col) - 1) if col else 0.0
        means.append(m)
        stds.append(var ** 0.5 or 1.0)
    return means, stds


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="scratch/splits_skempi_full_clustered_id60_kfold")
    ap.add_argument("--results", default="scratch/foldx_skempi_full/results_all",
                    help="unified SP FoldX dir (build_results_all.py: symlinks every complex to "
                         "its most-complete JSON across the full-SKEMPI + S1102 + benchmark runs; "
                         "SP coverage 28%%->58%%). Pass the old .../results for the pre-union set.")
    ap.add_argument("--outdir", default="scratch/foldx_skempi_full",
                    help="parent for splits_skempi_full_foldx / _foldxdec")
    args = ap.parse_args()

    folds = sorted(glob.glob(os.path.join(args.src, "fold_*")))
    if not folds:
        raise SystemExit(f"no fold_* under {args.src}")
    base_name = detect_base(folds[0])
    groups = parse_groups(args.src, base_name)
    fx = load_foldx(args.results, groups)
    trusted = load_trusted_groupings()
    print(f"[merge] {len(groups)} complexes with groups; "
          f"FoldX remapped pairs: {len(fx)}  (from {args.results})")
    print(f"[merge] grouping guard active: {len(trusted)} codes from {SKEMPI_CSV} "
          f"(denials counted per-row below)")
    denied = {}

    out_scalar = os.path.join(args.outdir, "splits_skempi_full_foldx")
    out_dec = os.path.join(args.outdir, "splits_skempi_full_foldxdec")

    tot_rows = tot_cov = 0
    for i in range(len(folds)):
        fd = os.path.join(args.src, f"fold_{i}")
        train = read_tsv(os.path.join(fd, f"{base_name}_train.tsv"))
        tv = [foldx_for(r, fx, trusted) for r in train]
        s_mean, s_std = standardize_fit(tv, 1)
        d_mean, d_std = standardize_fit(tv, len(TERMS))

        def zscalar(v):
            if v is None:
                return 0.0
            return max(-CLIP, min(CLIP, (v[0] - s_mean[0]) / s_std[0]))

        def zdec(v):
            if v is None:
                return [0.0] * len(TERMS)
            return [max(-CLIP, min(CLIP, (v[1][j] - d_mean[j]) / d_std[j]))
                    for j in range(len(TERMS))]

        os.makedirs(os.path.join(out_scalar, f"fold_{i}"), exist_ok=True)
        os.makedirs(os.path.join(out_dec, f"fold_{i}"), exist_ok=True)
        cov_fold = rows_fold = 0
        for split in ("train", "val", "test"):
            rows = read_tsv(os.path.join(fd, f"{base_name}_{split}.tsv"))
            fs = open(os.path.join(out_scalar, f"fold_{i}", f"{base_name}_{split}.tsv"), "w")
            fdz = open(os.path.join(out_dec, f"fold_{i}", f"{base_name}_{split}.tsv"), "w")
            for r in rows:
                v = foldx_for(r, fx, trusted)
                if v is None:
                    cid = r[0].split("_")[0].split(".")
                    w = trusted.get(cid[0])
                    if w is not None and len(cid) >= 3 and (cid[1], cid[2]) != w:
                        denied[r[0].split("_")[0]] = denied.get(r[0].split("_")[0], 0) + 1
                cov_fold += v is not None
                rows_fold += 1
                base = r[:4] if len(r) >= 4 else r
                fs.write("\t".join(base + [f"{zscalar(v):.5f}"]) + "\n")
                fdz.write("\t".join(base + [f"{x:.5f}" for x in zdec(v)]) + "\n")
            fs.close()
            fdz.close()
        tot_rows += rows_fold
        tot_cov += cov_fold
        print(f"[merge] fold {i}: coverage {cov_fold}/{rows_fold} "
              f"({100*cov_fold/rows_fold:.1f}%)  scalar mean={s_mean[0]:.3f} std={s_std[0]:.3f}")
    print(f"[merge] TOTAL coverage {tot_cov}/{tot_rows} ({100*tot_cov/tot_rows:.1f}%)")
    if denied:
        tot_d = sum(denied.values())
        print(f"[merge] grouping guard DENIED {tot_d} row-instances (wrong interface, would have "
              f"been silently wrong before): " + ", ".join(f"{k}={v}" for k, v in sorted(denied.items())))
    print(f"[merge]   scalar splits -> {out_scalar}")
    print(f"[merge]   decomp splits -> {out_dec}")


if __name__ == "__main__":
    main()
