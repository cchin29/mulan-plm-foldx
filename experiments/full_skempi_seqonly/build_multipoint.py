#!/usr/bin/env python3
"""Build MuLAN inputs for the full SKEMPI v2 MULTI-POINT set + the FoldX join key.

Mirror of build_skempi_full.py (single-point) for the comma-joined `Mutation(s)_cleaned`
rows RDE keeps but that build drops. Reuses build_skempi_full's helpers unchanged
(load_mapping / group_seq / THREE_TO_ONE / curation constants) so the sequence, offset
and WT-validation conventions are byte-identical to the single-point arm.

A variant is emitted only if **every** sub-mutation validates (WT identity at its
contiguous position, chain present in one of the #Pdb groups). Sub-muts whose real
chain is outside both groups are dropped — which auto-excludes exactly the symmetric-
homodimer copies the FoldX chain-map audit flagged (e.g. 2C5D `TA26D,TB26D`), keeping
the built set consistent with the trustworthy FoldX values.

Outputs to scratch/skempi_full/ (gitignored):
  multi_point.tsv            -- wt1_label  wt2_label  mut1,mut2,...  ddG   (MuLAN 4-col contract,
                                mutation col is comma-joined; parse_mutations applies each to its
                                own A/B partner)
  multi_point_foldx_key.tsv  -- wt1_label  wt2_label  mulan_muts  code  raw_variant
                                (the SIDECAR join: raw_variant = the original SKEMPI cleaned string,
                                == the FoldX results_multipoint/<code>.json `variants` key)
  multi_point_replicates.tsv -- wt1_label  wt2_label  mulan_muts  n  mean  min  max  spread  n_refs
                                (the ddG column above is a MEAN over n SKEMPI rows. This records
                                the collapse, because SKEMPI's Affinity_wt is a per-ROW reference
                                state, so a mutation can recur against different parental
                                backgrounds. n_refs>1 proves a group is not replicates; it is not
                                required -- a maturation ladder can sit inside one paper. Filter
                                or weight on `spread` and `n`, not on `n_refs` alone)
  wt_sequences_multipoint.fasta -- chain-group records for all labels used by multi-point rows
                                (superset overlaps the single-point fasta; kept separate so the
                                single-point outputs are untouched — union the two for training).

Run:  python3 experiments/full_skempi_seqonly/build_multipoint.py
"""
from __future__ import annotations

import argparse
import collections
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_skempi_full as b  # noqa: E402  (helpers + constants; single-point path untouched)


def curated_multipoint_entries():
    """RDE load_skempi_entries logic, restricted to MULTI-point rows (>=2 sub-muts)."""
    df = pd.read_csv(b.CSV, sep=";")
    df["ddG"] = b.RT * np.log(df["Affinity_mut_parsed"]) - b.RT * np.log(df["Affinity_wt_parsed"])
    entries = []
    for _, row in df.iterrows():
        code, g1, g2 = row["#Pdb"].split("_")
        if code in b.BLOCK:
            continue
        raw = ",".join(s for s in row["Mutation(s)_cleaned"].split(",") if s)
        subs = raw.split(",")
        if len(subs) < 2:                          # multi-point only
            continue
        if not os.path.exists(os.path.join(b.PDBDIR, f"{code.upper()}.pdb")):
            continue
        if not np.isfinite(row["ddG"]):
            continue
        entries.append({"code": code, "g1": g1, "g2": g2, "subs": subs,
                        "raw": raw, "ddG": float(row["ddG"]), "complex": row["#Pdb"],
                        "ref": str(row.get("Reference", ""))})
    return entries


def build(verbose=True):
    entries = curated_multipoint_entries()
    fasta = {}
    acc = collections.defaultdict(list)             # (labelA,labelB,mulan_mut) -> [(ddG, ref)]
    raw_of = {}                                     # (labelA,labelB,mulan_mut) -> raw_variant
    drops = collections.Counter()
    mapping_cache = {}
    kept = 0
    for e in entries:
        code, g1, g2 = e["code"], e["g1"], e["g2"]
        if code not in mapping_cache:
            try:
                mapping_cache[code] = b.load_mapping(code)
            except FileNotFoundError:
                mapping_cache[code] = None
        chains = mapping_cache[code]
        if chains is None:
            drops["no_mapping_file"] += 1
            continue
        seqA, offA = b.group_seq(chains, g1)
        seqB, offB = b.group_seq(chains, g2)
        if seqA is None or seqB is None:
            drops["missing_chain_in_group"] += 1
            continue
        labelA, labelB = f"{code}.{g1}.{g2}_{g1}", f"{code}.{g1}.{g2}_{g2}"

        mulan_subs, ok = [], True
        for sm in e["subs"]:
            wt, mchain, mt = sm[0], sm[1], sm[-1]
            try:
                num = int(sm[2:-1])
            except ValueError:
                drops["icode_or_bad_resseq"] += 1; ok = False; break
            if mchain in g1:
                side, off, tgt = "A", offA, seqA
            elif mchain in g2:
                side, off, tgt = "B", offB, seqB
            else:
                drops["mutchain_not_in_either_group"] += 1; ok = False; break   # e.g. 2C5D symm copy
            if not (0 < num <= len(chains[mchain]["seq"])):
                drops["pos_out_of_range"] += 1; ok = False; break
            pos = off[mchain] + num
            if tgt[pos - 1] != wt:
                drops["wt_mismatch"] += 1; ok = False; break
            mulan_subs.append(f"{wt}{side}{pos}{mt}")
        if not ok:
            continue

        mulan_mut = ",".join(mulan_subs)
        key = (labelA, labelB, mulan_mut)
        if key in raw_of and raw_of[key] != e["raw"]:
            drops["join_key_collision"] += 1        # two raw variants -> same MuLAN string (should not happen)
            continue
        fasta[labelA], fasta[labelB] = seqA, seqB
        acc[key].append((e["ddG"], e["ref"]))
        raw_of[key] = e["raw"]
        kept += 1

    # Mean kept, collapse recorded -- see the sidecar below and build_skempi_full.py for why
    # it is a sidecar rather than extra columns.
    rows = [(a, bl, m, sum(d for d, _ in v) / len(v)) for (a, bl, m), v in acc.items()]
    n_replicate = kept - len(rows)

    os.makedirs(b.OUTDIR, exist_ok=True)
    fa_path = os.path.join(b.OUTDIR, "wt_sequences_multipoint.fasta")
    with open(fa_path, "w") as fh:
        for label, seq in sorted(fasta.items()):
            fh.write(f">{label}\n{seq}\n")
    tsv_path = os.path.join(b.OUTDIR, "multi_point.tsv")
    key_path = os.path.join(b.OUTDIR, "multi_point_foldx_key.tsv")
    with open(tsv_path, "w") as ft, open(key_path, "w") as fk:
        fk.write("#wt1\twt2\tmulan_muts\tcode\traw_variant\n")
        for r in sorted(rows):
            ft.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]:.6f}\n")
            code = r[0].split(".")[0]
            fk.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{code}\t{raw_of[(r[0], r[1], r[2])]}\n")

    rep_path = os.path.join(b.OUTDIR, "multi_point_replicates.tsv")
    with open(rep_path, "w") as fh:
        fh.write("#wt1\twt2\tmulan_muts\tn\tmean\tmin\tmax\tspread\tn_refs\n")
        for (a, bl, mm), v in sorted(acc.items()):
            d = [x for x, _ in v]
            fh.write(f"{a}\t{bl}\t{mm}\t{len(d)}\t{sum(d)/len(d):.6f}\t{min(d):.6f}\t"
                     f"{max(d):.6f}\t{max(d)-min(d):.6f}\t{len({r for _, r in v})}\n")

    if verbose:
        n = len(entries)
        print(f"curated multi-point entries  : {n}")
        print(f"kept (all sub-muts validated): {kept}   coverage {100*kept/n:.1f}%")
        print(f"unique rows after replicate-avg: {len(rows)}   (merged {n_replicate} replicates)")
        print(f"dropped                      : {sum(drops.values())}")
        for k, v in drops.most_common():
            print(f"    {k:28s}: {v}")
        print(f"fasta labels (chain-groups)  : {len(fasta)}")
        print(f"distinct #Pdb complexes      : {len({r[0].split('_')[0] for r in rows})}")
        print(f"wrote {tsv_path}")
        print(f"wrote {key_path}")
        print(f"wrote {fa_path}")
    return rows, fasta, raw_of, drops


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    build(verbose=not ap.parse_args().quiet)
