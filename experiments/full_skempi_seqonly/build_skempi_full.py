#!/usr/bin/env python3
"""Build MuLAN inputs for the full SKEMPI v2 SINGLE-POINT set (step 1, sequence-only).

Entry set = RDE `load_skempi_entries` curation (verbatim logic; docs/history/PLAN_FULL_SKEMPI.md §2 🔒),
restricted to single-point mutations. Emits the two files MuLAN's data.py consumes:

  scratch/skempi_full/wt_sequences.fasta   -- one record per used (CODE, chain-group), label
                                              "<CODE>_<group>", seq = concat of member chains.
  scratch/skempi_full/single_point.tsv     -- rows: wt1_label \t wt2_label \t mutation \t ddG
                                              (MuLAN col contract 0=wt1,1=wt2,2=muts,3=ddG).

MuLAN convention (mulan/utils.py::parse_mutations): a mutation is "<wt><chain A|B><pos><mt>",
applied as seq[pos-1]=mt, chain 'A' -> partner1(seq1), 'B' -> partner2(seq2), pos = 1-based
index INTO the partner sequence (NOT the PDB author resseq). So we:
  * put the MUTATED group as partner1 (chain 'A'), the other as partner2 ('B') -- RDE ligand rule;
  * concat multi-chain groups (antibody H_L, multimers) in group-string order;
  * translate author resseq -> per-chain seq-index via the <CODE>.mapping file, then add the
    offset of the mutated chain within its concatenated group;
  * VALIDATE partner1[pos-1] == wt at build time (parse_mutations does not) and keep a row only
    if it passes -- reporting coverage %.

Chain sequences + author-resseq->seqindex both come from scratch/skempi2/PDBs/<CODE>.mapping
(cols: RESNAME3 CHAIN AUTHOR_RESSEQ SEQINDEX; seqindex 1-based, contiguous, per chain).

Outputs to scratch/skempi_full/ (gitignored). Nothing in mulan/ or the S1102 dirs is touched.
"""
from __future__ import annotations

import argparse
import collections
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

from skempi_foldx.exclusions import INTRACTABLE  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CSV = os.path.join(ROOT, "scratch/skempi_v2.csv")
PDBDIR = os.path.join(ROOT, "scratch/skempi2/PDBs")
OUTDIR = os.path.join(ROOT, "scratch/skempi_full")
BLOCK = set(INTRACTABLE)   # RDE block_list. Same ids as the FoldX compute guard, from
                           # one place, so curation and compute cannot drift apart --
                           # the reasons differ (RDE drops it; FoldX cannot repair it)
                           # but an id blocked for either reason must be blocked for both.
RT = (8.314 / 4184) * (273.15 + 25.0)

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


def curated_singlepoint_entries():
    """RDE load_skempi_entries logic, restricted to single-point rows."""
    df = pd.read_csv(CSV, sep=";")
    df["ddG"] = RT * np.log(df["Affinity_mut_parsed"]) - RT * np.log(df["Affinity_wt_parsed"])
    entries = []
    for _, row in df.iterrows():
        code, g1, g2 = row["#Pdb"].split("_")
        if code in BLOCK:
            continue
        muts = row["Mutation(s)_cleaned"].split(",")
        if len(muts) != 1:                       # single-point only
            continue
        if not os.path.exists(os.path.join(PDBDIR, f"{code.upper()}.pdb")):
            continue
        if not np.isfinite(row["ddG"]):
            continue
        entries.append({
            "code": code, "g1": g1, "g2": g2,
            "mut": muts[0], "ddG": float(row["ddG"]),
            "complex": row["#Pdb"],
            # Carried for the replicate sidecar. SKEMPI's Affinity_wt is a per-ROW reference
            # state, so a mutation can recur against different parental backgrounds and its mean
            # then averages across different baselines.
            #
            # n_refs > 1 is SUFFICIENT evidence a group is not replicates, but NOT necessary:
            # 3MZG.A.B HA167A is n=19 from a single reference with a 3.14 kcal/mol spread -- an
            # affinity-maturation ladder inside one paper. Filter on spread AND n, not on
            # n_refs alone.
            "ref": str(row.get("Reference", "")),
        })
    return entries


def load_mapping(code):
    """<CODE>.mapping -> {chain: {'seq': str, 'resseq2idx': {resseq_str: 1-based-idx}}}."""
    path = os.path.join(PDBDIR, f"{code.upper()}.mapping")
    chains = collections.OrderedDict()
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 4:
                continue
            resn, ch, resseq, seqidx = parts[0], parts[1], parts[2], int(parts[3])
            c = chains.setdefault(ch, {"resn": [], "resseq2idx": {}})
            c["resn"].append(THREE_TO_ONE.get(resn, "X"))
            c["resseq2idx"][resseq] = seqidx        # 1-based, per-chain
    for ch, c in chains.items():
        c["seq"] = "".join(c["resn"])
        del c["resn"]
    return chains


def group_seq(chains, group):
    """Concat member-chain sequences in group-string order; return (seq, {chain: offset})."""
    seq, offsets, off = "", {}, 0
    for ch in group:
        if ch not in chains:
            return None, None                       # missing chain -> caller drops
        offsets[ch] = off
        seq += chains[ch]["seq"]
        off += len(chains[ch]["seq"])
    return seq, offsets


def build(verbose=True):
    entries = curated_singlepoint_entries()
    fasta = {}                                      # label -> seq
    acc = collections.defaultdict(list)             # (wt1_label, wt2_label, mut) -> [ddG, ...]
    drops = collections.Counter()
    mapping_cache = {}
    kept = 0
    for e in entries:
        code, g1, g2, mut = e["code"], e["g1"], e["g2"], e["mut"]
        wt, mchain, mt = mut[0], mut[1], mut[-1]
        try:
            num = int(mut[2:-1])                     # 1-based index into the contiguous chain seq
        except ValueError:
            drops["icode_or_bad_resseq"] += 1        # insertion code -> non-int -> drop
            continue
        if code not in mapping_cache:
            try:
                mapping_cache[code] = load_mapping(code)
            except FileNotFoundError:
                mapping_cache[code] = None
        chains = mapping_cache[code]
        if chains is None:
            drops["no_mapping_file"] += 1
            continue
        # FIXED partner assignment: group1 -> partner 'A' (seq1), group2 -> partner 'B' (seq2).
        # (NOT "mutated side = A" -- that would make g1- and g2-side mutations both chain 'A' and
        # collide on the (complex, mutation) join key.) Mutation chain letter = the side it's on.
        if mchain in g1:
            side, grp = "A", g1
        elif mchain in g2:
            side, grp = "B", g2
        else:
            drops["mutchain_not_in_either_group"] += 1
            continue
        seqA, offA = group_seq(chains, g1)
        seqB, offB = group_seq(chains, g2)
        if seqA is None or seqB is None:
            drops["missing_chain_in_group"] += 1
            continue
        if not (0 < num <= len(chains[mchain]["seq"])):
            drops["pos_out_of_range"] += 1
            continue
        off = offA if side == "A" else offB
        tgt = seqA if side == "A" else seqB
        pos = off[mchain] + num                     # 1-based into the mutated partner
        if tgt[pos - 1] != wt:
            drops["wt_mismatch"] += 1
            continue
        # Complex id = full #Pdb (code + both chain groups), '.'-joined so it survives
        # rescore._pdb_of (split on '_' -> first token) as a UNIQUE per-#Pdb structure id.
        cid = f"{code}.{g1}.{g2}"
        labelA, labelB = f"{cid}_{g1}", f"{cid}_{g2}"
        fasta[labelA] = seqA
        fasta[labelB] = seqB
        acc[(labelA, labelB, f"{wt}{side}{pos}{mt}")].append((e["ddG"], e["ref"]))
        kept += 1

    # Average replicate measurements of the same (complex, mutation) -> one row, unique join key.
    #
    # The mean is kept, but it is NOT free of assumption: 609 (#Pdb, mutation) keys in SKEMPI
    # cover 1501 rows, 560 of them disagreeing on ddG by up to 5.37 kcal/mol, because
    # `Affinity_wt` is a per-row reference state rather than a per-complex constant. Averaging an
    # affinity-maturation ladder averages across different baselines. Measured on this build:
    # 362 of 585 averaged groups span more than one Reference, which true replicates would not.
    #
    # So the collapse is recorded rather than hidden -- see the sidecar written below. It is a
    # SIDECAR and not extra columns because `mulan/data.py` dispatches on column count: with
    # >4 columns, col3 becomes the label and col4.. are fed to the model as zero-shot score
    # features. Appending spread and n would silently become a 2-term input vector.
    rows = [(a, b, m, sum(d for d, _ in v) / len(v)) for (a, b, m), v in acc.items()]
    n_replicate = kept - len(rows)

    os.makedirs(OUTDIR, exist_ok=True)
    fa_path = os.path.join(OUTDIR, "wt_sequences.fasta")
    with open(fa_path, "w") as fh:
        for label, seq in sorted(fasta.items()):
            fh.write(f">{label}\n{seq}\n")
    tsv_path = os.path.join(OUTDIR, "single_point.tsv")
    with open(tsv_path, "w") as fh:
        for r in sorted(rows):
            fh.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]:.6f}\n")

    # Replicate provenance, joinable on the first three columns. Header-bearing, so it can never
    # be mistaken for a split file by a loader that keys on column count.
    rep_path = os.path.join(OUTDIR, "single_point_replicates.tsv")
    with open(rep_path, "w") as fh:
        fh.write("complexA\tcomplexB\tmutation\tn\tmean\tmin\tmax\tspread\tn_refs\n")
        for (a, b, m), v in sorted(acc.items()):
            d = [x for x, _ in v]
            fh.write(f"{a}\t{b}\t{m}\t{len(d)}\t{sum(d)/len(d):.6f}\t{min(d):.6f}\t"
                     f"{max(d):.6f}\t{max(d)-min(d):.6f}\t{len({r for _, r in v})}\n")

    if verbose:
        n = len(entries)
        print(f"curated single-point entries : {n}")
        print(f"kept (WT-validated)          : {kept}   coverage {100*kept/n:.1f}%")
        print(f"unique rows after replicate-avg: {len(rows)}   (merged {n_replicate} replicates)")
        print(f"dropped                      : {sum(drops.values())}")
        for k, v in drops.most_common():
            print(f"    {k:28s}: {v}")
        print(f"fasta labels (chain-groups)  : {len(fasta)}")
        print(f"distinct #Pdb complexes      : {len({r[0].split('_')[0] for r in rows})}")
        print(f"wrote {fa_path}")
        print(f"wrote {tsv_path}")
    return rows, fasta, drops


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    build(verbose=not ap.parse_args().quiet)
