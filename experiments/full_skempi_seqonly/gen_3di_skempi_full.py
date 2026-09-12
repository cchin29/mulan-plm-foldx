#!/usr/bin/env python3
"""WT 3Di fasta for the full-SKEMPI single-point set, aligned to build_skempi_full.py's
CONTIGUOUS multi-chain group sequences (for the SaProt structure-aware arm).

The stock experiments/gen_3di.py assumes the S1102 label scheme (role A/B, single chain per
partner, token placed at author-resnum-1). Our labels are "<CODE>_<group>" where the group is
the real chain letters (possibly multi-chain), and our AA sequence is the CONTIGUOUS ATOM order
from the <CODE>.mapping file (author resseq is gappy) -- so 3Di must be aligned the same way:
per chain, tokens placed by the mapping's seq-index, chains concatenated in group order.

    .venv/bin/python experiments/full_skempi_seqonly/gen_3di_skempi_full.py \
        --fasta scratch/skempi_full/wt_sequences.fasta \
        --out   scratch/skempi_full/wt_3di.fasta
"""
from __future__ import annotations

import argparse
import collections
import os
import sys

import numpy as np
import mini3di

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_skempi_full as b  # noqa: E402  (PDBDIR, THREE_TO_ONE)
sys.path.insert(0, os.path.join(b.ROOT, "mulan"))
from mulan.utils import parse_fasta  # noqa: E402

GAP_3DI = "d"   # valid Foldseek 3Di state for positions with no encodable backbone


def mapping_order(code):
    """<CODE>.mapping -> {chain: [author_resid(str), ...] in seq-index order}. The author-resid
    keeps its insertion code (e.g. '116A'), so it matches the PDB ATOM residue id exactly."""
    order = collections.OrderedDict()
    with open(os.path.join(b.PDBDIR, f"{code.upper()}.mapping")) as fh:
        for line in fh:
            p = line.split()
            if len(p) < 4:
                continue
            order.setdefault(p[1], []).append(p[2])   # resSeq(+iCode) as string
    return order


def parse_backbone(pdb_path, chain_id):
    """{author_resid(str): {N/CA/C/CB: xyz}} for a chain, first occurrence wins. Resid = resSeq
    + iCode (cols 23-27) stripped, matching the <CODE>.mapping author-resid keys."""
    res = {}
    with open(pdb_path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.startswith("ATOM") or line[21] != chain_id:
                continue
            atom = line[12:16].strip()
            if atom not in ("N", "CA", "C", "CB"):
                continue
            rid = line[22:27].strip()                 # resSeq(23-26) + iCode(27)
            res.setdefault(rid, {}).setdefault(
                atom, (float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return res


def chain_3di_contiguous(pdb_path, chain_id, resseq_order, encoder):
    """3Di string of length len(resseq_order), token at contiguous seq-index i for the
    residue whose author resseq is resseq_order[i]; missing backbone -> GAP_3DI."""
    coords = parse_backbone(pdb_path, chain_id)
    keep_idx, keep_rn = [], []
    for i, rn in enumerate(resseq_order):
        c = coords.get(rn)
        if c and all(a in c for a in ("N", "CA", "C")):
            keep_idx.append(i)
            keep_rn.append(rn)
    out = [GAP_3DI] * len(resseq_order)
    if keep_rn:
        def arr(atom):
            return np.array([coords[rn].get(atom, (np.nan,) * 3) for rn in keep_rn], float)
        states = encoder.encode_atoms(ca=arr("CA"), cb=arr("CB"), n=arr("N"), c=arr("C"))
        s3di = encoder.build_sequence(states).lower()
        for i, ch in zip(keep_idx, s3di):
            out[i] = ch
    return "".join(out), len(keep_rn)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fasta", default=os.path.join(b.OUTDIR, "wt_sequences.fasta"))
    ap.add_argument("--out", default=os.path.join(b.OUTDIR, "wt_3di.fasta"))
    a = ap.parse_args()

    aa = parse_fasta(a.fasta)
    encoder = mini3di.Encoder()
    order_cache = {}
    lines, made, skipped, mism = [], 0, [], []
    for label, seq in aa.items():
        cid, group = label.split("_", 1)            # cid = "<code>.<g1>.<g2>"
        code = cid.split(".")[0]
        if code not in order_cache:
            try:
                order_cache[code] = mapping_order(code)
            except FileNotFoundError:
                order_cache[code] = None
        chains = order_cache[code]
        pdb_path = os.path.join(b.PDBDIR, f"{code.upper()}.pdb")
        if chains is None or not os.path.exists(pdb_path):
            skipped.append(label); continue
        parts, ok = [], True
        for ch in group:
            if ch not in chains:
                ok = False; break
            s3di, _ = chain_3di_contiguous(pdb_path, ch, chains[ch], encoder)
            parts.append(s3di)
        if not ok:
            skipped.append(label); continue
        full = "".join(parts)
        if len(full) != len(seq):
            mism.append((label, len(full), len(seq))); continue
        lines.append(f">{label}\n{full}\n")
        made += 1

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        fh.writelines(lines)
    print(f"[3di] wrote {made}/{len(aa)} -> {a.out}")
    if skipped:
        print(f"[3di] skipped {len(skipped)}: {skipped[:8]}{'...' if len(skipped) > 8 else ''}")
    if mism:
        print(f"[3di] LENGTH MISMATCH {len(mism)} (not written): {mism[:8]}")


if __name__ == "__main__":
    main()
