"""Extract per-residue 3Di structure tokens for the S1102 wild-type chains.

For run6b/6c (PLAN_PROSTT5_STRUCTURE_v2.md): convert each WT chain's backbone in the
SKEMPI 2.0 cleaned PDBs into Foldseek-3Di tokens, **index-aligned to the same residue
numbering `build_wt_fasta.py` uses** so the 3Di string is per-residue parallel to the
AA `wt_sequences.fasta`. 3Di is computed with `mini3di` (pure-Python, Foldseek-3Di
compatible) directly from N/CA/C/CB coordinates, which lets us scatter each token to
its `auth_seq_id - 1` position and gap-fill missing-structure positions — exactly how
the AA fasta is built (`range(1, max_num+1)`, gaps -> 'X').

Output: a 3Di FASTA with one lowercase 3Di string per WT label (e.g. `1A22_A`), each of
length == the matching AA sequence. ProstT5 fold-mode (`<fold2AA>`) consumes lowercase
3Di. Mutant ids are NOT emitted: a single-point mutation leaves the backbone ~unchanged,
so every mutant reuses its WT chain's 3Di (handled at concat time).

Usage (from repo root, mulan venv):
    python experiments/gen_3di.py \
        --fasta   scratch/results/run1_s1102_ankh/wt_sequences.fasta \
        --pdb-dir scratch/skempi2/PDBs \
        --combos-csv scratch/skempi_v2.csv --pdb-ids scratch/pdb_ids.txt \
        --out     scratch/3di/wt_3di.fasta
"""
import argparse
import sys
import os

import numpy as np
import mini3di

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_wt_fasta import load_chain_combos  # noqa: E402

from mulan.utils import parse_fasta  # noqa: E402

# Placeholder 3Di state for positions with no encodable backbone (the AA channel already
# marks these as 'X'). Must be a VALID Foldseek/ProstT5 3Di letter: an invalid token (e.g.
# 'x') maps to <unk>, and consecutive <unk>s collapse into one token, breaking the
# per-residue length alignment. 'd' is a common, valid state; these gaps are ~0.04% of
# residues so the exact fill is immaterial — only length parity matters.
GAP_3DI = "d"


def parse_chain_atoms(pdb_path, chain_id):
    """Per-residue N/CA/C/CB coords keyed by auth_seq_id, first occurrence wins
    (skips alt locs / duplicates) — the same residue selection as build_wt_fasta."""
    res = {}
    order = []
    with open(pdb_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[21] != chain_id:
                continue
            atom = line[12:16].strip()
            if atom not in ("N", "CA", "C", "CB"):
                continue
            try:
                rn = int(line[22:26])
            except ValueError:
                continue
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            if rn not in res:
                res[rn] = {}
                order.append(rn)
            res[rn].setdefault(atom, xyz)
    return res, order


def chain_3di(pdb_path, chain_id, target_len, encoder):
    """3Di string of length `target_len`, token at index (resnum-1), gaps -> GAP_3DI."""
    res, order = parse_chain_atoms(pdb_path, chain_id)
    # residues mini3di can encode (need backbone N, CA, C); CB optional (NaN for Gly)
    keep = [rn for rn in order if all(a in res[rn] for a in ("N", "CA", "C"))
            and 1 <= rn <= target_len]
    if not keep:
        return None, 0
    def arr(atom):
        return np.array([res[rn].get(atom, (np.nan, np.nan, np.nan)) for rn in keep], float)
    states = encoder.encode_atoms(ca=arr("CA"), cb=arr("CB"), n=arr("N"), c=arr("C"))
    s3di = encoder.build_sequence(states).lower()
    out = [GAP_3DI] * target_len
    for rn, ch in zip(keep, s3di):
        out[rn - 1] = ch
    return "".join(out), len(keep)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fasta", required=True, help="AA wild-type sequences FASTA (for labels + lengths)")
    ap.add_argument("--pdb-dir", required=True)
    ap.add_argument("--combos-csv", required=True, help="skempi_v2.csv (partner-role -> chain mapping)")
    ap.add_argument("--pdb-ids", required=True, help="pdb_ids.txt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    aa = parse_fasta(args.fasta)
    pdb_ids = [l.strip() for l in open(args.pdb_ids) if l.strip()]
    combos = load_chain_combos(args.combos_csv, set(pdb_ids))
    encoder = mini3di.Encoder()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    lines, made, skipped, mismatches = [], 0, [], []
    for label in aa:                                # e.g. 1A22_A / 1A22_B
        pdbid, role = label.rsplit("_", 1)          # role in {A, B}
        combo = combos.get(pdbid)
        if combo is None:
            skipped.append(label); continue
        chain_id = combo[0] if role == "A" else combo[1]
        pdb_path = os.path.join(args.pdb_dir, f"{pdbid}.pdb")
        if not os.path.exists(pdb_path):
            skipped.append(label); continue
        target = len(aa[label])
        s3di, n_enc = chain_3di(pdb_path, chain_id, target, encoder)
        if s3di is None:
            skipped.append(label); continue
        if len(s3di) != target:                      # must stay per-residue parallel to AA
            mismatches.append((label, len(s3di), target)); continue
        gaps = s3di.count(GAP_3DI)
        lines.append(f">{label}\n{s3di}\n")
        made += 1
        if made <= 3 or gaps:
            print(f"  {label}: chain {chain_id}, len {target}, encoded {n_enc}, gaps {gaps}", flush=True)

    with open(args.out, "w") as f:
        f.writelines(lines)
    print(f"\n[gen_3di] wrote {made}/{len(aa)} 3Di sequences -> {args.out}", flush=True)
    if skipped:
        print(f"[gen_3di] skipped {len(skipped)}: {skipped[:10]}{'...' if len(skipped) > 10 else ''}", flush=True)
    if mismatches:
        print(f"[gen_3di] LENGTH MISMATCH (not written) {len(mismatches)}: {mismatches[:10]}", flush=True)


if __name__ == "__main__":
    main()
