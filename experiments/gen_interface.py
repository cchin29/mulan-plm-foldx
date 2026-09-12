"""Per-residue binding-interface indicator for the S1102 wild-type chains (C3).

For each WT chain, mark residues in contact with the partner chain (any heavy atom within
`--cutoff` Å of any partner heavy atom), **index-aligned to the AA fasta numbering** (same
residue selection as build_wt_fasta / gen_3di). Binding ΔΔG is dominated by the interface;
MuLAN consumes this as a learnable attention bias (config.interface_bias) so its pooling
focuses where the mutation can affect binding (see PROSTT5_STRUCTURE_OPTIONS.md, C3).

Output: one `[L]` float tensor per WT chain label (1.0 = interface residue, else 0.0),
saved as `<label>.pt`. Mutant ids reuse their WT chain's vector (interface is a structural
property of the complex).

Usage (from repo root, mulan venv):
    python experiments/gen_interface.py \
        --fasta   scratch/results/run1_s1102_ankh/wt_sequences.fasta \
        --pdb-dir scratch/skempi2/PDBs \
        --combos-csv scratch/skempi_v2.csv --pdb-ids scratch/pdb_ids.txt \
        --out-dir scratch/iface_prostt5 --cutoff 5.0
"""
import argparse
import os
import sys

import numpy as np
# MPS: route ops without a Metal kernel to CPU rather than erroring; must precede `import torch`.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_wt_fasta import load_chain_combos  # noqa: E402

from mulan.utils import parse_fasta  # noqa: E402


def parse_heavy_atoms(pdb_path):
    """{chain_id: {resnum: [xyz,...]}} for non-hydrogen ATOM records (first altloc per atom)."""
    chains = {}
    with open(pdb_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            element = line[76:78].strip()
            atom = line[12:16].strip()
            if element == "H" or (not element and atom.startswith("H")):
                continue
            ch = line[21]
            try:
                rn = int(line[22:26])
            except ValueError:
                continue
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            chains.setdefault(ch, {}).setdefault(rn, []).append(xyz)
    return chains


def interface_vector(chains, chain_id, partner_id, target_len, cutoff):
    """[target_len] float: 1.0 where a chain residue's heavy atoms are within `cutoff` of
    any partner heavy atom; index = resnum - 1."""
    res = chains.get(chain_id, {})
    partner = chains.get(partner_id, {})
    if not res or not partner:
        return None
    partner_xyz = np.array([a for atoms in partner.values() for a in atoms], float)  # [M,3]
    out = np.zeros(target_len, dtype=np.float32)
    for rn, atoms in res.items():
        if not (1 <= rn <= target_len):
            continue
        a = np.array(atoms, float)                              # [k,3]
        d2 = ((a[:, None, :] - partner_xyz[None, :, :]) ** 2).sum(-1)  # [k,M]
        if d2.min() <= cutoff * cutoff:
            out[rn - 1] = 1.0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--pdb-dir", required=True)
    ap.add_argument("--combos-csv", required=True)
    ap.add_argument("--pdb-ids", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cutoff", type=float, default=5.0, help="heavy-atom contact cutoff (Å)")
    args = ap.parse_args()

    aa = parse_fasta(args.fasta)
    pdb_ids = [l.strip() for l in open(args.pdb_ids) if l.strip()]
    combos = load_chain_combos(args.combos_csv, set(pdb_ids))
    os.makedirs(args.out_dir, exist_ok=True)

    made, skipped, frac = 0, [], []
    pdb_cache = {}
    for label in aa:
        pdbid, role = label.rsplit("_", 1)
        combo = combos.get(pdbid)
        pdb_path = os.path.join(args.pdb_dir, f"{pdbid}.pdb")
        if combo is None or not os.path.exists(pdb_path):
            skipped.append(label); continue
        chain_id = combo[0] if role == "A" else combo[1]
        partner_id = combo[1] if role == "A" else combo[0]
        if pdbid not in pdb_cache:
            pdb_cache[pdbid] = parse_heavy_atoms(pdb_path)
        vec = interface_vector(pdb_cache[pdbid], chain_id, partner_id, len(aa[label]), args.cutoff)
        if vec is None:
            skipped.append(label); continue
        torch.save(torch.from_numpy(vec), os.path.join(args.out_dir, f"{label}.pt"))
        made += 1
        frac.append(vec.mean())

    print(f"[gen_interface] wrote {made}/{len(aa)} interface vectors -> {args.out_dir} "
          f"(cutoff {args.cutoff} Å; mean interface fraction {np.mean(frac):.3f})", flush=True)
    if skipped:
        print(f"[gen_interface] skipped {len(skipped)}: {skipped[:10]}", flush=True)


if __name__ == "__main__":
    main()
