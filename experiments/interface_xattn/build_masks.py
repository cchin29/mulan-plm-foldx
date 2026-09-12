"""A1 step 2 — build per-complex interface contact masks for the S1102 set.

Writes a boolean [L1, L2] tensor per complex to `scratch/interface_masks/{s1}__{s2}.pt`
(gitignored scratch), where L1/L2 are the FASTA lengths of role-A / role-B and
mask[i, j] is True iff residue i of role-A and residue j of role-B are in contact
under the chosen cutoff. Index i/j is the FASTA sequence position (0-based), which the
alignment probe confirmed equals PDB ATOM resnum-1 for all 1100 S1102 rows (single
chain per role, no multi-chain groups in the filtered set).

Reuses the FoldX Stage-2 assets verbatim (no new structure prep):
  - role->real-chain via SKEMPI `Mutation(s)_cleaned` group field (scratch/skempi_v2.csv)
  - the SKEMPI-cleaned bound PDBs at scratch/foldx_s1102/work/<PDB>/<PDB>.pdb

Contact modes:
  heavy (default): min heavy-atom distance <= --cutoff (8 A) — standard interface def.
  cb            : Cb-Cb distance <= --cb-cutoff (5 A); Gly falls back to Ca.

Self-check (the plan's §2 #1 risk): for every residue present in BOTH the PDB chain and
the FASTA, assert the amino acids agree; residues present in one but not the other stay
False and are counted/logged (a real gap must not silently shift numbering). Any AA
mismatch aborts that complex loudly rather than emitting a misaligned mask.

Read-only w.r.t. mulan/ and the running queue: reads data + scratch, writes only scratch.
Run from repo root with the main venv:
    ./.venv/bin/python experiments/interface_xattn/build_masks.py
    ./.venv/bin/python experiments/interface_xattn/build_masks.py --mode cb --cb-cutoff 5
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import collections

import torch
from Bio.PDB import PDBParser, NeighborSearch

AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


VARIANT_SUFFIX = {"real": "", "dense": "_dense", "shuffle": "_shuffled"}


def repo_paths(variant="real"):
    root = os.getcwd()
    return {
        "s1102": f"{root}/experiments/embedding_sweep/data/S1102_filtered.tsv",
        "fasta": f"{root}/experiments/embedding_sweep/data/wt_sequences.fasta",
        "skempi": f"{root}/scratch/skempi_v2.csv",
        "work": f"{root}/scratch/foldx_s1102/work",
        "out": f"{root}/scratch/interface_masks{VARIANT_SUFFIX[variant]}",
    }


def load_fasta(path):
    seqs, key = {}, None
    for ln in open(path):
        ln = ln.rstrip("\n")
        if ln.startswith(">"):
            key = ln[1:].strip()
            seqs[key] = ""
        elif key:
            seqs[key] += ln.strip()
    return seqs


def load_skempi_groups(path):
    """pdb -> (g1, g2) real-chain groups from the SKEMPI pdb field (e.g. 1CSE_E_I)."""
    groups = {}
    for row in csv.reader(open(path, newline=""), delimiter=";"):
        if not row or not row[0]:
            continue
        parts = row[0].split("_")
        if len(parts) < 3:
            continue
        groups.setdefault(parts[0], (parts[1], parts[2]))
    return groups


def load_complexes(s1102_path):
    """Unique complexes -> (s1_label, s2_label); single-point rows define the set."""
    seen = {}
    for ln in open(s1102_path):
        p = ln.rstrip("\n").split("\t")
        if len(p) < 3:
            continue
        seen.setdefault(p[0].split("_")[0], (p[0], p[1]))
    return seen


def _residues(chain):
    """Standard amino-acid residues of a Biopython chain, in ATOM order."""
    return [r for r in chain if r.id[0].strip() == "" and r.resname.strip().upper() in AA3TO1]


def _rep_cb(residue):
    """Cb atom, or Ca for glycine / missing Cb."""
    if "CB" in residue:
        return residue["CB"]
    return residue["CA"] if "CA" in residue else None


def resolve_chain(model, group, seq, label):
    """Pick the single real chain of `group` whose ATOM sequence aligns to the FASTA `seq`.

    S1102 has one real chain per role (probe: 0 multi-chain groups), but `group` can still
    list several ids; choose the one whose residues best match the FASTA at resnum-1.
    Returns (chain, n_checked, n_match) for the winner, or raises if none present.
    """
    best = None
    for cid in group:
        if cid not in model:
            continue
        chain = model[cid]
        checked = match = 0
        for r in _residues(chain):
            idx = r.id[1] - 1  # ATOM resnum -> FASTA index
            if 0 <= idx < len(seq):
                checked += 1
                if AA3TO1[r.resname.strip().upper()] == seq[idx]:
                    match += 1
        score = (match, checked)
        if best is None or score > best[1]:
            best = (chain, score)
    if best is None:
        raise ValueError(f"none of chains {group!r} present for {label}")
    chain, (match, checked) = best
    if checked and match / checked < 0.9:
        raise ValueError(
            f"{label}: chain {chain.id} aligns poorly to FASTA ({match}/{checked}); "
            "numbering mismatch — refusing to emit a misaligned mask"
        )
    return chain, checked, match


def build_mask(chain1, chain2, L1, L2, mode, cutoff, cb_cutoff):
    """Boolean [L1,L2] contact mask; residue r indexes FASTA position r-1."""
    mask = torch.zeros((L1, L2), dtype=torch.bool)
    res1 = [r for r in _residues(chain1) if 0 <= r.id[1] - 1 < L1]
    res2 = [r for r in _residues(chain2) if 0 <= r.id[1] - 1 < L2]
    if mode == "cb":
        pts2 = [(r, _rep_cb(r)) for r in res2]
        pts2 = [(r, a) for r, a in pts2 if a is not None]
        ns = NeighborSearch([a for _, a in pts2])
        owner = {id(a): r for r, a in pts2}
        for r1 in res1:
            a1 = _rep_cb(r1)
            if a1 is None:
                continue
            i = r1.id[1] - 1
            for a2 in ns.search(a1.coord, cb_cutoff, level="A"):
                j = owner[id(a2)].id[1] - 1
                mask[i, j] = True
    else:  # heavy-atom min distance
        atoms2, owner = [], {}
        for r in res2:
            for a in r:
                if a.element != "H":
                    atoms2.append(a)
                    owner[id(a)] = r.id[1] - 1
        ns = NeighborSearch(atoms2)
        for r1 in res1:
            i = r1.id[1] - 1
            hit = set()
            for a1 in r1:
                if a1.element == "H":
                    continue
                for a2 in ns.search(a1.coord, cutoff, level="A"):
                    hit.add(owner[id(a2)])
            for j in hit:
                mask[i, j] = True
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["heavy", "cb"], default="heavy")
    ap.add_argument("--cutoff", type=float, default=8.0, help="heavy-atom min-dist cutoff (A)")
    ap.add_argument("--cb-cutoff", type=float, default=5.0, help="Cb-Cb cutoff (A) for --mode cb")
    ap.add_argument("--variant", choices=["real", "dense", "shuffle"], default="real",
                    help="real=8A interface mask; dense=all-pairs (no interface prior); "
                         "shuffle=column-permuted real mask (same per-row density, wrong pairing). "
                         "dense/shuffle are the §5 capacity/prior controls; each writes a distinct "
                         "dir (interface_masks / _dense / _shuffled).")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for --variant shuffle")
    ap.add_argument("--only", default=None, help="comma-sep PDB ids to limit (debug)")
    ap.add_argument("--dry-run", action="store_true", help="validate + report, do not write .pt")
    args = ap.parse_args()

    P = repo_paths(args.variant)
    gen = torch.Generator().manual_seed(args.seed)  # shuffle reproducibility (sorted iteration)
    fasta = load_fasta(P["fasta"])
    groups = load_skempi_groups(P["skempi"])
    complexes = load_complexes(P["s1102"])
    if args.only:
        want = set(args.only.split(","))
        complexes = {k: v for k, v in complexes.items() if k in want}
    os.makedirs(P["out"], exist_ok=True)
    parser = PDBParser(QUIET=True)

    summary = {}
    n_ok = n_err = 0
    for pdb, (s1, s2) in sorted(complexes.items()):
        try:
            g1, g2 = groups[pdb]
            L1, L2 = len(fasta[s1]), len(fasta[s2])
            pdb_path = f"{P['work']}/{pdb}/{pdb}.pdb"
            model = next(parser.get_structure(pdb, pdb_path).get_models())
            c1, chk1, m1 = resolve_chain(model, g1, fasta[s1], f"{s1}")
            c2, chk2, m2 = resolve_chain(model, g2, fasta[s2], f"{s2}")
            mask = build_mask(c1, c2, L1, L2, args.mode, args.cutoff, args.cb_cutoff)
            # §5 control variants derived from the real mask (keeps the alignment self-check).
            if args.variant == "dense":
                # all-pairs cross-attention: no interface prior (padded rows/cols masked by the model)
                mask = torch.ones((L1, L2), dtype=torch.bool)
            elif args.variant == "shuffle":
                # permute chain-2 columns: preserves per-row density + column marginal, breaks pairing
                mask = mask[:, torch.randperm(L2, generator=gen)]
            iface1 = int(mask.any(dim=1).sum())
            iface2 = int(mask.any(dim=0).sum())
            summary[f"{s1}__{s2}"] = {
                "pdb": pdb, "chains": [c1.id, c2.id], "L": [L1, L2],
                "contacts": int(mask.sum()), "iface_res": [iface1, iface2],
                "align_match": [f"{m1}/{chk1}", f"{m2}/{chk2}"],
            }
            if iface1 == 0 or iface2 == 0:
                print(f"  WARN {pdb} {c1.id}/{c2.id}: empty interface (0 contacts)", file=sys.stderr)
            if not args.dry_run:
                torch.save(mask, f"{P['out']}/{s1}__{s2}.pt")
            n_ok += 1
        except Exception as e:  # noqa: BLE001 — report and continue, don't abort the batch
            n_err += 1
            summary[f"{s1}__{s2}"] = {"pdb": pdb, "error": repr(e)}
            print(f"  ERROR {pdb}: {e}", file=sys.stderr)

    if not args.dry_run:
        with open(f"{P['out']}/_summary.json", "w") as fh:
            json.dump(summary, fh, indent=1)
    dens = [v["contacts"] for v in summary.values() if "contacts" in v]
    print(f"complexes ok={n_ok} err={n_err}  variant={args.variant}  mode={args.mode} "
          f"cutoff={args.cutoff if args.mode=='heavy' else args.cb_cutoff}A")
    if dens:
        dens.sort()
        print(f"contacts/complex: min={dens[0]} median={dens[len(dens)//2]} max={dens[-1]}")
    print(f"out: {P['out']}" + ("  (dry-run: nothing written)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
