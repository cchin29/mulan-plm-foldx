#!/usr/bin/env python3
"""Scaffold: build cross-chain interface masks for the full-SKEMPI complexes, so the
S1102 attention→interface head-to-head (attention→interface question) can be re-run on the broader,
leakage-controlled full-SKEMPI set (by-complex / clustered heads).

Analog of experiments/interface_xattn/build_masks.py (S1102), reusing its heavy-atom
contact logic, but adapted to full-SKEMPI naming and multi-chain groups:

  fasta key = "PDB.grp1.grp2_role"  e.g. 1A22.A.B_A (single chain) or 1AHW.AB.C_AB
              (role "AB" = chains A+B concatenated, antibody heavy+light).
  bound PDB = scratch/foldx_skempi_full/work/<PDB>/<PDB>.pdb

For each complex the two roles (s1, s2) become the two sides of a [L1,L2] contact mask.
A role's residues are the concatenation of its chains (in role-string order), each chain's
residues aligned to consecutive FASTA positions (verified ≥90% AA identity). 108 complexes
have both a fasta entry and a bound PDB.

Outputs scratch/interface_masks_skempi_full/{s1}__{s2}.pt (bool [L1,L2]) + _summary.json,
mirroring the S1102 mask layout so score_interface.py consumes it unchanged.

Usage:  ./.venv/bin/python experiments/attention_interface/build_masks_skempi_full.py [--only 1A22,1AHW] [--dry-run]
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from Bio.PDB import PDBParser, NeighborSearch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "interface_xattn"))
from build_masks import AA3TO1, _residues, load_fasta  # noqa: E402  (reuse S1102 helpers)

WT_FASTA = ROOT / "scratch" / "skempi_full" / "wt_sequences.fasta"
WORK = ROOT / "scratch" / "foldx_skempi_full" / "work"
OUT = ROOT / "scratch" / "interface_masks_skempi_full"
CUTOFF = 8.0  # heavy-atom min-dist (matches the S1102 masks)


def role_chains(role: str) -> list[str]:
    """Role string -> chain ids. 'AB' -> ['A','B']; 'C' -> ['C']. (SKEMPI chain ids are
    single characters, so each character is a chain.)"""
    return list(role)


def aligned_residues(model, chains: list[str], seq: str, label: str):
    """Residues of `chains` (in order) mapped to FASTA indices: chain k's residues occupy
    the next block of `seq` after the prior chains. Returns [(residue, fasta_idx)] and the
    identity fraction; refuses (raises) if alignment is poor."""
    out, offset, checked, match = [], 0, 0, 0
    for cid in chains:
        if cid not in model:
            raise ValueError(f"{label}: chain {cid} absent from PDB")
        res = _residues(model[cid])
        for k, r in enumerate(res):
            idx = offset + k
            if idx < len(seq):
                checked += 1
                if AA3TO1[r.resname.strip().upper()] == seq[idx]:
                    match += 1
                out.append((r, idx))
        offset += len(res)
    frac = match / checked if checked else 0.0
    if frac < 0.90:
        raise ValueError(f"{label}: concat aligns poorly to FASTA ({match}/{checked}={frac:.2f})")
    return out, frac


def build_mask(res1, res2, L1: int, L2: int) -> torch.Tensor:
    """Heavy-atom min-dist [L1,L2] mask over pre-aligned (residue, fasta_idx) lists."""
    mask = torch.zeros((L1, L2), dtype=torch.bool)
    atoms2, owner = [], {}
    for r, j in res2:
        for a in r:
            if a.element != "H":
                atoms2.append(a); owner[id(a)] = j
    if not atoms2:
        return mask
    ns = NeighborSearch(atoms2)
    for r1, i in res1:
        hit = set()
        for a1 in r1:
            if a1.element == "H":
                continue
            for a2 in ns.search(a1.coord, CUTOFF, level="A"):
                hit.add(owner[id(a2)])
        for j in hit:
            mask[i, j] = True
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma-sep PDB ids to limit (debug)")
    ap.add_argument("--dry-run", action="store_true", help="validate + report, do not write .pt")
    args = ap.parse_args()

    fasta = load_fasta(str(WT_FASTA))
    # group fasta keys by complex: PDB -> {role: key}
    complexes: dict[str, dict[str, str]] = {}
    for key in fasta:
        base, _, role = key.rpartition("_")          # 1A22.A.B_A -> (1A22.A.B, _, A)
        pdb = base.split(".")[0]
        complexes.setdefault(pdb, {})[role] = key
    want = set(args.only.split(",")) if args.only else None
    parser = PDBParser(QUIET=True)
    OUT.mkdir(exist_ok=True)

    summary, n_ok, n_skip, n_err = {}, 0, 0, 0
    for pdb in sorted(complexes):
        if want and pdb not in want:
            continue
        roles = complexes[pdb]
        pdb_file = WORK / pdb / f"{pdb}.pdb"
        if len(roles) != 2 or not pdb_file.exists():
            n_skip += 1
            continue
        (r1name, s1), (r2name, s2) = sorted(roles.items())
        try:
            model = next(parser.get_structure(pdb, str(pdb_file)).get_models())
            L1, L2 = len(fasta[s1]), len(fasta[s2])
            res1, f1 = aligned_residues(model, role_chains(r1name), fasta[s1], s1)
            res2, f2 = aligned_residues(model, role_chains(r2name), fasta[s2], s2)
            mask = build_mask(res1, res2, L1, L2)
            i1, i2 = int(mask.any(1).sum()), int(mask.any(0).sum())
            summary[f"{s1}__{s2}"] = dict(L1=L1, L2=L2, iface1=i1, iface2=i2,
                                          align1=round(f1, 3), align2=round(f2, 3))
            if not args.dry_run:
                torch.save(mask, OUT / f"{s1}__{s2}.pt")
            n_ok += 1
            print(f"  {pdb}: {s1}__{s2}  L=({L1},{L2}) iface=({i1},{i2}) align=({f1:.2f},{f2:.2f})")
        except Exception as e:
            n_err += 1
            print(f"  {pdb}: ERR {e}")
    print(f"\nOK {n_ok} | skip {n_skip} (not 2-role / no PDB) | err {n_err}")
    if not args.dry_run:
        (OUT / "_summary.json").write_text(json.dumps(summary, indent=2))
        print("wrote", OUT)


if __name__ == "__main__":
    main()
