"""A1 §2 alignment test — the #1 correctness risk (PDB numbering <-> FASTA index).

Asserts, for every S1102 single-point row, that the residue at the mutation position in
BOTH the FASTA sequence (seq[pos-1]) and the resolved PDB chain (ATOM resnum==pos) equals
the mutation's WT amino acid. A single off-by-one or wrong-chain anywhere fails the run.
Also sanity-checks that each emitted mask has the right shape and a non-empty interface.

Pure-read; no mulan/ imports, no GPU. Run from repo root with the main venv:
    ./.venv/bin/python -m pytest experiments/interface_xattn/test_build_masks.py -q
    ./.venv/bin/python experiments/interface_xattn/test_build_masks.py        # plain-run fallback
"""
import os

import torch
from Bio.PDB import PDBParser

import importlib.util

_here = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location("a1_build_masks", os.path.join(_here, "build_masks.py"))
bm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bm)

P = bm.repo_paths()


def _rows():
    out = []
    for ln in open(P["s1102"]):
        p = ln.rstrip("\n").split("\t")
        if len(p) < 3 or "," in p[2]:
            continue
        out.append((p[0], p[1], p[2]))
    return out


def test_fasta_and_pdb_numbering_match_wt():
    fasta = bm.load_fasta(P["fasta"])
    groups = bm.load_skempi_groups(P["skempi"])
    parser = PDBParser(QUIET=True)
    models = {}  # pdb -> Bio model (cache)
    n = 0
    for s1, s2, mut in _rows():
        wt, role, pos = mut[0], mut[1], int(mut[2:-1])
        seq = fasta[s1 if role == "A" else s2]
        assert 1 <= pos <= len(seq), f"{mut}: pos out of FASTA range for {s1}/{s2}"
        assert seq[pos - 1] == wt, f"{mut}: FASTA[{pos-1}]={seq[pos-1]!r} != wt {wt!r}"

        pdb = s1.split("_")[0]
        if pdb not in models:
            path = f"{P['work']}/{pdb}/{pdb}.pdb"
            models[pdb] = next(parser.get_structure(pdb, path).get_models())
        model = models[pdb]
        g = groups[pdb][0] if role == "A" else groups[pdb][1]
        # exactly one chain in the role group carries resnum==pos with the WT aa
        hits = []
        for cid in g:
            if cid not in model:
                continue
            for r in bm._residues(model[cid]):
                if r.id[1] == pos and bm.AA3TO1[r.resname.strip().upper()] == wt:
                    hits.append(cid)
        assert len(hits) == 1, f"{pdb} {mut}: expected 1 PDB chain with resnum {pos}=={wt}, got {hits}"
        n += 1
    assert n == 1100, f"expected 1100 single-point rows, checked {n}"


def test_masks_shape_and_nonempty():
    """Each emitted mask (if generated) is [L1,L2] with a non-empty interface both sides."""
    fasta = bm.load_fasta(P["fasta"])
    complexes = bm.load_complexes(P["s1102"])
    out = P["out"]
    if not os.path.isdir(out) or not any(f.endswith(".pt") for f in os.listdir(out)):
        import pytest
        pytest.skip("no masks generated yet — run build_masks.py first")
    for pdb, (s1, s2) in complexes.items():
        fp = f"{out}/{s1}__{s2}.pt"
        if not os.path.exists(fp):
            continue
        m = torch.load(fp)
        assert m.dtype == torch.bool and m.shape == (len(fasta[s1]), len(fasta[s2])), \
            f"{s1}__{s2}: bad shape {tuple(m.shape)} vs {(len(fasta[s1]), len(fasta[s2]))}"
        assert m.any(dim=1).sum() > 0 and m.any(dim=0).sum() > 0, f"{s1}__{s2}: empty interface"


if __name__ == "__main__":
    test_fasta_and_pdb_numbering_match_wt()
    print("OK  fasta+pdb numbering match WT for all 1100 rows")
    try:
        test_masks_shape_and_nonempty()
        print("OK  mask shapes + non-empty interfaces")
    except Exception as e:  # pytest.skip or assertion
        print("mask check:", e)
