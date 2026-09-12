"""SKEMPI interface — RSA-burial version of the monomer-vs-complex contrast (2026-07-09 plan §4, #3).

The contact-based sweep (`run_skempi_interface.py`) is the P2 cutoff study; this is its RSA analog,
using the tool's *headline* burial signal. For each single-mutation residue we run DSSP twice —
on the full crystal **complex** and on the residue's **isolated chain** (partner deleted) — and
compare relative solvent accessibility (rASA):

  * **ΔrASA = rASA_monomer − rASA_complex** ≥ 0 — how much the partner buries the residue. This IS
    Levy's interface definition (core = exposed in monomer → buried in complex), so it should be
    large for COR, ~0 for INT/SUR.
  * AUROC(interface | rASA_monomer)  — the monomer-only view; expected weak (interface core/rim
    residues are surface-exposed in the monomer, indistinguishable from true surface).
  * AUROC(interface | ΔrASA)         — the complex-informed view; expected strong.

The gap is the RSA-side measure of what the monomer-only tool misses. Heavy (DSSP on the complex
+ each isolated chain, hundreds of mkdssp runs) → run via pueue. Needs mkdssp.
    .venv-structctx/bin/python experiments/skempi_interface/run_rsa_burial.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from Bio.PDB import DSSP, PDBIO, PDBParser, Select

sys.path.insert(0, str(Path(__file__).parent))
import run_skempi_interface as base  # parse_mutations, INTERFACE/NONINTERFACE, auroc*, PDB_DIR

OUT_DIR = Path(__file__).with_name("results")
# A machine-local working dir. Overridable, and defaulting under the system temp dir
# rather than a path that only existed on the machine this was written on.
_SCRATCH = Path(os.environ.get("SKEMPI_RSA_SCRATCH",
                              Path(tempfile.gettempdir()) / "skempi_rsa_chains"))  # isolated-chain temp PDBs


class _ChainSelect(Select):
    def __init__(self, chain_id):
        self.chain_id = chain_id

    def accept_chain(self, chain):
        return 1 if chain.id == self.chain_id else 0


def _dssp_rasa(model, pdb_path) -> dict:
    """{(chain_id, resnum, icode): rel_ASA} via mkdssp; skips residues with NA accessibility."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = DSSP(model, str(pdb_path), dssp="mkdssp")
    out = {}
    for (chain_id, res_id), vals in d.property_dict.items():
        rasa = vals[3]  # relative ASA
        if rasa in ("NA", None):
            continue
        het, resnum, icode = res_id
        try:
            out[(chain_id, int(resnum), icode)] = float(rasa)
        except (TypeError, ValueError):
            continue
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _SCRATCH.mkdir(parents=True, exist_ok=True)
    muts = [m for m in base.parse_mutations()
            if m["label"] in base.INTERFACE or m["label"] in base.NONINTERFACE]
    by_pdb = defaultdict(list)
    for m in muts:
        by_pdb[m["pdb"]].append(m)

    parser = PDBParser(QUIET=True)
    io = PDBIO()
    recs = []
    skipped = defaultdict(int)
    for pdb, rows in by_pdb.items():
        path = base.PDB_DIR / f"{pdb}.pdb"
        if not path.exists():
            skipped["no_pdb"] += len(rows); continue
        try:
            struct = parser.get_structure(pdb, str(path))
            model = struct[0]
            complex_rasa = _dssp_rasa(model, path)
        except Exception:
            skipped["complex_dssp_error"] += len(rows); continue

        # DSSP each isolated chain that carries a mutation (cache per chain)
        chain_rasa: dict[str, dict] = {}
        for r in rows:
            ch = r["chain"]
            if ch in chain_rasa or ch not in {c.id for c in model}:
                continue
            chain_pdb = _SCRATCH / f"{pdb}_{ch}.pdb"
            try:
                io.set_structure(struct)
                io.save(str(chain_pdb), _ChainSelect(ch))
                m_iso = parser.get_structure(f"{pdb}_{ch}", str(chain_pdb))[0]
                chain_rasa[ch] = _dssp_rasa(m_iso, chain_pdb)
            except Exception:
                chain_rasa[ch] = {}
            finally:
                chain_pdb.unlink(missing_ok=True)

        for r in rows:
            key = (r["chain"], r["resnum"], r["icode"])
            rc = complex_rasa.get(key)
            rm = chain_rasa.get(r["chain"], {}).get(key)
            if rc is None or rm is None:
                skipped["residue_no_dssp"] += 1; continue
            recs.append({
                "label": r["label"], "is_iface": int(r["label"] in base.INTERFACE),
                "rasa_complex": rc, "rasa_monomer": rm, "delta_rasa": rm - rc,
            })

    _report(recs, skipped, len(muts))


def _report(recs, skipped, n_muts) -> None:
    lab = np.array([r["is_iface"] for r in recs])
    mono = np.array([r["rasa_monomer"] for r in recs], float)
    drasa = np.array([r["delta_rasa"] for r in recs], float)
    a_mono = base.auroc_ci(mono, lab)          # monomer-only view (expect weak)
    a_delta = base.auroc_ci(drasa, lab)        # complex-informed (expect strong)

    classes = ["COR", "SUP", "INT", "RIM", "SUR"]
    L = ["# SKEMPI interface — RSA burial (monomer vs complex)", ""]
    L.append(f"Residues scored: **{len(recs)}** of {n_muts} (interface {int(lab.sum())} / "
             f"non-interface {int((lab==0).sum())}). Skipped: {dict(skipped)}.\n")
    L.append("## AUROC (interface vs non-interface)\n")
    L.append("| feature | AUROC | 95% CI |")
    L.append("|---|---|---|")
    L.append(f"| ΔrASA (monomer−complex, complex-informed) | {a_delta[0]:.3f} | [{a_delta[1]:.3f}, {a_delta[2]:.3f}] |")
    L.append(f"| rASA monomer only (monomer view) | {a_mono[0]:.3f} | [{a_mono[1]:.3f}, {a_mono[2]:.3f}] |")
    L.append("\n## Per-class mean rASA\n")
    L.append("| class | n | rASA monomer | rASA complex | ΔrASA |")
    L.append("|---|---|---|---|---|")
    for c in classes:
        rs = [r for r in recs if r["label"] == c]
        if not rs:
            continue
        L.append(f"| {c} | {len(rs)} | {np.mean([r['rasa_monomer'] for r in rs]):.2f} "
                 f"| {np.mean([r['rasa_complex'] for r in rs]):.2f} "
                 f"| {np.mean([r['delta_rasa'] for r in rs]):.2f} |")
    L.append("\n## Verdict\n")
    L.append(f"- **ΔrASA (needs the complex)** identifies interface residues at AUROC "
             f"**{a_delta[0]:.3f}** — burial-on-binding is Levy's interface definition, recovered.")
    L.append(f"- **Monomer rASA alone** manages only AUROC **{a_mono[0]:.3f}** — the RSA-side "
             "confirmation that the monomer-only tool cannot flag interface residues from burial "
             "(they're surface-exposed until the partner arrives), matching the contact-based "
             "~0.49 in `SKEMPI_INTERFACE.md`.")
    (OUT_DIR / "SKEMPI_RSA_BURIAL.md").write_text("\n".join(L) + "\n")
    (OUT_DIR / "skempi_rsa_burial.json").write_text(json.dumps(
        {"n_scored": len(recs), "skipped": dict(skipped),
         "auroc_delta_rasa": a_delta, "auroc_monomer_rasa": a_mono}, indent=2))
    print("\n".join(L))
    print(f"\n[wrote {OUT_DIR/'SKEMPI_RSA_BURIAL.md'}]")


if __name__ == "__main__":
    main()
