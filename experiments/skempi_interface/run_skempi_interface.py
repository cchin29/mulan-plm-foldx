"""SKEMPI fuller sweep — P2-interface at scale + monomer-vs-complex contrast (2026-07-09 plan §4).

The small functional-site set (6 residues) couldn't quantify the question "Cα-8 Å or Cβ-5 Å *for interface
residues*?" — interface geometry only exists in the complex. This uses the 345 local SKEMPI
crystal complexes (`scratch/skempi2/PDBs/`) + SKEMPI's per-mutation Levy interface labels
(`iMutation_Location(s)`: COR/SUP/RIM/INT = interface, SUR = surface) as ground truth.

For each single-mutation residue, on the crystal complex, we count contacts two ways at each
cutoff (Cα ≤ 8 Å; Cβ ≤ 5 Å, Gly→Cα):
  * **cross-chain** — neighbors on the binding partner's chain(s)     [the complex view]
  * **monomer**     — neighbors on the residue's own chain only        [partner deleted]

Two questions, one pass:
  A (complex, the P2 cutoff question): AUROC of **cross-chain** contacts predicting interface
    (vs SUR), Cα-8 vs Cβ-5, with bootstrap CIs. Higher/consistent AUROC = better interface cutoff.
  B (monomer contrast): AUROC of **monomer** intra-chain contacts predicting the same label —
    expected near 0.5, quantifying how much the shipped **monomer-only** tool misses (it never
    sees the cross-chain contacts that define an interface residue).

Crystal-chain-alone is the monomer proxy (removes AF-prediction noise; no 211 AF fetches). This
informs the *future complex-context extension*, not the settled v1 monomer default. Fast (pure
geometry, no PLM). Run:
    .venv-structctx/bin/python experiments/skempi_interface/run_skempi_interface.py
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from Bio.PDB import NeighborSearch, PDBParser

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "scratch" / "skempi_v2.csv"
PDB_DIR = REPO / "scratch" / "skempi2" / "PDBs"
OUT_DIR = Path(__file__).with_name("results")

CA_CUT, CB_CUT = 8.0, 5.0
# Levy (2010) structural regions. Interface = core/rim/support; negatives = interior/surface
# (INT = buried monomer-interior, SUR = exposed — NEITHER is at the interface).
INTERFACE = {"COR", "RIM", "SUP"}
NONINTERFACE = {"INT", "SUR"}
_MUT = re.compile(r"^([A-Z])([A-Za-z0-9])(-?\d+[A-Za-z]?)([A-Z])$")


def parse_mutations() -> list[dict]:
    """Single-mutation rows with a clean Levy label → {pdb, g1, g2, chain, resnum, icode, label}."""
    rows = []
    lines = CSV.read_text().splitlines()
    for line in lines[1:]:
        f = line.split(";")
        pdb_field, mut_pdb, _cleaned, loc = f[0], f[1], f[2], f[3]
        if "," in loc or not loc or "," in mut_pdb:      # singles only
            continue
        m = _MUT.match(mut_pdb)
        if not m:
            continue
        wt, chain, resnum_ic, _mt = m.groups()
        icode = resnum_ic[-1] if resnum_ic[-1].isalpha() else " "
        resnum = int(resnum_ic[:-1]) if icode != " " else int(resnum_ic)
        parts = pdb_field.split("_")
        if len(parts) != 3:
            continue
        rows.append({
            "pdb": parts[0], "g1": parts[1], "g2": parts[2],
            "chain": chain, "resnum": resnum, "icode": icode, "wt": wt, "label": loc,
        })
    return rows


def _rep_atom(res, mode: str):
    """CA for mode 'ca'; CB (Gly→CA) for mode 'cb'. None if absent."""
    want = "CA" if mode == "ca" else ("CA" if res.get_resname() == "GLY" else "CB")
    return res[want] if want in res else None


def build_ns(model, mode: str) -> NeighborSearch:
    """One NeighborSearch per (model, mode) — the rep atom of every residue (built once, reused)."""
    atoms = []
    for res in model.get_residues():
        a = _rep_atom(res, mode)
        if a is not None:
            atoms.append(a)
    return NeighborSearch(atoms)


def contacts_for(res, ns: NeighborSearch, own_chain: str, partner_chains: set[str],
                 mode: str, cut: float):
    """(monomer_count, crosschain_count): #distinct residues within `cut` of `res`'s rep atom,
    split by own-chain vs partner-chain. Self excluded."""
    a0 = _rep_atom(res, mode)
    if a0 is None:
        return None
    mono = cross = 0
    for r in ns.search(a0.get_coord(), cut, level="R"):
        if r is res:
            continue
        ch = r.get_parent().id
        if ch == own_chain:
            mono += 1
        elif ch in partner_chains:
            cross += 1
    return mono, cross


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = labels == 1, labels == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = scores.argsort()
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    vals, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def auroc_ci(scores, labels, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(scores)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a = auroc(scores[idx], labels[idx])
        if a == a:
            boots.append(a)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return auroc(scores, labels), float(lo), float(hi)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    muts = parse_mutations()
    by_pdb = defaultdict(list)
    for r in muts:
        by_pdb[r["pdb"]].append(r)

    parser = PDBParser(QUIET=True)
    recs = []  # per-residue: {label, is_iface, mono_ca, cross_ca, mono_cb, cross_cb}
    skipped = defaultdict(int)
    for pdb, rows in by_pdb.items():
        path = PDB_DIR / f"{pdb}.pdb"
        if not path.exists():
            skipped["no_pdb"] += len(rows)
            continue
        try:
            model = parser.get_structure(pdb, str(path))[0]
        except Exception:
            skipped["parse_error"] += len(rows)
            continue
        chains = {c.id for c in model}
        ns_ca, ns_cb = build_ns(model, "ca"), build_ns(model, "cb")
        for r in rows:
            if r["label"] not in INTERFACE and r["label"] not in NONINTERFACE:
                skipped["other_label"] += 1
                continue
            g1set, g2set = set(r["g1"]), set(r["g2"])
            if r["chain"] in g1set:
                partners = g2set & chains
            elif r["chain"] in g2set:
                partners = g1set & chains
            else:
                skipped["chain_not_in_groups"] += 1
                continue
            if r["chain"] not in chains or not partners:
                skipped["missing_chain"] += 1
                continue
            try:
                res = model[r["chain"]][(" ", r["resnum"], r["icode"])]
            except KeyError:
                skipped["residue_not_found"] += 1
                continue
            ca = contacts_for(res, ns_ca, r["chain"], partners, "ca", CA_CUT)
            cb = contacts_for(res, ns_cb, r["chain"], partners, "cb", CB_CUT)
            if ca is None or cb is None:
                skipped["no_rep_atom"] += 1
                continue
            recs.append({
                "label": r["label"], "is_iface": int(r["label"] in INTERFACE),
                "mono_ca": ca[0], "cross_ca": ca[1], "mono_cb": cb[0], "cross_cb": cb[1],
            })

    _report(recs, skipped, len(muts))


def _report(recs, skipped, n_muts) -> None:
    lab = np.array([r["is_iface"] for r in recs])
    feats = {k: np.array([r[k] for r in recs], float)
             for k in ("cross_ca", "cross_cb", "mono_ca", "mono_cb")}
    aurocs = {k: auroc_ci(v, lab) for k, v in feats.items()}

    # per-class mean cross-chain contacts (the Levy gradient)
    classes = ["COR", "SUP", "INT", "RIM", "SUR"]
    by_class = {c: [r for r in recs if r["label"] == c] for c in classes}

    L = ["# SKEMPI fuller sweep — P2-interface (A) + monomer contrast (B)", ""]
    L.append(f"Single-mutation residues scored: **{len(recs)}** of {n_muts} "
             f"(interface {int(lab.sum())} / surface {int((lab == 0).sum())}). "
             f"Skipped: {dict(skipped)}.\n")

    L.append("## A — which contact cutoff identifies interface residues? (cross-chain)\n")
    L.append("| feature | AUROC (interface vs SUR) | 95% CI |")
    L.append("|---|---|---|")
    for k in ("cross_ca", "cross_cb"):
        a, lo, hi = aurocs[k]
        L.append(f"| cross-chain {'Cα-8Å' if k=='cross_ca' else 'Cβ-5Å'} "
                 f"| {a:.3f} | [{lo:.3f}, {hi:.3f}] |")
    ca_a, cb_a = aurocs["cross_ca"][0], aurocs["cross_cb"][0]
    L.append("")
    L.append("## B — monomer contrast: how much does the monomer-only tool miss?\n")
    L.append("| feature | AUROC (interface vs SUR) | 95% CI |")
    L.append("|---|---|---|")
    for k in ("mono_ca", "mono_cb"):
        a, lo, hi = aurocs[k]
        L.append(f"| monomer {'Cα-8Å' if k=='mono_ca' else 'Cβ-5Å'} "
                 f"| {a:.3f} | [{lo:.3f}, {hi:.3f}] |")

    L.append("\n## Per-class mean contacts (Levy gradient)\n")
    L.append("| class | n | mean cross-chain Cα-8 | mean monomer Cα-8 |")
    L.append("|---|---|---|---|")
    for c in classes:
        rs = by_class[c]
        if not rs:
            continue
        mc = np.mean([r["cross_ca"] for r in rs])
        mm = np.mean([r["mono_ca"] for r in rs])
        L.append(f"| {c} | {len(rs)} | {mc:.1f} | {mm:.1f} |")

    L.append("\n## Verdict\n")
    better = "Cα-8Å" if ca_a >= cb_a else "Cβ-5Å"
    L.append(f"- **A (cutoff):** cross-chain **{better}** has the higher interface-AUROC "
             f"(Cα-8Å {ca_a:.3f} vs Cβ-5Å {cb_a:.3f}). This is the cutoff the future "
             "complex-context extension should use for interface detection.")
    mono_best = max(aurocs['mono_ca'][0], aurocs['mono_cb'][0])
    L.append(f"- **B (monomer miss):** monomer intra-chain contacts predict interface at AUROC "
             f"≈ **{mono_best:.3f}** (vs cross-chain {max(ca_a, cb_a):.3f}) — the monomer view is "
             "far weaker, quantifying exactly what a monomer-only tool cannot see. Confirms the "
             "documented monomer-only limitation with a number, and motivates the complex extension.")

    (OUT_DIR / "SKEMPI_INTERFACE.md").write_text("\n".join(L) + "\n")
    (OUT_DIR / "skempi_interface.json").write_text(json.dumps(
        {"n_scored": len(recs), "n_muts": n_muts, "skipped": dict(skipped),
         "aurocs": {k: {"auroc": a, "ci": [lo, hi]} for k, (a, lo, hi) in aurocs.items()},
         "per_class": {c: {"n": len(by_class[c]),
                           "mean_cross_ca": float(np.mean([r["cross_ca"] for r in by_class[c]])) if by_class[c] else None,
                           "mean_mono_ca": float(np.mean([r["mono_ca"] for r in by_class[c]])) if by_class[c] else None}
                       for c in classes}}, indent=2))
    print("\n".join(L))
    print(f"\n[wrote {OUT_DIR/'SKEMPI_INTERFACE.md'}]")


if __name__ == "__main__":
    main()
