#!/usr/bin/env python3
"""Tier-0: reproduce the MuLAN paper's PDB-Interactome attention->interface AUROC.

Scores the paper's *released* Ankh-large attention (Zenodo record 18175031,
mulan_attentions_pdb.h5) against the released per-residue interface labels
(contacts_data_pdb.pkl.gz) and checks that we recover the published numbers:

    overall  0.642   |  protein-protein 0.612  |  DNA/RNA 0.763  |  ligand 0.660

The paper does not state whether "overall AUROC" is pooled over all residues or
macro-averaged over sequences (Fig 2 shows a single ROC per class, suggesting
pooled; Fig S1 for S1102 was explicitly "average over sequences"). We therefore
report BOTH aggregations per class so the matching one pins down the exact
protocol before we extend the analysis to other PLM backbones (Tier 1).

Ground truth (contacts_data_pdb.pkl.gz) is a dict of dicts keyed by PDB chain id
(e.g. "8h7f_A1"):
    sequences       chain -> str (length L)
    coords          chain -> float32 [L,3]  (C-alpha)
    labels_ppi      chain -> bool [L]  (interface with another protein)
    labels_dna_rna  chain -> bool [L]  (interface with DNA/RNA)
    labels_ligand   chain -> bool [L]  (interface with a small molecule)
    labels_all      chain -> bool [L]  (union of the above)
Attention (mulan_attentions_pdb.h5) is a dataset per chain -> float32 [L],
already MinMax-scaled per chain (paper Methods "Extraction of Attention weights").

Usage:  ./.venv/bin/python experiments/attention_interface/score_interactome.py
"""
from __future__ import annotations
import gzip
import pickle
import time
from pathlib import Path

import numpy as np
import h5py
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
ZEN = ROOT / "scratch" / "attention_interface" / "zenodo"
PKL = ZEN / "contacts_data_pdb.pkl.gz"
H5 = ZEN / "mulan_attentions_pdb.h5"
OUT = ROOT / "experiments" / "attention_interface" / "RESULTS_interactome.md"

# class key in pickle -> (human label, paper's published pooled AUROC)
CLASSES = [
    ("labels_all", "all interactions", 0.642),
    ("labels_ppi", "protein-protein", 0.612),
    ("labels_dna_rna", "DNA/RNA", 0.763),
    ("labels_ligand", "ligand", 0.660),
]


def macro_auroc(per_chain):
    """Mean of per-chain AUROCs over chains that contain both a positive and a
    negative residue (a chain with no negatives / no positives has no ROC)."""
    vals = [a for a in per_chain if a == a]  # drop NaN
    return (np.mean(vals), len(vals)) if vals else (np.nan, 0)


def main():
    t0 = time.time()
    print(f"loading labels  {PKL.name} ...", flush=True)
    with gzip.open(PKL, "rb") as f:
        d = pickle.load(f)
    print(f"loading attention {H5.name} ...", flush=True)
    h = h5py.File(H5, "r")

    chains = sorted(set(h.keys()) & set(d["sequences"].keys()))
    print(f"scoring {len(chains):,} chains present in both files "
          f"(load {time.time()-t0:.0f}s)", flush=True)

    # Accumulators. For each class: pooled residue scores/labels over qualifying
    # chains, and a list of per-chain AUROCs for the macro view.
    pooled = {k: ([], []) for k, _, _ in CLASSES}   # k -> (scores[], labels[])
    macro = {k: [] for k, _, _ in CLASSES}          # k -> [per-chain auroc]
    n_qual = {k: 0 for k, _, _ in CLASSES}          # chains with >=1 positive of that class
    skipped = 0

    for i, c in enumerate(chains):
        att = np.asarray(h[c], dtype=np.float64)
        if att.ndim != 1 or att.size == 0:
            skipped += 1
            continue
        L = att.size
        for k, _, _ in CLASSES:
            lab = np.asarray(d[k][c], dtype=bool)
            if lab.size != L:            # defensive; probe showed 0 mismatches
                continue
            npos = int(lab.sum())
            if npos == 0:                # chain has no positive of this class
                continue
            n_qual[k] += 1
            s, l = pooled[k]
            s.append(att)
            l.append(lab)
            if npos < L:                 # both classes present -> per-chain ROC defined
                macro[k].append(roc_auc_score(lab, att))
        if (i + 1) % 40000 == 0:
            print(f"  ...{i+1:,}/{len(chains):,}  ({time.time()-t0:.0f}s)", flush=True)
    h.close()

    # Finalize pooled AUROC per class (over the qualifying chains).
    rows = []
    for k, name, paper in CLASSES:
        s, l = pooled[k]
        if s:
            S = np.concatenate(s)
            Lb = np.concatenate(l)
            pooled_auc = roc_auc_score(Lb, S)
            n_res, n_pos = Lb.size, int(Lb.sum())
        else:
            pooled_auc, n_res, n_pos = np.nan, 0, 0
        mac, n_mac = macro_auroc(macro[k])
        rows.append((name, paper, pooled_auc, mac, n_qual[k], n_res, n_pos, n_mac))

    # ---- report ----
    hdr = (f"{'class':16s} {'paper':>7s} {'pooled':>8s} {'Δpool':>7s} "
           f"{'macro':>8s} {'Δmac':>7s} {'chains':>8s} {'residues':>11s} {'pos%':>6s}")
    lines = [hdr, "-" * len(hdr)]
    for name, paper, pool, mac, nch, nres, npos, nmac in rows:
        dp = pool - paper if pool == pool else float("nan")
        dm = mac - paper if mac == mac else float("nan")
        pospct = 100 * npos / nres if nres else float("nan")
        lines.append(f"{name:16s} {paper:7.3f} {pool:8.3f} {dp:+7.3f} "
                     f"{mac:8.3f} {dm:+7.3f} {nch:8,d} {nres:11,d} {pospct:5.1f}%")
    report = "\n".join(lines)
    print("\n" + report)
    print(f"\nskipped (empty/malformed attention): {skipped}")
    print(f"total wall time: {time.time()-t0:.0f}s")

    # ---- write markdown ----
    md = ["# Tier-0: PDB-Interactome attention->interface AUROC (reproduction)",
          "",
          "Released **Ankh-large** attention (Zenodo 18175031 `mulan_attentions_pdb.h5`) "
          "scored against the released per-residue interface labels "
          "(`contacts_data_pdb.pkl.gz`). Goal: recover the paper's published AUROCs "
          "and pin down the aggregation protocol (pooled-over-residues vs "
          "macro-averaged-over-chains).",
          "",
          f"- chains scored (h5 ∩ pickle): **{len(chains):,}**",
          "- attention: per-chain MinMax-scaled, mean over 3 heads × 64 filters (paper Methods)",
          "- per-class AUROC computed over chains with ≥1 positive residue of that class",
          "",
          "| class | paper | pooled | Δ | macro (per-chain) | Δ | chains | residues | pos% |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name, paper, pool, mac, nch, nres, npos, nmac in rows:
        dp = f"{pool-paper:+.3f}" if pool == pool else "—"
        dm = f"{mac-paper:+.3f}" if mac == mac else "—"
        pospct = f"{100*npos/nres:.1f}%" if nres else "—"
        md.append(f"| {name} | {paper:.3f} | {pool:.3f} | {dp} | {mac:.3f} | {dm} | "
                  f"{nch:,} | {nres:,} | {pospct} |")
    md += ["",
           "Whichever column (pooled / macro) matches the paper column fixes the "
           "protocol carried into Tier 1 (the PLM-backbone head-to-head).", ""]
    OUT.write_text("\n".join(md))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
