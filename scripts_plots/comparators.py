#!/usr/bin/env python3
"""Published comparator values, parsed out of experiments/BENCHMARK_MATRIX.md.

Why this module exists
----------------------
Until now every plot that drew a frontier band carried its own typed copy of the
same numbers: plot_ppS_scaling.py and plot_ppS_generalization.py each declared
`FRONTIER` with RDE-Network 0.401 / DiffAffinity 0.397 / ... written out by hand,
in two files, from a third file (BENCHMARK_MATRIX.md) that neither of them read.
Three copies of one fact is two chances to update it in only one place — and a
figure that quietly disagrees with the matrix is exactly the failure this repo
is trying to make impossible.

So the values are no longer typed anywhere. This module locates each table in
BENCHMARK_MATRIX.md *by its header row* (not by line number, so edits above it
cannot silently shift the parse — see mdtable.py, which owns that rule and is
shared with the generator that writes MuLAN's own rows into the same file),
strips the bold markup and the superscript provenance markers, and hands the
plots a plain {method: value} dict. If a method or a column named here stops
existing in the matrix, the import fails loudly rather than drawing a stale band.

What stays with the plots
-------------------------
Which comparator gets a labelled line, which one annotates the band floor, and
where the label hangs — that is presentation, it depends on what else occupies
that panel's margin, and it is not derivable from the matrix. The plots keep it,
but they now reference comparators *by name* and look the value up here.

Usage
-----
    import comparators
    comparators.FRONTIER_BYCOMPLEX   # {"RDE-Network": 0.401, ...}

    python3 scripts_plots/comparators.py     # print every set + its source line
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mdtable

ROOT = Path(__file__).resolve().parents[1]
BMX = ROOT / "experiments/BENCHMARK_MATRIX.md"


def _table(name: str, *required, keycols: int = 1):
    """One table from the matrix. Locating and parsing live in mdtable, which
    gen_benchmark_matrix_data.py also imports — so the program that writes MuLAN's
    rows into the matrix and the program that reads comparators out of it cannot
    drift apart on what a table is."""
    return mdtable.parse(BMX, mdtable.read(BMX), name, required, keycols=keycols)


# --------------------------------------------------------------------------- tables

PAPERS = _table("papers", "| Method — Date", "CATH")
AUROC = _table("AUROC companion", "| Method ", "CATH (all-muts)", "by-complex")
MULAN_AUROC_TBL = _table("MuLAN AUROC by embedding × arm",
                         "| Embedding ", "CATH", "by-complex (all-muts)", keycols=2)


# --------------------------------------------------------------------------- frontier sets
#
# Each set names the methods that report a comparable number in that column. The
# membership is deliberate, not "every row with a value":
#
#  - GearBind's byCplx 0.525 is POOLED Spearman, a different statistic under the
#    same column heading, so it is excluded from the by-complex band.
#  - MuLAN's own rows are excluded everywhere — the band is what we are measured
#    against, and including ourselves in it would flatter the comparison.
#
# Adding a method here without checking that its number is the same statistic on
# the same split is the mistake this list is shaped to make visible.

# Per-structure Spearman(T≥10), RDE 3-fold by-complex split (single+multi combined).
FRONTIER_BYCOMPLEX = PAPERS.column(
    "byCplx", ["RDE-Network", "DiffAffinity", "Prompt-DDG", "CATH-ddG", "BA-DDG"])

# Per-structure Spearman(T≥10), CATH-superfamily hold-out (USP-ddG Table 1 + its re-evals).
FRONTIER_CATH = PAPERS.column(
    "CATH", ["USP-ddG", "CATH-ddG", "BA-DDG", "Prompt-DDG",
             "RDE-Network", "DiffAffinity", "PPIformer"])

# Fold-averaged Spearman on the CD-HIT ≤60% split — the only form of the clustered
# metric any frontier paper reports, hence the only form MuLAN may be set against.
FRONTIER_CDHIT = PAPERS.column("CD-HIT Sp", ["ProtBFF", "ProMIM", "ProSST"])

# AUROC (sign-of-effect), CATH hold-out, all-muts.
FRONTIER_CATH_AUROC = AUROC.column(
    "CATH (all-muts)", ["USP-ddG", "CATH-ddG", "flex-ddG", "FoldX",
                        "RDE-Network", "BA-DDG", "DiffAffinity"])

# MuLAN's own CATH AUROC per embedding × arm. Not a comparator — but it lives in the
# matrix rather than in a CSV this repo can recompute, so it was typed into the plot
# for the same reason and carries the same drift risk.
_ARM = {"base": "base", "+ FoldX scalar": "fx_scalar", "+ FoldX MLP": "fx_mlp"}
MULAN_CATH_AUROC = {
    (model, _ARM[arm]): vals["CATH"]
    for (model, arm), vals in MULAN_AUROC_TBL.rows.items()
    if arm in _ARM and vals["CATH"] is not None
}


# --------------------------------------------------------------------------- audit


def main() -> None:
    """Print every exported set against the matrix line each value came from."""
    print(f"Comparator values parsed from {BMX.relative_to(ROOT)}")
    print("Nothing below is typed in a plot script; change the matrix and the figures follow.\n")

    for title, table, col, values in [
        ("FRONTIER_BYCOMPLEX  (per-structure Spearman, RDE 3-fold by-complex)",
         PAPERS, "byCplx", FRONTIER_BYCOMPLEX),
        ("FRONTIER_CATH  (per-structure Spearman, CATH-superfamily hold-out)",
         PAPERS, "CATH", FRONTIER_CATH),
        ("FRONTIER_CDHIT  (fold-averaged Spearman, CD-HIT ≤60%)",
         PAPERS, "CD-HIT Sp", FRONTIER_CDHIT),
        ("FRONTIER_CATH_AUROC  (AUROC, CATH hold-out, all-muts)",
         AUROC, "CATH (all-muts)", FRONTIER_CATH_AUROC),
    ]:
        print(title)
        for name, v in values.items():
            print(f"  {v:<8.3f} {name:<14} <- {BMX.name}:{table.lineno(name)}  "
                  f"{table.row(name)!r} / {col!r}")
        print()

    print("MULAN_CATH_AUROC  (MuLAN's own, AUROC companion details table)")
    for (model, arm), v in MULAN_CATH_AUROC.items():
        print(f"  {v:<8.3f} {model} {arm}")
    print(f"\n{len(MULAN_CATH_AUROC)} MuLAN (embedding, arm) values, "
          f"{sum(len(s) for s in (FRONTIER_BYCOMPLEX, FRONTIER_CATH, FRONTIER_CDHIT, FRONTIER_CATH_AUROC))} "
          f"comparator values, 0 typed.")


if __name__ == "__main__":
    main()
