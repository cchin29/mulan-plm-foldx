#!/usr/bin/env python3
"""Per-structure Spearman (T>=10) of MuLAN across embeddings, on the three
leakage-controlled full-SKEMPI splits — clustered (mmseqs id<=60%), by-complex
(whole-PDB hold-out) and CATH-superfamily (USP-ddG's exact hold-out). All three
panels are the COMBINED single+multi rung, which is the protocol every frontier
comparator on the figure was measured under (the same choice plot_ppS_scaling.py
makes, and for the same reason).

Sibling in spirit to plot_ddg_scaling.py's base-vs-aug view: here the FoldX
arms (scalar IE, 12-term MLP) play the "augmented" role against the PLM-only
base, so each embedding shows base / +FoldX-scalar / +FoldX-MLP side by side.

Single source of truth = scripts_plots/results_matrix_ps.csv (produced by
results_matrix.py --ps; the -ALL rungs are scored by
experiments/full_skempi_seqonly/score_sp_all.py).
Re-run any time results_matrix_ps.csv is refreshed.

Usage:  ./.venv/bin/python scripts_plots/plot_ppS_generalization.py
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import plot_common

ROOT = Path(__file__).resolve().parents[1]
PS_CSV = ROOT / "scripts_plots" / "results_matrix_ps.csv"

# The three panels: (ps-CSV split label, human title). Order = strictness of the split.
# The clustered and by-complex panels were the single-point (`-SP`) rungs until 2026-09-12. The
# comparators shaded behind the by-complex bars are each one model trained over single AND
# multi-point mutations, so the bars they sit against have to be the same; and once one panel
# moves, leaving the clustered one on `-SP` would make it the single step of the ladder measured
# on a different mutation set. The `-SP` rows stay in results_matrix_ps.csv.
SPLITS = [
    ("fullSK clustered-ALL", "clustered, single+multi\n(mmseqs id ≤ 60%)"),
    ("fullSK bycomplex-ALL", "by-complex, single+multi\n(whole-PDB hold-out)"),
    ("fullSK CATH-all",      "CATH-superfamily, all mutations\n(USP-ddG hold-out)"),
]
# Canonical embedding order (best/largest first); a model absent from a split is skipped there.
MODEL_ORDER = ["ESM-C 6B", "ESM-C 600M", "ESM2-3B", "SaProt", "ProstT5",
               "Ankh-large", "Ankh3-xl", "Ankh3-large", "AIDO-16B"]

# Arm styling — base = neutral grey, FoldX arms in the "aug" role (scalar orange, MLP blue),
# matching plot_generalization_ladder.py so the arms read identically across figures.
ARMS = [("base", "base (PLM only)", "#b8c2cc"),
        ("fx_scalar", "+ FoldX scalar", "#dd6b20"),
        ("fx_mlp", "+ FoldX 12-term MLP", "#3182ce")]

# Frontier per-structure Spearman(T≥10) comparators, from data/benchmarks/frontier.tsv — kept in
# sync with plot_ppS_scaling.py. Only the whole-PDB / structurally leakage-controlled panels have a
# per-structure comparator; the clustered (CD-HIT ≤60%) split has none (ProtBFF reports it pooled only).
# GearBind byCplx 0.525 is POOLED → excluded.
import benchmarks as _bench   # data/benchmarks/frontier.tsv, protocol-tagged

# Hoisted so the annotation tuples below read their values from the SAME dict the bars
# use. They used to restate the number inline, which is the drift this extraction exists
# to remove: a changed comparator would have moved a bar and left its label behind.
# `only=` is required, not stylistic: the by-complex protocol also carries the FoldX baseline row
# (Prompt-DDG's 0.369), which is a physics reference and not a frontier method. Read unfiltered,
# it would sit inside the band and pull the band's dashed floor down from DiffAffinity 0.397 to
# 0.369 under a label that still said DiffAffinity — which is what this script did until
# 2026-09-12. Same five names as plot_ppS_scaling.py.
_CMP_BYCPLX = _bench.comparators("ps_spearman_T10", "bycomplex_all",
                                 only=["RDE-Network", "DiffAffinity", "Prompt-DDG",
                                       "CATH-ddG", "BA-DDG"])
# Nine on the CATH hold-out: the seven learned methods plus the two physics protocols, flex ddG
# and FoldX, as USP-ddG's Table 1 re-evaluates them on the same split and metric (Per-PPI
# Spearman, all-mutation rows). The pair was left out until 2026-09-12 so that reading the
# comparators from the TSV did not widen the figure as a side effect; adding them was the
# deliberate change. Both fall inside the seven's min–max, so the band is unchanged — what the
# reader gains is that the two physics values bracket MuLAN's best CATH bar.
_CMP_CATH = _bench.comparators("ps_spearman_T10", "cath",
                               only=["USP-ddG", "CATH-ddG", "BA-DDG", "flex-ddG", "FoldX",
                                     "Prompt-DDG", "RDE-Network", "DiffAffinity", "PPIformer"])

FRONTIER_COLOR = "#7b5aa6"
FRONTIER = {
    # Keyed to the COMBINED single+multi by-complex rung — the protocol these numbers were measured
    # under (one model per method, trained over single and multi-point, whole-PDB hold-out). Do
    # NOT move them back onto `fullSK bycomplex-SP`: that rung trains on single-point only, so its
    # bars are not comparable to any of these. This panel plotted exactly that mismatch until
    # 2026-09-12; plot_ppS_scaling.py panel 2 had made the move on 2026-07-25.
    "fullSK bycomplex-ALL": dict(
        comparators=_CMP_BYCPLX,
        # RDE-Network (2023) introduced this by-complex per-structure-Spearman metric -> label its line.
        key=[("BA-DDG", _CMP_BYCPLX["BA-DDG"], "ceiling"),
             ("RDE-Network", _CMP_BYCPLX["RDE-Network"], "metric origin")],
        floor=("DiffAffinity", _CMP_BYCPLX["DiffAffinity"]),
    ),
    "fullSK CATH-all": dict(
        comparators=_CMP_CATH,
        # The two physics protocols get their own lines: they are the comparison most relevant to
        # a FoldX-augmented model, and they sit nearest MuLAN's best bar (ESM-C 6B + scalar, 0.438
        # — FoldX 0.430 just below it, flex ddG 0.454 just above). FoldX here is USP-ddG's own run
        # on their hold-out, not the 0.383 this repository measures on its split.
        key=[("CATH-ddG", _CMP_CATH["CATH-ddG"], "SOTA"),
             ("flex-ddG", _CMP_CATH["flex-ddG"], "physics"),
             ("FoldX", _CMP_CATH["FoldX"], "physics, per USP-ddG"),
             ("BA-DDG", _CMP_CATH["BA-DDG"], "")],
        floor=("PPIformer", _CMP_CATH["PPIformer"]),
        # Four key labels on this panel, and all four heights fall where ESM-C 6B's bar-value
        # labels stand at the left edge (its FoldX arms top out at 0.438 / 0.404). Start the
        # labels past that first group: nothing in the next two groups reaches 0.40.
        label_x=0.115,
    ),
}


def draw_frontier_band(ax, split):
    """Shade the frontier per-structure-Spearman range + draw the key (SOTA / rival) lines.
    Labels use the y-axis transform (x = axes fraction) since this panel's x-axis is categorical."""
    fr = FRONTIER.get(split)
    tr = ax.get_yaxis_transform()
    if not fr:
        ax.text(0.985, 0.985, "no per-structure frontier comparator\n(ProtBFF reports this split "
                "pooled only)", transform=ax.transAxes, ha="right", va="top",
                fontsize=7.3, color="#9aa3b2", style="italic")
        return
    vals = fr["comparators"]
    lo, hi = min(vals.values()), max(vals.values())
    ax.axhspan(lo, hi, color=FRONTIER_COLOR, alpha=0.09, zorder=0)
    ax.axhline(lo, color=FRONTIER_COLOR, lw=0.8, ls="--", alpha=0.45, zorder=1)
    # Semi-opaque backing, as plot_ppS_scaling.py gives its FoldX labels: the left margin is
    # also where the first bar group stands, and the CATH panel now labels four lines there.
    bbox = dict(facecolor="white", alpha=0.78, edgecolor="none", pad=1.2)
    fname, fval = fr["floor"]
    ax.text(0.015, lo, f"{fname} {fval:.3f}", transform=tr, ha="left", va="top",
            fontsize=7.3, color=FRONTIER_COLOR, style="italic", zorder=6, bbox=bbox)
    lx = fr.get("label_x", 0.015)   # axes fraction the key labels start at (see the CATH entry)
    for name, y, role in fr["key"]:
        ax.axhline(y, color=FRONTIER_COLOR, lw=1.4, zorder=1)
        lab = f"{name} {y:.3f}" + (f" ({role})" if role else "")
        ax.text(lx, y, lab, transform=tr, ha="left", va="bottom",
                fontsize=8, color=FRONTIER_COLOR, fontweight="bold", zorder=6, bbox=bbox)


def load():
    """{(split, model, arm): ps_spearman_T10} from the results_matrix ps CSV."""
    d = {}
    with open(PS_CSV) as fh:
        for r in csv.DictReader(fh):
            v = r.get("ps_spearman_T10", "")
            if v:
                d[(r["split"], r["model"], r["arm"])] = float(v)
    return d


def main():
    plot_common.apply_theme()
    D = load()

    fig, axes = plt.subplots(1, len(SPLITS), figsize=(18, 6.2), sharey=True)
    W = 0.26  # bar width

    for ax, (split, title) in zip(axes, SPLITS):
        draw_frontier_band(ax, split)   # frontier reference band behind the bars
        models = [m for m in MODEL_ORDER if any((split, m, a) in D for a, _, _ in ARMS)]
        x = np.arange(len(models))
        for j, (arm, _, color) in enumerate(ARMS):
            off = (j - 1) * W
            ys = [D.get((split, m, arm), np.nan) for m in models]
            bars = ax.bar(x + off, ys, W, color=color, edgecolor="white", lw=0.5, zorder=3)
            for xi, y in zip(x + off, ys):
                if y == y:  # not NaN
                    ax.annotate(f"{y:.2f}", (xi, y + 0.006), ha="center", va="bottom",
                                fontsize=6.3, color="#333", rotation=90, zorder=4)
        # mean-base guide + panel mean deltas
        base_mean = np.nanmean([D.get((split, m, "base"), np.nan) for m in models])
        ax.axhline(base_mean, ls=":", lw=1.1, color="#888", zorder=1)
        ax.annotate(f"mean base {base_mean:.3f}", (len(models) - 0.5, base_mean + 0.006),
                    ha="right", va="bottom", fontsize=8, color="#888")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=35, ha="right", fontsize=9)
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        ax.set_ylim(0, 0.62)
        ax.grid(False, axis="x")
        ax.grid(True, axis="y", alpha=0.18)

    axes[0].set_ylabel("per-structure Spearman ρ  (T ≥ 10)", fontsize=11.5)
    handles = [Patch(fc=c, label=lab) for _, lab, c in ARMS]
    handles.append(Patch(fc=FRONTIER_COLOR, alpha=0.18, ec=FRONTIER_COLOR,
                         label="frontier per-structure ρ range"))
    axes[-1].legend(handles=handles, loc="upper right", fontsize=9.5, framealpha=0.95)
    fig.suptitle("MuLAN per-structure Spearman across embeddings — FoldX (scalar / 12-term MLP) "
                 "vs PLM-only base on the leakage-controlled full-SKEMPI splits, single+multi\n"
                 "violet band = frontier per-structure ρ comparators (data/benchmarks/frontier.tsv), "
                 "each on the rung it was measured on",
                 fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    for ext in ("png", "svg"):
        out = ROOT / "scripts_plots" / f"ppS_generalization.{ext}"
        fig.savefig(out, dpi=140)
        print("wrote", out)


if __name__ == "__main__":
    main()
