#!/usr/bin/env python3
"""Per-structure Spearman (T>=10) vs PLM scale — a 4-panel figure that extends
plot_ddg_scaling.py's "performance vs Model Scale" view down the ladder.

Layout: 2 rows × 3 cols.
  Row 1 (centered): ΔΔG performance (Pearson r, S1102 10-fold CV) — the
     ddg_scaling_300ep_balanced panel itself, redrawn (same label style, marker
     sizes and legend as ddg_scaling_300ep_balanced.png).
  Row 2: per-structure Spearman ρ (T≥10) on the three leakage-controlled
     full-SKEMPI splits — all three COMBINED single+multi, which is the protocol
     every frontier comparator on the figure was measured under: by-complex,
     clustered (CD-HIT id60), CATH-superfamily (shared Spearman y-axis).
All panels share the same log-x range as ddg_scaling_300ep_balanced (8e7 … 2.4e10).
Each PLM is a base point with the two FoldX levers rising from it (scalar diamond,
12-term MLP triangle) — FoldX in the "augmented" role. Labels are placed in-plot.

Data: panel 1 <- ddg_scaling_data_300ep_balanced.csv; panels 2-4 <- results_matrix_ps.csv
(results_matrix.py --ps; SP + bycomplex-ALL rungs via
experiments/full_skempi_seqonly/score_sp_all.py).

Usage:  ./.venv/bin/python scripts_plots/plot_ppS_scaling.py
"""
from __future__ import annotations
import csv
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import plot_common

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "scripts_plots"
import benchmarks as _bench   # data/benchmarks/frontier.tsv, protocol-tagged

PS_CSV = HERE / "results_matrix_ps.csv"
DDG_CSV = HERE / "ddg_scaling_data_300ep_balanced.csv"
FXA_CSV = ROOT / "experiments/rescore_perstructure/foldx_alone_baseline_allplm_cov99.csv"

XLIM = (8e7, 2.4e10)   # identical to ddg_scaling_300ep_balanced

# Frontier per-structure Spearman(T≥10) comparators, from experiments/BENCHMARK_MATRIX.md.
# Only the two whole-PDB / structurally leakage-controlled panels have per-structure comparators:
#  - by-complex COMBINED single+multi (RDE-3-fold): RDE / DiffAffinity / Prompt-DDG / CATH-ddG /
#    BA-DDG (secondary tables) — all train ONE model over single+multi, hence the -ALL rung.
#  - CATH-superfamily hold-out: USP-ddG Table 1 (+re-evals of the baselines).
# The clustered (CD-HIT ≤60%) panel has NO per-structure comparator — ProtBFF reports one whole-set
# correlation on that split and no per-structure figure, so its comparable number lives on the
# fold-averaged companion (3b), not here. Row 1 is a MuLAN-only leaky-CV metric. GearBind byCplx
# 0.525 is POOLED, a third metric again → excluded.
# Hoisted so the annotation tuples below read their values from the SAME dict the bars
# use. They used to restate the number inline, which is the drift this extraction exists
# to remove: a changed comparator would have moved a bar and left its label behind.
# `only=` is required, not stylistic: this protocol also carries the FoldX baseline row, which
# panel 2 draws as its own reference line and must NOT be folded into the frontier band. An
# unfiltered read would have stretched the band's floor from DiffAffinity 0.397 down to 0.369 and
# added FoldX to the comparator set, silently, the moment that row was added to the TSV.
_CMP_BYCPLX = _bench.comparators("ps_spearman_T10", "bycomplex_all",
                                 only=["RDE-Network", "DiffAffinity", "Prompt-DDG",
                                       "CATH-ddG", "BA-DDG"])
# Nine on the CATH hold-out: the seven learned methods plus the two physics protocols, flex ddG
# and FoldX, as USP-ddG's Table 1 re-evaluates them on the same split and metric (Per-PPI
# Spearman, all-mutation rows). The pair was left out until 2026-09-12 so that reading the
# comparators from the TSV did not widen the figure as a side effect; adding them was the
# deliberate change. Both fall inside the seven's min–max, so the band is unchanged.
_CMP_CATH = _bench.comparators("ps_spearman_T10", "cath",
                               only=["USP-ddG", "CATH-ddG", "BA-DDG", "flex-ddG", "FoldX",
                                     "Prompt-DDG", "RDE-Network", "DiffAffinity", "PPIformer"])

FRONTIER_COLOR = "#7b5aa6"
FRONTIER = {
    # Keyed to the COMBINED single+multi by-complex rung — the protocol these numbers were measured
    # under. Do NOT move them back onto `fullSK bycomplex-SP`: that rung trains on single-point only,
    # so its bars are not comparable to any of these.
    "fullSK bycomplex-ALL": dict(
        comparators=_CMP_BYCPLX,
        # RDE-Network (2023) introduced this by-complex per-structure-Spearman metric -> label its line.
        key=[("BA-DDG", _CMP_BYCPLX["BA-DDG"], "ceiling"),
             ("RDE-Network", _CMP_BYCPLX["RDE-Network"], "metric origin")],
        floor=("DiffAffinity", _CMP_BYCPLX["DiffAffinity"]),        # lower band-edge annotation (≈RDE, drawn below the line)
    ),
    "fullSK CATH-all": dict(
        comparators=_CMP_CATH,
        # flex ddG gets a key line: it is the physics protocol nearest MuLAN's best CATH marker
        # (0.454 against ESM-C 6B + scalar 0.438). USP-ddG's FoldX 0.430 is drawn by
        # draw_foldx_reported (FX_REPORTED below), the figure's grammar for a comparator's own
        # FoldX baseline — beside, not instead of, the 0.383 this repository measures on its split.
        key=[("CATH-ddG", _CMP_CATH["CATH-ddG"], "SOTA"),
             ("flex-ddG", _CMP_CATH["flex-ddG"], "physics"),
             ("BA-DDG", _CMP_CATH["BA-DDG"], "")],
        floor=("PPIformer", _CMP_CATH["PPIformer"]),
    ),
}
PS_YMAX = 0.56   # headroom for BA-DDG 0.513 (byCplx) + CATH-ddG 0.494 labels
PS_YMIN = -0.055  # floor headroom: ProstT5 sits at 0.054 on the clustered panel and hangs its
                  # label BELOW the marker, which the old ymin=0 clipped in half.

# --- clustered fold-averaged Spearman (CD-HIT ≤60%) -------------------------
# The clustered per-structure panel has no frontier comparator, but the fold-averaged Spearman on
# the same CD-HIT ≤60% split IS frontier-comparable — that is the form ProtBFF / ProMIM / ProSST
# report, so it is the only form of the metric that may be set against them.
#
# Read from results_matrix_ps.csv (`spearman_foldavg`, written by score_sp_all.py), NOT hardcoded.
# It was a literal table until 2026-08-05, transcribed from BENCHMARK_MATRIX.md's "CD-HIT Sp"
# column — which is POOLED Spearman, and was besides frozen at the 28%-coverage FoldX lineage. So
# the panel was wrong twice over: the wrong statistic under a fold-averaged axis label, at numbers
# three coverage generations stale (ESM-C 6B + 12-term read 0.298 against a current 0.504). Its
# base points survived both errors intact, which is what made the staleness hard to see — base
# arms carry no FoldX channel, so coverage cannot move them.
# The COMBINED single+multi clustered rung, matching ProtBFF's own protocol. Was `clustered-SP`
# until 2026-08-05, on the strength of a `| S |` marking in BENCHMARK_MATRIX's papers table — which
# is a transcription error: the ~6,631-mutation set that marking's own source describes is the
# ProtBFF-convention set (SKEMPI2 minus the 10 largest complexes) taken over *single and multiple*
# mutations, and 6,631 is far more than SKEMPI2 holds in single-point entries alone. Same reasoning
# that put panel 2 on `bycomplex-ALL`: a comparator may only sit against the rung it was measured on.
CDHIT_FOLDAVG_SPLIT = "fullSK clustered-ALL"
# Frontier comparators on the CD-HIT ≤60% split, from data/benchmarks/frontier.tsv. This dict was
# the one comparator literal the extraction missed — every other script was rewired, so it read as
# done. `only=` keeps the panel to the three methods it has always drawn: the TSV also carries
# ProtBFF's re-evaluations of RDE-Network and of its own ESM2/ProSST variants, and picking the
# whole protocol up would silently widen the band.
_CMP_CDHIT = _bench.comparators("pooled_spearman", "clustered_id60",
                                only=["ProtBFF", "ProMIM", "ProSST (base)"])
FRONTIER_CDHIT = dict(
    comparators={"ProtBFF": _CMP_CDHIT["ProtBFF"], "ProMIM": _CMP_CDHIT["ProMIM"],
                 "ProSST": _CMP_CDHIT["ProSST (base)"]},
    key=[("ProtBFF", _CMP_CDHIT["ProtBFF"], "SOTA"), ("ProMIM", _CMP_CDHIT["ProMIM"], "")],
    floor=("ProSST", _CMP_CDHIT["ProSST (base)"]),
)
FOLDAVG_YMAX = 0.64   # the FoldX arms now top out at 0.505, above ProtBFF; the old 0.52 clipped
                      # both the highest markers' labels and the SOTA line's, and 0.58 still left
                      # too little room for the staggered Ankh-large label to stay inside the axes

# --- FoldX-only reference ---------------------------------------------------
# The unsupervised physics baseline. Frontier papers report against FoldX alone, so every panel
# carries it — measured on THAT panel's own tier and own metric, never borrowed across either.
# Both halves of that matter: fold-averaged and pooled correlations differ on a 3-fold tier, and
# FoldX-alone is partition-dependent through the uncovered rows, so a value lifted from a
# neighbouring row would be a different quantity wearing the same name.
FX_COLOR = "#8a5a00"          # dark amber — the figure grammar's non-neural-physics family, and
                              # deliberately not the FoldX-scalar orange (#dd6b20) of the markers
# Violet, matching the frontier band and the ProtBFF line it belongs to — this is ProtBFF's number,
# so colour carries provenance and the dotted stroke carries "baseline, not a method". Amber is
# reserved for physics WE measured; putting a transcribed FoldX in amber implied our pipeline
# produced it, which is exactly the confusion the 0.294-vs-0.521 gap makes expensive.
FX_REPORTED_COLOR = FRONTIER_COLOR

# FoldX as the panel's own comparator reports it, where that paper publishes a FoldX baseline.
# Drawn beside our measured line rather than instead of it, because the two disagree by far more
# than the model differences the panel is being used to read: on the clustered split ProtBFF puts
# FoldX at 0.294 and ours is 0.521, which is above ProtBFF's own headline. That gap is a protocol
# difference in an unsupervised baseline neither side trains — most likely RepairPDB before
# AnalyseComplex, and 6,631 mutations against our 5,801 — so showing only ours would let the panel
# imply a win that the FoldX rows do not support. Value + source travel together; do not add a row
# here without the table it came from.
def _fx_reported(metric, protocol):
    """The FoldX baseline row for one protocol -> (value, source table)."""
    return (_bench.comparators(metric, protocol, only=["FoldX"])["FoldX"],
            _bench.source_of("FoldX", metric, protocol))


FX_REPORTED = {
    "cdhit": _fx_reported("pooled_spearman", "clustered_id60"),
    "fullSK bycomplex-ALL": _fx_reported("ps_spearman_T10", "bycomplex_all"),
    # USP-ddG's own FoldX on their CATH hold-out (Table 1, Per-PPI Spearman, all mutations). It
    # lands 0.047 above ours (0.383): a different FoldX setup on the same split, the same kind of
    # gap the two rows above show, and it is what a reader comparing against that table sees.
    "fullSK CATH-all": _fx_reported("ps_spearman_T10", "cath"),
}
# Which margin each transcribed line labels on, and whether it hangs above or below. Not derivable:
# it depends on what that panel's frontier band already occupies. Panel 2's left margin is taken at
# this height by DiffAffinity's floor label, which hangs down from 0.397 onto 0.369.
FX_REPORTED_PLACE = {
    "cdhit": ("left", "bottom"),
    "fullSK bycomplex-ALL": ("right", "top"),
    # Panel 4a's right margin at this height is taken by our own FoldX-alone label hanging from
    # 0.383 and by AIDO-16B's markers; the left is clear between BA-DDG 0.402 and flex ddG 0.454.
    "fullSK CATH-all": ("left", "bottom"),
}
FX_ALONE = {                  # panel -> (tier in foldx_alone_baseline_allplm_cov99.csv, column)
    "ddg":       ("S1102 CV10-balanced", "pr_foldavg"),
    "cdhit":     ("fullSK clustered-ALL", "sp_foldavg"),
    "auroc":     ("fullSK CATH-all", "auroc_destab"),
    "fullSK bycomplex-ALL": ("fullSK bycomplex-ALL", "rho"),
    "fullSK clustered-ALL": ("fullSK clustered-ALL", "rho"),
    "fullSK CATH-all": ("fullSK CATH-all", "rho"),
}

# --- CATH-superfamily AUROC (sign-of-effect, all-muts) ----------------------
# The frontier's classification metric on the CATH hold-out — the "clean" comparison (both all-muts),
# where MuLAN ranks 2nd. MuLAN per-embedding CATH AUROC + frontier lines verbatim from
# experiments/BENCHMARK_MATRIX.md (the "AUROC companion" table's CATH column) — do NOT recompute.
CATH_AUROC = {  # (model, arm) -> CATH AUROC (all-muts)
    ("ESM-C 6B", "base"): 0.787, ("ESM-C 6B", "fx_scalar"): 0.791, ("ESM-C 6B", "fx_mlp"): 0.715,
    ("ESM-C 600M", "base"): 0.732, ("ESM-C 600M", "fx_scalar"): 0.786, ("ESM-C 600M", "fx_mlp"): 0.683,
    ("SaProt", "base"): 0.716, ("SaProt", "fx_scalar"): 0.768, ("SaProt", "fx_mlp"): 0.722,
    ("ProstT5", "base"): 0.700, ("ProstT5", "fx_scalar"): 0.737, ("ProstT5", "fx_mlp"): 0.676,
    ("ESM2-3B", "base"): 0.745, ("ESM2-3B", "fx_scalar"): 0.763, ("ESM2-3B", "fx_mlp"): 0.675,
    ("Ankh-large", "base"): 0.806, ("Ankh-large", "fx_scalar"): 0.776, ("Ankh-large", "fx_mlp"): 0.671,
    ("Ankh3-large", "base"): 0.724, ("Ankh3-large", "fx_scalar"): 0.774, ("Ankh3-large", "fx_mlp"): 0.708,
    ("Ankh3-xl", "base"): 0.781, ("Ankh3-xl", "fx_scalar"): 0.783, ("Ankh3-xl", "fx_mlp"): 0.749,
    ("AIDO-16B", "base"): 0.680, ("AIDO-16B", "fx_scalar"): 0.738, ("AIDO-16B", "fx_mlp"): 0.719,
}
# The last panel that owned a private copy of the comparator numbers. The values matched
# frontier.tsv exactly, which is the problem the centralisation exists to remove: a copy that
# agrees today is indistinguishable from one that has silently drifted. Read from the file, with
# the key labels and the floor derived from it rather than restated.
_CMP_CATH_AUROC = _bench.comparators("auroc", "cath")
_AUROC_RANKED = sorted(_CMP_CATH_AUROC.items(), key=lambda kv: -kv[1])
FRONTIER_CATH_AUROC = dict(
    comparators=_CMP_CATH_AUROC,
    key=[(_AUROC_RANKED[0][0], _AUROC_RANKED[0][1], "SOTA"),
         (_AUROC_RANKED[1][0], _AUROC_RANKED[1][1], "")],
    floor=_AUROC_RANKED[-1],
)
AUROC_YMIN, AUROC_YMAX = 0.56, 0.87   # AUROC 0.5 = random; band starts ~0.62; headroom for staggered labels
# AUROC values all cluster high (0.67–0.81), so the shared LABEL layout collides — stagger explicitly.
AUROC_LABEL = {
    "ESM-C 600M": ("top", -16), "SaProt": ("bottom", 16), "Ankh-large": ("bottom", 30),
    "Ankh3-large": ("top", -18), "ProstT5": ("top", -30), "ESM2-3B": ("top", -16),
    "Ankh3-xl": ("bottom", 16), "ESM-C 6B": ("bottom", 30),
}


def draw_frontier_band(ax, split):
    """Shade the frontier per-structure-Spearman range + draw the key (SOTA / rival) lines."""
    fr = FRONTIER.get(split)
    if not fr:
        return
    vals = fr["comparators"]
    lo, hi = min(vals.values()), max(vals.values())
    ax.axhspan(lo, hi, color=FRONTIER_COLOR, alpha=0.09, zorder=0)
    ax.axhline(lo, color=FRONTIER_COLOR, lw=0.8, ls="--", alpha=0.45, zorder=1)
    # Labels on the emptier LEFT margin (far-right is crowded by the ESM-C 6B annotations).
    fname, fval = fr["floor"]
    ax.annotate(f"{fname} {fval:.3f}", (XLIM[0], lo), ha="left", va="top",
                fontsize=7.3, color=FRONTIER_COLOR, style="italic")
    # key comparators: bold solid line + left-margin label (incl. the top band edge)
    for name, y, role in fr["key"]:
        ax.axhline(y, color=FRONTIER_COLOR, lw=1.4, zorder=1)
        lab = f"{name} {y:.3f}" + (f" ({role})" if role else "")
        ax.annotate(lab, (XLIM[0], y), ha="left", va="bottom",
                    fontsize=8, color=FRONTIER_COLOR, fontweight="bold")

# ---- ddg-panel colors (subset of plot_ddg_scaling.COLORS) -------------------
DDG_COL = {
    "seq_cv10_linux": "#16386f", "aug_cv10_linux": "#5b9bd5",
    "seq_cv10_mps": "#8E1B6B", "aug_cv10_mps": "#E39BC8",
    "seq_cv10_gpu": "#1b6b3a", "aug_cv10_gpu": "#74c48f",
    # FoldX arms match panels 2-4a ARM_STYLE exactly: scalar = orange diamond, MLP = blue triangle.
    "foldxscalar": "#dd6b20", "foldxmlp": "#3182ce", "reference": "#8a8a8a", "lineage": "#8f8f8f",
}


def _plat(p):
    p = str(p).strip().lower()
    return "mps" if p == "mps" else ("gpu" if p in ("gpu", "cuda", "linux_gpu", "linux gpu") else "linux")


def _pcolor(series_key, p):
    return DDG_COL[f"{series_key}_{_plat(p)}"]


def _canon(name):
    """Canonical model key (drop +aug/+FoldX, normalize '\\n' -> space) so an aug/foldx
    row anchors to its OWN base — several PLMs share a param count (see reference)."""
    s = str(name).replace("\\n", " ").replace("\n", " ")
    s = re.sub(r"\+\s*aug", "", s, flags=re.I)
    s = re.sub(r"\+\s*foldx(?:[-\s]?(?:mlp|scalar))?", "", s, flags=re.I)
    return re.sub(r"\s+", " ", s).strip()


def draw_ddg_panel(ax):
    """Reproduce the ddg_scaling_300ep_balanced performance panel (Pearson r vs params) —
    same marker sizes (seq 250 / aug 95 / FoldX 115), bold in-plot labels (name + emb_dim,
    dx/dy from CSV) and the same legend."""
    df = pd.read_csv(DDG_CSV, comment="#")
    for c in ["params", "pcc", "pcc_std", "dx", "dy", "emb_dim"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for _, r in df[df.series == "reference"].iterrows():
        ax.axhline(r.pcc, ls="--", lw=1.4, color=DDG_COL["reference"], zorder=1)
        ax.text(XLIM[1], r.pcc + 0.001, f"  {r.model}: {r.pcc:.3f}", va="bottom",
                ha="right", fontsize=11, color="#6f6f6f", style="italic")

    s = df[(df.series == "seq_cv10") & df.pcc.notna()].sort_values("params")
    ax.scatter(s.params, s.pcc, s=250, color=[_pcolor("seq_cv10", p) for p in s.run_platform],
               edgecolor="white", linewidth=1.5, zorder=3)
    lineage_drawn = False
    if "lineage" in df.columns:
        meas = df[df.pcc.notna() & df.lineage.notna() & (df.lineage.astype(str).str.strip() != "")]
        for _, g in meas.groupby("lineage"):
            g = g.sort_values("params")
            if len(g) >= 2:
                ax.plot(g.params, g.pcc, ":", lw=2.4, alpha=0.9, zorder=2, color=DDG_COL["lineage"])
                lineage_drawn = True
    for _, r in s.iterrows():
        if int(r.label) == 1:
            disp = str(r.model).replace("\\n", "\n")
            txt = disp if pd.isna(r.emb_dim) else f"{disp}\n{int(r.emb_dim)}"
            ax.annotate(txt, (r.params, r.pcc), textcoords="offset points",
                        xytext=((r.dx or 0), (r.dy or 0) + 10), ha="center", fontsize=13,
                        fontweight="bold", color=_pcolor("seq_cv10", r.run_platform))

    base_by = {_canon(br.model): float(br.pcc) for _, br in s.iterrows()}
    aug = df[(df.series == "aug_cv10") & df.pcc.notna()]
    for _, r in aug.iterrows():
        b = base_by.get(_canon(r.model))
        if b is not None:
            ax.plot([r.params, r.params], [b, r.pcc], "-", lw=1.6, alpha=0.9,
                    color=_pcolor("aug_cv10", r.run_platform), zorder=2)
    ax.scatter(aug.params, aug.pcc, s=80, marker="x", linewidths=2.2,  # x so it can't be read as a FoldX arm
               color=[_pcolor("aug_cv10", p) for p in aug.run_platform], zorder=4)

    # FoldX scalar (orange diamond) + FoldX 12-term MLP (blue triangle) — same glyphs/colors as panels 2-4a.
    fxs = df[(df.series == "foldxscalar_cv10") & df.pcc.notna()]
    for _, r in fxs.iterrows():
        b = base_by.get(_canon(r.model))
        if b is not None:
            ax.plot([r.params, r.params], [b, r.pcc], "-", lw=1.6, alpha=0.75,
                    color=DDG_COL["foldxscalar"], zorder=2)
    ax.scatter(fxs.params, fxs.pcc, s=95, marker="D", color=DDG_COL["foldxscalar"],
               edgecolor="white", linewidth=1.3, zorder=4)

    fx = df[(df.series == "foldxmlp_cv10") & df.pcc.notna()]
    for _, r in fx.iterrows():
        b = base_by.get(_canon(r.model))
        if b is not None:
            ax.plot([r.params, r.params], [b, r.pcc], "-", lw=1.6, alpha=0.75,
                    color=DDG_COL["foldxmlp"], zorder=2)
    ax.scatter(fx.params, fx.pcc, s=115, marker="^", color=DDG_COL["foldxmlp"],
               edgecolor="white", linewidth=1.3, zorder=4)

    # legend (same as ddg_scaling): shape = arm, color = run platform (seq rows gated on presence)
    rp = df.get("run_platform", pd.Series(dtype=object)).astype(str).str.lower()
    gpu_names = ("gpu", "cuda", "linux_gpu", "linux gpu")
    leg = [Line2D([0], [0], color=DDG_COL["seq_cv10_linux"], lw=0, marker="o",
                  markeredgecolor="white", markersize=11, label="Sequence-only CV10 (Linux CPU)")]
    if bool(((df.series == "seq_cv10") & df.pcc.notna() & (rp == "mps")).any()):
        leg.append(Line2D([0], [0], color=DDG_COL["seq_cv10_mps"], lw=0, marker="o",
                          markeredgecolor="white", markersize=11, label="Sequence-only CV10 (MPS, M4 Pro)"))
    if bool(((df.series == "seq_cv10") & df.pcc.notna() & rp.isin(gpu_names)).any()):
        leg.append(Line2D([0], [0], color=DDG_COL["seq_cv10_gpu"], lw=0, marker="o",
                          markeredgecolor="white", markersize=11, label="Sequence-only CV10 (Linux GPU)"))
    leg.append(Line2D([0], [0], color="#7d7d7d", lw=1.6, marker="x", markersize=8,
                      markeredgewidth=2.2, label="+ Tier-1 aug CV10"))
    if bool(((df.series == "foldxscalar_cv10") & df.pcc.notna()).any()):
        leg.append(Line2D([0], [0], color=DDG_COL["foldxscalar"], lw=1.6, marker="D",
                          markeredgecolor="white", markersize=8, label="+ FoldX scalar add_scores CV10"))
    if bool(((df.series == "foldxmlp_cv10") & df.pcc.notna()).any()):
        leg.append(Line2D([0], [0], color=DDG_COL["foldxmlp"], lw=1.6, marker="^",
                          markeredgecolor="white", markersize=9, label="+ FoldX 12-term MLP add_scores CV10"))
    if lineage_drawn:
        leg.append(Line2D([0], [0], color=DDG_COL["lineage"], lw=2.4, ls=":",
                          label="Same-family scaling (shared lineage)"))
    ax.legend(handles=leg, loc="lower left", fontsize=7.3, frameon=True, facecolor="white",
              edgecolor="#d9d9d9", framealpha=0.95, borderpad=0.3, labelspacing=0.25,
              handletextpad=0.4, markerscale=0.7, bbox_to_anchor=(0.01, 0.01))

    ax.set_ylim(0.780, 0.905)
    # FoldX alone is annotated, not drawn: on this leaky per-mutation CV10 partition it is ~0.45
    # against a 0.78–0.90 axis, so a line would either fall outside the panel or, if the axis were
    # opened to hold it, flatten the scaling trend this panel exists to show. The gap is the point
    # — it is the size of the leakage, not a MuLAN result — so it is stated in words instead.
    fx = foldx_alone_value("ddg")
    if fx is not None:
        ax.annotate(f"FoldX alone {fx:.3f} (fold-avg r, same CV10 folds) — below this axis",
                    (XLIM[1], 0.780), ha="right", va="bottom", fontsize=7.5,
                    color=FX_COLOR, style="italic")
    ax.set_ylabel("ΔΔG performance\n(Pearson r, S1102 10-fold CV)", fontsize=12)
    ax.set_title("1 · ΔΔG Pearson r (S1102) CV10 300 epochs",
                 fontsize=12, fontweight="bold", loc="left")


# ---- per-structure (ppS) panels --------------------------------------------
PARAMS = {
    "ESM-C 600M": 6.0e8, "SaProt": 6.5e8, "Ankh-large": 1.15e9, "Ankh3-large": 1.15e9,
    "ProstT5": 1.21e9, "ESM2-3B": 2.8e9, "Ankh3-xl": 3.48e9, "ESM-C 6B": 6.0e9,
    "AIDO-16B": 1.6e10,
}
EMB_DIM = {
    "ESM-C 600M": 1152, "SaProt": 1280, "Ankh-large": 1536, "Ankh3-large": 1536,
    "ProstT5": 1024, "ESM2-3B": 2560, "Ankh3-xl": 2560, "ESM-C 6B": 2560,
    "AIDO-16B": 2304,
}
# Only the exact-overlap pair is nudged apart (Ankh-large ≡ Ankh3-large at 1.15e9).
PLOT_X = dict(PARAMS, **{"Ankh-large": 1.02e9, "Ankh3-large": 1.35e9})
LINEAGE = [("ESM-C 600M", "ESM-C 6B"), ("Ankh3-large", "Ankh3-xl")]
# In-plot bold label placement: ('bottom', dy) = above top marker; ('top', dy) = below bottom.
LABEL = {
    # SaProt / ESM-C 600M pulled due WEST (into the empty left margin) — at ~6e8 params they crowd
    # each other + Ankh3-large above/below.
    "ESM-C 600M": ("west", -12), "SaProt": ("west", -12), "Ankh-large": ("bottom", 14),
    "ProstT5": ("top", -16), "Ankh3-large": ("bottom", 14), "ESM2-3B": ("top", -16),
    "Ankh3-xl": ("bottom", 14), "ESM-C 6B": ("bottom", 14), "AIDO-16B": ("bottom", 14),
}
# arm -> (marker, color, label, size) — sizes match the ddg panel (base 250 / D 95 / ^ 115).
ARM_STYLE = {
    "base":      ("o", "#16386f", "base (PLM only)",     250),
    "fx_scalar": ("D", "#dd6b20", "+ FoldX scalar",       95),
    "fx_mlp":    ("^", "#3182ce", "+ FoldX 12-term MLP", 115),
}


def _place_label(ax, m, x, lo, hi, labels=LABEL, y_base=None):
    """Bold in-plot model label. LABEL entry is ('bottom'|'top', dy) = above/below the marker
    stack (ha=center), or ('west', dx) = due west of it (ha=right, vertically centred) — used to
    pull crowded low-param labels (SaProt / ESM-C 600M) out into the empty left margin.

    A 'west' label sits level with the model's BASE marker (`y_base`), not the midpoint of its
    arm stack: with a tall FoldX lever the midpoint floats far above the point being named, which
    read as a label belonging to no marker and collided with the frontier lines. Falls back to the
    stack midpoint only when the model has no base arm."""
    place = labels.get(m, ("bottom", 14))
    txt = f"{m}\n{EMB_DIM.get(m, '')}"
    if place[0] == "west":
        y = y_base if y_base is not None else (lo + hi) / 2
        ax.annotate(txt, (x, y), textcoords="offset points", xytext=(place[1], 0),
                    ha="right", va="center", fontsize=10.5, fontweight="bold",
                    color="#16386f", zorder=5)
    else:
        va, dy = place
        ax.annotate(txt, (x, hi if va == "bottom" else lo), textcoords="offset points",
                    xytext=(0, dy), ha="center", va=va, fontsize=10.5, fontweight="bold",
                    color="#16386f", zorder=5)


def load_ps(field="ps_spearman_T10"):
    """(split, model, arm) -> one metric from results_matrix_ps.csv.

    A blank cell means "not scored on this rung" and is left out of the dict entirely, so a panel
    draws no marker rather than a marker at zero.
    """
    d = {}
    with open(PS_CSV) as fh:
        for r in csv.DictReader(fh):
            v = r.get(field, "")
            if v:
                d[(r["split"], r["model"], r["arm"])] = float(v)
    return d


def load_foldx_alone():
    """tier -> the unsupervised FoldX-only row, all metric columns kept as strings."""
    out = {}
    with open(FXA_CSV) as fh:
        for r in csv.DictReader(fh):
            if r.get("arm") == "FoldX alone":
                out[r["tier"]] = r
    return out


FXA = load_foldx_alone()


def foldx_alone_value(panel):
    """The FoldX-only value for a panel, or None if that tier/column is not in the sweep."""
    tier, field = FX_ALONE[panel]
    row = FXA.get(tier)
    if not row or not row.get(field):
        return None
    return float(row[field])


def draw_foldx_alone(ax, panel, va="top", side="right"):
    """Amber dashed FoldX-only line, labelled on one margin.

    Right margin by default: the frontier lines all label left, so the two never compete. It hangs
    BELOW its line (`va="top"`) because `mean base` is also right-margin and labels upward. Panels
    whose right edge is occupied by a model label pass side="left" instead — on those the frontier
    band's own left labels are the ones that must be checked for clearance.
    """
    v = foldx_alone_value(panel)
    if v is None:
        return None
    ax.axhline(v, color=FX_COLOR, lw=1.5, ls=(0, (7, 3)), zorder=1.5)
    x, ha = (XLIM[1], "right") if side == "right" else (XLIM[0], "left")
    # Semi-opaque backing: the line sits mid-panel by construction (that is the whole point of a
    # baseline), so on the per-structure panels the label lands on AIDO-16B's markers at the right
    # edge. Backing it keeps both readable without moving the reference off its own value.
    # Short label: the panel already carries a second, longer FoldX label where a comparator
    # publishes one, and the legend is what distinguishes measured from transcribed. Spelling
    # "measured here" into every panel pushed the text across the markers it sits among.
    ax.annotate(f"FoldX alone {v:.3f}", (x, v), ha=ha, va=va,
                fontsize=8.5, color=FX_COLOR, fontweight="bold", zorder=6,
                bbox=dict(facecolor="white", alpha=0.78, edgecolor="none", pad=1.2))
    return v


def draw_foldx_reported(ax, panel, side=None, va=None):
    """The comparator's OWN published FoldX baseline, dotted, in the lighter amber.

    Labelled on the opposite margin from our measured line so the two cannot be read as one
    reference with two values, and named by its source table — it is transcribed, not measured.
    """
    ent = FX_REPORTED.get(panel)
    if not ent:
        return None
    v, src = ent
    _side, _va = FX_REPORTED_PLACE.get(panel, ("left", "bottom"))
    side, va = (side or _side), (va or _va)
    ax.axhline(v, color=FX_REPORTED_COLOR, lw=1.4, ls=(0, (2, 2.5)), zorder=1.5)
    x, ha = (XLIM[0], "left") if side == "left" else (XLIM[1], "right")
    ax.annotate(f"FoldX per {src} {v:.3f}", (x, v), ha=ha, va=va,
                fontsize=8, color=FX_REPORTED_COLOR, fontweight="bold", zorder=6,
                bbox=dict(facecolor="white", alpha=0.78, edgecolor="none", pad=1.2))
    return v


def draw_ps_panel(ax, D, split, title, show_ylabel, label_override=None):
    labels = label_override or LABEL   # per-panel label nudges (LABEL is shared across all panels)
    draw_frontier_band(ax, split)   # frontier reference band behind the MuLAN markers
    draw_foldx_alone(ax, split)     # unsupervised physics baseline on this tier's own ppS
    draw_foldx_reported(ax, split)  # ...and the same baseline as this panel's comparator reports it
    models = [m for m in PARAMS if any((split, m, a) in D for a in ARM_STYLE)]
    xs = {m: PLOT_X[m] for m in models}
    for a, b in LINEAGE:
        if a in models and b in models and (split, a, "base") in D and (split, b, "base") in D:
            ax.plot([xs[a], xs[b]], [D[(split, a, "base")], D[(split, b, "base")]],
                    ls=":", color="#9aa3b2", lw=1.6, zorder=1)
    for m in models:
        x = xs[m]
        present = {a: D[(split, m, a)] for a in ARM_STYLE if (split, m, a) in D}
        lo, hi = min(present.values()), max(present.values())
        ax.plot([x, x], [lo, hi], color="#c2c9d2", lw=1.6, zorder=2)
        for a, y in present.items():
            mk, col, _, sz = ARM_STYLE[a]
            ax.scatter([x], [y], marker=mk, s=sz, color=col, edgecolor="white", lw=1.2, zorder=4)
        _place_label(ax, m, x, lo, hi, labels, y_base=present.get("base"))

    base_mean = np.nanmean([D.get((split, m, "base"), np.nan) for m in models])
    ax.axhline(base_mean, ls=":", lw=1.0, color="#888", zorder=1)
    ax.annotate(f"mean base {base_mean:.3f}", (XLIM[1], base_mean), ha="right", va="bottom",
                fontsize=8.5, color="#888")
    ax.set_ylim(PS_YMIN, PS_YMAX)
    if show_ylabel:
        ax.set_ylabel("per-structure Spearman ρ  (T ≥ 10)", fontsize=12)
    ax.set_title(title, fontsize=11.5, fontweight="bold", loc="left")
    # Panels without a per-structure frontier comparator: say so explicitly (avoid conflation).
    if split not in FRONTIER:
        # SE corner: the coverage-fixed FoldX arms (2026-07-27) lifted the clustered panel's points
        # into the upper half, so the legend moved back to the NW and this disclaimer follows it out
        # of the way. The floor below ~0.05 on the right is the only reliably empty region.
        ax.annotate("no per-structure frontier comparator\n(ProtBFF reports this split fold-averaged — see 3b)",
                    (XLIM[1], PS_YMIN), ha="right", va="bottom", fontsize=7.5,
                    color="#9aa3b2", style="italic")


def _draw_frontier(ax, fr):
    """Shade a frontier comparator band + key (SOTA/rival) lines, labels on the left margin."""
    vals = fr["comparators"]
    lo, hi = min(vals.values()), max(vals.values())
    ax.axhspan(lo, hi, color=FRONTIER_COLOR, alpha=0.09, zorder=0)
    ax.axhline(lo, color=FRONTIER_COLOR, lw=0.8, ls="--", alpha=0.45, zorder=1)
    fname, fval = fr["floor"]
    ax.annotate(f"{fname} {fval:.3f}", (XLIM[0], lo), ha="left", va="top",
                fontsize=7.3, color=FRONTIER_COLOR, style="italic")
    for name, y, role in fr["key"]:
        ax.axhline(y, color=FRONTIER_COLOR, lw=1.4, zorder=1)
        lab = f"{name} {y:.3f}" + (f" ({role})" if role else "")
        ax.annotate(lab, (XLIM[0], y), ha="left", va="bottom",
                    fontsize=8, color=FRONTIER_COLOR, fontweight="bold")


def _draw_scalar_panel(ax, data, fr, ylabel, title, ymin, ymax, label_override=None,
                       fx_panel=None, fx_side="right"):
    """MuLAN base/scalar/MLP markers (one scalar metric per (model, arm)) vs PLM scale, with a
    frontier band. Shared engine for the CD-HIT fold-avg Spearman (3b) and CATH AUROC (4b) panels."""
    labels = label_override or LABEL
    _draw_frontier(ax, fr)
    if fx_panel:
        draw_foldx_alone(ax, fx_panel, side=fx_side)
        draw_foldx_reported(ax, fx_panel)   # margin comes from FX_REPORTED_PLACE
    models = [m for m in PARAMS if any((m, a) in data for a in ARM_STYLE)]
    xs = {m: PLOT_X[m] for m in models}
    for a, b in LINEAGE:
        if (a, "base") in data and (b, "base") in data:
            ax.plot([xs[a], xs[b]], [data[(a, "base")], data[(b, "base")]],
                    ls=":", color="#9aa3b2", lw=1.6, zorder=1)
    for m in models:
        x = xs[m]
        present = {a: data[(m, a)] for a in ARM_STYLE if (m, a) in data}
        lo, hi = min(present.values()), max(present.values())
        ax.plot([x, x], [lo, hi], color="#c2c9d2", lw=1.6, zorder=2)
        for a, y in present.items():
            mk, col, _, sz = ARM_STYLE[a]
            ax.scatter([x], [y], marker=mk, s=sz, color=col, edgecolor="white", lw=1.2, zorder=4)
        _place_label(ax, m, x, lo, hi, labels, y_base=present.get("base"))

    base_mean = np.nanmean([data.get((m, "base"), np.nan) for m in models])
    ax.axhline(base_mean, ls=":", lw=1.0, color="#888", zorder=1)
    ax.annotate(f"mean base {base_mean:.3f}", (XLIM[1], base_mean), ha="right", va="bottom",
                fontsize=8.5, color="#888")
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=11.5, fontweight="bold", loc="left")


def draw_cdhit_panel(ax):
    """MuLAN fold-averaged Spearman on the CD-HIT ≤60% clustered split — the frontier-comparable
    form of the metric, and the only one that may be set against ProtBFF (vs the per-structure
    clustered panel below it, which shares the split but not the statistic)."""
    fa = load_ps("spearman_foldavg")
    data = {(m, a): v for (s, m, a), v in fa.items() if s == CDHIT_FOLDAVG_SPLIT}
    _draw_scalar_panel(ax, data, FRONTIER_CDHIT,
                       "Spearman ρ  (CD-HIT ≤ 60%, fold-avg)",
                       "3b · CD-HIT ≤60% fold-avg Spearman, single+multi (ProtBFF)",
                       0, FOLDAVG_YMAX, fx_panel="cdhit",
                       # Every FoldX arm on this panel lands in 0.41–0.51, so the label row above
                       # the stacks is far more crowded than on the per-structure panels: Ankh-large
                       # and Ankh3-large (1.02e9 / 1.35e9) overprinted outright. Stagger them by a
                       # text row rather than pulling one west, where SaProt and ESM-C 600M already sit.
                       label_override={**LABEL, "Ankh-large": ("bottom", 22)})


def draw_auroc_panel(ax):
    """MuLAN CATH-superfamily AUROC (sign-of-effect, all-muts) — the frontier's classification metric,
    the clean CATH comparison (both all-muts) where MuLAN ranks 2nd."""
    # The band already carries the literature's FoldX 0.754; the amber line is OUR FoldX run on
    # this exact hold-out, and the two landing together (0.754 vs 0.761) is the corroboration.
    _draw_scalar_panel(ax, CATH_AUROC, FRONTIER_CATH_AUROC,
                       "AUROC  (destabilizing, CATH all-muts)",
                       "4b · CATH-superfamily AUROC (CATH-ddG)",
                       AUROC_YMIN, AUROC_YMAX, label_override=AUROC_LABEL,
                       # left margin: 0.760 is the one height on this panel where the right edge is
                       # taken (AIDO-16B's label sits on it), and the left is clear between
                       # CATH-ddG 0.781 and DiffAffinity 0.625
                       fx_panel="auroc", fx_side="left")


def main():
    plot_common.apply_theme()
    D = load_ps()

    fig = plt.figure(figsize=(20, 11))
    gs = fig.add_gridspec(2, 6, left=0.055, right=0.99, top=0.9, bottom=0.085,
                          hspace=0.30, wspace=0.42)
    ax1 = fig.add_subplot(gs[0, 0:2])                       # upper-left: ΔΔG CV10 (Pearson y)
    axP = fig.add_subplot(gs[0, 2:4])                       # upper-center: CD-HIT fold-avg Sp (own y)
    axA = fig.add_subplot(gs[0, 4:6])                       # upper-right: CATH AUROC (own y)
    ax2 = fig.add_subplot(gs[1, 0:2])                       # row 2 (shared per-structure Spearman y)
    ax3 = fig.add_subplot(gs[1, 2:4], sharey=ax2)
    ax4 = fig.add_subplot(gs[1, 4:6], sharey=ax2)

    draw_ddg_panel(ax1)
    draw_cdhit_panel(axP)                                   # centered above the clustered per-structure panel
    draw_auroc_panel(axA)                                   # above the CATH per-structure panel
    # Row 2 ordered by increasing leakage control: by-complex -> clustered -> CATH.
    # Panel 2 plots the COMBINED single+multi by-complex tier, not the SP-only one: the frontier
    # comparators drawn on it (RDE-Network / DiffAffinity / Prompt-DDG / BA-DDG) are all measured on
    # one model trained over single+multi with a whole-PDB hold-out, so that is the only rung they
    # can be honestly compared against. Was `fullSK bycomplex-SP` until 2026-07-25, when the combined
    # tier's FoldX arms first existed to plot.
    # Ankh-large, Ankh3-large and ProstT5 sit within 5% of each other in x (1.15/1.15/1.21e9), so
    # which label needs moving is per-panel — it depends on where each model's arms land.
    draw_ps_panel(ax2, D, "fullSK bycomplex-ALL", "2 · by-complex, single+multi (RDE-Network)", True,
                  # ProstT5's label ran straight through Ankh3-large's base marker (0.158) — drop it
                  # a further text row so it clears.
                  # ESM-C 6B tops out just above Ankh3-xl a third of a decade away, so their two
                  # centred labels touch — lift ESM-C 6B by a text row.
                  label_override={**LABEL, "ProstT5": ("top", -32), "ESM-C 6B": ("bottom", 30)})
    # Combined single+multi, like panels 2 and 4a. This rung has no per-structure comparator at all
    # (see the in-panel note), so nothing had to be re-matched to move it — but leaving it on the
    # single-point rung made it the one step of the ladder measured on a different mutation set,
    # which is not a difference a reader should have to discover from a comment.
    draw_ps_panel(ax3, D, "fullSK clustered-ALL", "3a · CD-HIT ≤60%, single+multi (ProtBFF)", False)
    # CATH is the one panel where Ankh-large (hi 0.387) and Ankh3-large (hi 0.393) top out level, so
    # two centred labels at the same x overprint. Pull Ankh-large due west of its base instead.
    # Ankh3-large's label also has to clear ProstT5's FoldX-scalar diamond (0.436), which sits 5% away
    # in x and ABOVE Ankh3-large's own stack top (0.393) — hence the extra-tall offset.
    draw_ps_panel(ax4, D, "fullSK CATH-all", "4a · CATH-superfamily (CATH-ddG)", False,
                  label_override={**LABEL, "Ankh-large": ("west", -12),
                                  "Ankh3-large": ("bottom", 44)})
    # Show y tick values on all three Spearman panels (shared scale, but labels repeated).
    for ax in (ax3, ax4):
        ax.tick_params(labelleft=True)

    for ax in (ax1, axP, axA, ax2, ax3, ax4):
        ax.set_xscale("log")
        ax.set_xlim(*XLIM)
        ax.grid(True, which="both", alpha=0.16)
        ax.set_xlabel("Model Scale  (encoder parameters)", fontsize=11)

    # Shared arm/frontier legend — panel 3a (clustered) is the only per-structure panel with dead
    # space. It goes upper-LEFT: after the 2026-07-27 coverage fix the FoldX arms climbed to ~0.37,
    # so the upper-RIGHT box was sitting on the Ankh3-xl / ESM-C 6B markers and labels. The small
    # PLMs still bottom out the left edge, leaving the NW corner clear above ~0.4.
    handles = [Line2D([0], [0], marker=mk, color="w", markerfacecolor=col, markeredgecolor="white",
                      markersize=10, label=lab) for mk, col, lab, _ in ARM_STYLE.values()]
    handles.append(Line2D([0], [0], ls=":", color="#9aa3b2", lw=1.6, label="same-family scaling"))
    handles.append(Patch(fc=FRONTIER_COLOR, alpha=0.18, ec=FRONTIER_COLOR,
                         label="frontier range"))   # short: provenance is in the suptitle, and a
                                                    # wide legend box reached the Ankh-large label
    handles.append(Line2D([0], [0], color=FX_COLOR, lw=1.5, ls=(0, (7, 3)),
                          label="FoldX alone — measured (per panel)"))   # "per panel" because the
                                                    # value is re-measured on each tier and metric
    handles.append(Line2D([0], [0], color=FX_REPORTED_COLOR, lw=1.4, ls=(0, (2, 2.5)),
                          label="FoldX — comparator's own (2, 3b, 4a)"))
    # Moved off panel 3a on 2026-08-05: at six entries the box reached down past 0.40 and covered
    # the Ankh-large / Ankh3-large columns, which are the two the clustered panels are read for.
    # Panel 2's floor is the one region on this row that no marker enters — its lowest base sits at
    # 0.19. Lower-RIGHT, not left: ProstT5's label hangs below its marker into the left corner, and
    # the two smallest PLMs sit there too, while nothing on the right half is drawn under 0.25.
    ax2.legend(handles=handles, loc="lower right", fontsize=7.8, framealpha=0.95,
               title="MuLAN arms (panels 2–4)", title_fontsize=7.8, borderpad=0.35,
               labelspacing=0.32, handletextpad=0.5)

    fig.suptitle("MuLAN across PLM scale — ΔΔG performance (1) and per-structure Spearman on the "
                 "leakage-controlled full-SKEMPI ladder: by-complex single+multi (2), clustered (3a), CATH (4a)\n"
                 "frontier-comparable companions above: clustered CD-HIT fold-avg Spearman (3b), CATH AUROC "
                 "(4b);  FoldX scalar / MLP = lift over base;  violet band = frontier (data/benchmarks/frontier.tsv)",
                 fontsize=13, y=0.985)

    for ext in ("png", "svg"):
        out = HERE / f"ppS_scaling.{ext}"
        fig.savefig(out, dpi=150)
        print("wrote", out)


if __name__ == "__main__":
    main()
