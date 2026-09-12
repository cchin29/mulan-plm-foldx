#!/usr/bin/env python3
"""Frontier comparison: MuLAN's leakage-controlled arms against the published SKEMPI-v2 ddG
frontier, on the two apples-to-apples anchors.

  Panel A  Overall (pooled) Spearman  vs ProtBFF CD-HIT-60 peers  [SAME split family + metric]
  Panel B  Per-structure Spearman     vs USP-ddG / CATH-ddG       [SAME split family: literal
                                        CATH-superfamily hold-out, TM<0.6 on BOTH partners]

Panel B previously drew its MuLAN bars from experiments/retrain_split/results_clustered.csv --
the CD-HIT <=60% single-point tier -- and ranked them against a CATH-superfamily frontier band.
Its own subtitle admitted the mismatch and called a literal CATH rerun "the pending drop-in".
That rerun has since happened (full-SKEMPI CATH-superfamily hold-out, 687 mutations /
13 complexes at T>=10, FoldX coverage 99.1%), so panel B now reads the CATH tier directly.
The correction is material and moves every MuLAN bar down: esmc6b_foldx_scalar reads 0.438 on
the real CATH tier, not the 0.460 the CD-HIT tier gave. That places it BELOW flex-ddG (0.454)
rather than above it, and +0.008 over the published FoldX row (0.430).

Frontier numbers: data/benchmarks/frontier.tsv, read through `benchmarks.py` and keyed by
(method, metric, protocol) -- originally transcribed from USP-ddG Table 1, CATH-ddG Table 2 and
ProtBFF Table 2, which each row names in its `source` column. MuLAN numbers are measured here and
deliberately never enter that TSV: panel A <- experiments/retrain_split/results_clustered.csv
(single-point S1102, CD-HIT <=60%, complete 1100/1100); panel B <- scripts_plots/results_matrix_ps.csv
(fullSK CATH-all), with 95% cluster-bootstrap whiskers from
experiments/rescore_perstructure/foldx_alone_baseline_allplm_cov99.csv.

Run:  .venv/bin/python scripts_plots/plot_frontier_comparison.py
Out:  scripts_plots/frontier_comparison.png
"""
import csv
from pathlib import Path

import benchmarks as _bench   # data/benchmarks/frontier.tsv, protocol-tagged
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CLUST = ROOT / "experiments/retrain_split/results_clustered.csv"
PS = ROOT / "scripts_plots/results_matrix_ps.csv"
FXA = ROOT / "experiments/rescore_perstructure/foldx_alone_baseline_allplm_cov99.csv"
OUT = ROOT / "scripts_plots/frontier_comparison.png"

MU = "#1f6fb4"        # MuLAN + FoldX (ours) — blue
MU_FLOOR = "#9ec9e6"  # MuLAN base (no-FoldX floor) — light blue
FX_OURS = "#e08a3c"   # FoldX alone measured on OUR split — orange (the physics comparator)
FRONT = "#7a7a7a"     # published baselines — gray
SOTA = "#c44536"      # structure-integrated SOTA — red

# Exact match, so the separate "fullSK CATH-all+aug" rows cannot be picked up here.
CATH_TIER = "fullSK CATH-all"


def load_clustered():
    with open(CLUST) as fh:
        return {r["arm"]: r for r in csv.DictReader(fh)}


def load_cath_ps():
    """(model, arm) -> ps_spearman_T10 on the literal CATH-superfamily hold-out."""
    out = {}
    with open(PS) as fh:
        for r in csv.DictReader(fh):
            if r["split"] == CATH_TIER and r["ps_spearman_T10"]:
                out[(r["model"], r["arm"])] = float(r["ps_spearman_T10"])
    return out


def load_cath_ci():
    """(model_key, arm) -> (lo, hi) 95% cluster-bootstrap CI on the same tier."""
    out = {}
    with open(FXA) as fh:
        for r in csv.DictReader(fh):
            if r["tier"] == CATH_TIER and r["lo"] and r["hi"]:
                out[(r["model"], r["arm"])] = (float(r["lo"]), float(r["hi"]))
    return out


def load_cath_rho():
    """(model_key, arm) -> rho on the same tier. Only used for the FoldX-alone row, whose
    value lives here rather than in results_matrix_ps.csv (it is not a MuLAN arm)."""
    out = {}
    with open(FXA) as fh:
        for r in csv.DictReader(fh):
            if r["tier"] == CATH_TIER and r["rho"]:
                out[(r["model"], r["arm"])] = float(r["rho"])
    return out


def bar_panel(ax, rows, title, subtitle, xlabel):
    """rows: list of (label, value, color, is_ours, ci_or_None)."""
    rows = list(rows)
    ys = range(len(rows))
    vals = [r[1] for r in rows]
    bars = ax.barh(ys, vals, color=[r[2] for r in rows], edgecolor="white", height=0.72)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.invert_yaxis()

    # Whiskers where we have them; published rows are quoted without CIs and get none.
    for y, r in zip(ys, rows):
        ci = r[4] if len(r) > 4 else None
        if ci:
            ax.plot([ci[0], ci[1]], [y, y], color="#333", lw=1.0, solid_capstyle="butt", zorder=3)
            for x in ci:
                ax.plot([x, x], [y - 0.14, y + 0.14], color="#333", lw=1.0, zorder=3)

    right = max([r[1] for r in rows] + [r[4][1] for r in rows if len(r) > 4 and r[4]])
    ax.set_xlim(0, right * 1.18)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=26)
    ax.text(0, 1.012, subtitle, transform=ax.transAxes, fontsize=7.8, color="#444", va="bottom")
    for b, v, r in zip(bars, vals, rows):
        # Clear the whisker cap where there is one, so the label never sits on top of it.
        ci = r[4] if len(r) > 4 else None
        x = (ci[1] if ci else v) + right * 0.015
        ax.text(x, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=8, fontweight="bold" if r[3] else "normal",
                color=r[2] if r[3] else "#333")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis="x", lw=0.4, color="#ddd", zorder=0)
    ax.set_axisbelow(True)


def main():
    d = load_clustered()
    g = lambda arm, col: float(d[arm][col])
    ps = load_cath_ps()
    ci = load_cath_ci()
    rho = load_cath_rho()

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 6.2))

    # ---- Panel A: Overall (pooled) Spearman vs ProtBFF CD-HIT-60 (same split family + metric) ----
    # MuLAN single-point (ours); ProtBFF rows are all-mutations (flagged in caption).
    # Comparators from data/benchmarks/frontier.tsv. `only=` is explicit so a row added to the
    # file for another figure cannot silently appear in this one.
    cmpA = _bench.comparators("pooled_spearman", "clustered_id60",
                              only=["ProSST + ProtBFF", "ProMIM", "RDE-Network",
                                    "ESM2 + ProtBFF", "FoldX", "ESM2 (bare PLM)"])
    rowsA = [
        ("MuLAN·esmc6b_foldx_scalar (ours)", g("esmc6b_foldx_scalar", "spearman"), MU, True, None),
        ("ProSST + ProtBFF",                 cmpA["ProSST + ProtBFF"], FRONT, False, None),
        ("MuLAN·saprot_foldx (ours)",        g("saprot_foldx", "spearman"), MU, True, None),
        ("ProMIM",                           cmpA["ProMIM"], FRONT, False, None),
        ("RDE-Network",                      cmpA["RDE-Network"], FRONT, False, None),
        ("ESM2 + ProtBFF",                   cmpA["ESM2 + ProtBFF"], FRONT, False, None),
        ("FoldX",                            cmpA["FoldX"], FRONT, False, None),
        ("ESM2 (bare PLM floor)",            cmpA["ESM2 (bare PLM)"], FRONT, False, None),
    ]
    rowsA.sort(key=lambda r: r[1], reverse=True)
    bar_panel(axA, rowsA,
              "A · Overall (pooled) Spearman — CD-HIT ≤60% tier",
              "Clean anchor: SAME split family + metric as ProtBFF Table 2",
              "Overall Spearman ρ")

    # ---- Panel B: per-structure Spearman on the LITERAL CATH-superfamily hold-out ----
    # Same seven published rows as before; the MuLAN bars now come from the CATH tier, and
    # FoldX-alone measured on our own split is carried alongside the published FoldX row.
    cmpB = _bench.comparators("ps_spearman_T10", "cath",
                              only=["CATH-ddG", "USP-ddG", "flex-ddG", "FoldX",
                                    "Prompt-DDG", "RDE-Network", "PPIformer"])
    rowsB = [
        ("CATH-ddG  (structure SOTA)",              cmpB["CATH-ddG"], SOTA, False, None),
        ("USP-ddG  (structure SOTA)",               cmpB["USP-ddG"], SOTA, False, None),
        ("flex-ddG",                                cmpB["flex-ddG"], FRONT, False, None),
        ("MuLAN·ESM-C 6B + FoldX scalar (ours)",    ps[("ESM-C 6B", "fx_scalar")], MU, True,
         ci[("esmc6b", "fx_scalar")]),
        ("MuLAN·ESM-C 6B + FoldX 12-term (ours)",   ps[("ESM-C 6B", "fx_mlp")], MU, True,
         ci[("esmc6b", "fx_mlp")]),
        ("FoldX  (published)",                      cmpB["FoldX"], FRONT, False, None),
        ("MuLAN·Ankh-large + FoldX scalar (ours)",  ps[("Ankh-large", "fx_scalar")], MU, True,
         ci[("ankh", "fx_scalar")]),
        ("FoldX alone — measured on THIS split",    rho[("—", "FoldX alone")], FX_OURS, True,
         ci[("—", "FoldX alone")]),
        ("MuLAN·ESM-C 6B base (no FoldX, ours)",    ps[("ESM-C 6B", "base")], MU_FLOOR, True,
         ci[("esmc6b", "base")]),
        ("Prompt-DDG",                              cmpB["Prompt-DDG"], FRONT, False, None),
        ("RDE-Network",                             cmpB["RDE-Network"], FRONT, False, None),
        ("PPIformer",                               cmpB["PPIformer"], FRONT, False, None),
    ]
    rowsB.sort(key=lambda r: r[1], reverse=True)
    bar_panel(axB, rowsB,
              "B · Per-structure (per-PPI) Spearman — CATH-superfamily tier",
              "Clean anchor: literal CATH-superfamily hold-out (TM<0.6 both partners), 687 muts / 13 complexes T≥10",
              "Per-structure Spearman ρ (T≥10)")

    fig.suptitle("MuLAN (sequence-only + FoldX) vs the SKEMPI-v2 ΔΔG frontier — honest splits",
                 fontsize=13.5, fontweight="bold", x=0.5, y=0.99)
    cap = ("Blue = MuLAN + FoldX (ours).  Light blue = MuLAN base, the no-FoldX floor.  Orange = FoldX alone measured on our own split.  "
           "Red = structure-integrated SOTA.  Gray = published baselines.\n"
           "Caveats: (A) single-point S1102, CD-HIT ≤60% clustered retrain, complete 1100/1100; ProtBFF rows are all-mutations vs our single-point, "
           "and MuLAN·esmc6b Pearson is leverage-inflated — Spearman is the honest read.\n"
           "(B) full SKEMPI, single+multi, FoldX coverage 99.1%.  Whiskers = 95% cluster bootstrap on our rows; published rows are quoted without CIs.  "
           "Against FoldX alone (0.383) only the ESM-C 6B scalar arm clears with a CI excluding zero: +0.055 [+0.001, +0.118].")
    fig.text(0.5, 0.005, cap, ha="center", va="bottom", fontsize=7.3, color="#555")
    fig.tight_layout(rect=(0, 0.075, 1, 0.94))
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
