#!/usr/bin/env python3
"""Plot the rescore_perstructure results (reads results.csv, writes results.png).

Two panels:
  A — per-arm dumbbell: overall Spearman -> per-structure Spearman (T=10), sorted by the
      per-structure value. Line length = the leverage collapse; dot colour = scoring channel.
  B — per-PLM channel lift: per-structure Spearman (T=10) for base/aug/foldxmlp within each of
      the 6 PLMs, showing the FoldX-MLP lift is systematic, not a single-arm artefact.

Colours: dataviz categorical slots 1-4 (validated CVD-safe, light surface) mapped to channel.
Run:  experiments/rescore_perstructure/.venv/bin/python experiments/rescore_perstructure/plot_results.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent

# ---- design tokens (dataviz reference palette, light surface) --------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"       # text-primary
INK2 = "#52514e"      # text-secondary
MUTED = "#8a8984"     # overall-dot / grid ink
GRID = "#e6e5e1"
CH = {                # categorical slots 1-4 -> scoring channel
    "base": "#2a78d6",       # blue
    "aug": "#1baf7a",        # aqua
    "foldxmlp": "#eda100",   # yellow
    "mint": "#008300",       # green
}
CH_LABEL = {"base": "base (PLM only)", "aug": "augmented",
            "foldxmlp": "FoldX-MLP", "mint": "MINT"}
PLMS = ["esm2", "esmc600m", "prostt5", "saprot", "ankh", "esmc6b"]


def channel(arm: str) -> str:
    if arm.startswith("mint"):
        return "mint"
    for c in ("foldxmlp", "aug", "base"):
        if arm.endswith(c):
            return c
    raise ValueError(arm)


def main():
    df = pd.read_csv(HERE / "results.csv")
    df["channel"] = df["arm"].map(channel)

    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "font.size": 9,
        "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": GRID,
        "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    fig = plt.figure(figsize=(12.5, 8.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.65, 1.0], wspace=0.32,
                          left=0.15, right=0.975, top=0.86, bottom=0.11)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])

    # ---- Panel A: dumbbell overall -> per-structure Spearman ---------------
    d = df.sort_values("ps_spearman_T10", ascending=True).reset_index(drop=True)
    y = range(len(d))
    for yi, row in zip(y, d.itertuples()):
        col = CH[row.channel]
        axA.plot([row.ps_spearman_T10, row.spearman], [yi, yi],
                 color=GRID, lw=2, zorder=1, solid_capstyle="round")
        axA.scatter(row.spearman, yi, s=42, facecolor=SURFACE, edgecolor=MUTED,
                    lw=1.4, zorder=2)                                  # overall (open)
        axA.scatter(row.ps_spearman_T10, yi, s=70, facecolor=col, edgecolor="#33322f",
                    lw=0.6, zorder=3)                                  # per-structure (filled)
    axA.set_yticks(list(y))
    axA.set_yticklabels(d["arm"], fontsize=8)
    axA.set_xlim(0.24, 0.87)
    axA.set_xlabel("Spearman ρ  (pred vs true)")
    axA.set_title("A · Per-structure Spearman collapses vs overall",
                  loc="left", fontsize=11, fontweight="bold", color=INK, pad=8)
    axA.xaxis.grid(True, color=GRID, lw=0.8)
    axA.set_axisbelow(True)
    axA.tick_params(length=0)
    # annotate one collapse span for legibility
    top = d.iloc[-1]
    axA.annotate("", xy=(top.ps_spearman_T10, len(d) - 1), xytext=(top.spearman, len(d) - 1),
                 arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0, shrinkA=6, shrinkB=8))
    axA.text((top.ps_spearman_T10 + top.spearman) / 2, len(d) - 1 + 0.55,
             "removing mega-complex\nleverage", ha="center", va="bottom",
             fontsize=7.2, color=INK2, style="italic")

    legA = [Line2D([0], [0], marker="o", color="none", markerfacecolor=SURFACE,
                   markeredgecolor=MUTED, markeredgewidth=1.4, markersize=7,
                   label="overall (n=1100)"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#66655f",
                   markeredgecolor="#33322f", markeredgewidth=0.6, markersize=8.5,
                   label="per-structure (T≥10)")]
    axA.legend(handles=legA, loc="lower right", frameon=False, fontsize=8,
               handletextpad=0.3, borderaxespad=0.6)

    # ---- Panel B: per-PLM channel lift (ps_spearman_T10) -------------------
    bar_order = ["base", "aug", "foldxmlp"]
    h = 0.26
    for gi, plm in enumerate(PLMS):
        base_y = gi
        for j, ch in enumerate(bar_order):
            arm = f"{plm}_{ch}"
            v = float(df.loc[df["arm"] == arm, "ps_spearman_T10"].iloc[0])
            yy = base_y + (j - 1) * h
            axB.barh(yy, v, height=h * 0.9, color=CH[ch], edgecolor=SURFACE, lw=1.0, zorder=2)
            axB.text(v + 0.006, yy, f"{v:.2f}", va="center", ha="left",
                     fontsize=6.8, color=INK2)
    axB.set_yticks(range(len(PLMS)))
    axB.set_yticklabels(PLMS, fontsize=8.5)
    axB.set_xlim(0, 0.72)
    axB.set_xlabel("Per-structure Spearman ρ  (T≥10)")
    axB.set_title("B · FoldX-MLP lift is systematic per PLM",
                  loc="left", fontsize=11, fontweight="bold", color=INK, pad=8)
    axB.xaxis.grid(True, color=GRID, lw=0.8)
    axB.set_axisbelow(True)
    axB.tick_params(length=0)
    axB.invert_yaxis()
    # (channel identity comes from the shared figure legend at top — no per-panel legend)

    # channel legend + title/caption on the figure
    fig.legend(handles=[Patch(facecolor=CH[c], label=CH_LABEL[c]) for c in CH],
               loc="upper left", bbox_to_anchor=(0.15, 0.955), ncol=4, frameon=False,
               fontsize=8.5, columnspacing=1.4, handletextpad=0.5, title="scoring channel",
               title_fontsize=8.5)
    fig.suptitle("S1102 OOF re-score — FoldX-MLP channel leads on per-structure ranking",
                 x=0.15, y=0.985, ha="left", fontsize=13.5, fontweight="bold", color=INK)
    fig.text(0.065, 0.012,
             "⚠ Metric change, not split change: predictions are from the leaky per-mutation "
             "10-fold CV (same complex in train & test), so per-structure ρ is an UPPER BOUND,\n"
             "not generalization — the honest by-complex figure needs a retrain.  "
             "Metrics validated to <1e-9 against the RDE-PPI reference implementation.",
             ha="left", va="bottom", fontsize=7.6, color=INK2)

    out = HERE / "results.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
