#!/usr/bin/env python3
"""
Model timeline for the PLM sweep — release date × lineage, with per-model stats
(size, embedding dim, pretraining objective) and the headline S1102 MuLAN 10-fold
CV PCC.  Lead slide for the phase-1 review.

Data: RESULTS.md master table (PCC) + PHASE2_PRIMER / model cards (dates, objectives).
Dates are approximate first public release / preprint dates (see footnote).

Run:  python plot_model_timeline.py   ->  model_timeline.{png,svg}
Deps: matplotlib, numpy  (no pandas).
"""
import os
import datetime as dt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------- data
# lane order (top -> bottom); each family gets a horizontal lane
LANES = [
    ("ESM",     "ESM family",            "sequence-only",  "#1f6f8b"),
    ("Ankh",    "Ankh family",           "sequence-only",  "#6a4c93"),
    ("SaProt",  "SaProt family",         "AA + 3Di (structure)", "#2a9d8f"),
    ("ProstT5", "ProstT5",               "seq-only (3Di unused)", "#e76f51"),
    ("AIDO",    "AIDO.Protein",          "sequence-only (MoE)",   "#5c6b73"),
]
LANE_Y = {k: y for (k, *_), y in zip(LANES, [10.8, 8.1, 5.4, 2.7, 0])}
LANE_COLOR = {k: c for (k, _n, _t, c) in LANES}

def d(y, m):
    return dt.date(y, m, 15)

# key, display name, date, lineage, params, emb_dim, objective, pcc, pcc_std, best, hf_repo
# `date` = HuggingFace repo creation date (createdAt), fetched via the HF API 2026-07.
MODELS = [
    ("esm2",   "ESM2-3B",        d(2022,10), "ESM",    "2.8B",     2560, "Masked-LM, UniRef50", 0.816, 0.054, False, "facebook/esm2_t36_3B_UR50D"),
    ("esmc",   "ESM C 6B",       d(2025,12), "ESM",    "6B",       2560, "ESM Cambrian — newer pretraining", 0.859, 0.069, True, "EvolutionaryScale/esmc-6b-2024-12"),
    ("ankh",   "Ankh-large",     d(2022, 9), "Ankh",   "1.15B",    1536, "T5 span-denoising encoder", 0.832, 0.057, False, "ElnaggarLab/ankh-large"),
    ("ankh3",  "Ankh3 large / xl", d(2024, 9), "Ankh", "1.15B / 3.48B", 1536, "T5 multi-task: denoise + completion", 0.831, 0.056, False, "ElnaggarLab/ankh3-large · -xl"),
    ("saprot", "SaProt-650M",    d(2023,10), "SaProt", "650M",     1280, "ESM2 + AA·3Di structural vocab", 0.842, 0.050, False, "westlake-repl/SaProt_650M_AF2"),
    ("saprot13","SaProt-1.3B",   d(2025, 5), "SaProt", "1.3B",     1280, "AFDB/OMG/NCBI, 66 layers (deeper)", 0.837, 0.061, False, "westlake-repl/SaProt_1.3B_AFDB_OMG_NCBI"),
    ("prostt5","ProstT5",        d(2023, 7), "ProstT5","1.2B",     1024, "T5, bilingual AA↔​3Di", 0.805, 0.055, False, "Rostlab/ProstT5"),
    ("aido",   "AIDO.Protein-16B", d(2024,11), "AIDO", "16B MoE",  2304, "16B Mixture-of-Experts, MLM", 0.828, 0.046, False, "genbio-ai/AIDO.Protein-16B"),
]
PAPER_PCC = 0.868

# ------------------------------------------------------------------------- helpers
import matplotlib.colors as mcolors
_green = mcolors.LinearSegmentedColormap.from_list("perf", ["#e8f2e8", "#8fbf8f", "#2e7d32"])
def pcc_color(pcc):
    t = np.clip((pcc - 0.80) / (0.868 - 0.80), 0, 1)
    return _green(0.15 + 0.85 * t)

def to_num(date):
    return date.year + (date.month - 1) / 12.0

# ------------------------------------------------------------------------------ fig
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#cccccc",
})
fig, ax = plt.subplots(figsize=(16, 10.2), dpi=200)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

x_min, x_max = 2022.4, 2026.35
ax.set_xlim(x_min, x_max)
ax.set_ylim(-2.05, 13.0)

# --- lane baselines + labels
for key, name, tag, color in LANES:
    y = LANE_Y[key]
    ax.plot([x_min + 0.05, x_max - 0.05], [y, y], color=color, lw=1.1, alpha=0.35, zorder=1)
    ax.text(x_min - 0.02, y, name, ha="right", va="center", fontsize=12.5,
            fontweight="bold", color=color)
    ax.text(x_min - 0.02, y - 0.42, tag, ha="right", va="center", fontsize=8.5,
            style="italic", color="#888888")

# --- year gridlines
for yr in range(2022, 2027):
    ax.axvline(yr, color="#eeeeee", lw=1, zorder=0)
    ax.text(yr, 12.65, str(yr), ha="center", va="center", fontsize=11,
            color="#999999", fontweight="bold")

# --- lineage connectors (within family, arrow of time)
def connect(k1, k2):
    m1 = next(m for m in MODELS if m[0] == k1)
    m2 = next(m for m in MODELS if m[0] == k2)
    y = LANE_Y[m1[3]]
    ax.annotate("", xy=(to_num(m2[2]), y), xytext=(to_num(m1[2]), y),
                arrowprops=dict(arrowstyle="-|>", color=LANE_COLOR[m1[3]],
                                lw=1.6, alpha=0.55, shrinkA=14, shrinkB=14), zorder=2)
connect("esm2", "esmc")
connect("ankh", "ankh3")
connect("saprot", "saprot13")

# --- model nodes
for key, name, date, lin, params, dim, obj, pcc, std, best, hf in MODELS:
    x = to_num(date)
    y = LANE_Y[lin]
    col = LANE_COLOR[lin]

    # marker on the lane
    ax.scatter([x], [y], s=130, color=col, edgecolor="white", lw=1.5, zorder=5)

    # PCC chip just below the marker
    chip_y = y - 0.66
    chipc = pcc_color(pcc)
    txtc = "white" if pcc >= 0.835 else "#1b3a1b"
    chip = FancyBboxPatch((x - 0.135, chip_y - 0.17), 0.27, 0.34,
                          boxstyle="round,pad=0.015,rounding_size=0.06",
                          linewidth=1.6 if best else 0.8,
                          edgecolor="#d4af37" if best else "white",
                          facecolor=chipc, zorder=6)
    ax.add_patch(chip)
    ax.text(x, chip_y + 0.02, f"{pcc:.3f}", ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=txtc, zorder=7)
    ax.text(x, chip_y - 0.42, "S1102 PCC", ha="center", va="center",
            fontsize=6.8, color="#999999", zorder=7)

    # info card above the marker
    dd = date.strftime("%b %Y")
    l2 = f"{params} · {dim}-d"
    ax.text(x, y + 1.20, name, ha="center", va="bottom", fontsize=12,
            fontweight="bold", color=col, zorder=7)
    ax.text(x, y + 1.04, dd, ha="center", va="top", fontsize=8.4,
            color="#666666", fontweight="bold", zorder=7)
    ax.text(x, y + 0.80, l2, ha="center", va="top", fontsize=9,
            color="#333333", zorder=7)
    extra = ("  (large 0.823)" if key == "ankh3" else "")
    ax.text(x, y + 0.60, obj + extra, ha="center", va="top", fontsize=8.2,
            style="italic", color="#666666", zorder=7)
    ax.text(x, y + 0.42, hf, ha="center", va="top", fontsize=7.2,
            family="monospace", color="#7f7f7f", zorder=7)
    # stem from marker up to card
    ax.plot([x, x], [y + 0.12, y + 0.30], color=col, lw=0.9, alpha=0.5, zorder=3)
    if best:
        ax.text(x - 0.16, chip_y + 0.02, "★ best\nPLM", ha="right", va="center",
                fontsize=9, fontweight="bold", color="#b8860b", zorder=7,
                linespacing=0.95)

# --- title / subtitle
ax.text(x_min - 1.15, 14.0, "Embedding-model landscape for MuLAN ΔΔG",
        fontsize=19, fontweight="bold", color="#222222", ha="left")
ax.text(x_min - 1.15, 13.28,
        "HuggingFace release timeline (repo createdAt) · lineage · size / embedding dim / pretraining objective · headline S1102 10-fold-CV PCC (from-scratch MuLAN head)",
        fontsize=11, color="#555555", ha="left")

# --- performance colorbar-ish legend + paper ref
leg_items = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#888", markersize=11, label="model release (family lane)"),
    Line2D([0],[0], marker="s", color="w", markerfacecolor=_green(0.85), markersize=12, label="PCC chip — darker = higher"),
    Line2D([0],[0], color="#888", lw=1.6, label="within-family scaling / succession"),
]
ax.legend(handles=leg_items, loc="lower left", bbox_to_anchor=(0.005, 0.005),
          frameon=True, fontsize=9, framealpha=0.9, edgecolor="#dddddd")

ax.text(x_max - 0.03, -1.72,
        f"Paper MuLAN-Ankh reference = {PAPER_PCC:.3f}.  Dates = HuggingFace repo creation (createdAt, HF API).  ESM C 6B: open-weights repo dates to Dec 2025; the “esmc-6b-2024-12” model version trained Dec 2024.\n"
        "Ankh/Ankh3 weights predate their papers.  Ladder is non-monotonic in size — embedding quality tracks pretraining objective + data, not parameter count.",
        ha="right", va="center", fontsize=8.0, color="#999999", linespacing=1.5)

# cosmetics
for s in ["top", "right", "left", "bottom"]:
    ax.spines[s].set_visible(False)
ax.set_xticks([]); ax.set_yticks([])

plt.subplots_adjust(left=0.11, right=0.985, top=0.90, bottom=0.02)
fig.savefig(os.path.join(HERE, "model_timeline.png"), dpi=200, facecolor="white", bbox_inches="tight")
fig.savefig(os.path.join(HERE, "model_timeline.svg"), facecolor="white", bbox_inches="tight")
print("wrote model_timeline.png / .svg")
