#!/usr/bin/env python3
"""
Tier-1 augmentation — the two physically-exact rules as equations:
  (1) antisymmetry  (reverse-mutation)   ΔΔG(wt→mut) = −ΔΔG(mut→wt)
  (2) identity anchor                     ΔΔG(wt→wt) = 0

Run: python plot_augmentation.py  ->  augmentation.{png,svg}
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))

INK="#12303B"; TEAL="#1C7293"; GREEN="#2E7D32"; MUT="#6E7B82"; TXT="#21323A"
CARD="#F3F6F6"; ICE="#E3EEF0"; GREENT="#E4F0E4"; LINE="#D9E0E2"

fig, ax = plt.subplots(figsize=(13, 6.2), dpi=200)
fig.patch.set_facecolor("white"); ax.set_facecolor("white")
ax.set_xlim(0, 13); ax.set_ylim(0, 6.2); ax.axis("off")

# header
ax.text(0.7, 5.92, "TIER-1 AUGMENTATION", fontsize=12.5, fontweight="bold",
        color=TEAL, ha="left")

def card(cx, cy, w, h, badge_sym, badge_col, title, sub, eq, cap):
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor=CARD, edgecolor=LINE, linewidth=1.3, zorder=2))
    # title
    ax.text(cx-w/2+0.7, cy+h/2-0.5, title, ha="left", va="center", fontsize=15.5,
            fontweight="bold", color=INK, zorder=4)
    ax.text(cx-w/2+0.7, cy+h/2-0.86, sub, ha="left", va="center", fontsize=11,
            color=MUT, style="italic", zorder=4)
    # equation
    ax.text(cx+0.35, cy-0.05, eq, ha="center", va="center", fontsize=26,
            color=INK, zorder=4)
    # caption
    ax.text(cx+0.35, cy-h/2+0.42, cap, ha="center", va="center", fontsize=12,
            color=TXT, zorder=4)

card(6.5, 4.28, 11.8, 2.3, "⇄", TEAL,
     "Antisymmetry", "reverse-mutation",
     "ΔΔG(wt → mut)  =  − ΔΔG(mut → wt)",
     "for each labeled (wt→mut, ΔΔG = y),  add the reverse  (mut→wt, −y)")

card(6.5, 1.62, 11.8, 2.3, "0", GREEN,
     "Identity", "zero-change anchor",
     "ΔΔG(wt → wt)  =  0",
     "add identity anchors  (wt→wt, ΔΔG = 0)")

plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
fig.savefig(os.path.join(HERE, "augmentation.png"), dpi=200, facecolor="white", bbox_inches="tight")
fig.savefig(os.path.join(HERE, "augmentation.svg"), facecolor="white", bbox_inches="tight")
print("wrote augmentation.png / .svg")
