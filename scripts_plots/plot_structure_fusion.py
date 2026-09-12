#!/usr/bin/env python3
"""
Structure-channel data-flow diagram: ProstT5 (late / side fusion, negative) vs
SaProt (input fusion, positive). Shows inputs, per-residue vector dimensions,
the AA vs 3Di vectors, the FoldX mutant-structure route, the MuLAN siamese head,
and the mut-wt algebra that explains why one cancels and the other survives.

Run: python plot_structure_fusion.py  ->  structure_fusion.{png,svg}
Deps: matplotlib.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))

INK="#12303B"; TEAL="#1C7293"; GREEN="#2E7D32"; CORAL="#C65B43"; AMBER="#B8860B"
SLATE="#5C6B73"; TXT="#21323A"; MUT="#6E7B82"; ICE="#E3EEF0"; CARD="#F2F5F6"
CORALT="#F7E7E2"; GREENT="#E4F0E4"; AMBERT="#F6EBD3"; TEALT="#E3EEF2"; WHITE="#FFFFFF"

fig, ax = plt.subplots(figsize=(16, 11), dpi=200)
fig.patch.set_facecolor("white"); ax.set_facecolor("white")
ax.set_xlim(0, 16); ax.set_ylim(0, 11)
ax.axis("off")

def box(cx, cy, w, h, text, fc, ec, tc=TXT, fs=10.5, lw=1.4, family=None, weight="normal", rnd=0.02):
    ax.add_patch(FancyBboxPatch((cx-w/2, cy-h/2), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rnd}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tc,
            zorder=4, family=family, fontweight=weight, linespacing=1.25)

def arrow(x1, y1, x2, y2, color=SLATE, lw=1.9, style="-|>"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle=style, color=color, lw=lw, shrinkA=3, shrinkB=3,
                        mutation_scale=15), zorder=2)

def badge(cx, cy, sym, color):
    ax.add_patch(plt.Circle((cx, cy), 0.21, color=color, zorder=6))
    ax.text(cx, cy+0.005, sym, ha="center", va="center", fontsize=14, color="white",
            fontweight="bold", zorder=7)

def vbar(cx, cy, w, h, fc, ec, label, dim, tc, hatch=None, cell=None):
    """single vector bar with a dim caption underneath; optional highlighted cell."""
    ax.add_patch(Rectangle((cx-w/2, cy-h/2), w, h, facecolor=fc, edgecolor=ec,
                           linewidth=1.5, zorder=3, hatch=hatch))
    if cell is not None:  # highlight one residue cell
        cw = w/16
        ax.add_patch(Rectangle((cx-w/2+cell*cw, cy-h/2), cw, h, facecolor=GREEN,
                               edgecolor="white", linewidth=0.8, zorder=4))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=9.3, color=tc,
            zorder=5, fontweight="bold")
    ax.text(cx, cy-h/2-0.16, dim, ha="center", va="top", fontsize=9,
            color=MUT, zorder=5, family="monospace")

# ============================================================ header
ax.text(8.0, 10.68, "Structure channels in MuLAN ΔΔG:  ProstT5 (late fusion)  vs  SaProt (input fusion)",
        ha="center", va="center", fontsize=17, fontweight="bold", color=INK)
ax.plot([8.0, 8.0], [1.9, 9.75], color="#dfe6e7", lw=1.4, zorder=1)

# ============================================================ LEFT: ProstT5
LX = 4.0; aa, di = 2.15, 5.75
ax.text(LX, 9.55, "ProstT5  —  late / side fusion", ha="center", va="center",
        fontsize=14, fontweight="bold", color=CORAL)
badge(1.05, 9.55, "✗", CORAL)
ax.text(LX, 9.22, "sequence-only PLM (T5) with a bolted-on 3Di channel", ha="center", va="center",
        fontsize=9.8, color=MUT, style="italic")
box(LX, 8.82, 5.4, 0.44, "S1102 CV10:  0.663 – 0.705      <      AA-only  0.740",
    CORALT, CORAL, tc="#7a2f22", fs=10.5, weight="bold")

# sources
box(aa, 8.05, 2.75, 0.66, "WT amino-acid sequence\n(same string in mut & wt)", ICE, "#c9d6d9", fs=9.6)
box(di, 8.05, 2.75, 0.66, "WT structure\n(SKEMPI 2.0 PDB)", ICE, "#c9d6d9", fs=9.6)
# proc 1
box(aa, 6.95, 2.75, 0.66, "ProstT5 · AA mode\n⟨AA2fold⟩ prefix", TEALT, TEAL, fs=9.6)
box(di, 6.95, 2.75, 0.66, "mini3di\nbackbone → 3Di tokens", CARD, "#cfd8da", fs=9.6)
# proc 2 : AA vector  |  ProstT5 fold
vbar(aa, 5.95, 2.6, 0.34, TEALT, TEAL, "AA embedding", "ℝ^(L×1024)", TEAL)
box(di, 5.9, 2.75, 0.66, "ProstT5 · fold mode\n⟨fold2AA⟩", CARD, "#cfd8da", fs=9.6)
# proc 3 : 3Di vector (WT-fixed)
vbar(di, 4.85, 2.6, 0.34, CORALT, CORAL, "3Di embedding — WT-fixed", "ℝ^(L×1024)", "#7a2f22", hatch="////")
# concat segmented bar
cy=3.65; cw=4.9; ax.text(LX, 4.16, "concatenate", ha="center", fontsize=9.5, color=MUT, style="italic")
ax.add_patch(Rectangle((LX-cw/2, cy-0.19), cw/2, 0.38, facecolor=TEALT, edgecolor=TEAL, lw=1.5, zorder=3))
ax.add_patch(Rectangle((LX, cy-0.19), cw/2, 0.38, facecolor=CORALT, edgecolor=CORAL, lw=1.5, zorder=3, hatch="////"))
ax.text(LX-cw/4, cy, "AA · 1024", ha="center", va="center", fontsize=9, color=TEAL, fontweight="bold", zorder=5)
ax.text(LX+cw/4, cy, "3Di · 1024", ha="center", va="center", fontsize=9, color="#7a2f22", fontweight="bold", zorder=5)
ax.text(LX, cy-0.35, "ℝ^(L×2048)", ha="center", va="top", fontsize=9, color=MUT, family="monospace")
ax.text(LX, 2.98, "variants: learned gate (E1) · pooled WT-3Di context (B1) · interface-bias +1 dim (C3, 1025-d)",
        ha="center", va="center", fontsize=8.3, color=MUT, style="italic")
# head
box(LX, 2.4, 3.7, 0.62, "MuLAN Light-Attention head\nsiamese:  out = f(mut) − f(wt)", INK, INK, tc="white", fs=9.6)
# arrows
arrow(aa,7.72,aa,7.28); arrow(di,7.72,di,7.28)
arrow(aa,6.62,aa,6.13); arrow(di,6.62,di,6.24)
arrow(di,5.57,di,5.03)
arrow(aa,5.78,LX-0.7,3.86); arrow(di,4.68,LX+0.8,3.86)
arrow(LX,3.46,LX,2.72)
# algebra callout
box(LX,1.6,7.0,0.74,
    "[ ΔAA·1024  ‖  (3Di_WT − 3Di_WT) = 0·1024 ]\n→  3Di cancels in mut − wt;  extra width dilutes AA",
    CORALT, CORAL, tc="#7a2f22", fs=9.4, family="monospace")

# ============================================================ RIGHT: SaProt
RX = 12.0
ax.text(RX, 9.55, "SaProt  —  input fusion", ha="center", va="center",
        fontsize=14, fontweight="bold", color=GREEN)
badge(9.05, 9.55, "✓", GREEN)
ax.text(RX, 9.22, "structure-aware vocabulary — AA and 3Di fused per residue", ha="center", va="center",
        fontsize=9.8, color=MUT, style="italic")
box(RX, 8.82, 5.4, 0.44, "S1102 CV10:  0.842   vs   0.816 seq-only ctrl    (+0.026)",
    GREENT, GREEN, tc="#1e4620", fs=10.5, weight="bold")

# two separate inputs (mirrors ProstT5) — but fused at the INPUT
saa, sdi = 10.15, 13.85
box(saa, 8.05, 2.7, 0.72, "AA sequence\n(changes on mutation)", TEALT, TEAL, tc=TXT, fs=9.4)
box(sdi, 8.05, 3.0, 0.72, "WT structure → mini3di\n3Di state  (FoldX identical)", CORALT, CORAL, tc="#7a2f22", fs=9.4)
ax.text(RX, 7.08, "fuse per residue:  AA char ⊕ 3Di state  →  one SA token  (446-vocab)",
        ha="center", fontsize=9.3, color=MUT, style="italic")
# per-residue fused-token row: each cell = AA (top) over 3Di (bottom)
toks=[("L","p"),("V","d"),("A","p"),("K","a"),("G","s"),("T","d"),("I","v")]
cw=0.6; gp=0.06; n=len(toks); tot=n*cw+(n-1)*gp; x0=RX-tot/2; ty=6.42
for i,(a,dd) in enumerate(toks):
    cx=x0+cw/2+i*(cw+gp)
    ax.add_patch(Rectangle((cx-cw/2, ty), cw, 0.24, facecolor=TEALT, edgecolor=TEAL, lw=1.1, zorder=3))
    ax.add_patch(Rectangle((cx-cw/2, ty-0.24), cw, 0.24, facecolor=CORALT, edgecolor=CORAL, lw=1.1, zorder=3))
    ax.text(cx, ty+0.12, a, ha="center", va="center", fontsize=10.5, color=TEAL, fontweight="bold", zorder=4)
    ax.text(cx, ty-0.12, dd, ha="center", va="center", fontsize=10.5, color="#7a2f22", fontweight="bold", zorder=4)
ax.text(x0-0.14, ty+0.12, "AA", ha="right", va="center", fontsize=8.6, color=TEAL, fontweight="bold")
ax.text(x0-0.14, ty-0.12, "3Di", ha="right", va="center", fontsize=8.6, color="#7a2f22", fontweight="bold")
ax.text(RX, ty-0.42, "AA + 3Di concatenated per residue — fused at the INPUT (vs ProstT5's late concat of the two embeddings)",
        ha="center", va="top", fontsize=8.5, color=MUT, style="italic")

box(RX, 5.15, 4.6, 0.62, "SaProt  (ESM2-650M backbone)\nsingle fused stream", CARD, "#cfd8da", fs=9.8)
ax.text(RX, 4.4, "SA-token embedding", ha="center", va="bottom", fontsize=9.3,
        color="#1e4620", fontweight="bold", zorder=5)
vbar(RX, 4.08, 4.4, 0.30, GREENT, GREEN, "", "ℝ^(L×1280)", "#1e4620", cell=13)
box(RX, 3.25, 5.6, 0.52, "mutated site: only the AA half flips  (A·p) → (V·p)  —  3Di half stays WT", GREENT, GREEN, tc="#1e4620", fs=9.3, weight="bold")
box(RX, 2.4, 4.6, 0.5, "MuLAN Light-Attention head   ( out = f(mut) − f(wt) )", INK, INK, tc="white", fs=9.4)
arrow(saa,7.69,RX-1.5,6.74); arrow(sdi,7.69,RX+1.5,6.74)
arrow(RX,6.14,RX,5.48); arrow(RX,4.84,RX,4.46); arrow(RX,3.93,RX,3.53); arrow(RX,2.99,RX,2.67)
box(RX,1.6,7.0,0.74,
    "token(mut)=(mutAA,3Di)  ≠  token(wt)=(wtAA,3Di)\n→  emb_mut − emb_wt ≠ 0;  structure signal SURVIVES",
    GREENT, GREEN, tc="#1e4620", fs=9.4, family="monospace")

# ============================================================ shared FoldX strip
fy=0.58
ax.add_patch(FancyBboxPatch((0.5, fy-0.42), 15.0, 0.84, boxstyle="round,pad=0.02,rounding_size=0.03",
    facecolor=AMBERT, edgecolor=AMBER, linewidth=1.5, zorder=3))
badge(1.15, fy, "✗", AMBER)
ax.text(2.0, fy+0.15, "FoldX on mutant structure:",
        ha="left", va="center", fontsize=10.5, color="#6b5210", fontweight="bold", zorder=5)
ax.text(2.0, fy-0.17, "FoldX moves side chains, not the backbone — and 3Di encodes the backbone, so the mutant's 3Di ≡ WT's 3Di  (0/39 mutations changed).",
        ha="left", va="center", fontsize=10, color="#6b5210", zorder=5)

plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
fig.savefig(os.path.join(HERE, "structure_fusion.png"), dpi=200, facecolor="white", bbox_inches="tight")
fig.savefig(os.path.join(HERE, "structure_fusion.svg"), facecolor="white", bbox_inches="tight")
print("wrote structure_fusion.png / .svg")
