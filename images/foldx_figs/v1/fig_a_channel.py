"""Figure A — The FoldX channel: how a physics ΔΔG enters MuLAN."""
from pathlib import Path as _Path

# Output beside this script, not in whatever directory it was authored in.
_OUT = _Path(__file__).resolve().parent
import sys
# `mulanfig` lives one level up, in images/foldx_figs/; anything local to this variant
# directory (e.g. fig_c2_common) is found via _OUT. Both entries are resolved from
# __file__ so the script runs from any working directory.
sys.path[:0] = [str(_OUT), str(_OUT.parent)]
from mulanfig import *

W, H = 1700, 1072
o = []

o.append(text(56, 50, "The FoldX channel — physics pipeline, two arms, one head slot",
              25, INK, anchor="start", weight=650))
o.append(text(56, 80, "Two arms feed one head slot. The channel is computed once from structure, so it is reused unchanged across every PLM backbone and every split.",
              13.5, INK_SOFT, anchor="start"))

# ================================================================ LEFT: physics
LX, LY, LW, LH = 56, 108, 588, 624
o.append(rrect(LX, LY, LW, LH, 14, "#fffdf7", PHYS_S, 2.0))
o.append(text(LX + 22, LY + 26, "FoldX — the physics side, no learning", 15, PHYS_D, anchor="start", weight=650))

cx = LX + LW/2

# --- crystal structure glyph
gy = LY + 46
o.append(f'<path d="M {cx-124} {gy+34} q 20,-34 52,-29 q 36,6 27,36 q -7,24 -40,22 q -34,-2 -39,-29 z" '
         f'fill="#cfe0f4" stroke="#2f6fb5" stroke-width="1.8"/>')
o.append(f'<path d="M {cx+120} {gy+34} q -20,-34 -52,-29 q -36,6 -27,36 q 7,24 40,22 q 34,-2 39,-29 z" '
         f'fill="#d5eeea" stroke="#4b9d94" stroke-width="1.8"/>')
o.append(text(cx - 82, gy + 32, "g1", 12, "#2f6fb5", weight=650))
o.append(text(cx + 80, gy + 32, "g2", 12, "#33807a", weight=650))
o.append(f'<line x1="{cx}" y1="{gy}" x2="{cx}" y2="{gy+66}" stroke="{MUTED}" stroke-width="1.5" stroke-dasharray="4 3"/>')
o.append(text(cx, gy + 80, "SKEMPI crystal complex · two interface chain groups", 11.5, INK_SOFT))

def step(y, title, sub, h=42):
    b = [rrect(cx - 202, y, 404, h, 9, PHYS_F, PHYS_S, 1.6)]
    b.append(text(cx, y + 16, title, 13, PHYS_D, weight=650))
    b.append(text(cx, y + 31, sub, 10.5, "#9a7a30"))
    return "".join(b)

Y1 = gy + 94
o.append(arrow(cx, Y1 - 12, Y1 and cx, Y1 - 2, PHYS_S, 1.8))
o.append(step(Y1, "RepairPDB", "one-time per structure (~3 min) · 211 already on disk"))
o.append(arrow(cx, Y1 + 42, cx, Y1 + 58, PHYS_S, 1.8))
Y2 = Y1 + 60
o.append(step(Y2, "BuildModel", "side-chain repack → mutant PDB · ~8 s per mutation"))
o.append(arrow(cx, Y2 + 42, cx, Y2 + 58, PHYS_S, 1.8))
Y3 = Y2 + 60
o.append(step(Y3, "AnalyseComplex  (g1 | g2)", "run on the wild-type and on the mutant"))

# --- the subtraction
YS = Y3 + 42
o.append(arrow(cx, YS, cx, YS + 16, PHYS_S, 1.8))
o.append(circ_op(cx - 118, YS + 34, "−", 11, 15))
o.append(text(cx + 6, YS + 30, "ΔΔG\u1d62\u2099\u209c  =  IE(mutant) − IE(wild-type)", 13, INK, weight=600))
o.append(text(cx + 6, YS + 46, "IE = interaction energy, kcal/mol", 10, MUTED))

# --- the two outputs, side by side
YO = YS + 62
o.append(f'<line x1="{cx}" y1="{YO-10}" x2="{cx}" y2="{YO}" stroke="{PHYS_S}" stroke-width="1.8"/>')
o.append(f'<path d="M {cx-128} {YO} L {cx+128} {YO}" stroke="{PHYS_S}" stroke-width="1.8" fill="none"/>')
o.append(arrow(cx - 128, YO, cx - 128, YO + 18, PHYS_S, 1.8))
o.append(arrow(cx + 128, YO, cx + 128, YO + 18, PHYS_S, 1.8))
YV = YO + 22
# scalar
o.append(f'<rect x="{cx-160}" y="{YV}" width="64" height="20" fill="{SLOT_F}" stroke="{PHYS_S}" stroke-width="1.8"/>')
o.append(text(cx - 128, YV + 10, "1 scalar", 10.5, PHYS_D, weight=600))
o.append(text(cx - 128, YV + 36, "ΔΔG interaction energy", 11, INK, weight=600))
o.append(text(cx - 128, YV + 51, "kcal/mol  ·  → Arm A", 10.5, MUTED))
# 12-term horizontal strip
sxx = cx + 128 - 6*17
for i in range(12):
    o.append(f'<rect x="{sxx+i*17}" y="{YV}" width="17" height="20" fill="{SLOT_F}" stroke="{PHYS_S}" stroke-width="1.2"/>')
o.append(text(cx + 128, YV + 36, "12 decomposed ΔΔG terms", 11, INK, weight=600))
o.append(text(cx + 128, YV + 51, "kcal/mol each  ·  → Arm B", 10.5, MUTED))

# --- the term list, 3 columns
ty = YV + 76
o.append(f'<line x1="{LX+22}" y1="{ty-12}" x2="{LX+LW-22}" y2="{ty-12}" stroke="{PHYS_S}" stroke-width="1" opacity="0.5"/>')
o.append(text(LX + 22, ty + 2, "The 12 terms — each a ΔΔG in kcal/mol, mutant − wild-type:",
              11.5, INK_SOFT, anchor="start", weight=600))
TERMS = ["1.  ΔΔG interaction energy", "2.  ΔΔG backbone H-bond", "3.  ΔΔG sidechain H-bond",
         "4.  ΔΔG Van der Waals", "5.  ΔΔG electrostatics", "6.  ΔΔG solvation polar",
         "7.  ΔΔG solvation hydrophobic", "8.  ΔΔG Van der Waals clashes", "9.  ΔΔG entropy sidechain",
         "10. ΔΔG entropy mainchain", "11. ΔΔG torsional clash", "12. ΔΔG backbone clash"]
for i, t in enumerate(TERMS):
    col, row = i // 6, i % 6
    o.append(text(LX + 30 + col*272, ty + 24 + row*16, t, 10,
                  PHYS_D if i == 0 else INK_SOFT, anchor="start", weight=650 if i == 0 else 400))
o.append(text(LX + 22, ty + 128, "Term 1 is the scalar Arm A uses — Arm B strictly contains Arm A.",
              10.5, PHYS_D, anchor="start", style="font-style:italic"))
o.append(text(LX + 22, ty + 145, "Names as emitted by FoldX AnalyseComplex.",
              10, MUTED, anchor="start"))

# ================================================================ RIGHT: MuLAN
RX, RY, RW, RH = 676, 108, 968, 624
o.append(rrect(RX, RY, RW, RH, 14, "#fcfcfb", INK, 2.0))
o.append(text(RX + 22, RY + 26, "MuLAN — the entry point", 15, INK, anchor="start", weight=650))

sx = RX + 26
for k, (lab, y0, mi) in enumerate([("Wild-type protein pair", RY + 56, None),
                                   ("Mutant protein pair", RY + 162, 4)]):
    o.append(rrect(sx, y0, 232, 84, 8, "#fff", INK_SOFT, 1.3, dash="4 3"))
    o.append(seqbox(sx + 14, y0 + 12, "NEVFHTGIK", 20, mi if k else None, 11))
    o.append(seqbox(sx + 14, y0 + 46, "NRLEVVPT" if k == 0 else "NRLEVVLT", 20, 7 if k else None, 11))
    o.append(text(sx + 116, y0 + 96, lab + "   [ L ]", 10.5, INK_SOFT))

ex = sx + 258
o.append(rrect(ex - 10, RY + 48, 132, 216, 8, "none", "#7a8ed6", 1.4))
o.append(snowflake(ex + 106, RY + 60))
for i in range(4):
    o.append(box(ex, RY + 60 + i*52, 106, 34, "Encoder", PLM_F, PLM_S, 12, 8))
o.append(text(ex + 53, RY + 280, "frozen PLM", 10.5, "#5a6fb5", weight=600))
o.append(text(ex + 53, RY + 294, "output [ L × H ]", 9.5, MUTED))

ax = ex + 132
for i in range(4):
    o.append(box(ax, RY + 60 + i*52, 130, 34, "Light Attention", ATT_F, ATT_S, 11, 8))

bx = ax + 140
for i in range(4):
    o.append(slab(bx, RY + 66 + i*52, 34, 20, 6))
o.append(f'<path d="M {bx+44} {RY+76} L {bx+62} {RY+76} L {bx+62} {RY+128} L {bx+44} {RY+128}" fill="none" stroke="{INK}" stroke-width="1.3"/>')
o.append(f'<path d="M {bx+44} {RY+180} L {bx+62} {RY+180} L {bx+62} {RY+232} L {bx+44} {RY+232}" fill="none" stroke="{INK}" stroke-width="1.3"/>')
o.append(slab(bx + 70, RY + 92, 30, 20, 6))
o.append(slab(bx + 70, RY + 196, 30, 20, 6))
o.append(text(bx + 85, RY + 82, "wt  [ H ]", 10.5, MUTED))
o.append(text(bx + 85, RY + 186, "mut  [ H ]", 10.5, MUTED))

dx = bx + 124
o.append(f'<path d="M {bx+106} {RY+102} L {dx} {RY+102} L {dx} {RY+144}" fill="none" stroke="{INK}" stroke-width="1.4"/>')
o.append(f'<path d="M {bx+106} {RY+206} L {dx} {RY+206} L {dx} {RY+166}" fill="none" stroke="{INK}" stroke-width="1.4"/>')
o.append(circ_op(dx, RY + 155, "−", 12, 16))
o.append(text(dx - 2, RY + 128, "mut − wt", 10.5, MUTED))

# pooled difference as a HORIZONTAL vector; the slot is literally the appended cell
VX, VY, CELL = dx + 26, RY + 145, 21
o.append(arrow(dx + 13, RY + 155, VX - 4, RY + 155, INK, 1.5))
for i in range(7):
    o.append(f'<rect x="{VX+i*CELL}" y="{VY}" width="{CELL}" height="20" fill="{TENS_F}" stroke="{TENS_S}" stroke-width="1.2"/>')
SX = VX + 7*CELL
o.append(f'<rect x="{SX}" y="{VY}" width="{CELL}" height="20" fill="{SLOT_F}" stroke="{PHYS_S}" stroke-width="2.4"/>')

# the physics arrives from above — top-down, so it never competes with the left-right path
o.append(arrow(SX + CELL/2, VY - 40, SX + CELL/2, VY - 4, PHYS_S, 1.8))
o.append(text(SX + CELL/2 - 4, VY - 52, "ΔΔG from FoldX", 10.5, PHYS_D, anchor="end", weight=650))
o.append(text(SX + CELL/2 - 4, VY - 40, "kcal/mol", 9.5, MUTED, anchor="end"))

# labels sit under the marks they name — nothing hangs off the right edge
o.append(f'<path d="M {VX} {VY+26} L {VX} {VY+31} L {SX-2} {VY+31} L {SX-2} {VY+26}" fill="none" stroke="{TENS_S}" stroke-width="1.2"/>')
o.append(text(VX + (SX-VX)/2, VY + 44, "pooled difference", 10.5, MUTED))
o.append(text(VX + (SX-VX)/2, VY + 57, "[ H ]", 10, MUTED, weight=600))
o.append(f'<path d="M {SX} {VY+26} L {SX} {VY+31} L {SX+CELL} {VY+31} L {SX+CELL} {VY+26}" fill="none" stroke="{PHYS_S}" stroke-width="1.2"/>')
o.append(text(SX + CELL/2, VY + 44, "add_scores", 10.5, PHYS_D, weight=700))
o.append(text(SX + CELL/2, VY + 57, "[ 1 ]", 10, PHYS_D, weight=600))

hy = RY + 258
o.append(f'<path d="M {VX+(SX+CELL-VX)/2} {VY+64} L {VX+(SX+CELL-VX)/2} {hy}" stroke="{INK}" stroke-width="1.5" fill="none"/>')
o.append(arrow(VX + (SX+CELL-VX)/2, hy, VX + (SX+CELL-VX)/2, hy + 14, INK, 1.5))
o.append(para(VX + (SX+CELL-VX)/2 - 67, hy + 16, 134, 32, "Linear"))
o.append(text(VX + (SX+CELL-VX)/2, hy + 62, "H + 1  →  1", 10, MUTED))
o.append(arrow(VX + (SX+CELL-VX)/2, hy + 70, VX + (SX+CELL-VX)/2, hy + 88, INK, 1.6))
o.append(text(VX + (SX+CELL-VX)/2, hy + 102, "ΔΔG", 16, INK, weight=650))
o.append(text(VX + (SX+CELL-VX)/2, hy + 120, "kcal/mol", 10, MUTED))

# --- fact strip filling the lower right panel
fy = RY + RH - 148
o.append(rrect(RX + 26, fy, RW - 52, 92, 10, "#fff", HAIRLINE, 1.4))
facts = [
    ("Head input width", "H + 1 — identical for both arms; the regression head is unchanged."),
    ("Only structural input", "Everything left of the slot is the published sequence-only model."),
    ("Computed once", "Model-independent physics: reused across 10 backbones × 9 split tiers, no re-run."),
]
for i, (k, v) in enumerate(facts):
    o.append(text(RX + 44, fy + 24 + i*24, k, 11, PHYS_D, anchor="start", weight=700))
    o.append(text(RX + 178, fy + 24 + i*24, v, 11, INK_SOFT, anchor="start"))
o.append(text(RX + RW/2, RY + RH - 32,
              "The channel adds one input, not a new pathway.",
              11.5, INK_SOFT, style="font-style:italic"))

# ================================================================ BOTTOM: the two arms
BY = 758
o.append(rrect(56, BY, 1588, 292, 14, "#fcfcfb", HAIRLINE, 1.8))
o.append(text(78, BY + 28, "The two arms — one slot, two producers", 16, INK, anchor="start", weight=650))

def arm(x, tag, name, cfg, params, body_fn, w=740):
    b = [rrect(x, BY + 48, w, 224, 11, "#fff", PHYS_S, 1.6)]
    b.append(f'<rect x="{x}" y="{BY+48}" width="6" height="224" fill="{PHYS_S}" rx="3"/>')
    b.append(text(x + 26, BY + 76, tag, 13, PHYS_D, anchor="start", weight=700))
    b.append(text(x + 26 + 62, BY + 76, name, 13.5, INK, anchor="start", weight=600))
    b.append(text(x + 26, BY + 258, cfg, 10.5, MUTED, anchor="start", style="font-family:ui-monospace,monospace"))
    b.append(text(x + w - 26, BY + 258, params, 11.5, PHYS_D, anchor="end", weight=650))
    b.append(body_fn(x))
    return "".join(b)

def slot_glyph(x, y, n=5):
    g, _ = hslot(x, y, n, 16, 18, labels=False)
    b = [g]
    sx = x + n*16
    b.append(text(x + n*8, y + 31, "[ H ]", 9.5, MUTED))
    b.append(text(sx + 8, y + 31, "[ 1 ]", 9.5, PHYS_D, weight=600))
    b.append(text(x + (n+1)*8, y + 47, "same slot", 9.5, PHYS_D, weight=650))
    return "".join(b)

def bodyA(x):
    b = []
    cy = BY + 158
    b.append(f'<rect x="{x+42}" y="{cy-11}" width="86" height="24" fill="{SLOT_F}" stroke="{PHYS_S}" stroke-width="1.8"/>')
    b.append(text(x + 85, cy + 1, "1 scalar", 11, PHYS_D, weight=600))
    b.append(text(x + 85, cy - 30, "ΔΔG interaction energy", 11, INK, weight=600))
    b.append(text(x + 85, cy + 30, "kcal/mol, standardised per fold", 10, MUTED))
    b.append(arrow(x + 136, cy, x + 246, cy, PHYS_S, 1.8))
    b.append(text(x + 194, cy - 14, "straight in", 10.5, INK_SOFT))
    b.append(slot_glyph(x + 254, cy - 9))
    b.append(text(x + 430, cy - 34, "Head cost — one weight", 12, INK, anchor="start", weight=600))
    b.append(text(x + 430, cy - 12, "The head learns one additional weight", 11.5, INK_SOFT, anchor="start"))
    b.append(text(x + 430, cy + 4, "on the physics number.", 11.5, INK_SOFT, anchor="start"))
    b.append(text(x + 430, cy + 34, "Safe default: +0.006–0.013 on every", 11, MUTED, anchor="start"))
    b.append(text(x + 430, cy + 48, "benchmark tested.", 11, MUTED, anchor="start"))
    return "".join(b)

def bodyB(x):
    b = []
    cy = BY + 158
    b.append(vstack(x + 46, cy - 46, 34, 8, 12, SLOT_F, PHYS_S))
    b.append(text(x + 63, cy - 60, "12 ΔΔG terms", 11, INK, weight=600))
    b.append(arrow(x + 86, cy, x + 128, cy, PHYS_S, 1.8))
    b.append(rrect(x + 132, cy - 34, 152, 68, 9, "#eef2fb", "#7a8ed6", 1.6))
    b.append(text(x + 208, cy - 16, "MLP", 12, "#3a4b8f", weight=700))
    b.append(text(x + 208, cy + 2, "12 → 16 → ReLU", 10.5, INK_SOFT))
    b.append(text(x + 208, cy + 16, "→ Dropout → 1", 10.5, INK_SOFT))
    b.append(text(x + 208, cy + 48, "trained jointly with the head", 10, MUTED))
    b.append(arrow(x + 288, cy, x + 326, cy, PHYS_S, 1.8))
    b.append(slot_glyph(x + 334, cy - 9))
    b.append(text(x + 476, cy - 34, "Head cost — one small module", 12, INK, anchor="start", weight=600))
    b.append(text(x + 476, cy - 12, "Clears the linear ceiling the scalar", 11.5, INK_SOFT, anchor="start"))
    b.append(text(x + 476, cy + 4, "sits exactly on (+0.011 out-of-fold).", 11.5, INK_SOFT, anchor="start"))
    b.append(text(x + 476, cy + 34, "Upside on ≥2k-mutation sets, and the", 11, MUTED, anchor="start"))
    b.append(text(x + 476, cy + 48, "arm that clears FoldX-alone on ESM-C 6B.", 11, MUTED, anchor="start"))
    return "".join(b)

o.append(arm(78, "Arm A", "FoldX scalar  (Stage 1)", "models/config/lightatt_addscores_config.json",
             "+1 parameter", bodyA, 740))
o.append(arm(860, "Arm B", "FoldX 12-term MLP  (Stage 2)", "models/config/lightatt_addscores_mlp_config.json",
             "+226 parameters", bodyB, 762))

write(str(_OUT / "fig_a_foldx_channel.svg"), W, H, "".join(o))
print("ok")
