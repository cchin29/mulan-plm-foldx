"""Figure A1 (appendix) — the FoldX channel reference sheet.

Panel 1d is the main slide: it shows *where* the arms attach. This sheet is the
reference behind it — how the number is computed, what the twelve terms are, and
the exact two configurations. It stands alone if sent on its own to answer Q1.
"""
from pathlib import Path as _Path

# Output beside this script, not in whatever directory it was authored in.
_OUT = _Path(__file__).resolve().parent
import sys; sys.path.insert(0, str(_OUT))
from mulanfig import *

W, H = 1700, 850
PX, PW = 56, 1588
o = []

o.append(text(56, 40, "APPENDIX  ·  FIGURE A1", 11, MUTED, anchor="start", weight=700))
o.append(text(56, 66, "The FoldX channel — definitions and provenance", 24, INK, anchor="start", weight=650))
o.append(text(56, 92,
              "Everything behind panel 1d: how the number is computed, what the twelve terms are, and the exact two configurations. "
              "Encoder-independent — the channel is physics, reused unchanged across every backbone and every split.",
              12.5, INK_SOFT, anchor="start"))

# ============================================================ computation
CY, CH_ = 118, 296
o.append(rrect(PX, CY, 828, CH_, 13, "#fffdf7", PHYS_S, 1.8))
o.append(text(PX + 22, CY + 26, "Computation", 14.5, PHYS_D, anchor="start", weight=650))
o.append(text(PX + 22, CY + 46, "per complex, serial, resumable — one cached JSON per complex", 10.5, MUTED, anchor="start"))

STEPS = [("RepairPDB", "one-time per structure", "~3 min · 211 on disk"),
         ("BuildModel", "side-chain repack → mutant PDB", "~8 s per mutation"),
         ("AnalyseComplex", "on the two SKEMPI chain groups", "run on WT and mutant")]
for i, (lab, sub, cost) in enumerate(STEPS):
    sx = PX + 26 + i*258
    o.append(rrect(sx, CY + 72, 226, 58, 9, PHYS_F, PHYS_S, 1.6))
    o.append(text(sx + 113, CY + 92, lab, 13, PHYS_D, weight=650))
    o.append(text(sx + 113, CY + 110, sub, 9.5, "#9a7a30"))
    o.append(text(sx + 113, CY + 146, cost, 10, MUTED))
    if i < 2:
        o.append(arrow(sx + 230, CY + 101, sx + 254, CY + 101, PHYS_S, 1.8))

o.append(f'<line x1="{PX+26}" y1="{CY+166}" x2="{PX+802}" y2="{CY+166}" stroke="{PHYS_S}" stroke-width="1" opacity="0.45"/>')
o.append(circ_op(PX + 44, CY + 192, "−", 11, 14))
o.append(text(PX + 66, CY + 186, "ΔΔGᵢₙₜ  =  IE(mutant) − IE(wild-type)", 13, INK, anchor="start", weight=650))
o.append(text(PX + 66, CY + 204, "IE = interaction energy, kcal/mol.  The same subtraction is taken term by term for the 12-vector.",
              10.5, MUTED, anchor="start"))

o.append(text(PX + 26, CY + 232, "Chain mapping is validated, not assumed", 10.5, PHYS_D, anchor="start", weight=650))
o.append(text(PX + 26, CY + 248, "The dataset mutation carries a role chain (A/B = seq1/seq2); SKEMPI's Mutation(s)_cleaned carries the real PDB chain.",
              10.5, INK_SOFT, anchor="start"))
o.append(text(PX + 26, CY + 263, "Matched on (wt_aa, pos, mut_aa), duplicates disambiguated role→group, every mapping checked against the repaired PDB.",
              10.5, INK_SOFT, anchor="start"))
o.append(text(PX + 26, CY + 281, "Coverage — 99.2 % single-point, 98.7 % multi-point, 99.1 % combined. Uncovered rows fall back to the per-fold standardised mean (0).",
              10.5, MUTED, anchor="start"))

# ============================================================ the 12 terms
TX, TW = PX + 852, PW - 852
o.append(rrect(TX, CY, TW, CH_, 13, "#fcfcfb", HAIRLINE, 1.8))
o.append(text(TX + 22, CY + 26, "The twelve terms", 14.5, INK, anchor="start", weight=650))
o.append(text(TX + 22, CY + 46, "each a ΔΔG in kcal/mol, mutant − wild-type · names as emitted by AnalyseComplex",
              10.5, MUTED, anchor="start"))
TERMS = ["1.  ΔΔG interaction energy", "2.  ΔΔG backbone H-bond", "3.  ΔΔG sidechain H-bond",
         "4.  ΔΔG Van der Waals", "5.  ΔΔG electrostatics", "6.  ΔΔG solvation polar",
         "7.  ΔΔG solvation hydrophobic", "8.  ΔΔG Van der Waals clashes", "9.  ΔΔG entropy sidechain",
         "10. ΔΔG entropy mainchain", "11. ΔΔG torsional clash", "12. ΔΔG backbone clash"]
for i, t in enumerate(TERMS):
    col, row = i // 6, i % 6
    tx, ty = TX + 32 + col*354, CY + 84 + row*25
    if i == 0:
        o.append(rrect(tx - 10, ty - 12, 330, 24, 5, SLOT_F, PHYS_S, 1.4))
    o.append(text(tx, ty, t, 11.5, PHYS_D if i == 0 else INK_SOFT, anchor="start", weight=650 if i == 0 else 400))
o.append(f'<line x1="{TX+22}" y1="{CY+244}" x2="{TX+TW-22}" y2="{CY+244}" stroke="{HAIRLINE}" stroke-width="1"/>')
o.append(text(TX + 22, CY + 264, "Term 1 is the scalar Arm A uses, so Arm B strictly contains Arm A.", 11, INK, anchor="start", weight=600))
o.append(text(TX + 22, CY + 281, "FoldX's Interaction Energy is a weighted sum of the components — the 12-vector is near-collinear by construction.",
              10.5, MUTED, anchor="start"))

# ============================================================ the spec table
SY = CY + CH_ + 26
o.append(rrect(PX, SY, PW, 358, 13, "#fcfcfb", HAIRLINE, 1.8))
o.append(text(PX + 22, SY + 26, "The two arms — specification", 14.5, INK, anchor="start", weight=650))
o.append(text(PX + 22, SY + 46, "both write into the same add_scores slot; the regression head width is identical between them",
              10.5, MUTED, anchor="start"))

C0, C1_, C2_ = PX + 26, PX + 286, PX + 880
GX = PX + PW - 300          # the "where it lands" gutter
TBL_R = GX - 26             # table content stops here
ROWS = [
    ("input", "1 scalar — ΔΔG interaction energy (kcal/mol)", "12-vector — all twelve ΔΔG terms (kcal/mol)", 0),
    ("transform", "none; standardised per fold", "MLP  12 → 16 → ReLU → Dropout → 1, trained jointly with the head", 0),
    ("added parameters", "1", "226", 0),
    ("head input width", "H + 1", "H + 1   (unchanged)", 0),
    ("config", "models/config/lightatt_addscores_config.json", "models/config/lightatt_addscores_mlp_config.json", 1),
    ("flags", "add_scores: true", "add_scores: true · zs_mlp: true", 1),
    ("", "", "zs_input_dim: 12 · zs_mlp_hidden: 16", 1),
    ("measured lift", "+0.009 PCC, 8 / 10 folds", "+0.017 PCC, 8 / 10 folds  (p ≈ 0.02)", 0),
]
o.append(f'<line x1="{C0}" y1="{SY+74}" x2="{TBL_R}" y2="{SY+74}" stroke="{INK}" stroke-width="1.2"/>')
o.append(text(C1_, SY + 64, "Arm A — FoldX scalar   (Stage 1)", 12.5, PHYS_D, anchor="start", weight=700))
o.append(text(C2_, SY + 64, "Arm B — FoldX 12-term MLP   (Stage 2)", 12.5, PHYS_D, anchor="start", weight=700))
ry = SY + 98
for i, (k, a, b, mono) in enumerate(ROWS):
    st = "font-family:ui-monospace,monospace" if mono else ""
    if k:
        if i % 2 == 0:
            o.append(f'<rect x="{C0}" y="{ry-14}" width="{TBL_R-C0}" height="28" fill="#f4f4f2"/>')
        o.append(text(C0 + 6, ry, k, 11, INK, anchor="start", weight=650))
        o.append(text(C1_, ry, a, 10 if mono else 11, INK_SOFT, anchor="start", style=st))
        o.append(text(C2_, ry, b, 10 if mono else 11, INK_SOFT, anchor="start", style=st))
        ry += 30
    else:
        o.append(text(C2_, ry - 12, b, 10, INK_SOFT, anchor="start", style=st))
        ry += 16

o.append(f'<line x1="{C0}" y1="{SY+308}" x2="{TBL_R}" y2="{SY+308}" stroke="{HAIRLINE}" stroke-width="1"/>')
o.append(f'<line x1="{TBL_R+12}" y1="{SY+64}" x2="{TBL_R+12}" y2="{SY+308}" stroke="{HAIRLINE}" stroke-width="1"/>')
o.append(text(C0 + 6, SY + 330, "Measured on Ankh-large, S1102 CV10, 300 epochs / patience 30, paired folds.", 10.5, INK_SOFT, anchor="start"))
o.append(text(C0 + 470, SY + 330, "Scalar is the safe default across the four SKEMPI benchmarks (+0.006–0.013); the MLP is the upside on ≥2k-mutation sets, and the arm that clears FoldX-alone on ESM-C 6B.",
              10.5, MUTED, anchor="start"))

# --- where it lands, in its own gutter so it never reads as a table row
gx = GX
o.append(text(gx, SY + 64, "Where it lands", 11.5, INK, anchor="start", weight=650))
g, _ = hslot(gx, SY + 88, 5, 15, 18, labels=False)
o.append(g)
o.append(text(gx + 37, SY + 120, "[ H ]", 9.5, MUTED))
o.append(text(gx + 82, SY + 120, "[ 1 ]", 9.5, PHYS_D, weight=650))
o.append(text(gx, SY + 146, "add_scores — one number,", 10.5, PHYS_D, anchor="start", weight=600))
o.append(text(gx, SY + 161, "appended to the pooled vector", 10.5, INK_SOFT, anchor="start"))
o.append(text(gx, SY + 184, "→  Linear (H + 1 → 1)", 10.5, INK_SOFT, anchor="start"))
o.append(text(gx, SY + 199, "→  ΔΔG, kcal/mol", 10.5, INK_SOFT, anchor="start"))
o.append(text(gx, SY + 236, "See Figure 1d for the arms", 10.5, MUTED, anchor="start", style="font-style:italic"))
o.append(text(gx, SY + 251, "in context.", 10.5, MUTED, anchor="start", style="font-style:italic"))

write(str(_OUT / "figA1_foldx_channel.svg"), W, H, "".join(o))
print("ok")
