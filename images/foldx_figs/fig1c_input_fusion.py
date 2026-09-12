"""Figure 1c — structure fused into the input token, and the attempt to make it mutation-specific."""
from pathlib import Path as _Path

# Output beside this script, not in whatever directory it was authored in.
_OUT = _Path(__file__).resolve().parent
import sys; sys.path.insert(0, str(_OUT))
from mulanfig import *
from fig1_common import *

H = 1046
o = [header("1c")]

# ---- what 1c adds: the fused token, before the embedder -----------------
pre = []
pre.append(rrect(PRE_X - 14, ROW(0) - 18, PRE_W + 28, 4*ROW_P + 8, 10, "#f2faf4", GOOD, 2.2, dash="6 4"))
pre.append(arrow(SEQ_X + SEQ_W + 6, ROW(1) + 34, PRE_X - 18, ROW(1) + 34, INK, 1.5))
# the four encoder rows are WT-chain1 / WT-chain2 / mutant-chain1 / mutant-chain2,
# so the AA halves alternate — and the 3Di halves repeat, because chain 1's 3Di is
# the same string for the wild type and the mutant (the point panel 1a makes).
ROW_AA  = ["NEV", "NRL", "NEV", "NRL"]
ROW_3DI = ["dvq", "pvl", "dvq", "pvl"]
for i in range(4):
    y = ROW(i) + 17
    for k in range(3):
        bx = PRE_X + 6 + k*32
        pre.append(f'<rect x="{bx}" y="{y-13}" width="30" height="13" fill="#e8eef7" stroke="#8ea9c9" stroke-width="1.1"/>')
        pre.append(f'<rect x="{bx}" y="{y}" width="30" height="13" fill="#f4ecdf" stroke="#c2ab84" stroke-width="1.1"/>')
        pre.append(text(bx + 15, y - 6.5, ROW_AA[i][k], 9, INK))
        pre.append(text(bx + 15, y + 6.5, ROW_3DI[i][k], 9, PHYS_D))
    pre.append(arrow(PRE_X + PRE_W - 2, y, ENC_X - 8, y, INK, 1.3))
pre.append(text(PRE_X + PRE_W/2, ROW(0) - 26, "one fused AA + 3Di token", 9.5, GOOD, weight=650))
pre.append(text(PRE_X + PRE_W/2, ROW(3) + 50, "SaProt", 10.5, GOOD, weight=700))
pre.append(text(PRE_X + PRE_W/2, ROW(3) + 64, "structure enters at the input", 9.5, MUTED))

o.append(arch("1c", pre="".join(pre)))
o.append(leader(PRE_X + PRE_W/2, 468, 340, 500))

DY = 500
o.append(detail(DY, 504, "Structure at the input — the gain, and the mutation-specificity limit",
                "The same WT 3Di that hurt ProstT5 downstream helps SaProt at the input. The remaining question is whether the 3Di can be made to change with the mutation."))

# ================= left: SaProt result =====================================
ax = PANEL_X + 46
o.append(text(ax, DY + 82, "SaProt-650M, S1102, 10-fold", 12.5, INK, anchor="start", weight=650))
o.append(text(ax, DY + 102, "the embedder owns the fusion — it was pretrained on the joint vocabulary", 10.5, MUTED, anchor="start"))
BARX, BARW = ax + 176, 300
for i, (lab, val, sd, col) in enumerate([("3Di masked with #", 0.816, 0.069, "#c9c9c7"),
                                         ("real WT 3Di", 0.842, 0.050, GOOD)]):
    ry = DY + 142 + i*46
    o.append(text(BARX - 14, ry, lab, 11, INK_SOFT, anchor="end"))
    w = (val - 0.75) / 0.12 * BARW
    o.append(f'<rect x="{BARX}" y="{ry-10}" width="{w:.0f}" height="20" rx="3" fill="{col}" opacity="0.55"/>')
    o.append(f'<line x1="{BARX+w-sd/0.12*BARW:.0f}" y1="{ry}" x2="{BARX+w+sd/0.12*BARW:.0f}" y2="{ry}" stroke="{col}" stroke-width="2"/>')
    ew = sd/0.12*BARW
    o.append(text(BARX + w + ew + 14, ry, f"{val:.3f}", 12, INK, anchor="start", weight=650 if i else 400))
o.append(text(ax, DY + 244, "+0.026 PCC   ·   8 / 10 folds, paired", 13, GOOD, anchor="start", weight=700))
o.append(text(ax, DY + 272, "The same WT 3Di that cost ProstT5 −0.077 when concatenated", 11, INK_SOFT, anchor="start"))
o.append(text(ax, DY + 288, "downstream gains +0.026 when the embedder consumes it at the", 11, INK_SOFT, anchor="start"))
o.append(text(ax, DY + 304, "input. It is the fusion, not the data, that decides.", 11, INK_SOFT, anchor="start"))
o.append(text(ax, DY + 336, "Best sub-1B arm. Stacks with Tier-1 augmentation → 0.852.", 10.5, MUTED, anchor="start"))

# ================= right: the FoldX mutant-3Di attempt =====================
bx = PANEL_X + 720
o.append(f'<line x1="{bx-44}" y1="{DY+66}" x2="{bx-44}" y2="{DY+472}" stroke="{HAIRLINE}" stroke-width="1"/>')
o.append(text(bx, DY + 82, "The mutation-specific 3Di attempt", 12.5, INK, anchor="start", weight=650))
o.append(text(bx, DY + 102, "model each mutant with FoldX, then recompute its 3Di — probed on 1A22 (36 muts) + 1ACB L38P / L38G", 10.5, MUTED, anchor="start"))

# the attempted pipeline, left to right
py = DY + 148
steps = [("WT structure", "#cfe0f4", "#2f6fb5"), ("BuildModel", PHYS_F, PHYS_S), ("mutant structure", "#d5eeea", "#4b9d94")]
for i, (lab, f, st) in enumerate(steps):
    sx = bx + i*180
    o.append(rrect(sx, py, 148, 38, 8, f, st, 1.6))
    o.append(text(sx + 74, py + 19, lab, 11, INK, weight=600))
    if i < 2:
        o.append(arrow(sx + 152, py + 19, sx + 176, py + 19, PHYS_S, 1.6))
o.append(arrow(bx + 2*180 + 152, py + 19, bx + 2*180 + 176, py + 19, PHYS_S, 1.6))
o.append(text(bx + 2*180 + 182, py + 14, "mini3di", 10.5, INK_SOFT, anchor="start", weight=600))
o.append(text(bx + 2*180 + 182, py + 28, "→ 3Di", 10.5, MUTED, anchor="start"))

# the two 3Di strings
sy = DY + 226
for r, lab in enumerate(["WT 3Di", "mutant 3Di"]):
    ry = sy + r*24
    o.append(text(bx + 78, ry + 10, lab, 10.5, MUTED, anchor="end"))
    for i, ch in enumerate("pvqlpvncl"):
        cx = bx + 88 + i*22
        o.append(f'<rect x="{cx}" y="{ry}" width="20" height="20" fill="#f4ecdf" stroke="#c2ab84" stroke-width="1.1"/>')
        o.append(text(cx + 10, ry + 10, ch, 10.5, PHYS_D))
o.append(f'<path d="M {bx+290} {sy-6} L {bx+300} {sy-6} L {bx+300} {sy+50} L {bx+290} {sy+50}" fill="none" stroke="{MUTED}" stroke-width="1.3"/>')
o.append(text(bx + 312, sy + 22, "identical at every position", 11, BAD, anchor="start", weight=650))

o.append(text(bx, DY + 306, "0 / 39", 15, BAD, anchor="start", weight=700))
o.append(text(bx + 66, DY + 306, "mutations changed the 3Di anywhere — including 1ACB Leu38→Pro and →Gly,", 11, INK_SOFT, anchor="start"))
o.append(text(bx + 66, DY + 322, "the two most backbone-perturbing substitutions available", 11, INK_SOFT, anchor="start"))
o.append(text(bx, DY + 350, "0 / 232", 15, BAD, anchor="start", weight=700))
o.append(text(bx + 78, DY + 350, "backbone atoms moved between the WT re-model and the mutant (1KNE)", 11, INK_SOFT, anchor="start"))

o.append(rrect(bx, DY + 374, 780, 92, 9, "#fdf4f4", BAD, 1.6))
o.append(text(bx + 18, DY + 396, "Mechanism — backbone rigidity", 11.5, "#a33", anchor="start", weight=650))
o.append(text(bx + 18, DY + 418, "BuildModel is a side-chain repacker with sub-Ångström backbone motion; 3Di is a coarse 20-state", 10.5, INK_SOFT, anchor="start"))
o.append(text(bx + 18, DY + 434, "backbone descriptor. Sub-Å moves never cross a state boundary. Encoder-independent: a property of FoldX + 3Di,", 10.5, INK_SOFT, anchor="start"))
o.append(text(bx + 18, DY + 450, "so it holds for any 3Di-consuming embedder ⇒ SaProt's WT-3Di run (0.842) is the ceiling for this whole route.", 10.5, INK_SOFT, anchor="start"))

write(str(_OUT / "fig1c_input_fusion.svg"), W, H, "".join(o))
print("ok")
