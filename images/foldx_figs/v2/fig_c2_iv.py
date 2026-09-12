"""Civ — a physics ΔΔG into the head's add_scores slot."""
from pathlib import Path as _Path

# Output beside this script, not in whatever directory it was authored in.
_OUT = _Path(__file__).resolve().parent
import sys
# `mulanfig` lives one level up, in images/foldx_figs/; anything local to this variant
# directory (e.g. fig_c2_common) is found via _OUT. Both entries are resolved from
# __file__ so the script runs from any working directory.
sys.path[:0] = [str(_OUT), str(_OUT.parent)]
from mulanfig import *
from fig_c2_common import *

H = 1046
o = [header("Civ")]
o.append(arch("Civ", slot_mark=True))

DY = 500
o.append(detail(DY, 504, "The FoldX arms — computation and entry point",
                "Physics computed from structure alone: no learning, model-independent, computed once and reused across every backbone and every split.   ·   Full term list, configs and coverage: appendix reference sheet."))

# ================= the physics chain, left to right ========================
py = DY + 118
CH = [("RepairPDB", "one-time per structure"),
      ("BuildModel", "mutant PDB, ~8 s/mutation"),
      ("AnalyseComplex", "on chain groups g1 | g2")]
for i, (lab, sub) in enumerate(CH):
    sx = PANEL_X + 46 + i*212
    o.append(rrect(sx, py, 184, 46, 9, PHYS_F, PHYS_S, 1.6))
    o.append(text(sx + 92, py + 17, lab, 12, PHYS_D, weight=650))
    o.append(text(sx + 92, py + 33, sub, 9.5, "#9a7a30"))
    if i < 2:
        o.append(arrow(sx + 188, py + 23, sx + 208, py + 23, PHYS_S, 1.7))
sx = PANEL_X + 46 + 3*212
o.append(arrow(sx - 24, py + 23, sx - 4, py + 23, PHYS_S, 1.7))
o.append(circ_op(sx + 14, py + 23, "−", 11, 14))
o.append(text(sx + 34, py + 16, "ΔΔGᵢₙₜ = IE(mutant) − IE(wild-type)", 12, INK, anchor="start", weight=600))
o.append(text(sx + 34, py + 34, "IE = interaction energy, kcal/mol", 9.5, MUTED, anchor="start"))

# ================= the two arms ============================================
AY = DY + 208
ARM_W = 500
for j, (tag, name, params, cfg) in enumerate([
        ("Arm A", "FoldX scalar", "+1 parameter", "lightatt_addscores_config.json"),
        ("Arm B", "FoldX 12-term MLP", "+226 parameters", "lightatt_addscores_mlp_config.json")]):
    x = PANEL_X + 46 + j*(ARM_W + 40)
    o.append(rrect(x, AY, ARM_W, 174, 10, "#fff", PHYS_S, 1.6))
    o.append(f'<rect x="{x}" y="{AY}" width="6" height="174" fill="{PHYS_S}" rx="3"/>')
    o.append(text(x + 24, AY + 26, tag, 12, PHYS_D, anchor="start", weight=700))
    o.append(text(x + 78, AY + 26, name, 13, INK, anchor="start", weight=600))
    o.append(text(x + ARM_W - 22, AY + 26, params, 11.5, PHYS_D, anchor="end", weight=650))
    o.append(text(x + 24, AY + 158, cfg, 10, MUTED, anchor="start", style="font-family:ui-monospace,monospace"))
    cy = AY + 84
    if j == 0:
        o.append(f'<rect x="{x+32}" y="{cy-12}" width="88" height="24" fill="{SLOT_F}" stroke="{PHYS_S}" stroke-width="1.8"/>')
        o.append(text(x + 76, cy, "1 scalar", 11, PHYS_D, weight=600))
        o.append(text(x + 76, cy - 26, "ΔΔG interaction energy", 11, INK, weight=600))
        o.append(text(x + 76, cy + 26, "kcal/mol, standardised per fold", 9.5, MUTED))
        o.append(arrow(x + 126, cy, x + 186, cy, PHYS_S, 1.7))
        o.append(text(x + 156, cy - 14, "straight in", 9.5, INK_SOFT))
        o.append(text(x + 210, cy - 12, "No extra module — the head learns", 11, INK_SOFT, anchor="start"))
        o.append(text(x + 210, cy + 4, "one additional weight on the number.", 11, INK_SOFT, anchor="start"))
        o.append(text(x + 210, cy + 30, "Safe default: +0.006–0.013 across the four SKEMPI benchmarks.", 10.5, MUTED, anchor="start"))
    else:
        for i in range(12):
            o.append(f'<rect x="{x+32}" y="{cy-33+i*5.6}" width="46" height="5.1" fill="{SLOT_F}" stroke="{PHYS_S}" stroke-width="0.9"/>')
        o.append(text(x + 55, cy + 44, "12 ΔΔG terms", 10.5, INK, weight=600))
        o.append(text(x + 55, cy + 57, "kcal/mol each", 9.5, MUTED))
        o.append(arrow(x + 84, cy, x + 112, cy, PHYS_S, 1.7))
        o.append(rrect(x + 116, cy - 30, 140, 60, 9, "#eef2fb", "#7a8ed6", 1.6))
        o.append(text(x + 186, cy - 14, "MLP", 12, "#3a4b8f", weight=700))
        o.append(text(x + 186, cy + 2, "12 → 16 → ReLU", 10, INK_SOFT))
        o.append(text(x + 186, cy + 16, "→ Dropout → 1", 10, INK_SOFT))
        o.append(text(x + 186, cy + 44, "trained jointly with the head", 9.5, MUTED))
        o.append(arrow(x + 260, cy, x + 288, cy, PHYS_S, 1.7))
        o.append(text(x + 296, cy - 12, "Clears the linear ceiling the", 11, INK_SOFT, anchor="start"))
        o.append(text(x + 296, cy + 4, "scalar sits exactly on (+0.011).", 11, INK_SOFT, anchor="start"))
        o.append(text(x + 296, cy + 30, "Clears FoldX-alone on ESM-C 6B.", 10.5, MUTED, anchor="start"))

# ================= the feed into the slot ==================================
# both arms converge, then one solid amber line rises into the slot: this IS data flow
JX = PANEL_X + 46 + ARM_W + 20
RISER = PANEL_X + 46 + (ARM_W + 40) + ARM_W + 18
o.append(f'<path d="M {PANEL_X+46+ARM_W/2} {AY+174} L {PANEL_X+46+ARM_W/2} {AY+198} L {JX} {AY+198}" fill="none" stroke="{PHYS_S}" stroke-width="2"/>')
o.append(f'<path d="M {PANEL_X+46+(ARM_W+40)+ARM_W/2} {AY+174} L {PANEL_X+46+(ARM_W+40)+ARM_W/2} {AY+198} L {JX} {AY+198}" fill="none" stroke="{PHYS_S}" stroke-width="2"/>')
o.append(f'<circle cx="{JX}" cy="{AY+198}" r="4" fill="{PHYS_S}"/>')
o.append(text(JX + 14, AY + 214, "one number per mutation → the same slot", 11, PHYS_D, anchor="start", weight=600))
o.append(f'<path d="M {JX} {AY+198} L {RISER} {AY+198} L {RISER} {DY+80} L {SLOT_X+VCELL/2} {DY+80} '
         f'L {SLOT_X+VCELL/2} {VEC_Y+22}" fill="none" stroke="{PHYS_S}" stroke-width="2.2"/>')
o.append(arrow(SLOT_X + VCELL/2, VEC_Y + 20, SLOT_X + VCELL/2, VEC_Y + 11, PHYS_S, 2.2))
o.append(text(SLOT_X + VCELL/2 + 14, DY + 68, "into add_scores", 10.5, PHYS_D, anchor="start", weight=650))

# ================= the ceiling note ========================================
o.append(rrect(PANEL_X + 1124, DY + 208, 464, 174, 9, "#fffaf0", PHYS_S, 1.6))
o.append(text(PANEL_X + 1144, DY + 228, "Channel value, out-of-fold", 12, PHYS_D, anchor="start", weight=700))
o.append(text(PANEL_X + 1144, DY + 244, "S1102, out-of-fold, vs Ankh-large", 9.5, MUTED, anchor="start"))
FACTS = [("+0.407", "raw FoldX ΔΔG vs the experimental label"),
         ("+0.35",  "already inside Ankh-large's own predictions"),
         ("+0.20",  "orthogonal to its residual — the part worth having"),
         ("+0.011", "out-of-fold ceiling from all 12 terms")]
for i, (n, d) in enumerate(FACTS):
    fy = DY + 268 + i*26
    o.append(text(PANEL_X + 1144, fy, n, 12, INK, anchor="start", weight=700))
    o.append(text(PANEL_X + 1202, fy, d, 10.5, INK_SOFT, anchor="start"))
o.append(text(PANEL_X + 1144, DY + 376, "Measured lift +0.009 (scalar) · +0.017 (MLP):", 10.5, MUTED, anchor="start"))
o.append(text(PANEL_X + 1144, DY + 391, "Ankh-large, S1102 CV10, 300 ep / patience 30.", 10.5, MUTED, anchor="start"))

o.append(rrect(PANEL_X + 1124, DY + 404, 464, 66, 9, "#fff", PHYS_S, 1.6))
o.append(text(PANEL_X + 1144, DY + 424, "Against FoldX alone", 11.5, PHYS_D, anchor="start", weight=700))
o.append(text(PANEL_X + 1144, DY + 443, "the channel is indistinguishable on 8 of 9 split", 10.5, INK_SOFT, anchor="start"))
o.append(text(PANEL_X + 1144, DY + 458, "tiers (Ankh-large). Only ESM-C 6B clears it.", 10.5, INK_SOFT, anchor="start"))

write(str(_OUT / "fig_c2_iv_score_channel.svg"), W, H, "".join(o))
print("ok")
