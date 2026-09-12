"""Figure 1a — late concat of a WT-3Di block, between the embedder and Light Attention."""
from pathlib import Path as _Path

# Output beside this script, not in whatever directory it was authored in.
_OUT = _Path(__file__).resolve().parent
import sys; sys.path.insert(0, str(_OUT))
from mulanfig import *
from fig1_common import *

H = 1046
o = [header("1a")]

# ---- what 1a adds: the concat, in the reserved gap ------------------------
gap = []
gap.append(rrect(GAP_X - 12, ROW(0) - 18, GAP_W + 24, 4*ROW_P + 8, 10, "#fdf4f4", BAD, 2.2, dash="6 4"))
for i in range(4):
    y = ROW(i) + 17
    gap.append(arrow(ENC_X + ENC_W + 6, y, GAP_X - 2, y, INK, 1.3))
    gap.append(f'<rect x="{GAP_X+2}" y="{y-9}" width="26" height="18" fill="#e8eef7" stroke="#8ea9c9" stroke-width="1.2"/>')
    gap.append(circ_op(GAP_X + 44, y, "⊕", 8.5, 10))
    gap.append(f'<rect x="{GAP_X+60}" y="{y-9}" width="26" height="18" fill="#f4ecdf" stroke="#c2ab84" stroke-width="1.2"/>')
    gap.append(arrow(GAP_X + 92, y, GAP_X + 108, y, INK, 1.3))
    gap.append(f'<rect x="{GAP_X+110}" y="{y-11}" width="34" height="22" fill="{TENS_F}" stroke="{TENS_S}" stroke-width="1.2"/>')
    gap.append(arrow(GAP_X + 146, y, ATT_X - 6, y, INK, 1.3))
gap.append(text(GAP_X + 15, ROW(0) - 26, "AA", 9.5, "#3a5d8f", weight=600))
gap.append(text(GAP_X + 73, ROW(0) - 26, "3Di", 9.5, "#8a6207", weight=600))
gap.append(text(GAP_X + GAP_W/2 + 6, ROW(3) + 50, "[ L×1024 ] ⊕ [ L×1024 ] → [ L×2048 ]", 10, BAD, weight=600))
gap.append(text(GAP_X + GAP_W/2 + 6, ROW(3) + 64, "3Di half — one string, all four rows", 9.5, BAD, weight=600))

o.append(arch("1a", gap="".join(gap)))
o.append(leader(GAP_X + GAP_W/2, 468, 420, 500))

# ---- detail --------------------------------------------------------------
DY = 500
o.append(detail(DY, 520, "The 3Di block — provenance and cancellation",
                "3Di is computed from backbone coordinates. For a point mutant those coordinates do not exist: SKEMPI ships one solved structure per complex, the wild type."))

C1, C2, C3 = PANEL_X + 46, PANEL_X + 610, PANEL_X + 1054
o.append(f'<line x1="{C2-44}" y1="{DY+72}" x2="{C2-44}" y2="{DY+488}" stroke="{HAIRLINE}" stroke-width="1"/>')
o.append(f'<line x1="{C3-44}" y1="{DY+72}" x2="{C3-44}" y2="{DY+488}" stroke="{HAIRLINE}" stroke-width="1"/>')

# ===================== column 1 — where each string comes from ============
o.append(text(C1, DY + 86, "Provenance of the two 3Di strings", 12.5, INK, anchor="start", weight=650))

def dna(x, y):
    b = []
    for i, ch in enumerate("pvqlpv"):
        cx = x + i*21
        b.append(f'<rect x="{cx}" y="{y}" width="19" height="19" fill="#f4ecdf" stroke="#c2ab84" stroke-width="1.1"/>')
        b.append(text(cx + 9.5, y + 9.5, ch, 10, PHYS_D))
    return "".join(b)

# --- wild type: a solved structure
ry = DY + 118
o.append(text(C1, ry + 18, "wild type", 11, INK_SOFT, anchor="start", weight=600))
o.append(f'<path d="M {C1+84} {ry+26} q 11,-22 30,-18 q 22,4 15,22 q -6,16 -25,14 q -20,-2 -20,-18 z" '
         f'fill="#cfe0f4" stroke="#2f6fb5" stroke-width="1.5"/>')
o.append(f'<path d="M {C1+156} {ry+26} q -11,-22 -30,-18 q -22,4 -15,22 q 6,16 25,14 q 20,-2 20,-18 z" '
         f'fill="#d5eeea" stroke="#4b9d94" stroke-width="1.5"/>')
o.append(text(C1 + 120, ry + 48, "solved crystal structure", 9.5, MUTED))
o.append(arrow(C1 + 166, ry + 18, C1 + 196, ry + 18, INK, 1.5))
o.append(text(C1 + 181, ry + 4, "mini3di", 9, MUTED))
o.append(dna(C1 + 202, ry + 8))
o.append(text(C1 + 265, ry + 42, "WT 3Di", 10, PHYS_D, weight=650))

# --- mutant: no structure
ry2 = DY + 210
o.append(text(C1, ry2 + 18, "mutant", 11, INK_SOFT, anchor="start", weight=600))
o.append(rrect(C1 + 78, ry2 - 4, 84, 44, 8, "#f6f6f5", HAIRLINE, 1.4, dash="5 3"))
o.append(f'<line x1="{C1+92}" y1="{ry2+6}" x2="{C1+148}" y2="{ry2+30}" stroke="{BAD}" stroke-width="2.2"/>')
o.append(f'<line x1="{C1+148}" y1="{ry2+6}" x2="{C1+92}" y2="{ry2+30}" stroke="{BAD}" stroke-width="2.2"/>')
o.append(text(C1 + 120, ry2 + 52, "never solved — no coordinates", 9.5, BAD, weight=600))
o.append(arrow(C1 + 166, ry2 + 18, C1 + 196, ry2 + 18, HAIRLINE, 1.5, dash="4 3"))
o.append(dna(C1 + 202, ry2 + 8))
o.append(text(C1 + 265, ry2 + 42, "the wild type's string, reused", 10, BAD, weight=650))

# the fallback loop, drawn explicitly
o.append(f'<path d="M {C1+332} {ry+18} C {C1+392} {ry+18}, {C1+392} {ry2+18}, {C1+352} {ry2+18}" '
         f'fill="none" stroke="{BAD}" stroke-width="1.8" stroke-dasharray="5 3"/>')
o.append(arrow(C1 + 352, ry2 + 18, C1 + 336, ry2 + 18, BAD, 1.8))
o.append(text(C1 + 402, ry + 56, "nothing else", 10, BAD, anchor="start", weight=650))
o.append(text(C1 + 402, ry + 70, "to compute", 10, BAD, anchor="start", weight=650))
o.append(text(C1 + 402, ry + 84, "it from", 10, BAD, anchor="start", weight=650))

o.append(rrect(C1, DY + 300, 470, 88, 9, "#fdf4f4", BAD, 1.6))
o.append(text(C1 + 18, DY + 324, "The data constraint", 11.5, "#a33", anchor="start", weight=650))
o.append(text(C1 + 18, DY + 346, "Every variant of a complex has to share one structure, because", 10.5, INK_SOFT, anchor="start"))
o.append(text(C1 + 18, DY + 362, "only one was ever determined. So the 3Di half of the input is", 10.5, INK_SOFT, anchor="start"))
o.append(text(C1 + 18, DY + 378, "constant across the pair before training even begins — for any encoder.", 10.5, INK_SOFT, anchor="start"))

o.append(text(C1, DY + 418, "The obvious way out — model the mutant structure and recompute", 11, INK_SOFT, anchor="start"))
o.append(text(C1, DY + 434, "its 3Di — is panel 1c. It is a structurally guaranteed no-op.", 11, INK_SOFT, anchor="start"))

# ===================== column 2 — the algebra =============================
o.append(text(C2, DY + 86, "Effect on  mut − wt", 12.5, INK, anchor="start", weight=650))
for r, (lab, mut) in enumerate([("wild type", False), ("mutant", True)]):
    ry3 = DY + 122 + r*52
    o.append(text(C2 + 62, ry3 + 17, lab, 10.5, INK_SOFT, anchor="end"))
    o.append(f'<rect x="{C2+72}" y="{ry3}" width="112" height="34" fill="#e8eef7" stroke="#8ea9c9" stroke-width="1.4"/>')
    o.append(text(C2 + 128, ry3 + 17, "AA (mutated)" if mut else "AA", 10.5, "#3a5d8f", weight=600))
    o.append(circ_op(C2 + 200, ry3 + 17, "⊕", 9, 11))
    o.append(f'<rect x="{C2+216}" y="{ry3}" width="112" height="34" fill="#f4ecdf" stroke="#c2ab84" stroke-width="1.4"/>')
    o.append(text(C2 + 272, ry3 + 17, "WT 3Di", 10.5, PHYS_D, weight=600))
o.append(f'<path d="M {C2+336} {DY+122} L {C2+346} {DY+122} L {C2+346} {DY+208} L {C2+336} {DY+208}" fill="none" stroke="{BAD}" stroke-width="1.4"/>')
o.append(text(C2 + 352, DY + 165, "same", 10, BAD, anchor="start", weight=650))
o.append(circ_op(C2 + 52, DY + 232, "−", 10, 13))
o.append(f'<line x1="{C2+66}" y1="{DY+232}" x2="{C2+330}" y2="{DY+232}" stroke="{INK}" stroke-width="1.4"/>')
o.append(text(C2 + 128, DY + 260, "ΔAA  ≠ 0", 12.5, INK, weight=650))
o.append(text(C2 + 272, DY + 260, "Δ3Di  =  0", 12.5, BAD, weight=700))
o.append(text(C2, DY + 300, "The structure channel contributes a constant", 11, INK_SOFT, anchor="start"))
o.append(text(C2, DY + 316, "complex-level offset and zero mutation-specific", 11, INK_SOFT, anchor="start"))
o.append(text(C2, DY + 332, "signal — while doubling the width dilutes the", 11, INK_SOFT, anchor="start"))
o.append(text(C2, DY + 348, "AA signal that was working.", 11, INK_SOFT, anchor="start"))
o.append(text(C2, DY + 386, "run6b (3Di only) was skipped as algebraically", 10.5, MUTED, anchor="start"))
o.append(text(C2, DY + 402, "degenerate: fixed 3Di ⇒ mut_emb ≡ wt_emb.", 10.5, MUTED, anchor="start"))

# ===================== column 3 — the runs ================================
o.append(text(C3, DY + 86, "The four fusion variants", 12.5, INK, anchor="start", weight=650))
o.append(text(C3, DY + 106, "all five runs are ProstT5 — same split, optimizer and budget", 10.5, MUTED, anchor="start"))
RUNS = [("run2", "ProstT5 AA-only baseline", 0.740, True),
        ("run6c", "per-residue concat → 2048-d", 0.663, False),
        ("run6-E1", "+ learned gate on the 3Di block", 0.705, False),
        ("run6-B1", "3Di as head context, outside mut − wt", 0.673, False),
        ("run6-C3", "interface contacts as an attention bias", 0.662, False)]
BARX, BARW = C3 + 232, 200
BX0 = BARX + (0.740-0.60)/0.20*BARW
o.append(f'<line x1="{BX0}" y1="{DY+132}" x2="{BX0}" y2="{DY+330}" stroke="{HAIRLINE}" stroke-width="1.4" stroke-dasharray="4 3"/>')
o.append(text(BX0, DY + 122, "baseline", 10, MUTED))
for i, (rid, desc, val, base) in enumerate(RUNS):
    ry4 = DY + 146 + i*42
    o.append(text(C3, ry4, rid, 11.5, INK, anchor="start", weight=650))
    o.append(text(C3, ry4 + 15, desc, 10, INK_SOFT, anchor="start"))
    w = (val - 0.60) / 0.20 * BARW
    o.append(f'<rect x="{BARX}" y="{ry4-6}" width="{max(w,2):.0f}" height="16" rx="3" '
             f'fill="{"#9fb6d4" if base else BAD}" opacity="{0.9 if base else 0.55}"/>')
    o.append(text(BARX + w + 10, ry4 + 2, f"{val:.3f}", 11.5, INK, anchor="start", weight=650 if base else 400))
o.append(text(C3, DY + 364, "The learnable structure knobs", 11, INK, anchor="start", weight=600))
o.append(text(C3, DY + 384, "the E1 gate collapsed to 0.46 and the C3 attention bias to", 11, INK_SOFT, anchor="start"))
o.append(text(C3, DY + 400, "≈ 0.017 — the head was given the option to use structure", 11, INK_SOFT, anchor="start"))
o.append(text(C3, DY + 416, "and declined.", 11, INK_SOFT, anchor="start"))
o.append(text(C3, DY + 448, "Encoder: ProstT5 (AA mode) throughout.  Test PCC on S1102, single split.", 10.5, MUTED, anchor="start"))

write(str(_OUT / "fig1a_late_concat.svg"), W, H, "".join(o))
print("ok")
