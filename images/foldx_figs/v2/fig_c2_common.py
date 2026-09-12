"""Figure C v2 — shared main-architecture diagram for the four tabs Ci–Civ.

The top of every tab is the same MuLAN architecture (the visual abstract's left
panel, redrawn in the shared grammar). Each tab adds exactly one thing to it, and
expands that thing in a detail panel below. Comparing tabs side by side shows what
each route changed.

Conventions: solid arrow = data flow · dashed leader = "expanded below".
"""
import sys
from pathlib import Path as _Path
# `mulanfig` lives one level up, in images/foldx_figs/; anything local to this variant
# directory (e.g. this module) is found via _OUT. Both entries are resolved from
# __file__ so the four tabs that import this one run from any working directory.
_OUT = _Path(__file__).resolve().parent
sys.path[:0] = [str(_OUT), str(_OUT.parent)]
from mulanfig import *

W = 1700
PANEL_X, PANEL_W = 56, 1588

# ------------------------------------------------------------------ geometry
ARCH_Y   = 214
ROW_P    = 52
def ROW(i): return ARCH_Y + i*ROW_P          # top of encoder row i

SEQ_X,  SEQ_W  = 84,  228
PRE_X,  PRE_W  = 330, 104                     # reserved: Ciii's AA+3Di fusion
ENC_X,  ENC_W  = 452, 104
GAP_X,  GAP_W  = 578, 150                     # reserved: Ci's late concat
ATT_X,  ATT_W  = 748, 128
SLAB_X         = 908
CMB_X          = 978
MINUS_X        = 1064
VEC_X, VCELL, VN = 1100, 20, 7
SLOT_X         = VEC_X + VN*VCELL             # 1240
VEC_Y          = ROW(1) + 34
LIN_X          = 1330
DDG_X          = 1470

TABS = [("Ci",   "late concat"),
        ("Cii",  "hidden layer"),
        ("Ciii", "input fusion"),
        ("Civ",  "score channel")]
TAB_SUB = {
    "Ci":   "① Late concat of a WT-3Di block, between the embedder and Light Attention   ·   encoder: ProstT5",
    "Cii":  "② A different hidden layer of the embedder   ·   probed on ProstT5 and Ankh-large, run in ProstT5",
    "Ciii": "③ Structure fused into the input token, and the attempt to make it mutation-specific   ·   encoder: SaProt-650M",
    "Civ":  "④ A physics ΔΔG into the head's add_scores slot   ·   encoder: Ankh-large",
}
TAB_OUT = {
    "Ci":   ("bad",  "hurts",             "0.740 → 0.663"),
    "Cii":  ("bad",  "hurts",             "0.648  ·  0.429"),
    "Ciii": ("good", "helps  /  no-op",   "0.842   ·   0 / 39"),
    "Civ":  ("warn", "real, but bounded", "+0.009  ·  +0.017"),
}


def header(active):
    o = [text(56, 48, "Four routes for structure into MuLAN", 24, INK, anchor="start", weight=650),
         text(56, 74, TAB_SUB[active], 13, INK_SOFT, anchor="start")]
    # tab strip
    x = 56
    for tag, nm in TABS:
        on = (tag == active)
        w = 172
        o.append(rrect(x, 96, w, 34, 7, INK if on else "#fff",
                       INK if on else HAIRLINE, 1.6))
        o.append(text(x + 20, 113, tag, 12.5, "#fff" if on else MUTED,
                      anchor="start", weight=700))
        o.append(text(x + 62, 113, nm, 11.5, "#fff" if on else MUTED, anchor="start"))
        x += w + 8
    kind, word, nums = TAB_OUT[active]
    o.append(badge(PANEL_X + PANEL_W - 460, 98, word, kind, 190, 30))
    o.append(text(PANEL_X + PANEL_W, 113, nums, 13, INK, anchor="end", weight=600))
    return "".join(o)


def arch(active, gap=None, pre=None, enc_mark=False, slot_mark=False, dim_others=False):
    """The shared architecture. `gap` / `pre` are extra SVG drawn in the reserved
    columns; the *_mark flags highlight what this tab changes."""
    o = [rrect(PANEL_X, 150, PANEL_W, 318, 13, "#fcfcfb", HAIRLINE, 1.6)]
    o.append(text(PANEL_X + 22, 172, "MuLAN", 12, MUTED, anchor="start", weight=700))

    dim = 0.30 if dim_others else 1.0

    def g(inner, faded=True):
        return f'<g opacity="{dim if faded else 1}">{inner}</g>'

    # --- sequences
    b = []
    for k, (lab, y0, mi) in enumerate([("wild-type pair", ROW(0) - 12, None),
                                       ("mutant pair",    ROW(2) - 12, 4)]):
        b.append(rrect(SEQ_X, y0, SEQ_W, 82, 8, "#fff", INK_SOFT, 1.2, dash="4 3"))
        b.append(seqbox(SEQ_X + 12, y0 + 11, "NEVFHTGIK", 20, mi if k else None, 11))
        b.append(seqbox(SEQ_X + 12, y0 + 45, "NRLEVVLT" if k else "NRLEVVPT", 20, 7 if k else None, 11))
        b.append(text(SEQ_X + SEQ_W/2, y0 + 94, lab + "   [ L ]", 10, MUTED))
    o.append(g("".join(b)))

    # --- pre-embedder column
    if pre is not None:
        o.append(pre)
    else:
        o.append(g(arrow(SEQ_X + SEQ_W + 8, ROW(1) + 34, ENC_X - 8, ROW(1) + 34, INK, 1.5)))

    # --- frozen encoders
    b = [rrect(ENC_X - 11, ROW(0) - 12, ENC_W + 22, 4*ROW_P - 4, 8, "none", "#7a8ed6", 1.4)]
    b.append(snowflake(ENC_X + ENC_W + 2, ROW(0) - 2))
    for i in range(4):
        b.append(box(ENC_X, ROW(i), ENC_W, 34, "Encoder", PLM_F, PLM_S, 12, 8))
    b.append(text(ENC_X + ENC_W/2, ROW(3) + 50, "frozen PLM", 10.5, "#5a6fb5", weight=600))
    b.append(text(ENC_X + ENC_W/2, ROW(3) + 64, "[ L × H ]", 9.5, MUTED))
    o.append(g("".join(b), faded=not enc_mark))
    if enc_mark:
        o.append(rrect(ENC_X - 16, ROW(0) - 18, ENC_W + 32, 4*ROW_P + 8, 10, "none", WARN, 2.4, dash="6 4"))

    # --- reserved gap (Ci's concat)
    if gap is not None:
        o.append(gap)
    else:
        for i in range(4):
            o.append(g(arrow(ENC_X + ENC_W + 8, ROW(i) + 17, ATT_X - 8, ROW(i) + 17, INK, 1.4)))

    # --- light attention
    b = []
    for i in range(4):
        b.append(box(ATT_X, ROW(i), ATT_W, 34, "Light Attention", ATT_F, ATT_S, 11, 8))
    o.append(g("".join(b)))

    # --- slabs, pairing, difference
    b = []
    for i in range(4):
        b.append(slab(SLAB_X, ROW(i) + 6, 32, 20, 6))
    b.append(f'<path d="M {SLAB_X+42} {ROW(0)+16} L {SLAB_X+58} {ROW(0)+16} L {SLAB_X+58} {ROW(1)+16} L {SLAB_X+42} {ROW(1)+16}" fill="none" stroke="{INK}" stroke-width="1.3"/>')
    b.append(f'<path d="M {SLAB_X+42} {ROW(2)+16} L {SLAB_X+58} {ROW(2)+16} L {SLAB_X+58} {ROW(3)+16} L {SLAB_X+42} {ROW(3)+16}" fill="none" stroke="{INK}" stroke-width="1.3"/>')
    b.append(slab(CMB_X, ROW(0) + 32, 28, 20, 6))
    b.append(slab(CMB_X, ROW(2) + 32, 28, 20, 6))
    b.append(text(CMB_X + 14, ROW(0) + 20, "wt  [ H ]", 9.5, MUTED))
    b.append(text(CMB_X + 14, ROW(2) + 20, "mut  [ H ]", 9.5, MUTED))
    b.append(f'<path d="M {CMB_X+34} {ROW(0)+42} L {MINUS_X} {ROW(0)+42} L {MINUS_X} {VEC_Y-12}" fill="none" stroke="{INK}" stroke-width="1.3"/>')
    b.append(f'<path d="M {CMB_X+34} {ROW(2)+42} L {MINUS_X} {ROW(2)+42} L {MINUS_X} {VEC_Y+12}" fill="none" stroke="{INK}" stroke-width="1.3"/>')
    b.append(circ_op(MINUS_X, VEC_Y, "−", 11, 15))
    o.append(g("".join(b)))

    # --- pooled vector + the slot
    b = [arrow(MINUS_X + 12, VEC_Y, VEC_X - 4, VEC_Y, INK, 1.4)]
    for i in range(VN):
        b.append(f'<rect x="{VEC_X+i*VCELL}" y="{VEC_Y-10}" width="{VCELL}" height="20" fill="{TENS_F}" stroke="{TENS_S}" stroke-width="1.1"/>')
    b.append(text(VEC_X + VN*VCELL/2, VEC_Y + 26, "pooled  mut − wt", 9.5, MUTED))
    b.append(text(VEC_X + VN*VCELL/2, VEC_Y + 38, "[ H ]", 9.5, MUTED, weight=600))
    o.append(g("".join(b)))
    o.append(f'<rect x="{SLOT_X}" y="{VEC_Y-10}" width="{VCELL}" height="20" fill="{SLOT_F}" '
             f'stroke="{PHYS_S}" stroke-width="{2.6 if slot_mark else 2.0}" opacity="{1 if slot_mark else dim}"/>')
    ly = (VEC_Y - 34, VEC_Y - 22) if slot_mark else (VEC_Y + 26, VEC_Y + 38)
    o.append(f'<g opacity="{1 if slot_mark else dim}">'
             + text(SLOT_X + VCELL/2, ly[0], "add_scores", 9.5, PHYS_D, weight=650)
             + text(SLOT_X + VCELL/2, ly[1], "[ 1 ]", 9.5, PHYS_D, weight=600) + '</g>')
    if slot_mark:
        o.append(rrect(SLOT_X - 7, VEC_Y - 17, VCELL + 14, 34, 6, "none", PHYS_S, 2.2, dash="5 3"))

    # --- head
    b = [arrow(SLOT_X + VCELL + 6, VEC_Y, LIN_X - 6, VEC_Y, INK, 1.4)]
    b.append(para(LIN_X, VEC_Y - 16, 108, 32, "Linear", 12, size=12))
    b.append(text(LIN_X + 54, VEC_Y + 30, "H + 1 → 1", 9.5, MUTED))
    b.append(arrow(LIN_X + 112, VEC_Y, DDG_X - 12, VEC_Y, INK, 1.5))
    b.append(text(DDG_X - 4, VEC_Y - 4, "ΔΔG", 15, INK, anchor="start", weight=650))
    b.append(text(DDG_X - 4, VEC_Y + 13, "kcal/mol", 9.5, MUTED, anchor="start"))
    o.append(g("".join(b)))
    return "".join(o)


def leader(x1, y1, x2, y2):
    """Dashed 'expanded below' leader — never a flow arrow."""
    return (f'<path d="M {x1} {y1} L {x1} {(y1+y2)/2} L {x2} {(y1+y2)/2} L {x2} {y2}" '
            f'fill="none" stroke="{MUTED}" stroke-width="1.4" stroke-dasharray="5 4"/>'
            f'<circle cx="{x2}" cy="{y2}" r="3" fill="{MUTED}"/>')


def detail(y, h, title, sub=""):
    o = [rrect(PANEL_X, y, PANEL_W, h, 13, "#fcfcfb", HAIRLINE, 1.6)]
    o.append(text(PANEL_X + 22, y + 24, title, 14.5, INK, anchor="start", weight=650))
    if sub:
        o.append(text(PANEL_X + 22, y + 44, sub, 11.5, INK_SOFT, anchor="start"))
    return "".join(o)

ARCH_BOTTOM = 468


# ---- sequential blue ramp, used to shade encoder layers by probe PCC -------
_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
         "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281"]
def ramp(pcc, lo=0.65, hi=0.81):
    t = max(0.0, min(1.0, (pcc - lo) / (hi - lo)))
    return _BLUE[int(round(t * (len(_BLUE) - 1)))]
