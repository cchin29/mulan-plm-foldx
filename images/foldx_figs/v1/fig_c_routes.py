"""Figure C — Four routes for structure into MuLAN, located on the pipeline."""
from pathlib import Path as _Path

# Output beside this script, not in whatever directory it was authored in.
_OUT = _Path(__file__).resolve().parent
import sys
# `mulanfig` lives one level up, in images/foldx_figs/; anything local to this variant
# directory (e.g. fig_c2_common) is found via _OUT. Both entries are resolved from
# __file__ so the script runs from any working directory.
sys.path[:0] = [str(_OUT), str(_OUT.parent)]
from mulanfig import *

W, H = 1700, 1262
o = []

o.append(text(56, 50, "Four routes for structure into MuLAN", 25, INK, anchor="start", weight=650))
o.append(text(56, 80,
              "Each route is the diagnosis of the previous failure. The rail below shows where in the pipeline each one intervenes. "
              "Outcome column: test PCC (pooled Pearson) on S1102 unless noted.",
              13.5, INK_SOFT, anchor="start"))

# ============================================================ the reference rail
RAIL_Y = 152
STAGE_W, GAP = 162, 24
SX0 = 90
STAGES = [
    ("structure",        "PDB · 3Di",        "struct"),
    ("input tokens",     "AA  (+ 3Di)",      "tok"),
    ("frozen PLM",       "layers 1 … N",     "plm"),
    ("embedding",        "[ L × H ]",        "emb"),
    ("Light Attention",  "per-residue",      "att"),
    ("pooled  mut − wt", "[ H ]",            "pool"),
    ("head slot",        "add_scores [ 1 ]", "slot"),
]
def stage_x(i): return SX0 + i*(STAGE_W + GAP)

o.append(rrect(56, 110, 1588, 182, 13, "#fcfcfb", HAIRLINE, 1.6))
o.append(text(78, 130, "THE PIPELINE", 10.5, MUTED, anchor="start", weight=700))

for i, (nm, sub, kind) in enumerate(STAGES):
    x, cy = stage_x(i), RAIL_Y + 48
    if kind == "struct":
        o.append(f'<path d="M {x+42} {cy+6} q 12,-24 33,-20 q 24,5 17,24 q -6,17 -27,15 q -22,-2 -23,-19 z" '
                 f'fill="#cfe0f4" stroke="#2f6fb5" stroke-width="1.6"/>')
        o.append(f'<path d="M {x+120} {cy+6} q -12,-24 -33,-20 q -24,5 -17,24 q 6,17 27,15 q 22,-2 23,-19 z" '
                 f'fill="#d5eeea" stroke="#4b9d94" stroke-width="1.6"/>')
        o.append(f'<line x1="{x+81}" y1="{cy-17}" x2="{x+81}" y2="{cy+15}" stroke="{MUTED}" stroke-width="1.3" stroke-dasharray="3 2"/>')
    elif kind == "tok":
        for k, ch in enumerate("NEVF"):
            bx = x + 43 + k*19
            o.append(f'<rect x="{bx}" y="{cy-10}" width="17" height="17" fill="#fff" stroke="{INK}" stroke-width="1.1"/>')
            o.append(text(bx + 8.5, cy - 1.5, ch, 10, INK))
    elif kind == "plm":
        for k in range(3):
            o.append(box(x + 42, cy - 24 + k*15, 78, 12, "", PLM_F, PLM_S, 9, 4))
        o.append(text(x + 81, cy - 3, "Encoder", 9.5, INK))
        o.append(snowflake(x + 128, cy - 20, 6))
    elif kind == "emb":
        o.append(slab(x + 54, cy - 11, 54, 23, 7))
    elif kind == "att":
        o.append(box(x + 32, cy - 13, 98, 26, "attention", ATT_F, ATT_S, 10.5, 7))
    elif kind == "pool":
        o.append(f'<rect x="{x+43}" y="{cy-9}" width="{5*15}" height="18" fill="{TENS_F}" stroke="{TENS_S}" stroke-width="1.1"/>')
        for k in range(1, 5):
            o.append(f'<line x1="{x+43+k*15}" y1="{cy-9}" x2="{x+43+k*15}" y2="{cy+9}" stroke="{TENS_S}" stroke-width="0.8"/>')
    elif kind == "slot":
        g, _ = hslot(x + 36, cy - 9, 4, 15, 18, labels=False)
        o.append(g)
    o.append(text(x + STAGE_W/2, cy + 34, nm, 11.5, INK, weight=600))
    o.append(text(x + STAGE_W/2, cy + 49, sub, 10, MUTED))
    if i < len(STAGES) - 1:
        o.append(arrow(x + STAGE_W - 6, cy, x + STAGE_W + GAP - 4, cy, HAIRLINE, 1.8))

tx = stage_x(len(STAGES) - 1) + STAGE_W - 2
o.append(arrow(tx - 8, RAIL_Y + 48, tx + 12, RAIL_Y + 48, HAIRLINE, 1.8))
o.append(para(tx + 16, RAIL_Y + 34, 78, 28, "Linear", 10, size=11))
o.append(arrow(tx + 98, RAIL_Y + 48, tx + 122, RAIL_Y + 48, HAIRLINE, 1.8))
o.append(text(tx + 128, RAIL_Y + 44, "ΔΔG", 14, INK, anchor="start", weight=650))
o.append(text(tx + 128, RAIL_Y + 60, "kcal/mol", 9.5, MUTED, anchor="start"))

ROUTE_STAGE = {"1": 3, "1b": 2, "2": 1, "3": 0, "4": 6}
for tag, si in ROUTE_STAGE.items():
    mx = stage_x(si) + STAGE_W/2
    o.append(f'<line x1="{mx}" y1="{RAIL_Y+9}" x2="{mx}" y2="{RAIL_Y+26}" stroke="{INK}" stroke-width="1.4" stroke-dasharray="3 2"/>')
    o.append(f'<circle cx="{mx}" cy="{RAIL_Y}" r="13" fill="#fff" stroke="{INK}" stroke-width="1.8"/>')
    o.append(text(mx, RAIL_Y, tag, 11.5, INK, weight=700))

o.append(text(78, 312,
              "Routes 1 → 1b → 2 → 3 walk progressively earlier along the rail — each failure pushed the intervention further upstream. "
              "Route 4 abandons that direction and goes to the head instead.",
              12, INK_SOFT, anchor="start", style="font-style:italic"))

# ============================================================ the route timeline
SPINE = 78
CARD_X, CARD_W = 112, 486
MINI_X = 646
RES_X = 1092
CARD_H = 92

def spine_seg(y1, y2):
    return f'<line x1="{SPINE}" y1="{y1}" x2="{SPINE}" y2="{y2}" stroke="{HAIRLINE}" stroke-width="2"/>'

def locator(x, y, si, col):
    """Compact 7-tick copy of the rail, with the touched stage lit."""
    b = [f'<line x1="{x}" y1="{y}" x2="{x+6*17+9}" y2="{y}" stroke="{HAIRLINE}" stroke-width="1.2"/>']
    for i in range(7):
        cxx = x + 5 + i*17
        if i == si:
            b.append(f'<rect x="{cxx-6}" y="{y-8}" width="13" height="16" rx="2.5" fill="{col}" stroke="{col}" stroke-width="1.4"/>')
        else:
            b.append(f'<rect x="{cxx-4}" y="{y-4}" width="9" height="8" rx="1.5" fill="#fff" stroke="{HAIRLINE}" stroke-width="1.2"/>')
    b.append(text(x + 6*17 + 22, y, f"acts on:  {STAGES[si][0]}", 10.5, INK_SOFT, anchor="start", weight=600))
    return "".join(b)

def node(y, n, title, sub, si, loc_col, mini, badge_kind, badge_txt, nums, nums2=""):
    b = []
    cy = y + 34
    b.append(f'<circle cx="{SPINE}" cy="{cy}" r="15" fill="#fff" stroke="{INK}" stroke-width="1.8"/>')
    b.append(text(SPINE, cy, n, 14, INK, weight=700))
    b.append(f'<line x1="{SPINE+15}" y1="{cy}" x2="{CARD_X}" y2="{cy}" stroke="{HAIRLINE}" stroke-width="1.6"/>')
    b.append(rrect(CARD_X, y, CARD_W, CARD_H, 10, "#fcfcfb", HAIRLINE, 1.4))
    b.append(text(CARD_X + 18, y + 22, title, 14.5, INK, anchor="start", weight=600))
    b.append(text(CARD_X + 18, y + 44, sub, 12, INK_SOFT, anchor="start"))
    b.append(f'<line x1="{CARD_X+18}" y1="{y+58}" x2="{CARD_X+CARD_W-18}" y2="{y+58}" stroke="{HAIRLINE}" stroke-width="1" opacity="0.6"/>')
    b.append(locator(CARD_X + 18, y + 76, si, loc_col))
    b.append(mini)
    b.append(badge(RES_X, y + 8, badge_txt, badge_kind))
    b.append(text(RES_X, y + 48, nums, 13, INK, anchor="start", weight=500))
    if nums2:
        b.append(text(RES_X, y + 66, nums2, 11.5, MUTED, anchor="start"))
    return "".join(b)

def diag(y, s):
    b = [f'<path d="M {SPINE} {y-14} L {SPINE} {y+10} L {CARD_X-6} {y+10}" fill="none" '
         f'stroke="{HAIRLINE}" stroke-width="1.4" stroke-dasharray="3 3"/>']
    b.append(text(CARD_X + 4, y + 10, s, 12.5, INK_SOFT, anchor="start", style="font-style:italic"))
    return "".join(b)

# ---------------------------------------------------------------- premise
y = 342
o.append(rrect(CARD_X - 56, y, 980, 58, 10, "#f4f7fb", "#9fb6d4", 1.6))
o.append(text(CARD_X - 34, y + 21, "Premise — a structural quantity, a sequence-only model",
              14, INK, anchor="start", weight=600))
o.append(text(CARD_X - 34, y + 41,
              "ProstT5 has a 3Di structure channel and still loses to sequence-only Ankh-large (CV10 0.805 vs 0.832). So: feed the structure in.",
              12, INK_SOFT, anchor="start"))
o.append(text(RES_X, y + 14, "OUTCOME", 9.5, MUTED, anchor="start", weight=700))
o.append(text(RES_X, y + 32, "test PCC", 9.5, MUTED, anchor="start"))
o.append(spine_seg(y + 58, y + 92))

# ---------------------------------------------------------------- route 1
y1 = y + 92
mini = []
mini.append(text(MINI_X + 122, y1 + 4, "[ L × 1024 ] ⊕ [ L × 1024 ] → [ L × 2048 ]", 10, MUTED))
mini.append(vstack(MINI_X + 40, y1 + 24, 26, 8, 4, "#e8eef7", "#8ea9c9"))
mini.append(text(MINI_X + 53, y1 + 64, "AA", 10.5, INK_SOFT))
mini.append(circ_op(MINI_X + 86, y1 + 40, "⊕", 9, 11))
mini.append(vstack(MINI_X + 106, y1 + 24, 26, 8, 4, "#eeeae4", "#b9ac97"))
mini.append(text(MINI_X + 119, y1 + 64, "3Di", 10.5, INK_SOFT))
mini.append(arrow(MINI_X + 146, y1 + 40, MINI_X + 182, y1 + 40, HAIRLINE, 1.5))
mini.append(vstack(MINI_X + 186, y1 + 16, 26, 8, 8, TENS_F, TENS_S))
o.append(node(y1, "1", "Late concat of a separate WT-3Di block",
              "run6c — 3Di appended per residue, downstream of the encoder",
              3, BAD, "".join(mini), "bad", "hurts", "0.740 → 0.663",
              "and 3 repair variants, all below baseline"))
o.append(diag(y1 + 104, "every learnable structure knob trained itself off — gate → 0.46, contact bias → 0.017"))
o.append(spine_seg(y1 + CARD_H, y1 + 148))

# ---------------------------------------------------------------- route 1b
y2 = y1 + 148
mini = []
mini.append(text(MINI_X + 122, y2 + 4, "a linear probe ranks ProstT5's layers — MuLAN disagrees", 10, MUTED))
BASE = y2 + 68
for i, (lab, hgt, val, col, ec) in enumerate([
        ("layer 3", 42, "0.765", "#cfe0f4", "#8ea9c9"),
        ("layer 10", 38, "0.754", "#cfe0f4", "#8ea9c9"),
        ("layer 24", 20, "0.702", "#e3e3e3", TENS_S)]):
    bx = MINI_X + 52 + i*72
    mini.append(f'<rect x="{bx}" y="{BASE-hgt}" width="38" height="{hgt}" fill="{col}" stroke="{ec}" stroke-width="1.2"/>')
    mini.append(text(bx + 19, BASE - hgt + 10, val, 9.5, INK_SOFT, weight=600))
    mini.append(text(bx + 19, BASE + 12, lab, 9.5, MUTED))
mini.append(f'<line x1="{MINI_X+40}" y1="{BASE}" x2="{MINI_X+214}" y2="{BASE}" stroke="{HAIRLINE}" stroke-width="1.4"/>')
mini.append(text(MINI_X + 252, BASE - 16, "probe PCC,\nsite delta", 9.5, MUTED, anchor="start"))
o.append(node(y2, "1b", "Swap to a better hidden layer",
              "run3a / run3b — ProstT5 layer 7, and concat 7⊕10",
              2, BAD, "".join(mini), "bad", "hurts", "0.648  ·  0.429",
              "Ankh's final layer already near-optimal → not tried"))
o.append(diag(y2 + 104, "the layer ranking does not survive contact with the attention head"))
o.append(spine_seg(y2 + CARD_H, y2 + 152))

# ---------------------------------------------------------------- shared diagnosis
y3 = y2 + 156
o.append(rrect(CARD_X - 56, y3, 980, 60, 10, "#fdf6f6", "#dba9a9", 1.6))
o.append(text(CARD_X - 34, y3 + 21, "Diagnosis — the shared failure mode", 13.5, "#a33", anchor="start", weight=650))
o.append(text(CARD_X - 34, y3 + 42,
              "No mutant structures exist, so every mutant reuses its WT 3Di. Held constant, it cancels in  mut − wt  — zero mutation-specific signal.",
              12, INK_SOFT, anchor="start"))
yf = y3 + 60
o.append(spine_seg(yf, yf + 26))
o.append(text(CARD_X + 300, yf + 20, "two ways out", 12, MUTED, anchor="start", style="font-style:italic"))

# ---------------------------------------------------------------- route 2
y4 = yf + 32
mini = []
mini.append(text(MINI_X + 122, y4 + 4, "AA + 3Di fused into one token, pretrained that way", 10, MUTED))
for i, ch in enumerate("NEVF"):
    bx = MINI_X + 70 + i*40
    mini.append(f'<rect x="{bx}" y="{y4+26}" width="36" height="20" fill="#e8eef7" stroke="#8ea9c9" stroke-width="1.2"/>')
    mini.append(f'<rect x="{bx}" y="{y4+46}" width="36" height="18" fill="#f4ecdf" stroke="#c2ab84" stroke-width="1.2"/>')
    mini.append(text(bx + 18, y4 + 36, ch, 11.5, INK))
    mini.append(text(bx + 18, y4 + 55, "d", 11, PHYS_D))
mini.append(text(MINI_X + 58, y4 + 36, "AA", 10, MUTED, anchor="end"))
mini.append(text(MINI_X + 58, y4 + 55, "3Di", 10, MUTED, anchor="end"))
o.append(node(y4, "2", "Let the embedder own the fusion — SaProt",
              "structure enters at the input, not bolted on downstream",
              1, GOOD, "".join(mini), "good", "helps", "0.816 → 0.842",
              "+0.026, 8/10 folds, paired · best sub-1B arm"))
o.append(spine_seg(y4 + CARD_H, y4 + 126))

# ---------------------------------------------------------------- route 3
y5 = y4 + 126
mini = []
mini.append(text(MINI_X + 122, y5 + 4, "FoldX-modelled mutant 3Di vs WT 3Di", 10, MUTED))
for r, lab in enumerate(["WT", "mut"]):
    ry = y5 + 20 + r*22
    mini.append(text(MINI_X + 42, ry + 9, lab, 10.5, MUTED, anchor="end"))
    for i, ch in enumerate("pvqlpvncl"):
        bx = MINI_X + 50 + i*21
        mini.append(f'<rect x="{bx}" y="{ry}" width="19" height="18" fill="#f4ecdf" stroke="#c2ab84" stroke-width="1"/>')
        mini.append(text(bx + 9.5, ry + 9, ch, 10.5, PHYS_D))
mini.append(text(MINI_X + 144, y5 + 76, "identical at every position", 10.5, BAD, weight=600))
o.append(node(y5, "3", "Make the structure mutation-specific — FoldX",
              "RepairPDB → BuildModel(mutant) → recompute 3Di",
              0, MUTED, "".join(mini), "flat", "no-op", "0 / 39 mutations changed",
              "0 / 232 backbone atoms moved (1KNE)"))
o.append(diag(y5 + 104, "BuildModel repacks side chains only; 3Di reads backbone only ⇒ mutant 3Di ≡ WT 3Di"))
o.append(spine_seg(y5 + CARD_H, y5 + 150))

# ---------------------------------------------------------------- route 4
y6 = y5 + 150
mini = []
mini.append(text(MINI_X + 122, y6 + 4, "FoldX's native output is an energy, not a structure", 10, MUTED))
mini.append(box(MINI_X + 30, y6 + 26, 130, 30, "FoldX ΔΔG", PHYS_F, PHYS_S, 12, 8, PHYS_D, 600))
mini.append(text(MINI_X + 95, y6 + 66, "kcal/mol", 9.5, MUTED))
mini.append(arrow(MINI_X + 164, y6 + 41, MINI_X + 196, y6 + 41, PHYS_S, 1.6))
g4, _ = hslot(MINI_X + 200, y6 + 32, 5, 14, 16, labels=False)
mini.append(g4)
mini.append(text(MINI_X + 235, y6 + 60, "[ H ]", 9, MUTED))
mini.append(text(MINI_X + 277, y6 + 60, "[ 1 ]  add_scores", 9, PHYS_D, anchor="start", weight=650))
o.append(node(y6, "4", "Repurpose FoldX as a physics score channel",
              "one scalar, or 12 decomposed ΔΔG terms → MLP, into the head",
              6, WARN, "".join(mini), "warn", "real, but bounded", "+0.009  ·  +0.017",
              "scalar · 12-term MLP, vs MuLAN without it"))

# ---------------------------------------------------------------- coda
y7 = y6 + CARD_H + 20
o.append(rrect(CARD_X - 56, y7, 1532, 60, 10, "#fffaf0", PHYS_S, 1.8))
o.append(text(CARD_X - 34, y7 + 21, "Coda — the FoldX-alone comparator", 13.5, PHYS_D, anchor="start", weight=650))
o.append(text(CARD_X - 34, y7 + 42,
              "Against FoldX used alone as an unsupervised predictor, MuLAN + the channel is statistically indistinguishable on 8 of 9 split tiers "
              "(Ankh-large). Only ESM-C 6B clears it.",
              12, INK_SOFT, anchor="start"))

write(str(_OUT / "fig_c_structure_routes.svg"), W, H, "".join(o))
print("ok")
