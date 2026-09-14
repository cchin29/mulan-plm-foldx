"""Figure 2 — what each predictor is given at test time, and what that buys."""
from pathlib import Path as _Path

# Output beside this script, not in whatever directory it was authored in.
_OUT = _Path(__file__).resolve().parent
import sys; sys.path.insert(0, str(_OUT))
from mulanfig import *

W, H = 1700, 952
o = []

o.append(text(56, 50, "Figure 2  ·  Predictor inputs at test time — FoldX alone, MuLAN base, MuLAN + channel", 25, INK, anchor="start", weight=650))
o.append(text(56, 80,
              "One held-out complex, three ways to rank its mutations. FoldX alone is unsupervised and requires no training; it is the comparator the SKEMPI frontier reports against.",
              13.5, INK_SOFT, anchor="start"))

LANE_X, LANE_W = 56, 1032
GIVEN_X, PIPE_X, OUT_X = 84, 372, 730

def ranked(x, y, hi_col):
    """Small 'ranked candidate mutations' glyph."""
    b = []
    for i, wd in enumerate([84, 66, 52, 38]):
        b.append(f'<rect x="{x}" y="{y+i*15}" width="{wd}" height="10" rx="2" '
                 f'fill="{hi_col if i==0 else "#e4e4e2"}" stroke="none"/>')
    b.append(text(x + 100, y + 16, "mutations ranked", 10.5, MUTED, anchor="start"))
    b.append(text(x + 100, y + 30, "by ΔΔG (kcal/mol)", 10.5, MUTED, anchor="start"))
    return "".join(b)

def lane(y, h, tag, tagcol, title, given_fn, pipe_fn, note_txt, fill="#fcfcfb", stroke=HAIRLINE):
    b = [rrect(LANE_X, y, LANE_W, h, 12, fill, stroke, 1.8)]
    b.append(f'<rect x="{LANE_X}" y="{y}" width="7" height="{h}" fill="{tagcol}" rx="3.5"/>')
    b.append(text(LANE_X + 26, y + 26, tag, 11, tagcol, anchor="start", weight=700))
    b.append(text(LANE_X + 26, y + 47, title, 15, INK, anchor="start", weight=650))
    b.append(text(LANE_X + 26, y + h - 20, note_txt, 11, MUTED, anchor="start"))
    b.append(given_fn(y))
    b.append(pipe_fn(y))
    return "".join(b)

# ------------------------------------------------------------------ lane 1
Y1, LH = 122, 208
def given1(y):
    b = [text(GIVEN_X + 112, y + 74, "the test structure", 11.5, INK_SOFT, weight=600)]
    cx0 = GIVEN_X + 112
    b.append(f'<path d="M {cx0-84} {y+130} q 16,-30 42,-25 q 30,5 22,31 q -6,21 -33,19 q -28,-2 -31,-25 z" '
             f'fill="#cfe0f4" stroke="#2f6fb5" stroke-width="1.7"/>')
    b.append(f'<path d="M {cx0+84} {y+130} q -16,-30 -42,-25 q -30,5 -22,31 q 6,21 33,19 q 28,-2 31,-25 z" '
             f'fill="#d5eeea" stroke="#4b9d94" stroke-width="1.7"/>')
    b.append(f'<line x1="{cx0}" y1="{y+100}" x2="{cx0}" y2="{y+156}" stroke="{MUTED}" stroke-width="1.4" stroke-dasharray="4 3"/>')
    b.append(text(cx0, y + 166, "coordinates only", 10.5, MUTED))
    return "".join(b)
def pipe1(y):
    b = [arrow(GIVEN_X + 210, y + 128, PIPE_X - 12, y + 128, PHYS_S, 1.8)]
    b.append(rrect(PIPE_X, y + 96, 262, 64, 10, PHYS_F, PHYS_S, 1.8))
    b.append(text(PIPE_X + 131, y + 120, "FoldX", 14, PHYS_D, weight=700))
    b.append(text(PIPE_X + 131, y + 140, "RepairPDB → BuildModel → AnalyseComplex", 9.5, "#9a7a30"))
    b.append(badge(PIPE_X + 74, y + 34, "no training at all", "warn", 128))
    b.append(arrow(PIPE_X + 268, y + 128, OUT_X - 12, y + 128, PHYS_S, 1.8))
    b.append(ranked(OUT_X, y + 100, PHYS_S))
    return "".join(b)
o.append(lane(Y1, LH, "PREDICTOR 1", PHYS_S, "FoldX alone", given1, pipe1,
              "unsupervised physics, computed from the held-out structure itself — so homology control cannot lower its score",
              "#fffdf7", PHYS_S))

# ------------------------------------------------------------------ lane 2
Y2 = Y1 + LH + 22
def seqpair(x, y, mutated=False):
    b = [seqbox(x, y, "NEVFHTGIK", 18, 4 if mutated else None, 10)]
    b.append(seqbox(x, y + 26, "NRLEVVLT" if mutated else "NRLEVVPT", 18, 7 if mutated else None, 10))
    return "".join(b)
def given2(y):
    b = [text(GIVEN_X + 112, y + 74, "the two sequences", 11.5, INK_SOFT, weight=600)]
    b.append(seqpair(GIVEN_X + 30, y + 96, False))
    b.append(text(GIVEN_X + 112, y + 166, "wild-type + mutant", 10.5, MUTED))
    return "".join(b)
def pipe2(y):
    b = [arrow(GIVEN_X + 210, y + 128, PIPE_X - 12, y + 128, INK, 1.6)]
    b.append(box(PIPE_X, y + 100, 118, 40, "frozen PLM", PLM_F, PLM_S, 12, 8))
    b.append(arrow(PIPE_X + 122, y + 120, PIPE_X + 142, y + 120, INK, 1.5))
    b.append(box(PIPE_X + 146, y + 100, 116, 40, "MuLAN head", ATT_F, ATT_S, 12, 8))
    b.append(badge(PIPE_X + 74, y + 34, "trained on SKEMPI", "good", 132))
    b.append(arrow(PIPE_X + 268, y + 120, OUT_X - 12, y + 120, INK, 1.6))
    b.append(ranked(OUT_X, y + 92, "#8ea9c9"))
    return "".join(b)
o.append(lane(Y2, LH, "PREDICTOR 2", "#8ea9c9", "MuLAN base", given2, pipe2,
              "sequence only — nothing in the input tells it where the interface is, or what the physics says"))

# ------------------------------------------------------------------ lane 3
Y3 = Y2 + LH + 22
def given3(y):
    b = [text(GIVEN_X + 112, y + 74, "the sequences  +  FoldX's answer", 11.5, INK_SOFT, weight=600)]
    b.append(seqpair(GIVEN_X + 30, y + 88, False))
    b.append(f'<rect x="{GIVEN_X+84}" y="{y+142}" width="56" height="20" fill="{SLOT_F}" stroke="{PHYS_S}" stroke-width="2"/>')
    b.append(text(GIVEN_X + 112, y + 152, "ΔΔG", 11, PHYS_D, weight=650))
    b.append(text(GIVEN_X + 112, y + 172, "kcal/mol — the number from lane 1", 10.5, PHYS_D))
    return "".join(b)
def pipe3(y):
    b = [arrow(GIVEN_X + 210, y + 128, PIPE_X - 12, y + 128, INK, 1.6)]
    b.append(box(PIPE_X, y + 100, 118, 40, "frozen PLM", PLM_F, PLM_S, 12, 8))
    b.append(arrow(PIPE_X + 122, y + 120, PIPE_X + 142, y + 120, INK, 1.5))
    b.append(box(PIPE_X + 146, y + 100, 116, 40, "MuLAN head", ATT_F, ATT_S, 12, 8))
    b.append(f'<rect x="{PIPE_X+192}" y="{y+143}" width="24" height="15" fill="{SLOT_F}" stroke="{PHYS_S}" stroke-width="2"/>')
    b.append(text(PIPE_X + 224, y + 151, "add_scores slot", 9.5, PHYS_D, anchor="start", weight=600))
    # the FoldX number from lane 1 enters the head here
    b.append(arrow(GIVEN_X + 146, y + 151, PIPE_X + 188, y + 151, PHYS_S, 1.6))
    b.append(badge(PIPE_X + 60, y + 34, "trained  +  reads FoldX", "good", 160))
    b.append(arrow(PIPE_X + 268, y + 120, OUT_X - 12, y + 120, INK, 1.6))
    b.append(ranked(OUT_X, y + 92, ATT_S))
    return "".join(b)
o.append(lane(Y3, LH, "PREDICTOR 3", ATT_S, "MuLAN + FoldX channel", given3, pipe3,
              "the only arm with access to both — so the question is what it adds beyond lane 1"))

# ================================================================ results panel
PX, PW = 1120, 524
o.append(rrect(PX, Y1, PW, Y3 + LH - Y1, 12, "#fcfcfb", INK, 1.8))
o.append(text(PX + 24, Y1 + 28, "Per-tier results", 15, INK, anchor="start", weight=650))
o.append(text(PX + 24, Y1 + 50, "per-structure Spearman ρ (T ≥ 10) — unitless, 0 = no ranking skill", 11, MUTED, anchor="start"))

def scale(v, x0, wdt):  return x0 + v / 0.60 * wdt

def result_block(y, title, sub, vals, note_txt):
    """vals = [(label, value, colour, lane_no)] with FoldX first."""
    b = [text(PX + 24, y, title, 13, INK, anchor="start", weight=650)]
    b.append(text(PX + 24, y + 18, sub, 10.5, MUTED, anchor="start"))
    x0, wdt = PX + 132, 292
    for t in (0, 0.2, 0.4, 0.6):
        gx = scale(t, x0, wdt)
        b.append(f'<line x1="{gx}" y1="{y+34}" x2="{gx}" y2="{y+134}" stroke="{HAIRLINE}" stroke-width="1" opacity="0.6"/>')
        b.append(text(gx, y + 146, f"{t:.1f}", 9.5, MUTED))
    fx = vals[0][1]
    fxx = scale(fx, x0, wdt)
    b.append(f'<line x1="{fxx}" y1="{y+34}" x2="{fxx}" y2="{y+138}" stroke="{PHYS_S}" stroke-width="1.8" stroke-dasharray="4 3"/>')
    for i, (lab, v, col, ln) in enumerate(vals):
        yy = y + 48 + i*30
        vx = scale(v, x0, wdt)
        b.append(f'<line x1="{x0}" y1="{yy}" x2="{vx}" y2="{yy}" stroke="{col}" stroke-width="3" stroke-linecap="round" opacity="0.35"/>')
        b.append(f'<circle cx="{vx}" cy="{yy}" r="6" fill="{col}" stroke="#fff" stroke-width="2"/>')
        b.append(text(PX + 124, yy, lab, 10.5, INK_SOFT, anchor="end"))
        b.append(text(vx + 16, yy, f"{v:.3f}", 11.5, INK, anchor="start",
                      weight=700 if i in (0, 2) else 400))
    b.append(text(PX + 24, y + 168, note_txt, 11, INK_SOFT, anchor="start"))
    return "".join(b)

o.append(result_block(Y1 + 92,
                      "Ankh-large, clustered split",
                      "CD-HIT ≤60 % sequence identity · 96 complexes",
                      [("lane 1  FoldX", 0.418, PHYS_S, 1),
                       ("lane 2  base", 0.142, "#8ea9c9", 2),
                       ("lane 3  + channel", 0.406, ATT_S, 3)],
                      "Lane 3 lands on lane 1. Δ = −0.007 [−0.035, +0.022]."))

o.append(f'<line x1="{PX+24}" y1="{Y2+112}" x2="{PX+PW-24}" y2="{Y2+112}" stroke="{HAIRLINE}" stroke-width="1"/>')

o.append(result_block(Y2 + 138,
                      "ESM-C 6B, CATH superfamily",
                      "single-point, out-of-superfamily · 11 complexes",
                      [("lane 1  FoldX", 0.398, PHYS_S, 1),
                       ("lane 2  base", 0.377, "#8ea9c9", 2),
                       ("lane 3  + MLP", 0.495, ATT_S, 3)],
                      "Lane 3 clears lane 1. Δ = +0.097 [+0.017, +0.194]."))

o.append(f'<line x1="{PX+24}" y1="{Y3+104}" x2="{PX+PW-24}" y2="{Y3+104}" stroke="{HAIRLINE}" stroke-width="1"/>')
o.append(text(PX + 24, Y3 + 130, "The dashed line", 12.5, PHYS_D, anchor="start", weight=700))
o.append(text(PX + 24, Y3 + 154, "It marks lane 1. A learned model is worth adding", 11, INK_SOFT, anchor="start"))
o.append(text(PX + 24, Y3 + 171, "only if it sits clearly to the right of it —", 11, INK_SOFT, anchor="start"))
o.append(text(PX + 24, Y3 + 188, "which happens on 19 of 232 contrasts.", 11, INK_SOFT, anchor="start"))

# ================================================================ conclusion band
CY = Y3 + LH + 24
o.append(rrect(56, CY, 1588, 96, 12, "#fffaf0", PHYS_S, 1.8))
o.append(text(80, CY + 28, "The missing comparator", 14.5, PHYS_D, anchor="start", weight=700))
o.append(text(80, CY + 52,
              "Every earlier result compared lane 3 against lane 2. Against lane 1 — the alternative that requires no training — across 10 backbones and 9 split tiers:",
              12, INK_SOFT, anchor="start"))
stats = [("0 of 78", "base arms that beat FoldX alone", BAD),
         ("19 of 232", "contrasts that significantly beat it — all carry the channel", GOOD),
         ("3 of 76", "wins on the clustered tiers, where the method comes closest to failing", WARN)]
for i, (n, lab, col) in enumerate(stats):
    x = 96 + i*512
    o.append(text(x, CY + 78, n, 15, col, anchor="start", weight=700))
    o.append(text(x + 84, CY + 78, lab, 11.5, INK_SOFT, anchor="start"))

write(str(_OUT / "fig2_predictor_inputs.svg"), W, H, "".join(o))
print("ok")
