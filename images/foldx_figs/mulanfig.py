"""Shared SVG drawing grammar for the MuLAN FoldX figures.

Inherits the visual language of `images/visual_abstract.png`:
  grey isometric slab = tensor · pink rounded rect = frozen PLM encoder ·
  cyan rounded rect = light attention · blue parallelogram = operation ·
  small white rect = named scalar op · circled glyph = elementwise op ·
  dotted rounded rect = grouping · thick grey arrow = major flow.

Adds one new family, for the modality the paper's figure has no word for:
  AMBER rounded rect = non-neural physics (FoldX). Deliberately not pink, so it
  never reads as another learned module.
"""

FONT = "'Helvetica Neue', Helvetica, Arial, sans-serif"

# ---------------------------------------------------------------- palette
INK        = "#231f20"
INK_SOFT   = "#5a5754"
MUTED      = "#8a8782"
HAIRLINE   = "#b8b5b0"

PLM_F,   PLM_S   = "#f9ccd2", "#d9838f"   # frozen encoder (pink)
ATT_F,   ATT_S   = "#d3f0ec", "#74c4ba"   # light attention (cyan)
OP_F,    OP_S    = "#4472c4", "#2f56a0"   # operation (blue parallelogram)
TENS_F,  TENS_S  = "#dadada", "#8e8e8e"   # tensor slab (grey)
TENS_T           = "#efefef"              # slab top face
TENS_R           = "#c2c2c2"              # slab right face
PHYS_F,  PHYS_S  = "#fbe6bd", "#c9911a"   # NEW: physics / FoldX (amber)
PHYS_D           = "#8a6207"              # amber text
SLOT_F,  SLOT_S  = "#fdf3dd", "#c9911a"   # the add_scores slot
MUT              = "#e03127"              # mutated residue
GOOD, BAD, WARN  = "#0ca30c", "#d03b3b", "#e0900c"

def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

# ---------------------------------------------------------------- primitives
def rrect(x, y, w, h, r=8, fill="#fff", stroke=INK, sw=1.6, dash=None, extra=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" ry="{r}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d} {extra}/>')

def text(x, y, s, size=13, fill=INK, anchor="middle", weight=400, style="", lh=1.25):
    """Multi-line via \\n."""
    lines = str(s).split("\n")
    out = []
    y0 = y - (len(lines) - 1) * size * lh / 2
    for i, ln in enumerate(lines):
        out.append(f'<text x="{x}" y="{y0 + i*size*lh:.1f}" font-family="{FONT}" '
                   f'font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
                   f'font-weight="{weight}" style="{style}" '
                   f'dominant-baseline="central">{esc(ln)}</text>')
    return "".join(out)

def box(x, y, w, h, label, fill, stroke, size=13, r=9, tc=INK, weight=400, sw=1.6):
    return rrect(x, y, w, h, r, fill, stroke, sw) + text(x + w/2, y + h/2, label, size, tc, weight=weight)

def para(x, y, w, h, label, skew=14, fill=OP_F, stroke=OP_S, tc="#fff", size=12.5):
    """Blue parallelogram = an operation."""
    p = f"{x+skew},{y} {x+w},{y} {x+w-skew},{y+h} {x},{y+h}"
    return (f'<polygon points="{p}" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>'
            + text(x + w/2, y + h/2, label, size, tc, weight=500))

def slab(x, y, w, h, d=7, fill=TENS_F, stroke=TENS_S, cells=0):
    """Isometric tensor slab: front face + top + right, optional cell divisions."""
    o = []
    o.append(f'<polygon points="{x},{y} {x+d},{y-d} {x+w+d},{y-d} {x+w},{y}" '
             f'fill="{TENS_T}" stroke="{stroke}" stroke-width="1.2"/>')
    o.append(f'<polygon points="{x+w},{y} {x+w+d},{y-d} {x+w+d},{y+h-d} {x+w},{y+h}" '
             f'fill="{TENS_R}" stroke="{stroke}" stroke-width="1.2"/>')
    o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" '
             f'stroke="{stroke}" stroke-width="1.2"/>')
    if cells > 1:
        for i in range(1, cells):
            cy = y + h * i / cells
            o.append(f'<line x1="{x}" y1="{cy}" x2="{x+w}" y2="{cy}" stroke="{stroke}" stroke-width="0.7"/>')
    return "".join(o)

def vstack(x, y, w, cell, n, fill=TENS_F, stroke=TENS_S):
    """A vertical vector drawn as n stacked cells (the paper's 1-D tensor look)."""
    o = [f'<rect x="{x}" y="{y}" width="{w}" height="{cell*n}" fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>']
    for i in range(1, n):
        o.append(f'<line x1="{x}" y1="{y+cell*i}" x2="{x+w}" y2="{y+cell*i}" stroke="{stroke}" stroke-width="0.7"/>')
    return "".join(o)

def hslot(x, y, n=6, cell=16, h=18, labels=True, lab="add_scores"):
    """The add_scores slot, drawn consistently everywhere: a horizontal pooled
    vector with ONE amber cell appended at the right end (concatenation, literally)."""
    b = []
    for i in range(n):
        b.append(f'<rect x="{x+i*cell}" y="{y}" width="{cell}" height="{h}" '
                 f'fill="{TENS_F}" stroke="{TENS_S}" stroke-width="1.1"/>')
    sx = x + n*cell
    b.append(f'<rect x="{sx}" y="{y}" width="{cell}" height="{h}" '
             f'fill="{SLOT_F}" stroke="{PHYS_S}" stroke-width="2.2"/>')
    if labels:
        b.append(text(x + n*cell/2, y + h + 13, "[ H ]", 9.5, MUTED))
        b.append(text(sx + cell/2, y + h + 13, "[ 1 ]", 9.5, PHYS_D, weight=600))
        b.append(text(sx + cell/2, y + h + 26, lab, 9.5, PHYS_D, weight=650))
    return "".join(b), sx + cell/2


def seqbox(x, y, letters, cell=21, mut_idx=None, size=12):
    """Boxed sequence letters; mutated residue in red."""
    o = []
    for i, ch in enumerate(letters):
        cx = x + i*cell
        o.append(f'<rect x="{cx}" y="{y}" width="{cell}" height="{cell}" fill="#fff" '
                 f'stroke="{INK}" stroke-width="1.1"/>')
        col = MUT if (mut_idx is not None and i == mut_idx) else INK
        o.append(text(cx + cell/2, y + cell/2, ch, size, col, weight=600 if col == MUT else 400))
    return "".join(o)

def arrow(x1, y1, x2, y2, stroke=INK, sw=1.6, dash=None, head=7, opacity=1):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    import math
    a = math.atan2(y2-y1, x2-x1)
    bx, by = x2 - head*math.cos(a)*0.9, y2 - head*math.sin(a)*0.9
    p1 = (x2, y2)
    p2 = (bx - head*0.55*math.sin(a), by + head*0.55*math.cos(a))
    p3 = (bx + head*0.55*math.sin(a), by - head*0.55*math.cos(a))
    return (f'<line x1="{x1}" y1="{y1}" x2="{bx:.1f}" y2="{by:.1f}" stroke="{stroke}" '
            f'stroke-width="{sw}"{d} opacity="{opacity}"/>'
            f'<polygon points="{p1[0]:.1f},{p1[1]:.1f} {p2[0]:.1f},{p2[1]:.1f} {p3[0]:.1f},{p3[1]:.1f}" '
            f'fill="{stroke}" opacity="{opacity}"/>')

def elbow(x1, y1, x2, y2, stroke=INK, sw=1.6, dash=None, mid=None, arrowhead=True):
    """Orthogonal connector: horizontal, vertical, horizontal."""
    mx = mid if mid is not None else (x1 + x2) / 2
    d = f' stroke-dasharray="{dash}"' if dash else ""
    path = f"M {x1} {y1} L {mx} {y1} L {mx} {y2} L {x2-(8 if arrowhead else 0)} {y2}"
    o = [f'<path d="{path}" fill="none" stroke="{stroke}" stroke-width="{sw}"{d}/>']
    if arrowhead:
        o.append(arrow(x2-9, y2, x2, y2, stroke, sw))
    return "".join(o)

def circ_op(cx, cy, glyph, r=11, size=14):
    return (f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#fff" stroke="{INK}" stroke-width="1.5"/>'
            + text(cx, cy, glyph, size, INK, weight=500))

def fatarrow(x, y, w=54, h=17, fill="#7a7a7a"):
    """The visual abstract's thick grey flow arrow."""
    hh = h/2; t = h*0.42
    p = (f"{x},{y-t/2} {x+w-h},{y-t/2} {x+w-h},{y-hh} {x+w},{y} "
         f"{x+w-h},{y+hh} {x+w-h},{y+t/2} {x},{y+t/2}")
    return f'<polygon points="{p}" fill="{fill}"/>'

def snowflake(cx, cy, r=7, col="#4aa3df"):
    o = []
    import math
    for k in range(3):
        a = math.radians(60*k)
        o.append(f'<line x1="{cx-r*math.cos(a):.1f}" y1="{cy-r*math.sin(a):.1f}" '
                 f'x2="{cx+r*math.cos(a):.1f}" y2="{cy+r*math.sin(a):.1f}" '
                 f'stroke="{col}" stroke-width="1.6" stroke-linecap="round"/>')
    return "".join(o)

def badge(x, y, label, kind="good", w=None, h=22):
    """Outcome badge: status colour + glyph + word (never colour alone)."""
    col = {"good": GOOD, "bad": BAD, "warn": WARN, "flat": MUTED}[kind]
    gly = {"good": "↑", "bad": "↓", "warn": "△", "flat": "="}[kind]
    w = w if w else 15 + 7.6*len(label) + 16
    o = [rrect(x, y, w, h, h/2, "#fff", col, 1.6)]
    o.append(text(x + 13, y + h/2, gly, 12.5, col, weight=700))
    o.append(text(x + 13 + (w-20)/2, y + h/2, label, 11.5, col, anchor="middle", weight=600))
    return "".join(o)

def note(x, y, w, s, size=11.5, fill=INK_SOFT, anchor="start"):
    return text(x if anchor == "start" else x, y, s, size, fill, anchor=anchor)

def panel(x, y, w, h, r=16, stroke=INK, sw=2.0, fill="none"):
    return rrect(x, y, w, h, r, fill, stroke, sw)

def svg(w, h, body, bg="#ffffff"):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" font-family="{FONT}">'
            f'<rect width="{w}" height="{h}" fill="{bg}"/>{body}</svg>')

def write(path, w, h, body):
    open(path, "w").write(svg(w, h, body))
    return path
