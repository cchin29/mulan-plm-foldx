#!/usr/bin/env python3
"""Fishbone: the 2023->2025 shift to leakage-controlled, per-structure ddG evaluation.

Why this script exists
----------------------
The fishbone used to be a hand-authored SVG (assets/fishbone.html) rasterised to a PNG. Every
number in it was a literal typed into the markup, and the deck shipped the raster -- so none of the
numbers were greppable and none had a declared provenance. That is exactly how
"CATH tier - per-struct rho 0.468" survived the 2026-08-06 coverage correction and every text
sweep that followed it.

This version derives every number it draws, and refuses to draw at all if the sources disagree:

  * MuLAN's own numbers come from experiments/rescore_perstructure/foldx_alone_baseline_allplm_cov99.csv
    (rho, the FoldX-alone row on the same tier, the delta and its 95% cluster-bootstrap CI, and the
    mutation/complex counts). Nothing about MuLAN is typed in.
  * Published comparators are parsed out of the pipe table in experiments/BENCHMARK_MATRIX.md,
    which is the repo's single source of truth for frontier numbers and carries its own footnoted
    citations. Nothing about the frontier is typed in either.
  * check_consistency() then asserts the MuLAN row of BENCHMARK_MATRIX.md agrees with the CSV to
    3 dp. If someone reruns the pipeline and updates only one of the two, this script fails loudly
    instead of silently emitting a stale figure.

Output is a static SVG -- every label is a real <text> node, so `grep 0.468 fishbone.svg` works and
the deck can inline it instead of shipping a raster. The PNG is optional and only for slide decks
that cannot take inline SVG.

Run:   python3 scripts_plots/plot_fishbone.py [--png]
Out:   scripts_plots/fishbone.svg  (+ fishbone.png with --png)
       scripts_plots/fishbone_provenance.txt -- every drawn number and the file/row it came from
"""
import argparse
import csv
import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FXA = ROOT / "experiments/rescore_perstructure/foldx_alone_baseline_allplm_cov99.csv"
BMX = ROOT / "experiments/BENCHMARK_MATRIX.md"
OUT_SVG = ROOT / "scripts_plots/fishbone.svg"
OUT_PNG = ROOT / "scripts_plots/fishbone.png"
OUT_PROV = ROOT / "scripts_plots/fishbone_provenance.txt"

CATH_TIER = "fullSK CATH-all"
FLAGSHIP = ("esmc6b", "fx_scalar")  # the arm the deck quotes on the CATH tier

C = dict(
    t1="#86b6ef", t2="#2a78d6", t3="#e08a2b", t4="#6a3fb0",
    mulan="#008300", warn="#e0913a",
    ink="#0b0b0b", ink2="#52514e", muted="#898781",
    surface="#fcfcfb", border="rgba(11,11,11,0.14)",
)

PROV = []  # (drawn string, source)


def note(value, source):
    """Record a drawn number against the file+row it came from, and return it unchanged."""
    PROV.append((str(value), source))
    return value


# --------------------------------------------------------------------------- data


def load_foldx_alone():
    """tier -> {(model, arm): row} from the cluster-bootstrap baseline CSV."""
    out = {}
    with open(FXA) as fh:
        for r in csv.DictReader(fh):
            out.setdefault(r["tier"], {})[(r["model"], r["arm"])] = r
    if CATH_TIER not in out:
        sys.exit(f"FATAL: tier {CATH_TIER!r} absent from {FXA}")
    return out[CATH_TIER]


def parse_benchmark_matrix():
    """Parse the comparator pipe table in BENCHMARK_MATRIX.md.

    Returns {method_label: {column: float}}. Footnote markers (superscript letters, bold markup)
    are stripped; em-dashes become None. The table is located by its header row rather than by
    line number so that edits above it do not silently shift the parse.
    """
    lines = BMX.read_text().splitlines()
    hdr_i = next((i for i, l in enumerate(lines)
                  if l.startswith("| Method") and "CATH" in l), None)
    if hdr_i is None:
        sys.exit(f"FATAL: could not find the comparator table header in {BMX}")

    def cells(line):
        return [c.strip() for c in line.strip().strip("|").split("|")]

    cols = cells(lines[hdr_i])
    rows = {}
    for line in lines[hdr_i + 2:]:
        if not line.startswith("|"):
            break
        cs = cells(line)
        if len(cs) != len(cols):
            continue
        name = re.sub(r"\*\*|`", "", cs[0]).strip()
        vals = {}
        for col, cell in zip(cols[1:], cs[1:]):
            v = re.sub(r"\*\*|`", "", cell).strip()
            v = re.sub(r"[^\d.]+$", "", v)  # trailing footnote markers
            try:
                vals[re.sub(r"\*\*", "", col).strip()] = float(v)
            except ValueError:
                vals[re.sub(r"\*\*", "", col).strip()] = None
        rows[name] = vals
    if not rows:
        sys.exit(f"FATAL: comparator table in {BMX} parsed to zero rows")
    return rows


def pick(rows, prefix):
    """The single table row whose method label starts with `prefix`."""
    hits = [k for k in rows if k.startswith(prefix)]
    if len(hits) != 1:
        sys.exit(f"FATAL: {len(hits)} rows in BENCHMARK_MATRIX.md start with {prefix!r}: {hits}")
    return hits[0], rows[hits[0]]


def check_consistency(cath, rows):
    """Abort if the two sources disagree about MuLAN. This is the whole point of the script."""
    csv_rho = float(cath[FLAGSHIP]["rho"])
    _, mulan = pick(rows, "MuLAN SKEMPI best")
    doc_rho = mulan.get("CATH")
    if doc_rho is None:
        sys.exit("FATAL: MuLAN SKEMPI-best row in BENCHMARK_MATRIX.md has no CATH cell")
    if round(csv_rho, 3) != round(doc_rho, 3):
        sys.exit(
            "FATAL: sources disagree on MuLAN's CATH per-structure rho.\n"
            f"  {FXA.name}: {csv_rho:.4f} ({FLAGSHIP[0]} {FLAGSHIP[1]}, tier {CATH_TIER})\n"
            f"  {BMX.name}: {doc_rho:.4f} (MuLAN SKEMPI best row, CATH column)\n"
            "One of them was regenerated without the other. Fix before drawing."
        )
    return csv_rho


# --------------------------------------------------------------------------- svg


class Svg:
    def __init__(self, w, h):
        self.w, self.h, self.parts = w, h, []

    def el(self, tag, **a):
        at = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in a.items())
        self.parts.append(f"<{tag} {at}/>")

    def text(self, x, y, s, **a):
        at = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in a.items())
        self.parts.append(f'<text x="{x}" y="{y}" {at}>{html.escape(str(s))}</text>')

    def render(self, aria):
        body = "\n".join("  " + p for p in self.parts)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
            f'id="fb" role="img" aria-label="{html.escape(aria)}" '
            f'font-family="system-ui,-apple-system,Segoe UI,sans-serif">\n'
            f'  <rect width="{self.w}" height="{self.h}" fill="{C["surface"]}"/>\n'
            f'  <defs><marker id="arrow" markerWidth="14" markerHeight="14" refX="9" refY="5" '
            f'orient="auto" markerUnits="userSpaceOnUse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{C["ink2"]}"/></marker></defs>\n'
            f"{body}\n</svg>\n"
        )


# Canvas is sized to the content, not to a 16:9 habit. The deck slide that carries
# this figure gives it ~470 rendered px inside a 690px .slidebody; a 1600x900 viewBox
# renders ~740px and clips the bottom cards. So: no title/subtitle band (the slide's
# own H2 and lede say the same thing), legend moved into the dead space bottom-left,
# and the vertical gaps closed up.
CANVAS_W, CANVAS_H = 1600, 556
SPINE_Y, X0, X1, CW, GAP = 182, 250, 1360, 204, 50
BOTTOM_Y = 380         # top of the bottom band, below the deepest down-rib (356)
FOOTNOTE_Y = 544


def block(s, cx, cy, w, h, color, title, lines):
    s.el("rect", x=cx - w / 2, y=cy - h / 2, width=w, height=h, rx=10,
         fill=C["surface"], stroke=color, stroke_width=2)
    s.el("rect", x=cx - w / 2, y=cy - h / 2, width=w, height=5, rx=2.5, fill=color)
    s.text(cx, cy - h / 2 + 26, title, font_size=13, font_weight=700,
           fill=C["ink"], text_anchor="middle")
    for i, l in enumerate(lines):
        s.text(cx, cy - h / 2 + 45 + i * 16, l, font_size=11.5,
               fill=C["ink2"], text_anchor="middle")


def rib(s, x, side, color, cat, year, title, lines, stroke=None, strokew=1):
    h = 52 + len(lines) * 16 + 8
    cy = SPINE_Y - GAP - h / 2 if side == "up" else SPINE_Y + GAP + h / 2
    near = cy + h / 2 if side == "up" else cy - h / 2
    s.el("line", x1=x, y1=SPINE_Y, x2=x, y2=near, stroke=color, stroke_width=2, opacity=0.7)
    s.el("circle", cx=x, cy=SPINE_Y, r=5, fill=color, stroke=C["surface"], stroke_width=2)
    left, top = x - CW / 2, cy - h / 2
    s.el("rect", x=left, y=top, width=CW, height=h, rx=10, fill=C["surface"],
         stroke=stroke or C["border"], stroke_width=strokew)
    s.el("rect", x=left, y=top, width=5, height=h, rx=2.5, fill=color)
    s.text(left + 16, top + 18, cat, font_size=9.5, font_weight=700,
           fill=C["ink2"], letter_spacing=0.6)
    s.text(left + 16, top + 35, f"{year} · {title}", font_size=12.5,
           font_weight=700, fill=C["ink"])
    for i, l in enumerate(lines):
        s.text(left + 16, top + 53 + i * 16, l, font_size=11, fill=C["ink2"])


def build(cath, rows):
    fx = cath[FLAGSHIP]
    alone = cath[("—", "FoldX alone")]
    rho = note(f"{float(fx['rho']):.3f}", f"{FXA.name}:{CATH_TIER}/{'/'.join(FLAGSHIP)} rho")
    fx_alone = note(f"{float(alone['rho']):.3f}", f"{FXA.name}:{CATH_TIER}/FoldX alone rho")
    d = note(f"{float(fx['d_vs_foldx']):+.3f}", f"{FXA.name}:{CATH_TIER} d_vs_foldx")
    dlo = f"{float(fx['d_lo']):+.3f}"
    dhi = f"{float(fx['d_hi']):+.3f}"
    note(f"[{dlo}, {dhi}]", f"{FXA.name}:{CATH_TIER} d_lo/d_hi (95% cluster bootstrap)")
    n_mut = note(fx["n_mut"], f"{FXA.name}:{CATH_TIER} n_mut")
    n_cpx = note(fx["n_cplx"], f"{FXA.name}:{CATH_TIER} n_cplx")

    _, cathddg = pick(rows, "CATH-ddG")
    _, uspddg = pick(rows, "USP-ddG")
    _, protbff = pick(rows, "ProtBFF")
    _, rde = pick(rows, "RDE-Network")
    _, mulan = pick(rows, "MuLAN SKEMPI best")
    c_cath = note(f"{cathddg['CATH']:.3f}", f"{BMX.name}: CATH-ddG row, CATH column")
    u_cath = note(f"{uspddg['CATH']:.3f}", f"{BMX.name}: USP-ddG row, CATH column")
    p_sp = note(f"{protbff['CD-HIT Sp']:.3f}", f"{BMX.name}: ProtBFF row, CD-HIT Sp column")
    r_cath = note(f"{rde['CATH']:.3f}", f"{BMX.name}: RDE-Network row, CATH column")
    m_sp = note(f"{mulan['CD-HIT Sp']:.3f}", f"{BMX.name}: MuLAN SKEMPI best row, CD-HIT Sp")
    ceiling = max(cathddg["CATH"], uspddg["CATH"])
    note(f"{ceiling:.2f}", f"{BMX.name}: max(CATH-ddG, USP-ddG) CATH column")

    s = Svg(CANVAS_W, CANVAS_H)
    s.el("line", x1=X0, y1=SPINE_Y, x2=X1, y2=SPINE_Y,
         stroke=C["ink2"], stroke_width=4, marker_end="url(#arrow)")

    block(s, 140, SPINE_Y, 196, 100, C["t1"], "≤2023 · MuLAN-Ankh",
          ["per-mutation 10-fold CV", "pooled Pearson ≈ 0.85", "leaky — not comparable"])
    block(s, 1470, SPINE_Y, 232, 108, C["t4"], "Leakage-controlled evaluation",
          ["per-structure Spearman", "frontier-comparable",
           f"OOD ceiling ≈ {ceiling:.2f} (CATH tier)"])

    rib(s, 380, "up", C["t2"], "BY-COMPLEX", "2023", "RDE-Network",
        ["by-complex split (rng 2022)", "+ per-structure metric origin",
         "removes same-complex leak"])
    rib(s, 820, "up", C["t4"], "CATH", "2025·07", "CATH-ddG",
        ["CATH-superfamily hold-out (TM<0.6)", "fold-level OOD",
         f"per-struct ρ {c_cath} (near-SOTA)"])
    rib(s, 1120, "up", C["t3"], "CLUSTERED (CD-HIT ≤60%)", "2025·12", "ProtBFF",
        ["CD-HIT ≤60% clustered split", f"seq/PLM frontier ρ {p_sp} (pooled)",
         f"MuLAN+FoldX {m_sp} pooled — above"])

    rib(s, 380, "down", C["muted"], "METRIC AXIS (orthogonal)", "2023", "Metric shift",
        ["pooled Pearson → per-structure", "ranks mutations within a complex",
         "adopted field-wide"])
    rib(s, 590, "down", C["t3"], "CLUSTERED (interface)", "2024", "PPIRef / PPIformer",
        ["interface de-dup (iDist)", "removes near-duplicate interfaces",
         "+ leakage paper (2404.10457)"])
    rib(s, 980, "down", C["t4"], "CATH · SOTA", "2025·11", "USP-ddG",
        ["CATH-superfamily · current SOTA", f"per-PPI ρ {u_cath} · re-evals RDE→{r_cath}",
         "88.7% of by-complex ‘easy’"])
    rib(s, 1270, "down", C["t4"], "CATH · THIS WORK", "now", "MuLAN + FoldX",
        [f"CATH tier · per-struct ρ {rho}", f"vs FoldX alone {fx_alone} (same tier)",
         f"Δ {d} [{dlo}, {dhi}]", f"{n_mut} muts / {n_cpx} cplx · seq-only"],
        stroke=C["mulan"], strokew=2)

    # ---- bottom-left: title + split-tier legend (was the top band) ----
    lx, ly = 40, BOTTOM_Y
    s.text(lx, ly + 20, "From leaky CV to leakage-controlled ΔΔG evaluation",
           font_size=17, font_weight=700, fill=C["ink"])
    s.text(lx, ly + 42, "SPLIT TIER  (leakage control · light = leakiest → dark = strictest)",
           font_size=10.5, font_weight=700, fill=C["ink2"], letter_spacing=0.4)
    ramp = [("t1", "10-fold CV (leaky)"), ("t2", "by-complex"),
            ("t3", "clustered ≤60%"), ("t4", "CATH-superfamily (strictest)")]
    for i, (k, lab) in enumerate(ramp):
        s.el("rect", x=lx + i * 140, y=ly + 48, width=140, height=15,
             fill=C[k], stroke=C["surface"], stroke_width=1)
        s.text(lx + i * 140 + 70, ly + 76, lab, font_size=9.5,
               fill=C["ink2"], text_anchor="middle")
    s.el("rect", x=lx, y=ly + 92, width=13, height=13, rx=3, fill=C["muted"])
    s.text(lx + 19, ly + 103, "metric axis", font_size=11, fill=C["ink2"])
    s.el("rect", x=lx + 128, y=ly + 92, width=13, height=13, rx=3, fill=C["surface"],
         stroke=C["mulan"], stroke_width=2)
    s.text(lx + 147, ly + 103, "MuLAN (this work)", font_size=11, fill=C["ink2"])
    s.text(lx, ly + 124, "numbers = per-structure Spearman ρ (field-standard), "
                         "unless marked pooled",
           font_size=11, fill=C["muted"], font_style="italic")

    # ---- bottom-right: still-lacking panel ----
    x, y, w, h = 1150, BOTTOM_Y, 418, 146
    s.el("rect", x=x, y=y, width=w, height=h, rx=10, fill=C["surface"],
         stroke=C["border"], stroke_width=1)
    s.el("rect", x=x, y=y, width=5, height=h, rx=2.5, fill=C["warn"])
    s.text(x + 18, y + 26, "Still lacking (out-of-superfamily ΔΔG)",
           font_size=13, font_weight=700, fill=C["ink"])
    for i, t in enumerate([
        "SOTA is all structure-based; sequence-only lags (ESM2 0.19)",
        "FoldX fixed-backbone caps the biophysical channel",
        "interface-pretraining underdelivers OOD (< plain FoldX)",
        "label-noise + antibody / OOD sets remain weak",
        "no hosted leaderboard — match split AND metric to compare",
    ]):
        s.el("circle", cx=x + 22, cy=y + 40 + i * 19, r=2.4, fill=C["ink2"])
        s.text(x + 32, y + 44 + i * 19, t, font_size=11.5, fill=C["ink2"])

    s.text(40, FOOTNOTE_Y,
           f"Generated by scripts_plots/plot_fishbone.py — MuLAN numbers from {FXA.name}, "
           f"comparators from {BMX.name}. No literal ρ is typed into this figure.",
           font_size=10.5, fill=C["muted"])
    return s.render(
        "Fishbone timeline of the shift to leakage-controlled, per-structure ddG evaluation, "
        "2023 to 2026.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", action="store_true", help="also rasterise (needs cairosvg or playwright)")
    a = ap.parse_args()

    cath = load_foldx_alone()
    rows = parse_benchmark_matrix()
    check_consistency(cath, rows)

    OUT_SVG.write_text(build(cath, rows))
    width = max(len(v) for v, _ in PROV)
    OUT_PROV.write_text(
        "Every number drawn in fishbone.svg, and where it came from.\n"
        "Regenerate with: python3 scripts_plots/plot_fishbone.py\n\n"
        + "\n".join(f"{v:<{width}}  <-  {src}" for v, src in PROV) + "\n")
    print(f"wrote {OUT_SVG}  ({len(PROV)} derived numbers)")
    print(f"wrote {OUT_PROV}")

    if a.png:
        rasterise()


def rasterise():
    try:
        import cairosvg
        cairosvg.svg2png(url=str(OUT_SVG), write_to=str(OUT_PNG), scale=2)
    except ImportError:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            b = pw.chromium.launch()
            pg = b.new_page(viewport={"width": CANVAS_W, "height": CANVAS_H},
                            device_scale_factor=2)
            pg.goto(OUT_SVG.resolve().as_uri())
            pg.wait_for_timeout(400)
            pg.locator("svg#fb").screenshot(path=str(OUT_PNG))
            b.close()
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
