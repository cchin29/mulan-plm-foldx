#!/usr/bin/env python3
"""
Audit every data-bearing number drawn in images/foldx_figs/fig1_panels.html.

WHY THIS EXISTS
---------------
fig1_panels.html is hand-authored SVG. Its architecture diagrams are schematics
(nothing to audit), but ~25 of its text nodes are *results* — PCC values, paired
deltas, fold counts — and those were typed by hand with no declared source. That
is the same failure mode plot_fishbone.py was built to close: once the figure is
rendered, a retracted number survives every text sweep because it exists only as
a glyph.

Rewriting 186 KB of hand-tuned SVG as a generator would be the wrong trade. So
instead of deriving the *drawing* from data, this script derives the *check*:
every number in the figure is declared here with the source line it came from,
and the script fails loudly if the figure and the source ever disagree.

Net effect: `python3 scripts_plots/check_fig1_numbers.py` is now the thing that
breaks when RESULTS.md moves, instead of nothing breaking at all.

USAGE
-----
    python3 scripts_plots/check_fig1_numbers.py              # check + write sidecar
    python3 scripts_plots/check_fig1_numbers.py --html PATH  # check a copy (e.g. a deck asset)
    python3 scripts_plots/check_fig1_numbers.py --quiet      # only print failures

Exit 0 = every number matches its source. Exit 1 = at least one disagreement.

HOW A CHECK WORKS
-----------------
Each entry pins a number from two directions at once:

  html_after : the *label* text node it sits with in the figure. The value must
               appear within `window` text nodes of it — forward for a caption
               that follows its label, negative for the panels that draw the
               number first and the gloss after. Anchoring on the label (not on
               the bare number) means "0.740" cannot silently pass by matching
               some unrelated 0.740 elsewhere in the file.
  source     : file + regex with a (?P<v>...) group giving the authoritative value.

A number with no honest source is not allowed in here. If you cannot write the
locator, the figure should not be asserting the number.
"""

import argparse
import html as _html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "images" / "foldx_figs" / "fig1_panels.html"
SIDECAR = Path(__file__).resolve().parent / "fig1_provenance.txt"

RESULTS = "RESULTS.md"
FINDINGS = "FINDINGS_FOLDX_MUTANT_3DI.md"

# The two source documents sit at the repository root in the working tree and under docs/ in the
# published one, so resolve by name rather than by a fixed path. Checking a list beats hardcoding
# either layout: this script's whole purpose is to fail when a number moves, and it cannot do that
# from a tree it refuses to open.
_SEARCH = ("", "docs", "docs/history")


def find_source(name):
    for d in _SEARCH:
        p = ROOT / d / name if d else ROOT / name
        if p.exists():
            return p
    return None

# A markdown table row, e.g.  | run2 | ProstT5 (AA) | ... | single split | 0.740 | ...
# Bold markers are optional because RESULTS.md bolds its headline rows.
def _row(name, ncols_before_value):
    """Regex for the value cell `ncols_before_value` cells after the row key."""
    key = re.escape(name)
    skip = r"[^|\n]*\|" * ncols_before_value
    return rf"^\|\s*\*{{0,2}}{key}\*{{0,2}}\s*\|{skip}\s*\*{{0,2}}(?P<v>[\d.]+)\*{{0,2}}\s*\|"


# (panel, what, displayed, html_after, window, source_file, source_regex)
CHECKS = [
    # ---------------- panel 1a: ProstT5 3Di late concat -------------------
    ("1a", "run2 ProstT5 AA-only baseline PCC", "0.740",
     "ProstT5 AA-only baseline", 2, RESULTS, _row("run2", 4)),
    ("1a", "run6c per-residue AA+3Di concat PCC", "0.663",
     "per-residue concat → 2048-d", 2, RESULTS, _row("run6c", 4)),
    ("1a", "run6-E1 learned gate on 3Di block PCC", "0.705",
     "+ learned gate on the 3Di block", 2, RESULTS, _row("run6-E1", 4)),
    ("1a", "run6-B1 3Di as head context PCC", "0.673",
     "3Di as head context, outside mut − wt", 2, RESULTS, _row("run6-B1", 4)),
    ("1a", "run6-C3 interface-bias attention PCC", "0.662",
     "interface contacts as an attention bias", 2, RESULTS, _row("run6-C3", 4)),
    ("1a", "E1 learned gate settled value", "0.46",
     "the E1 gate collapsed to", 1, RESULTS,
     r"reached 0\.705 with the gate at (?P<v>[\d.]+)"),
    ("1a", "C3 attention-bias strength alpha", "0.017",
     "the E1 gate collapsed to", 2, RESULTS,
     r"the\s*\n?bias strength \*\*α → (?P<v>[\d.]+)"),
    ("1a", "panel headline: run2 → run6c (left)", "0.740",
     "hurts", 2, RESULTS, _row("run2", 4)),
    ("1a", "panel headline: run2 → run6c (right)", "0.663",
     "hurts", 2, RESULTS, _row("run6c", 4)),

    # ---------------- panel 1b: layer readout -----------------------------
    ("1b", "ProstT5 probe best layer PCC", "0.765",
     "— probe best", 1, RESULTS,
     r"\|\s*ProstT5 \(24 layers\)\s*\|\s*\d+ / (?P<v>[\d.]+)\s*\|"),
    ("1b", "ProstT5 default-layer probe PCC", "0.702",
     "— probe best", 1, RESULTS,
     r"\|\s*ProstT5 \(24 layers\)\s*\|[^|\n]*\|\s*\*{0,2}(?P<v>[\d.]+)\*{0,2}\s*\("),
    ("1b", "Ankh probe best layer PCC", "0.799",
     "— probe best", 1, RESULTS,
     r"\|\s*Ankh \(48 layers\)\s*\|\s*\d+ / (?P<v>[\d.]+)\s*\|"),
    ("1b", "Ankh default-layer probe PCC", "0.796",
     "— probe best", 1, RESULTS,
     r"\|\s*Ankh \(48 layers\)\s*\|[^|\n]*\|\s*\*{0,2}(?P<v>[\d.]+)\*{0,2}\s*\("),
    ("1b", "run3a mid-layer swap PCC", "0.648",
     "ProstT5, layer 7 · 1024-d", 2, RESULTS, _row("run3a", 4)),
    ("1b", "run3b multi-layer concat PCC", "0.429",
     "ProstT5, concat 7 ⊕ 10 · 2048-d", 2, RESULTS, _row("run3b", 4)),
    ("1b", "run1 Ankh-large baseline PCC (probe-validity note)", "0.757",
     "Probe validity", 3, RESULTS, _row("run1", 4)),

    # ---------------- panel 1c: SaProt input fusion -----------------------
    ("1c", "SaProt #-masked seq-only control PCC", "0.816",
     "3Di masked with", 2, RESULTS,
     r"seq-only ctrl (?P<v>[\d.]+) → WT-3Di"),
    ("1c", "SaProt real WT-3Di PCC", "0.842",
     "real WT 3Di", 2, RESULTS,
     r"seq-only ctrl [\d.]+ → WT-3Di (?P<v>[\d.]+)"),
    ("1c", "SaProt WT-3Di + Tier-1 aug PCC", "0.852",
     "Best sub-1B arm", 1, RESULTS,
     r"WT-3Di [\d.]+ →\s*\n?WT-3Di \+ aug (?P<v>[\d.]+)"),
    ("1c", "SaProt structure lift (paired)", "0.026",
     "PCC   ·   8 / 10 folds, paired", 1, RESULTS,
     r"WT-3Di \(mini3di\); \+(?P<v>[\d.]+) vs its #-masked seq-only ctrl"),
    ("1c", "SaProt structure lift, folds won", "8",
     "PCC   ·   8 / 10 folds, paired", 1, RESULTS,
     r"vs its #-masked seq-only ctrl \((?P<v>\d+)/10\)"),
    ("1c", "ProstT5 cost of the same WT 3Di (derived: run2 − run6c)", "0.077",
     "The same WT 3Di that cost ProstT5", 1, "DERIVED", "run2 - run6c"),
    ("1c", "mutant-3Di mutations that changed anything", "0 / 39",
     "mutations changed the 3Di anywhere", -1, RESULTS,
     r"\*\*(?P<v>0/39) changed the 3Di at any"),
    ("1c", "backbone atoms that moved (1KNE)", "0 / 232",
     "backbone atoms moved between the WT re-model", -1, FINDINGS,
     r"\*\*(?P<v>0 of 232) backbone"),

    # ---------------- panel 1d: FoldX score channel -----------------------
    ("1d", "raw FoldX ΔΔG vs experimental label (Pearson)", "0.407",
     "raw FoldX ΔΔG vs the experimental label", -1, RESULTS,
     r"Raw FoldX ΔΔG vs experimental label \(n=1100\) \| \*\*Pearson \+(?P<v>[\d.]+)"),
    ("1d", "r(FoldX, Ankh OOF prediction) — redundant part", "0.35",
     "already inside Ankh-large's own predictions", -1, RESULTS,
     r"out-of-fold Ankh \*\*prediction\*\*\) — redundant part \| \+(?P<v>[\d.]+)"),
    ("1d", "r(FoldX, Ankh OOF residual) — orthogonal part", "0.20",
     "orthogonal to its residual", -1, RESULTS,
     r"out-of-fold Ankh \*\*residual\*\*\) — orthogonal part \| \*\*\+(?P<v>[\d.]+)"),
    ("1d", "OOF linear ceiling, all 12 terms", "0.011",
     "out-of-fold ceiling from all 12 terms", -1, RESULTS,
     r"OOF ceiling, all 12 terms \| \+(?P<v>[\d.]+)"),
    ("1d", "measured lift, FoldX scalar (Ankh-large S1102)", "0.009",
     "Measured lift", 0, RESULTS,
     r"FoldX binding ΔΔG in add_scores; \*\*paired \+(?P<v>[\d.]+)"),
    ("1d", "measured lift, FoldX 12-term MLP", "0.017",
     "Measured lift", 0, RESULTS,
     r"Stage-2 decomposed MLP head; \*\*paired \+(?P<v>[\d.]+) vs base"),
    ("1d", "MLP head extra parameter count", "226",
     "+226 parameters", 0, RESULTS,
     r"Dropout\(0\.1\) → 1, \+(?P<v>\d+) params\)"),
    ("1d", "scalar safe-default range, low", "0.006",
     "Safe default: +0.006–0.013 across the four SKEMPI benchmarks.", 0, RESULTS,
     r"the scalar is the safe default \(\+(?P<v>[\d.]+)–[\d.]+ everywhere\)"),
    ("1d", "scalar safe-default range, high", "0.013",
     "Safe default: +0.006–0.013 across the four SKEMPI benchmarks.", 0, RESULTS,
     r"the scalar is the safe default \(\+[\d.]+–(?P<v>[\d.]+) everywhere\)"),
]


def text_nodes(path):
    s = path.read_text()
    out = []
    for m in re.finditer(r"<text[^>]*>(.*?)</text>", s, re.S):
        t = _html.unescape(re.sub(r"<[^>]+>", "", m.group(1))).strip()
        t = re.sub(r"\s+", " ", t)
        if t:
            out.append(t)
    return out


def in_figure(nodes, anchor, window, value):
    """Is `value` drawn within `window` text nodes of `anchor`?

    window >= 0 looks forward from the anchor (label, then value);
    window <  0 looks backward (value, then the label that glosses it).
    """
    anchor = re.sub(r"\s+", " ", anchor)
    value = re.sub(r"\s+", " ", value)
    hits = [i for i, n in enumerate(nodes) if anchor in n]
    if not hits:
        return None  # anchor itself is gone — the figure was restructured
    for i in hits:
        span = nodes[i:i + window + 1] if window >= 0 else nodes[max(0, i + window):i + 1]
        for n in span:
            if value in n:
                return True
    return False


def from_source(sources, fname, pattern):
    """Return (value, lineno, the source line it was read from) or (None, 0, '')."""
    text = sources[fname]
    m = re.search(pattern, text, re.M)
    if not m:
        return None, 0, ""
    lineno = text.count("\n", 0, m.start()) + 1
    line = text.splitlines()[lineno - 1].strip()
    if len(line) > 96:
        line = line[:95] + "…"
    return m.group("v"), lineno, line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", default=str(FIG))
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--no-sidecar", action="store_true")
    args = ap.parse_args()

    fig = Path(args.html)
    if not fig.exists():
        sys.exit(f"FATAL: figure not found: {fig}")
    nodes = text_nodes(fig)

    sources = {}
    for f in {c[5] for c in CHECKS} - {"DERIVED"}:
        p = find_source(f)
        if p is None:
            sys.exit(f"FATAL: source not found anywhere in {_SEARCH}: {f}")
        sources[f] = p.read_text()

    # values needed by the DERIVED checks, resolved from their own locators
    run2 = float(from_source(sources, RESULTS, _row("run2", 4))[0])
    run6c = float(from_source(sources, RESULTS, _row("run6c", 4))[0])
    derived = {
        "run2 - run6c": (f"{run2 - run6c:.3f}",
                         f"RESULTS.md run2 ({run2:.3f}) − run6c ({run6c:.3f})"),
    }

    failures, lines = [], []
    for panel, what, shown, anchor, window, fname, pattern in CHECKS:
        if fname == "DERIVED":
            truth, origin = derived[pattern][0], f"DERIVED  {derived[pattern][1]}"
        else:
            truth, lineno, srcline = from_source(sources, fname, pattern)
            origin = f"{fname}:{lineno}  {srcline}"

        if truth is None:
            failures.append(f"[{panel}] {what}\n"
                            f"    LOCATOR MISS — nothing in {fname} matches:\n"
                            f"    {pattern}")
            continue

        # compare numerically where both sides are plain numbers
        try:
            ok_value = abs(float(shown) - float(truth)) < 5e-4
        except ValueError:
            ok_value = shown.replace(" ", "") == truth.replace(" ", "").replace("of", "/")

        drawn = in_figure(nodes, anchor, window, shown)
        if drawn is None:
            failures.append(f"[{panel}] {what}\n"
                            f"    ANCHOR MISS — no text node contains {anchor!r}.\n"
                            f"    The figure was restructured; re-point this check.")
            continue
        if not drawn:
            failures.append(f"[{panel}] {what}\n"
                            f"    NOT DRAWN — {shown!r} is not within {window} node(s) "
                            f"of {anchor!r}.")
            continue
        if not ok_value:
            failures.append(f"[{panel}] {what}\n"
                            f"    DISAGREE — figure draws {shown!r}, "
                            f"{fname} says {truth!r}\n"
                            f"    locator: {pattern}")
            continue

        lines.append(f"{panel}  {shown:<8}  {what}\n"
                     f"          <- {origin}")

    if not args.quiet:
        print(f"fig1_panels.html — {len(CHECKS)} declared numbers, "
              f"{len(lines)} verified, {len(failures)} failed")
        print(f"figure: {fig}")
        print()
        for l in lines:
            print("  OK  " + l.replace("\n", "\n  "))

    if failures:
        print()
        print("=" * 72)
        for f in failures:
            print("FAIL " + f)
            print()
        sys.exit(f"FATAL: {len(failures)} of {len(CHECKS)} numbers in "
                 f"fig1_panels.html do not match their source.")

    if not args.no_sidecar:
        SIDECAR.write_text(
            "Every data-bearing number drawn in images/foldx_figs/fig1_panels.html,\n"
            "and the source line it came from.\n"
            "Re-verify with: python3 scripts_plots/check_fig1_numbers.py\n"
            "\n"
            "The architecture diagrams in this figure are schematics and carry no data.\n"
            "\n" + "\n".join(lines) + "\n")
        if not args.quiet:
            print(f"\nwrote {SIDECAR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
