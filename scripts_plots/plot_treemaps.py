#!/usr/bin/env python3
"""Composition of the benchmark subsets, drawn as treemaps under four different units.

`docs/SPLITS_AND_METRICS.md` argues that a by-complex hold-out is much weaker than it sounds and
that the per-structure metric averages over a minority of the complexes. Both claims are about the
*shape* of SKEMPI rather than about any model, and both are hard to believe from a table: what
makes them obvious is seeing that a handful of protease and antibody complexes occupy most of the
area, and that the units get dramatically coarser as the hold-out gets stricter.

Each view keeps the same encoding -- one tile per group, area proportional to the mutations in it
-- and changes only what a group is:

    scale       one complex.                Where the data actually lives.
    clusters    one <=60% sequence family.  MuLAN's clustered split unit; complexes single-linked
                                            when they share a chain cluster.
    cath        one CATH superfamily group. The strictest rung; both partners must share a
                                            superfamily, via SIFTS.
    coverage    one complex, coloured by    What the per-structure metric averages over, and what
                whether it clears T>=10.    it silently drops.

Reading them in that order is the leakage ladder: the same mutations, regrouped into progressively
fewer and larger units, each of which a strict split has to hold out whole.

    python scripts_plots/plot_treemaps.py --view all --out scripts_plots/treemaps --plots --html

Every input is in the repository -- `data/splits/`, `data/splits/clusters_id60/`,
`experiments/cath_leakage/cath_families.json`, `examples/S1102.tsv`. Nothing needs a GPU, a FoldX
licence or a download. A view whose input is missing is skipped and named, rather than silently
producing a smaller picture.

The treemap layout is duplicated from skempi-foldx's `experiments/store_composition.py` rather
than shared: the two repositories are published separately and neither should depend on the other
for a figure.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Size classes and their colours, used wherever a view has nothing else to say with colour.
CLASSES = (
    (100, "giant: >=100 mutations", "#0e7c66"),
    (10, "large: 10-99", "#2a9d7f"),
    (2, "medium: 2-9", "#6b7b8c"),
    (1, "singleton: 1", "#3d4652"),
)
SCORED = "#0e7c66"
DROPPED = "#4a5462"


def size_class(n):
    for threshold, label, colour in CLASSES:
        if n >= threshold:
            return label, colour
    return CLASSES[-1][1], CLASSES[-1][2]


# -- inputs --------------------------------------------------------------------------------------
# Each subset names the fold-0 files that, concatenated, are the whole subset -- train + val + test
# partition it, so the union is every row exactly once regardless of which fold is read.
SUBSETS = (
    ("S1102", ("examples/S1102.tsv",), "s1102"),
    ("SKEMPI single-point",
     tuple(f"data/splits/splits_skempi_full_bycomplex_seed42/fold_0/skempi_sp_{s}.tsv"
           for s in ("train", "val", "test")), "skempi_full_single"),
    ("SKEMPI multi-point",
     tuple(f"data/splits/splits_skempi_full_mp_bycomplex_seed42/fold_0/skempi_mp_{s}.tsv"
           for s in ("train", "val", "test")), "skempi_full_multi"),
    ("SKEMPI single + multi",
     tuple(f"data/splits/splits_skempi_full_bycomplex_all_seed42/fold_0/skempi_all_{s}.tsv"
           for s in ("train", "val", "test")), None),
)


def complex_key(a, b):
    """``PDB_chainsA_chainsB`` from the two partner labels, whichever convention they use.

    The full-SKEMPI splits label a partner ``1A4Y.A.B_A``, carrying the author chains in the
    complex name; S1102 labels it ``1A22_A``, carrying only the role letter. Both have to reduce
    to the one key that ``cath_families.json`` and the cluster files are written against.
    """
    head = a.rsplit("_", 1)[0]
    if "." in head:
        return head.replace(".", "_")
    pdb, ca = a.split("_", 1)
    _, cb = b.split("_", 1)
    return f"{pdb}_{ca}_{cb}"


def read_subset(paths):
    """``{complex key: mutation count}``, or ``None`` if any file is missing."""
    resolved = [ROOT / p for p in paths]
    if not all(p.exists() for p in resolved):
        return None
    counts = Counter()
    for path in resolved:
        with open(path) as fh:
            for row in csv.reader(fh, delimiter="\t"):
                if len(row) >= 3:
                    counts[complex_key(row[0], row[1])] += 1
    return counts


def read_clusters(name):
    """``{chain label: cluster representative}`` from a vendored mmseqs ``easy-cluster`` table."""
    if name is None:
        return None
    path = ROOT / "data" / "splits" / "clusters_id60" / f"{name}.tsv"
    if not path.exists():
        return None
    chain2rep = {}
    with open(path) as fh:
        for row in csv.reader(fh, delimiter="\t"):
            if len(row) >= 2:
                chain2rep[row[1]] = row[0]
    return chain2rep


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def cluster_families(counts, chain2rep):
    """``{complex key: family}`` by single-linkage over shared chain clusters.

    The same rule as `experiments/retrain_split/cluster_split.py` builds the clustered split
    with: two complexes are one family if either chain of one falls in a cluster containing a
    chain of the other. Reproduced here rather than imported because that module runs mmseqs on
    import path assumptions this script does not want.
    """
    chains = {}
    for key in counts:
        parts = key.split("_")
        if len(parts) != 3:
            # complex_key builds PDB_chainsA_chainsB from labels that carry no underscores of
            # their own. Nothing in the shipped splits violates that, but a hand-made table could,
            # and silently mis-clustering is worse than saying which key was unreadable.
            raise ValueError(f"complex key {key!r} is not PDB_chainsA_chainsB")
        pdb, ca, cb = parts
        # The cluster tables are keyed by the label form their subset was clustered under, so try
        # the dotted form first and fall back to the flat one.
        dotted = f"{pdb}.{ca}.{cb}"
        reps = set()
        for chain in (ca, cb):
            for label in (f"{dotted}_{chain}", f"{pdb}_{chain}"):
                if label in chain2rep:
                    reps.add(chain2rep[label])
                    break
            else:
                reps.add(f"{key}:{chain}")  # unclustered chain: its own singleton
        chains[key] = reps

    rep_to_complexes = defaultdict(list)
    for key, reps in chains.items():
        for rep in reps:
            rep_to_complexes[rep].append(key)
    uf = UnionFind(list(counts))
    for members in rep_to_complexes.values():
        for other in members[1:]:
            uf.union(members[0], other)
    return {key: uf.find(key) for key in counts}


def cath_families(counts):
    """``{complex key: family}`` from the SIFTS-derived assignment, or ``None`` if absent.

    This is our own derivation (`experiments/cath_leakage/cath_leakage.py`, both-partners rule
    over SIFTS `pdb_chain_cath_uniprot`) and not the fixed partition USP-ddG publish -- that one
    is `data/splits/splits_skempi_full_cath_kfold`, and it is theirs rather than ours.
    """
    path = ROOT / "experiments" / "cath_leakage" / "cath_families.json"
    if not path.exists():
        return None
    per_complex = json.load(open(path))["per_complex"]
    out, missing = {}, 0
    for key in counts:
        entry = per_complex.get(key)
        if entry is None:
            missing += 1
            out[key] = f"unmapped:{key}"
        else:
            out[key] = entry["family_id"]
    if missing:
        print(f"[cath] {missing} complexes absent from the assignment; each is its own family",
              file=sys.stderr)
    return out


# -- grouping ------------------------------------------------------------------------------------
def tiles_by_complex(counts, threshold=None):
    """One tile per complex. With a threshold, colour says whether it clears it."""
    tiles = []
    for key, n in counts.items():
        scored = None if threshold is None else n >= threshold
        tiles.append({
            "label": key, "mutations": n, "members": 1, "member_labels": [key],
            "scored": scored,
            "colour": (SCORED if scored else DROPPED) if threshold is not None
                      else size_class(n)[1],
        })
    return sorted(tiles, key=lambda t: (-t["mutations"], t["label"]))


def tiles_by_family(counts, families):
    """One tile per family, named for its largest member so the tile is identifiable."""
    grouped = defaultdict(list)
    for key, family in families.items():
        grouped[family].append(key)
    tiles = []
    for members in grouped.values():
        members = sorted(members, key=lambda k: (-counts[k], k))
        total = sum(counts[k] for k in members)
        label = members[0] if len(members) == 1 else f"{members[0]} +{len(members) - 1}"
        tiles.append({
            "label": label, "mutations": total, "members": len(members),
            "member_labels": members, "scored": None, "colour": size_class(total)[1],
        })
    return sorted(tiles, key=lambda t: (-t["mutations"], t["label"]))


VIEWS = ("scale", "clusters", "cath", "coverage")
VIEW_TITLES = {
    "scale": "Mutations per complex",
    "clusters": "Complexes under <=60% sequence clustering",
    "cath": "Complexes under CATH superfamily grouping",
    "coverage": "Per-structure Spearman coverage at T>=10",
}
VIEW_SUBTITLES = {
    "scale": "Tile = one complex, area proportional to its mutations; colour = size class",
    "clusters": ("Tile = one <=60% sequence-identity family (mmseqs, single-linkage over shared "
                 "chain clusters); area proportional to its mutations"),
    "cath": ("Tile = one CATH superfamily group (both partners share a superfamily, via SIFTS); "
             "area proportional to its mutations"),
    "coverage": ("Tile = one complex, area proportional to its mutations; teal = clears T>=10 and "
                 "is scored, grey = dropped by the metric"),
}


def build(view, threshold):
    """``[(subset label, tiles, note)]`` for one view, skipping what the repository cannot supply."""
    panels = []
    for label, paths, cluster_name in SUBSETS:
        counts = read_subset(paths)
        if counts is None:
            print(f"[{view}] {label}: input files absent; skipped", file=sys.stderr)
            continue
        if view in ("scale", "coverage"):
            panels.append((label, tiles_by_complex(counts, threshold if view == "coverage" else None),
                           None))
        elif view == "clusters":
            chain2rep = read_clusters(cluster_name)
            if chain2rep is None:
                why = ("the combined set was never clustered as one -- single- and multi-point "
                       "were clustered separately, and merging two runs is not the same partition"
                       if cluster_name is None else "cluster table absent")
                print(f"[clusters] {label}: {why}; skipped", file=sys.stderr)
                continue
            panels.append((label, tiles_by_family(counts, cluster_families(counts, chain2rep)),
                           f"{len(counts)} complexes"))
        elif view == "cath":
            families = cath_families(counts)
            if families is None:
                print("[cath] cath_families.json absent; skipped", file=sys.stderr)
                return []
            panels.append((label, tiles_by_family(counts, families), f"{len(counts)} complexes"))
    return panels


# -- layout --------------------------------------------------------------------------------------
def squarify(sizes, x, y, dx, dy):
    """Squarified treemap layout: ``[(x, y, dx, dy)]``, one per size, in the order given.

    Bruls, Huizing & van Wijk (2000). Sizes must be sorted descending and scaled to sum to
    ``dx * dy``. Iterative rather than recursive, since a subset can carry 300+ tiles.
    """
    sizes = [float(s) for s in sizes]
    rects = []

    def row(batch, x, y, dx, dy):
        covered = sum(batch)
        out = []
        if dx >= dy:
            width = covered / dy if dy else 0.0
            for s in batch:
                out.append((x, y, width, s / width if width else 0.0))
                y += s / width if width else 0.0
        else:
            height = covered / dx if dx else 0.0
            for s in batch:
                out.append((x, y, s / height if height else 0.0, height))
                x += s / height if height else 0.0
        return out

    def worst(batch, x, y, dx, dy):
        ratios = [max(w / h, h / w) for _, _, w, h in row(batch, x, y, dx, dy) if w > 0 and h > 0]
        return max(ratios) if ratios else float("inf")

    while sizes:
        if dx <= 0 or dy <= 0:
            rects.extend((x, y, 0.0, 0.0) for _ in sizes)
            break
        i = 1
        while i < len(sizes) and worst(sizes[:i], x, y, dx, dy) >= worst(sizes[:i + 1], x, y, dx, dy):
            i += 1
        batch, sizes = sizes[:i], sizes[i:]
        rects.extend(row(batch, x, y, dx, dy))
        covered = sum(batch)
        if dx >= dy:
            width = covered / dy if dy else 0.0
            x, dx = x + width, dx - width
        else:
            height = covered / dx if dx else 0.0
            y, dy = y + height, dy - height
    return rects


def label_colour(colour):
    """Black or white, whichever the tile can actually be read against."""
    from matplotlib.colors import to_rgb

    r, g, b = to_rgb(colour)
    return "black" if (0.299 * r + 0.587 * g + 0.114 * b) > 0.6 else "white"


_WIDTHS = {}


def label_width(text, fig):
    """Width of ``text`` in points per point of font size, measured by the renderer.

    A per-character average is wrong -- in DejaVu Sans a digit is 0.65 em while ``M`` is 0.86 and
    ``W`` is 1.01 -- and ``TextPath`` is wrong too, returning the ink box rather than the advance
    width. Since a width-limited label is sized to fill its tile exactly, either error puts it in
    the neighbouring tile.
    """
    if text not in _WIDTHS:
        probe = fig.text(0, 0, text, fontsize=10)
        extent = probe.get_window_extent(fig.canvas.get_renderer())
        probe.remove()
        _WIDTHS[text] = extent.width * 72 / fig.dpi / 10
    return _WIDTHS[text]


# -- summary -------------------------------------------------------------------------------------
def stats(tiles, threshold):
    counts = [t["mutations"] for t in tiles]
    total = sum(counts)
    scored = [t for t in tiles if t["mutations"] >= threshold]
    return {
        "groups": len(tiles),
        "mutations": total,
        "top1_share": counts[0] / total if total else 0.0,
        "top3_share": sum(counts[:3]) / total if total else 0.0,
        "singletons": sum(1 for n in counts if n == 1),
        "scored_groups": len(scored),
        "scored_mutations": sum(t["mutations"] for t in scored),
        "members": sum(t["members"] for t in tiles),
    }


def write_report(built, out: Path, threshold):
    lines = []
    w = lines.append
    w("Composition of the benchmark subsets, under four grouping units.")
    w("")
    w("Every view holds the mutations fixed and changes only what a group is. Going down the")
    w("list is the leakage ladder: the stricter the split, the coarser the unit it must hold out")
    w("whole, and the fewer independent units the benchmark actually contains.")
    w("")

    for view in VIEWS:
        panels = built.get(view) or []
        if not panels:
            continue
        w(VIEW_TITLES[view])
        w("-" * 88)
        if view == "coverage":
            w(f"{'':<24}{'complexes':>11}{'scored':>9}{'mutations':>11}{'scored':>9}"
              f"{'largest':>9}")
        else:
            w(f"{'':<24}{'groups':>9}{'from':>9}{'mutations':>11}{'largest':>9}{'top 3':>8}"
              f"{'singletons':>12}")
        for label, tiles, _ in panels:
            s = stats(tiles, threshold)
            if view == "coverage":
                w(f"{label:<24}{s['groups']:>11}{s['scored_groups']:>9}{s['mutations']:>11}"
                  f"{s['scored_mutations']:>9}{s['top1_share'] * 100:>8.0f}%")
            else:
                w(f"{label:<24}{s['groups']:>9}{s['members']:>9}{s['mutations']:>11}"
                  f"{s['top1_share'] * 100:>8.0f}%{s['top3_share'] * 100:>7.0f}%"
                  f"{s['singletons']:>12}")
        w("")
        if view == "clusters":
            w("The 'from' column is the collapse: that many complexes become that many families.")
            w("A by-complex hold-out treats the left number as its supply of independent units;")
            w("the right number is how many there are once homology is taken seriously.")
            w("")
        if view == "cath":
            w("Finer than the sequence clustering rather than coarser, because single-linkage")
            w("merges aggressively through shared inhibitor chains while the both-partners rule")
            w("keeps distinct antigen/antibody pairings apart. Definition-sensitive, and the")
            w("strictest rung only in the sense that it controls a different axis.")
            w("")
        if view == "coverage":
            w(f"The metric averages over complexes with at least {threshold} mutations. It keeps")
            w("most of the mutations and a minority of the complexes -- and complexes are the")
            w("unit it averages over, so the effective sample size is the smaller number.")
            w("")

    path = out / "TREEMAPS.txt"
    path.write_text("\n".join(lines) + "\n")
    return path


def write_csv(built, out: Path):
    path = out / "treemap_groups.csv"
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["view", "subset", "group", "mutations", "complexes", "members", "scored"])
        for view in VIEWS:
            for label, tiles, _ in built.get(view) or []:
                for t in tiles:
                    wr.writerow([view, label, t["label"], t["mutations"], t["members"],
                                 " ".join(t["member_labels"]),
                                 "" if t["scored"] is None else int(t["scored"])])
    return path


# -- figures -------------------------------------------------------------------------------------
def write_plots(built, out: Path, threshold):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch, Rectangle
    except ImportError:
        print("[plots] matplotlib not installed; skipping", file=sys.stderr)
        return []

    written = []
    W = H = 100.0
    for view in VIEWS:
        panels = built.get(view) or []
        if not panels:
            continue
        cols = 2 if len(panels) > 1 else 1
        rows_n = (len(panels) + cols - 1) // cols
        fig, axes = plt.subplots(rows_n, cols, figsize=(8.4 * cols, 6.6 * rows_n), squeeze=False)
        flat = [ax for row in axes for ax in row]
        for ax in flat[len(panels):]:
            ax.axis("off")

        for ax, (label, tiles, _) in zip(flat, panels):
            s = stats(tiles, threshold)
            if view == "coverage":
                sub = (f"{s['scored_groups']} of {s['groups']} complexes scored, "
                       f"holding {s['scored_mutations']} of {s['mutations']} mutations")
            elif view == "scale":
                sub = (f"{s['groups']} complexes, {s['mutations']} mutations; "
                       f"largest holds {s['top1_share'] * 100:.0f}%")
            else:
                sub = (f"{s['members']} complexes to {s['groups']} families; "
                       f"largest holds {s['top1_share'] * 100:.0f}%")
            ax.set_xlim(0, W)
            ax.set_ylim(0, H)
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{label}\n{sub}", fontsize=10)

        if view == "coverage":
            handles = [Patch(facecolor=SCORED, edgecolor="white",
                             label=f"scored (T>={threshold})"),
                       Patch(facecolor=DROPPED, edgecolor="white", label="dropped by the metric")]
        else:
            handles = [Patch(facecolor=col, edgecolor="white", label=lab)
                       for _, lab, col in CLASSES]
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
                   fontsize=9)
        fig.suptitle(f"{VIEW_TITLES[view]}\n{VIEW_SUBTITLES[view]}", fontsize=12)
        # Settle the layout before drawing a tile, so what fits inside one is measured off the
        # final axes rather than estimated. An estimate overflows the narrow tiles, and a label
        # spilling into its neighbour is worse than no label.
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.canvas.draw()

        for ax, (_, tiles, _) in zip(flat, panels):
            box = ax.get_window_extent()
            px = box.width * 72 / fig.dpi / W
            py = box.height * 72 / fig.dpi / H
            total = sum(t["mutations"] for t in tiles)
            scaled = [t["mutations"] * W * H / total for t in tiles]
            for t, (x, y, dx, dy) in zip(tiles, squarify(scaled, 0.0, 0.0, W, H)):
                ax.add_patch(Rectangle((x, y), dx, dy, facecolor=t["colour"],
                                       edgecolor="white", linewidth=0.5))
                # 0.92 of the tile, not all of it: the probe measures at 10pt and these render at
                # 4-7pt, where hinting rounds advance widths non-linearly.
                size = min(7.0, 0.92 * dx * px / label_width(t["label"], fig))
                if size >= 4.2 and dy * py >= 2.4 * size:
                    ax.text(x + dx / 2, y + dy / 2, f"{t['label']}\n{t['mutations']}",
                            ha="center", va="center", fontsize=size,
                            color=label_colour(t["colour"]), linespacing=1.15)

        path = out / f"treemap_{view}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
    return written


def write_interactive(built, out: Path, threshold, fragment: bool = False):
    """One page per view, a button per subset.

    The static figure can label only the tiles big enough to hold text. The tail is where a
    reader's own complex usually is, so it has to be readable by pointing at it -- and for the
    family views the membership is the whole point, which no static tile has room for.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("[html] plotly not installed (pip install plotly); skipping", file=sys.stderr)
        return []

    written = []
    for view in VIEWS:
        panels = built.get(view) or []
        if not panels:
            continue
        traces, buttons = [], []
        for i, (label, tiles, _) in enumerate(panels):
            total = sum(t["mutations"] for t in tiles)
            custom = [[t["label"], t["mutations"], t["mutations"] / total * 100, t["members"],
                       ", ".join(t["member_labels"][:12])
                       + (" ..." if len(t["member_labels"]) > 12 else ""),
                       "scored" if t["scored"] else "dropped by the metric"]
                      for t in tiles]
            traces.append(go.Treemap(
                labels=[t["label"] for t in tiles],
                parents=[""] * len(tiles),
                values=[t["mutations"] for t in tiles],
                marker=dict(colors=[t["colour"] for t in tiles],
                            line=dict(color="white", width=1)),
                customdata=custom,
                texttemplate="%{customdata[0]}<br>%{customdata[1]}",
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "%{customdata[1]} mutations   ·   %{customdata[2]:.1f}% of this subset<br>"
                    + ("%{customdata[5]}<br>" if view == "coverage" else
                       "%{customdata[3]} complexes: %{customdata[4]}<br>")
                    + "<extra></extra>"
                ),
                visible=(i == 0),
                tiling=dict(packing="squarify"),
            ))
            buttons.append(dict(label=label, method="update",
                                args=[{"visible": [j == i for j in range(len(panels))]},
                                      {"title": {"text": title_for(view, label, tiles, threshold)}}]))

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=dict(text=title_for(view, panels[0][0], panels[0][1], threshold),
                       x=0.5, xanchor="center"),
            updatemenus=[dict(type="buttons", direction="right", buttons=buttons,
                              x=0.5, xanchor="center", y=1.09, yanchor="bottom",
                              showactive=True)],
            margin=dict(t=150, l=20, r=20, b=20), height=780,
        )
        path = out / f"treemap_{view}.html"
        fig.write_html(path, include_plotlyjs="cdn", full_html=not fragment)
        written.append(path)
    return written


def title_for(view, label, tiles, threshold):
    s = stats(tiles, threshold)
    if view == "coverage":
        detail = (f"{s['scored_groups']} of {s['groups']} complexes clear T>={threshold}, "
                  f"holding {s['scored_mutations']} of {s['mutations']} mutations")
    elif view == "scale":
        detail = (f"{s['groups']} complexes, {s['mutations']} mutations; the largest holds "
                  f"{s['top1_share'] * 100:.0f}% and {s['singletons']} carry a single mutation")
    else:
        detail = (f"{s['members']} complexes collapse into {s['groups']} families; the largest "
                  f"holds {s['top1_share'] * 100:.0f}%, the top three {s['top3_share'] * 100:.0f}%")
    return (f"<b>{VIEW_TITLES[view]} — {label}</b><br>"
            f"<span style='font-size:12px'>{detail}<br>{VIEW_SUBTITLES[view]}</span>")


# -- entry point ---------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--view", default="all", choices=("all",) + VIEWS,
                    help="which grouping unit to draw (default: all four)")
    ap.add_argument("--threshold", type=int, default=10,
                    help="mutations a complex needs for the per-structure metric (default: 10)")
    ap.add_argument("--out", type=Path, default=ROOT / "scripts_plots" / "treemaps",
                    help="output directory")
    ap.add_argument("--csv", action="store_true", help="write one row per group")
    ap.add_argument("--plots", action="store_true", help="write the static PNGs (matplotlib)")
    ap.add_argument("--html", action="store_true", help="write the hoverable pages (plotly)")
    ap.add_argument("--html-fragment", action="store_true",
                    help="emit each page as an embeddable div rather than a whole document")
    args = ap.parse_args()

    views = VIEWS if args.view == "all" else (args.view,)
    built = {v: build(v, args.threshold) for v in views}
    if not any(built.values()):
        print("nothing to draw: no view found its inputs", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    written = [write_report(built, args.out, args.threshold)]
    if args.csv:
        written.append(write_csv(built, args.out))
    if args.plots:
        written += write_plots(built, args.out, args.threshold)
    if args.html or args.html_fragment:
        written += write_interactive(built, args.out, args.threshold, args.html_fragment)
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
