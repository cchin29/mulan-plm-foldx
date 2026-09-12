#!/usr/bin/env python3
"""MuLAN ΔΔG scaling figure — performance (and cost) vs model scale.

Reads ddg_scaling_data.csv (next to this file) and renders, in the reference
"emergent-property vs parameters" style:
  • Top panel  — ΔΔG performance (Pearson r on S1102, 10-fold CV) vs encoder params.
  • Bottom panel — cost: peak RAM (GB) vs params, with embedding-gen runtime annotated.

Reproducible & extensible: to add a model, append a row to the CSV and re-run
    python plot_ddg_scaling.py
No code changes needed. Pending rows (blank pcc) render as a vertical placeholder
so planned runs (e.g. AIDO-16B) show up until their number lands.

Deps: matplotlib, pandas  (pip install matplotlib pandas)
Outputs: ddg_scaling.png and ddg_scaling.svg next to this script.
"""
import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- series selection -------------------------------------------------------
# One script renders three training protocols from three CSVs, chosen by the env var
# MULAN_PLOT_SERIES (default "50ep"):
#   50ep           — from-scratch suite, 50 epochs / patience 10, balanced splitter
#                    (ddg_scaling_data.csv)
#   300ep_balanced — embedding sweep, 300 epochs / patience 30, seed-42 BALANCED splitter
#                    (fork default, equal-size folds) (ddg_scaling_data_300ep_balanced.csv)
#   300ep          — paper-faithful embedding sweep, 300 epochs / patience 30, seed-42
#                    rng.integers splitter (ddg_scaling_data_300ep.csv)
# Each series has its own CSV, output basename, subtitle, and y-range; the plotting logic
# is identical. `python plot_ddg_scaling.py` renders 50ep;
# `MULAN_PLOT_SERIES=300ep_balanced python plot_ddg_scaling.py` renders the balanced 300ep;
# `MULAN_PLOT_SERIES=300ep python plot_ddg_scaling.py` renders the rng.integers 300ep.
SERIES = os.environ.get("MULAN_PLOT_SERIES", "50ep").lower()
SERIES_CFG = {
    "50ep": dict(
        csv="ddg_scaling_data.csv", out="ddg_scaling", ylim=(0.785, 0.885),
        subtitle="From-scratch training · 50 epochs, patience 10 "
                 "· S1102 balanced-splitter mutation-based 10-fold CV",
    ),
    "300ep_balanced": dict(
        csv="ddg_scaling_data_300ep_balanced.csv", out="ddg_scaling_300ep_balanced",
        ylim=(0.780, 0.905),   # floor lowered to seat ESM3 (0.7938, lowest of the suite) + its label
        subtitle="Balanced-splitter training · 300 epochs, patience 30 "
                 "· S1102 seed-42 balanced (equal-fold) 10-fold CV (1100 muts)",
    ),
    "300ep": dict(
        csv="ddg_scaling_data_300ep.csv", out="ddg_scaling_300ep", ylim=(0.775, 0.885),
        subtitle="Paper-faithful training · 300 epochs, patience 30 "
                 "· S1102 seed-42 rng.integers 10-fold CV (1100 muts)",
    ),
}[SERIES]
CSV = os.path.join(HERE, SERIES_CFG["csv"])

# ---- style (mimics the reference figure) -----------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 13,
    "figure.facecolor": "white",
    "axes.facecolor": "#f7f7f7",
    "axes.edgecolor": "#c9c9c9",
    "axes.linewidth": 1.0,
    "axes.grid": True,
    "grid.color": "#e2e2e2",
    "grid.linewidth": 0.9,
})

COLORS = {
    # Linux 40c CPU box (blue family) — unchanged from the original scheme.
    "seq_cv10_linux": "#16386f",   # dark navy — sequence-only CV (primary)
    "aug_cv10_linux": "#5b9bd5",   # medium blue — + Tier-1 augmentation
    # Apple M4 Pro / MPS (magenta family) — same seq/aug shade logic, new hue.
    "seq_cv10_mps": "#8E1B6B",     # dark magenta — sequence-only CV (primary)
    "aug_cv10_mps": "#E39BC8",     # light magenta — + Tier-1 augmentation
    # Linux GPU box (green family) — same seq/aug shade logic, 3rd hue. CUDA CV10
    # (e.g. ESM C 6B on the Precision-7920 RTX 24G box), distinct from the Linux
    # 40c *CPU* blues.
    "seq_cv10_gpu": "#1b6b3a",     # dark green — sequence-only CV (primary)
    "aug_cv10_gpu": "#74c48f",     # light green — + Tier-1 augmentation
    # FoldX-MLP add_scores arm (§19): a distinct physics-channel lever, NOT a compute
    # platform — so it gets ONE amber hue regardless of run_platform (all ran on MPS).
    "foldxmlp": "#C77C0E",         # amber — + FoldX decomposed-12-term MLP add_scores
    "reference": "#8a8a8a",  # grey dashed — paper target
    "pending": "#b0b0b0",    # grey — planned/not-yet-run
    "lineage": "#8f8f8f",    # neutral grey — family-lineage connector (may span platforms)
    # runtime-label greys — a muted echo of the seq/aug marker hue, keyed on where
    # the CV10 itself ran (run_platform): blue-grey = Linux CPU, magenta-grey = MPS,
    # green-grey = Linux GPU.
    "rt_linux": "#7c8aa6",   # blue-grey  — runtime for a Linux CPU CV10 run
    "rt_mps": "#a67c96",     # magenta-grey — runtime for an MPS CV10 run
    "rt_gpu": "#7ca68c",     # green-grey — runtime for a Linux GPU CV10 run
    # cost-panel RAM ceilings, one per box: blue = Linux 40c (62 GB), pale magenta
    # = Apple M4 Pro (48 GB) — echoing the two platform hue families.
    "ram_linux": "#3a6ea5",  # blue — Linux 40c box RAM ceiling
    "ram_mac": "#d98cbb",    # pale magenta — Apple M4 Pro RAM ceiling
}


def _plat(run_platform):
    """Normalize a row's run_platform to one of the three color families:
    mps (Apple M4/MPS) | gpu (Linux CUDA box) | linux (Linux 40c CPU, default)."""
    p = str(run_platform).strip().lower()
    if p == "mps":
        return "mps"
    if p in ("gpu", "cuda", "linux_gpu", "linux gpu"):
        return "gpu"
    return "linux"


def platform_color(series_key, run_platform):
    """Pick the seq/aug color for a row, keyed on WHERE ITS CV10 RAN
    (`run_platform`: mps | gpu | linux), not where embedding-gen was timed.
    Falls back to the linux (original) shade for anything unrecognized."""
    return COLORS[f"{series_key}_{_plat(run_platform)}"]


def runtime_text_color(run_platform):
    """Grey-toned color for a row's small runtime label, keyed on where its CV10
    ran (`run_platform`): blue-grey Linux CPU, magenta-grey MPS, green-grey Linux
    GPU — a muted echo of the seq/aug marker hue. Unknown -> Linux (blue-grey)."""
    return COLORS[f"rt_{_plat(run_platform)}"]


def load():
    df = pd.read_csv(CSV, comment="#")
    for c in ["params", "pcc", "pcc_std", "peak_ram_gb", "emb_gen_min",
              "emb_gen_min_mps", "dx", "dy", "emb_dim", "est_ram_gb",
              "rt_dx", "rt_dy"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def runtime_label(r):
    """Return the 'H:MM' runtime string for a row, or '' if not logged."""
    v = r.get("cv10_runtime_hhmm")
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    v = str(v).strip()
    return "" if v == "" or v.lower() == "nan" else v


def runtime_offset(r, default_dx):
    """(dx, dy, ha, va) for a row's small runtime annotation: honors a per-row
    rt_dx/rt_dy override (see CSV header doc), else sits at marker height due
    `default_dx` points to the side (both base and aug default to +11 = due EAST
    so a base/aug pair's timings share one side; a per-row rt_dx=-11 flips a whole
    pair due WEST). ha/va are picked from the offset's sign so the text anchors
    away from the point in whichever direction it was nudged."""
    dx = r.get("rt_dx")
    dy = r.get("rt_dy")
    dx = default_dx if pd.isna(dx) else dx
    dy = 0 if pd.isna(dy) else dy
    ha = "left" if dx > 0 else ("right" if dx < 0 else "center")
    va = "bottom" if dy > 0 else ("top" if dy < 0 else "center")
    return dx, dy, ha, va


def plat_short(p):
    """Abbreviate a platform tag for compact annotation, e.g.
    'Linux 40c CPU / 62G' -> 'Linux 40c'."""
    if p is None or (isinstance(p, float) and pd.isna(p)):
        return ""
    return " ".join(str(p).split()[:2])


def fmt_min(x):
    """1.97 -> '2', 69.7 -> '69.7', 37.0 -> '37'."""
    return f"{x:g}"


def canon_model(name):
    """Canonical model key for matching an aug row to its OWN base. Drops the
    ' +aug' marker, normalizes the literal '\\n' (used in some CSV names for label
    wrapping, e.g. 'SaProt-650M\\n(WT-3Di)') and real newlines to a space, and
    collapses whitespace. Matching on this identity — not on params — is required
    because several PLMs share a param count (Ankh-large and Ankh3-large are both
    1.15e9) and a base/aug pair can even straddle platforms (ESM2-3B base=Linux,
    +aug=MPS), so a params- or platform-keyed match anchors the lift to the wrong
    base. Base and aug names must canonicalize to the same string (see the CSV)."""
    s = str(name).replace("\\n", " ").replace("\n", " ")
    s = re.sub(r"\+\s*aug", "", s, flags=re.I)
    s = re.sub(r"\+\s*foldx(?:[-\s]?mlp)?", "", s, flags=re.I)
    return re.sub(r"\s+", " ", s).strip()


def plot():
    df = load()
    fig, (ax, axc) = plt.subplots(
        2, 1, figsize=(11, 9.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.25], "hspace": 0.08})

    xmin, xmax = 8e7, 2.4e10

    # ===================== TOP: performance =====================
    ref = df[df.series == "reference"]
    for _, r in ref.iterrows():
        ax.axhline(r.pcc, ls="--", lw=1.4, color=COLORS["reference"], zorder=1)
        ax.text(xmax, r.pcc + 0.001, f"  {r.model}: {r.pcc:.3f}",
                va="bottom", ha="right", fontsize=11, color="#6f6f6f", style="italic")

    # sequence-only CV points (no connecting line by default — these are distinct PLMs,
    # not a within-family scaling sweep, so a line would imply a trajectory that isn't
    # there). Same-family scaling lines are opt-in via the `lineage` column below.
    s = df[(df.series == "seq_cv10") & df.pcc.notna()].sort_values("params")
    seq_colors = [platform_color("seq_cv10", p) for p in s.get("run_platform", "")]
    ax.scatter(s.params, s.pcc, s=250, color=seq_colors,
               edgecolor="white", linewidth=1.5, zorder=3)

    # optional lineage lines: connect measured points that share a non-blank `lineage`
    # tag (reference-figure "within-family scaling"), e.g. ESM2 -> ESM C. Dormant until
    # such rows exist. `lineage_drawn` gates the matching legend entry below.
    # NB: a family can span BOTH platforms (e.g. Ankh-large is a Linux run while
    # Ankh3-large/Ankh3-xl are MPS runs), so the connector itself is drawn in a
    # neutral grey rather than either platform's hue — it shouldn't read as
    # "belonging" to just one box.
    lineage_drawn = False
    if "lineage" in df.columns:
        meas = df[df.pcc.notna() & df.lineage.notna() &
                  (df.lineage.astype(str).str.strip() != "")]
        for k, (tag, g) in enumerate(meas.groupby("lineage")):
            g = g.sort_values("params")
            if len(g) >= 2:
                ax.plot(g.params, g.pcc, ":", lw=2.4, alpha=0.9, zorder=2,
                        color=COLORS["lineage"], label=f"{tag} (lineage)")
                lineage_drawn = True
    for _, r in s.iterrows():
        if int(r.label) == 1:
            emb = r.get("emb_dim")
            # a literal "\n" in the CSV model cell becomes a real line break, so a long
            # name (e.g. "SaProt-1.3B\n(AFDB)") can wrap to compact the label's width.
            model_disp = str(r.model).replace("\\n", "\n")
            label_text = model_disp if pd.isna(emb) else f"{model_disp}\n{int(emb)}"
            ax.annotate(label_text, (r.params, r.pcc), textcoords="offset points",
                        xytext=(r.get("dx", 0) or 0, (r.get("dy", 0) or 0) + 10),
                        ha="center", fontsize=13, fontweight="bold",
                        color=platform_color("seq_cv10", r.get("run_platform")))

    # training wall-clock (hh:mm), small, colored by where the CV10 ran (blue-grey
    # Linux / magenta-grey MPS). Placed DUE EAST of the marker by default (marker
    # height, horizontal); the aug diamond defaults to the same side so a base/aug
    # pair's two timings sit together on one flank of the vertical connector. A
    # model that would collide with a neighbor there is flipped DUE WEST (together
    # with its aug) via a per-row rt_dx=-11 in the CSV.
    for _, r in s.iterrows():
        rt = runtime_label(r)
        if rt:
            dx, dy, ha, va = runtime_offset(r, default_dx=11)
            ax.annotate(rt, (r.params, r.pcc), textcoords="offset points",
                        xytext=(dx, dy), ha=ha, va=va, fontsize=8.5,
                        color=runtime_text_color(r.get("run_platform")), zorder=5)

    # augmentation: connect each augmented point to its OWN base (same PLM), matched
    # by MODEL IDENTITY (canon_model) — shows the per-model aug lift, not a spurious
    # cross-model line. Matching by params (the old approach) mis-anchored the lift
    # when two PLMs share a param count (e.g. Ankh3-large +aug drew from Ankh-large's
    # 0.832 instead of its own base 0.823). Falls back to a params match ONLY when no
    # base name matches, and warns — so a naming drift is loud, not silently wrong.
    # Each connector/point is colored by that row's OWN run_platform (mps vs linux).
    aug = df[(df.series == "aug_cv10") & df.pcc.notna()]
    base_by_key = {canon_model(br.model): float(br.pcc) for _, br in s.iterrows()}
    base_pts = list(zip(s.params.tolist(), s.pcc.tolist()))
    for _, r in aug.iterrows():
        base_pcc = base_by_key.get(canon_model(r.model))
        if base_pcc is None:
            cand = [bp for (pp, bp) in base_pts if np.isclose(pp, r.params, rtol=1e-3)]
            base_pcc = cand[0] if cand else None
            if base_pcc is not None:
                print(f"WARN: aug row {r.model!r} matched no base by name "
                      f"(canon={canon_model(r.model)!r}); fell back to a params match "
                      f"-> base pcc {base_pcc} (this may be the wrong base — fix the "
                      f"CSV name so it canonicalizes to its base)", file=sys.stderr)
            else:
                print(f"WARN: aug row {r.model!r} has no base point; connector skipped",
                      file=sys.stderr)
        if base_pcc is not None:
            ax.plot([r.params, r.params], [base_pcc, r.pcc], "-",
                    color=platform_color("aug_cv10", r.get("run_platform")),
                    lw=1.6, alpha=0.9, zorder=2)
    aug_colors = [platform_color("aug_cv10", p) for p in aug.get("run_platform", "")]
    ax.scatter(aug.params, aug.pcc, s=95, color=aug_colors, marker="D",
               edgecolor="white", linewidth=1.3, zorder=4)

    # training wall-clock (hh:mm) for each aug diamond, colored by run_platform,
    # placed DUE EAST by default -- the SAME side as its base point so the pair's
    # two timings sit together on one flank of the connector (flipped west with the
    # base via a per-row rt_dx=-11 for models that would otherwise collide).
    for _, r in aug.iterrows():
        rt = runtime_label(r)
        if rt:
            dx, dy, ha, va = runtime_offset(r, default_dx=11)
            ax.annotate(rt, (r.params, r.pcc), textcoords="offset points",
                        xytext=(dx, dy), ha=ha, va=va, fontsize=8.5,
                        color=runtime_text_color(r.get("run_platform")), zorder=5)

    # FoldX-MLP add_scores arm (§19): a 3rd lever plotted at each PLM's OWN params-x,
    # matched to its base by canon_model (foldxmlp names end ' +FoldX', which canon
    # strips like ' +aug'). Vertical connector base -> foldxmlp shows the lift (amber),
    # triangle marker on top. One amber hue for all (physics channel, not a platform).
    fx = df[(df.series == "foldxmlp_cv10") & df.pcc.notna()]
    for _, r in fx.iterrows():
        base_pcc = base_by_key.get(canon_model(r.model))
        if base_pcc is None:
            print(f"WARN: foldxmlp row {r.model!r} (canon={canon_model(r.model)!r}) "
                  f"has no base point; connector skipped", file=sys.stderr)
            continue
        ax.plot([r.params, r.params], [base_pcc, r.pcc], "-",
                color=COLORS["foldxmlp"], lw=1.6, alpha=0.75, zorder=2)
    ax.scatter(fx.params, fx.pcc, s=115, color=COLORS["foldxmlp"], marker="^",
               edgecolor="white", linewidth=1.3, zorder=4)

    # pending placeholders (blank pcc) -> vertical guide at params. Label honors
    # dx/dy so several near-coincident pending guides don't overprint each other.
    for _, r in df[df.status == "pending"].iterrows():
        ax.axvline(r.params, ls=":", lw=1.6, color=COLORS["pending"], zorder=1)
        # a pending row may already have a measured MPS embedding-gen runtime even
        # though its PCC is still running — surface it on a 3rd line so the platform
        # timing is visible before the performance point lands.
        mps = "" if pd.isna(r.get("emb_gen_min_mps")) else \
            f"\ngen {fmt_min(r.emb_gen_min_mps)} min (M4/MPS)"
        ax.annotate(f"{r.model}\n(pending){mps}", (r.params, 0.792),
                    textcoords="offset points",
                    xytext=(r.get("dx", 0) or 0, (r.get("dy", 0) or 0)),
                    ha="center", va="bottom", fontsize=10.5, color="#8a8a8a")

    ax.set_ylabel("ΔΔG performance  (Pearson r, S1102 10-fold CV)", fontsize=13.5)
    ax.set_ylim(*SERIES_CFG["ylim"])
    ax.set_xscale("log")
    ax.set_xlim(xmin, xmax)

    # legend: shape encodes the ARM (circle=seq base, diamond=+aug, triangle=+FoldX-MLP);
    # marker COLOR encodes the run platform (blue=Linux CPU, magenta=MPS, green=Linux GPU),
    # keyed once by the seq circles + the caption. So the +aug / +FoldX overlays get ONE row
    # each (neutral / amber swatch), while seq keeps its per-platform rows (gated on presence).
    rp = df.get("run_platform", pd.Series(dtype=object)).astype(str).str.lower()
    gpu_names = ("gpu", "cuda", "linux_gpu", "linux gpu")
    mps_seq_present = bool(((df.series == "seq_cv10") & df.pcc.notna() & (rp == "mps")).any())
    gpu_seq_present = bool(((df.series == "seq_cv10") & df.pcc.notna() & rp.isin(gpu_names)).any())

    legend = [
        Line2D([0], [0], color=COLORS["seq_cv10_linux"], lw=0, marker="o",
               markeredgecolor="white", markersize=11,
               label="Sequence-only CV10 (Linux CPU)"),
    ]
    if mps_seq_present:
        legend.append(Line2D([0], [0], color=COLORS["seq_cv10_mps"], lw=0, marker="o",
                              markeredgecolor="white", markersize=11,
                              label="Sequence-only CV10 (MPS, M4 Pro)"))
    if gpu_seq_present:
        legend.append(Line2D([0], [0], color=COLORS["seq_cv10_gpu"], lw=0, marker="o",
                              markeredgecolor="white", markersize=11,
                              label="Sequence-only CV10 (Linux GPU)"))
    # +aug: ONE row (shape = arm). Each diamond's color on the plot keys its run platform
    # via the seq circles + caption, so a neutral swatch avoids 3 redundant per-platform rows.
    legend.append(Line2D([0], [0], color="#7d7d7d", lw=1.6, marker="D",
                          markeredgecolor="white", markersize=8,
                          label="+ Tier-1 aug CV10 (lift from base)"))
    if bool(((df.series == "foldxmlp_cv10") & df.pcc.notna()).any()):
        legend.append(Line2D([0], [0], color=COLORS["foldxmlp"], lw=1.6, marker="^",
                              markeredgecolor="white", markersize=9,
                              label="+ FoldX-MLP add_scores CV10 (lift from base)"))
    if lineage_drawn:
        legend.append(Line2D([0], [0], color=COLORS["lineage"], lw=2.4, ls=":",
                              label="Same-family scaling (shared lineage)"))
    # (The "MuLAN paper target" dashed line and any "pending" guide are still drawn +
    # annotated in-panel; they're dropped from the legend box to keep it compact.)
    ax.legend(handles=legend, loc="lower left", fontsize=9, frameon=True,
              facecolor="white", edgecolor="#d9d9d9", framealpha=0.95,
              borderpad=0.4, labelspacing=0.3, handletextpad=0.5, markerscale=0.85,
              bbox_to_anchor=(0.02, 0.02))

    # ===================== BOTTOM: cost =====================
    c = df[df.peak_ram_gb.notna()].sort_values("params")
    axc.plot(c.params, c.peak_ram_gb, "-o", color="#3f6b3f", lw=2.0, ms=9,
             markeredgecolor="white", markeredgewidth=1.3, zorder=3)
    for i, (_, r) in enumerate(c.iterrows()):
        # runtime lines carry BOTH the wall time and the platform that produced it,
        # so a viewer can tell the CPU box (`emb_gen_min` on `platform`) from the
        # Apple MPS run (`emb_gen_min_mps`).
        lines = [f"{r.peak_ram_gb:.0f} GB"]
        if pd.notna(r.emb_gen_min):
            tag = plat_short(r.get("platform")) or "?"
            lines.append(f"gen {fmt_min(r.emb_gen_min)} min ({tag})")
        if pd.notna(r.get("emb_gen_min_mps")):
            lines.append(f"gen {fmt_min(r.emb_gen_min_mps)} min (M4/MPS)")
        up = (i % 2 == 1)  # stagger to separate the two near-coincident ~5 GB points
        axc.annotate("\n".join(lines), (r.params, r.peak_ram_gb),
                     textcoords="offset points", xytext=(0, 12 if up else -24),
                     ha="center", va="bottom" if up else "top",
                     fontsize=10, color="#3f6b3f")
    # pending guide continues into the cost panel; each row annotates its OWN
    # estimated peak RAM (est_ram_gb), placed at that height.
    for _, r in df[df.status == "pending"].iterrows():
        axc.axvline(r.params, ls=":", lw=1.6, color=COLORS["pending"], zorder=1)
        est = r.get("est_ram_gb")
        if pd.notna(est):
            note = " (bf16)" if r.params >= 1e10 else ""
            axc.annotate(f"~{est:.0f} GB{note}\nest, pending", (r.params, est),
                         textcoords="offset points", xytext=(0, 6), ha="center",
                         fontsize=9.5, color="#8a8a8a")
    # per-box RAM ceilings: Linux 40c (62 GB, blue) and Apple M4 Pro (48 GB, pale
    # magenta). Labels flank their own line (62 above, 48 below) to keep clear.
    axc.axhline(62, ls="--", lw=1.1, color=COLORS["ram_linux"], alpha=0.85, zorder=1)
    axc.text(xmax, 62 * 1.04, "  linux box RAM (62 GB)", va="bottom", ha="right",
             fontsize=9.5, color=COLORS["ram_linux"])
    axc.axhline(48, ls="--", lw=1.1, color=COLORS["ram_mac"], alpha=0.9, zorder=1)
    # Mac label anchored at the LEFT (empty half of the cost panel) so it clears
    # the ESM C 6B "gen 69.7 min" annotation crowding the top-right.
    axc.text(xmin, 48 * 1.015, "Mac M4 Pro RAM (48 GB)  ", va="bottom", ha="left",
             fontsize=9.5, color=COLORS["ram_mac"])
    axc.set_yscale("log")
    axc.set_ylim(3, 90)
    axc.set_yticks([5, 10, 30, 60])
    axc.set_yticklabels(["5", "10", "30", "60"])
    # log-scale minor ticks default to LogFormatterSciNotation (e.g. "3 x 10^0"),
    # which clashes with the plain major labels above; RAM values here are all
    # small single/double-digit GB, so scientific notation is never needed —
    # blank out the minor tick labels entirely.
    axc.yaxis.set_minor_formatter(NullFormatter())
    axc.set_ylabel("Peak RAM (GB)", fontsize=12.5)
    axc.set_xlabel("Model Scale (Parameters)", fontsize=14)
    axc.set_xscale("log")
    axc.set_xlim(xmin, xmax)

    fig.suptitle("MuLAN ΔΔG: performance and cost vs PLM scale",
                 fontsize=15.5, fontweight="bold", x=0.5, y=0.955)
    fig.text(0.5, 0.918, SERIES_CFG["subtitle"], ha="center", fontsize=11,
             style="italic", color="#555555")
    fig.text(0.5, 0.051,
             "Point/marker color = where CV10 was TRAINED (run_platform), not where "
             "embeddings were generated — magenta: Apple M4 Pro (MPS)  /  "
             "blue: Linux 40c CPU box  /  green: Linux GPU box (RTX 24G).",
             ha="center", fontsize=9.5, color="#777777")
    fig.text(0.5, 0.028,
             "Runtime platforms — Linux 40c: 40-thread Linux CPU box, 62 GB;  "
             "M4/MPS: Apple M4 Pro, 48 GB, MPS;  Linux GPU: Precision-7920, RTX 24G. "
             "Each runtime is tagged with the box that produced it.",
             ha="center", fontsize=9.5, color="#777777")
    fig.text(0.5, 0.005,
             "Same 70/15/15→10-fold protocol per PLM; ± std is cross-fold and overlaps — "
             "ranking is by paired tests (see RESULTS.md). Params = encoder size.",
             ha="center", fontsize=9.5, color="#777777")

    out_png = os.path.join(HERE, SERIES_CFG["out"] + ".png")
    out_svg = os.path.join(HERE, SERIES_CFG["out"] + ".svg")
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    print("wrote", out_png)
    print("wrote", out_svg)


if __name__ == "__main__":
    plot()
