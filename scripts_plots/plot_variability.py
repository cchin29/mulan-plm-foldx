#!/usr/bin/env python3
"""MuLAN ΔΔG — JMP-style variability chart, nested PLM -> aug -> benchmark.

Response = per-fold Pearson r, grouped hierarchically:
  outer  = PLM (model)
  middle = augmentation  (base  |  +Tier-1 aug)
  inner  = benchmark dataset  (S1102 / S1131 / S4169 / S2003)

Each cell shows all fold values as jittered points with an overlaid box plot
(q25 / median / q75 + 1.5·IQR whiskers; grey line = median). Means are drawn at
two levels: a JMP-style mean diamond per benchmark box (horizontal diagonal at
the mean; top/bottom vertices = 95% CI of the mean), and a medium-grey dotted
line across each aug subgroup = its pooled mean (complete cells only). The S1102
box mean is also spelled out as a dark-grey number just left of its box. Color
encodes the benchmark. No global mean. Two dashed-grey horizontals span the panel
as MuLAN-paper targets (Table 1, mutation-based 10-fold CV, S1102): MuLAN-Ankh
0.868 and MuLAN-ESM2 0.854 — same device as the ddg_scaling scatterplot.

Coverage (this snapshot, 2026-07-08): Tier-1 aug benchmarks are landing —
ProstT5 +aug S1131 AND S4169 are now complete (S4169 = 0.766 ± 0.062), its S2003
+aug is partial (faded, 7/10). SaProt +aug S1131 is complete; its S4169 +aug is
partial (faded, 1/10) and S2003 +aug pending. ESM C 6B base benchmarks landed
from the Linux snapshot (S1131, S4169 & now S2003 all complete 10/10; S2003 =
0.852 ± 0.038); its +aug benchmarks aren't run yet -> those cells render as
"running". AIDO.Protein-16B is an S1102-only base+aug block (base 0.828 ± 0.046,
+aug 0.836 ± 0.057, 10/10 each; no benchmark sweeps).

Input: benchmarks_folds.csv (from aggregate_folds.py). Shares theme/palette with
plot_common.py. Re-run aggregate_folds.py then this script as folds land.

Deps: matplotlib, pandas, numpy.  Outputs: ddg_variability.{png,svg}.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon

import plot_common as cm

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- series selection -------------------------------------------------------
# One script renders three training protocols, chosen by env var MULAN_PLOT_SERIES
# (default "50ep"): the 50ep from-scratch suite (full benchmark sweep, balanced
# splitter), the 300ep BALANCED-splitter embedding sweep, and the 300ep paper-faithful
# rng.integers embedding sweep (both S1102-only). Each has its own per-fold CSV, model
# order, output basename, and subtitle; layout logic is shared. Parallels the three
# sets in plot_ddg_scaling.py.
#   python plot_variability.py                                  -> 50ep
#   MULAN_PLOT_SERIES=300ep_balanced python plot_variability.py -> 300ep balanced
#   MULAN_PLOT_SERIES=300ep python plot_variability.py          -> 300ep rng.integers
_SERIES = os.environ.get("MULAN_PLOT_SERIES", "50ep").lower()
# outer PLM order (S1102 rank); ESM C 600M is placed immediately right of ESM C 6B
# (ESM C family grouped), not by S1102 rank — the rest stay in S1102-mean order.
_ORDER_50EP = ["ESM C 6B", "ESM C 600M", "SaProt (WT-3Di)", "SaProt-1.3B (AFDB)",
               "Ankh3-xl", "Ankh3-large", "Ankh-large", "AIDO.Protein-16B",
               "ESM2-3B", "ProstT5"]
# 300ep rng.integers sweep so far: base done for these 3 (S1102-mean order); saprot/esm2
# base and all +aug still queued — append to the list as each lands.
_ORDER_300EP = ["ESM C 6B", "Ankh-large", "ESM C 600M", "ProstT5"]
# 300ep BALANCED sweep: base+aug now COMPLETE (10/10) for all 6 (S1102-mean order). ESM C 6B
# leads (base 0.881, aug 0.880 -- GPU run, from results_esmc6b_gpu_20260714). ESM C
# 600M slots by rank between Ankh and ESM2 (base 0.8336, aug 0.8309 -- a slight aug drop,
# the esmc600m "aug no lift"). Ankh3-large base is now COMPLETE (10/10, 0.835, landed
# 2026-07-14) but is base-only in this series (no balanced aug arm), so it stays out of this
# paired base+aug figure and off this list (dropped via the by-dict MODEL_ORDER filter below).
# ESM3-open (1.4B) base LANDED 2026-07-14 (10/10, 0.7938) -- the lowest-ranked, appended last; its
# +aug arm hasn't run (queue typo no-op'd it), so it's in BASE_S1102_ONLY -> single base cell, no
# misleading +aug placeholder, until the aug arm lands.
_ORDER_300EP_BAL = ["ESM C 6B", "SaProt (WT-3Di)", "Ankh-large", "ESM C 600M", "ESM2-3B", "ProstT5", "ESM3-open (1.4B)"]
_SERIES_CFG = {
    "50ep": dict(csv="benchmarks_folds.csv", out="ddg_variability", order=_ORDER_50EP,
                 subtitle="From-scratch training · 50 epochs, patience 10 · "
                          "S1102 balanced-splitter mutation-based 10-fold CV"),
    "300ep_balanced": dict(csv="benchmarks_folds_300ep_balanced.csv",
                           out="ddg_variability_300ep_balanced", order=_ORDER_300EP_BAL,
                           subtitle="Balanced-splitter training · 300 epochs, patience 30 · "
                                    "S1102 seed-42 balanced (equal-fold) 10-fold CV (1100 muts)"),
    "300ep": dict(csv="benchmarks_folds_300ep.csv", out="ddg_variability_300ep",
                  order=_ORDER_300EP,
                  subtitle="Paper-faithful training · 300 epochs, patience 30 · "
                           "S1102 seed-42 rng.integers 10-fold CV (1100 muts)"),
}[_SERIES]
# 300ep series (either splitter) are S1102-only; used for layout / title / caption.
_IS_300EP = _SERIES in ("300ep", "300ep_balanced")
CSV = os.path.join(HERE, _SERIES_CFG["csv"])
MODEL_ORDER = _SERIES_CFG["order"]
MODEL_ABBR = {"ESM C 6B": "ESM C 6B", "ESM C 600M": "ESM C 600M",
              "SaProt (WT-3Di)": "SaProt", "SaProt-1.3B (AFDB)": "SaProt-1.3B",
              "Ankh-large": "Ankh", "Ankh3-xl": "Ankh3-XL", "Ankh3-large": "Ankh3-L",
              "AIDO.Protein-16B": "AIDO", "ESM2-3B": "ESM2-3B", "ProstT5": "ProstT5",
              "ESM3-open (1.4B)": "ESM3"}
# reserved inner slots per aug subgroup (keeps model blocks aligned). base/aug =
# all four benchmarks, EXCEPT models in S1102_ONLY (only that cell is reserved, so
# their block is narrow instead of padded with "running" placeholders).
SUBGROUPS = [("aug", ["S1102", "S1131", "S4169", "S2003"]),
             ("base", ["S1102", "S1131", "S4169", "S2003"])]
# PLMs we only ran on S1102 (no benchmark sweeps planned) — drop their non-1102 cells.
# ESM C 600M has S1102 base+aug (both 50ep/patience10), no benchmark sweeps. In the
# 300ep series NO benchmark sweeps were run, so every model is S1102-only.
S1102_ONLY = (set(MODEL_ORDER) if _IS_300EP
              else {"ESM C 600M", "Ankh3-xl", "Ankh3-large", "SaProt-1.3B (AFDB)",
                    "AIDO.Protein-16B"})
# PLMs run on S1102 BASE only — no +aug either. Get a single base-S1102 block (no
# reserved aug cell), so they never show a misleading "running" +aug placeholder.
BASE_S1102_ONLY = {"ESM3-open (1.4B)"}   # esm3-open: base-only until its +aug arm re-runs (queue typo no-op'd it)
AUG_LABEL = {"base": "base", "aug": "+aug"}
DS_ABBR = {"S1102": "1102", "S1131": "1131", "S4169": "4169", "S2003": "2003"}


def subgroups_for(m):
    """Inner dataset slots for model m — trimmed to S1102 for S1102-only PLMs,
    and to a single base cell for base-S1102-only PLMs (no +aug)."""
    if m in BASE_S1102_ONLY:
        return [("base", ["S1102"])]
    if m in S1102_ONLY:
        return [(aug, ["S1102"]) for aug, _ in SUBGROUPS]
    return SUBGROUPS

# outer-label metadata: encoder params + per-residue embedding dim (RESULTS §7 /
# ddg_scaling_data.csv). Shown as a second line under each model name.
MODEL_META = {
    "ESM C 6B": ("6.0B", 2560), "ESM C 600M": ("600M", 1152),
    "SaProt (WT-3Di)": ("650M", 1280),
    "SaProt-1.3B (AFDB)": ("1.3B", 1280),
    "Ankh-large": ("1.15B", 1536), "Ankh3-xl": ("3.48B", 2560),
    "Ankh3-large": ("1.15B", 1536), "AIDO.Protein-16B": ("16B", 2304),
    "ESM2-3B": ("2.8B", 2560), "ProstT5": ("1.21B", 1024),
    "ESM3-open (1.4B)": ("1.4B", 1536),
}
EXPECT = 10
# t_{0.975, df} critical values for the mean's 95% CI (df = n-1), n up to 10 folds
# — used for the JMP-style mean diamonds (avoids a scipy dependency).
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
       6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}

SLOT = 1.0        # x width per inner cell
AUG_GAP = 0.55    # gap between base and +aug subgroups within a model
MODEL_GAP = 1.5   # gap between models
BOX_W = 0.6       # box plot width
JIT = 0.15        # half-spread of jittered fold points


def layout():
    """centers[(model,aug,ds)] -> x; sub_span[(model,aug)] -> (x0,x1);
    model_span[model] -> (x0,x1). Reserves a slot for every template cell."""
    centers, sub_span, model_span = {}, {}, {}
    x = 0.0
    for m in MODEL_ORDER:
        m0 = x
        subs = subgroups_for(m)
        for si, (aug, dsets) in enumerate(subs):
            s0 = x
            for ds in dsets:
                centers[(m, aug, ds)] = x + SLOT / 2
                x += SLOT
            sub_span[(m, aug)] = (s0, x)
            if si < len(subs) - 1:
                x += AUG_GAP
        model_span[m] = (m0, x)
        x += MODEL_GAP
    return centers, sub_span, model_span


def plot():
    cm.apply_theme()
    df = pd.read_csv(CSV)
    # Guard: MODEL_ORDER is hand-maintained separately from aggregate_folds.py's model
    # map, so a model can land in the CSV yet be missing from this series' _ORDER_* list
    # — in which case it is silently dropped from the figure (no cell, no warning). Flag
    # that loudly so a newly-aggregated model can't vanish from a paper figure unnoticed.
    dropped = sorted(set(df["model"].unique()) - set(MODEL_ORDER))
    if dropped:
        print(f"WARN: {len(dropped)} model(s) in {os.path.basename(CSV)} are NOT in this "
              f"series' MODEL_ORDER and will be OMITTED from the figure: {dropped}. Add "
              f"them to _ORDER_* (+ MODEL_ABBR / MODEL_META) to plot them.", file=sys.stderr)
    # Build cells only from in-order models: a model dropped above (e.g. a mid-flight
    # partial like Ankh3-large that isn't in this series' _ORDER_*) must not leak into the
    # y-range, subgroup means, or the partial-note (its abbr isn't even in MODEL_ABBR).
    by = {(m, a, d): g.sort_values("fold")["pcc"].to_numpy()
          for (m, a, d), g in df[df["model"].isin(MODEL_ORDER)].groupby(["model", "aug", "dataset"])}

    centers, sub_span, model_span = layout()
    xmax = list(centers.values())[-1]

    # width scales with model count so cells stay legible as PLMs are added.
    fig, ax = plt.subplots(figsize=(3.3 * len(MODEL_ORDER) + 0.5, 8.2))

    # ---- cells: box plots + jittered fold points, colored by benchmark ----
    for m in MODEL_ORDER:
        for aug, dsets in subgroups_for(m):
            for ds in dsets:
                cx = centers[(m, aug, ds)]
                v = by.get((m, aug, ds))
                col = cm.dataset_color(ds)
                if v is None or len(v) == 0:
                    ax.text(cx, 0.80, "running", rotation=90, ha="center", va="center",
                            fontsize=8, style="italic", color="#b3b3b3", zorder=2)
                    continue
                complete = len(v) >= EXPECT
                a_box = 0.18 if complete else 0.08
                a_pt = 0.85 if complete else 0.35
                bp = ax.boxplot([v], positions=[cx], widths=BOX_W, patch_artist=True,
                                showfliers=False, whis=1.5, zorder=3,
                                medianprops=dict(color="#7d7d7d", lw=1.8),
                                boxprops=dict(facecolor=col, alpha=a_box,
                                              edgecolor=col, lw=1.4),
                                whiskerprops=dict(color=col, lw=1.3),
                                capprops=dict(color=col, lw=1.3))
                jx = cx + np.linspace(-JIT, JIT, len(v))
                ax.scatter(jx, v, s=26, color=col, alpha=a_pt, edgecolor="white",
                           linewidth=0.4, zorder=4)
                # JMP-style mean diamond: horizontal diagonal at the benchmark mean
                # (spanning the box width), top/bottom vertices at the 95% CI of the
                # mean. Faded for partial cells to match the box/point treatment.
                mv = float(np.mean(v))
                ea = 0.9 if complete else 0.45           # edge/line alpha
                x0d, x1d = cx - BOX_W / 2, cx + BOX_W / 2
                if len(v) >= 2:
                    se = float(np.std(v, ddof=1)) / np.sqrt(len(v))
                    ci = T95.get(len(v) - 1, 2.262) * se
                    ax.add_patch(Polygon(
                        [(cx, mv + ci), (x1d, mv), (cx, mv - ci), (x0d, mv)],
                        closed=True, edgecolor=(0.55, 0.55, 0.55, ea),
                        facecolor=(0.55, 0.55, 0.55, 0.08 if complete else 0.04),
                        lw=0.8, zorder=5))
                # mean line = the diamond's horizontal diagonal (also covers n=1,
                # where the CI is undefined and no diamond is drawn).
                ax.plot([x0d, x1d], [mv, mv], "-", color="#8c8c8c", lw=0.8,
                        alpha=ea, zorder=6)
                if not complete:
                    ax.text(cx, float(np.max(v)) + 0.006, f"{len(v)}/{EXPECT}",
                            ha="center", va="bottom", fontsize=7.5, style="italic",
                            color="#999999", zorder=5)
                # dark-grey per-box mean label for the primary S1102 benchmark,
                # placed just left of the box (its own cell mean, distinct from the
                # pooled subgroup-mean line).
                if ds == "S1102":
                    ax.text(cx - BOX_W / 2 - 0.06, float(np.mean(v)), f"{np.mean(v):.3f}",
                            ha="right", va="center", fontsize=8, color="#333333",
                            zorder=7)

    # ---- reported group means: base & +aug subgroup means (over COMPLETE cells) ----
    def pooled(keys):
        vals = [by[k] for k in keys if k in by and len(by[k]) >= EXPECT]
        return float(np.concatenate(vals).mean()) if vals else None

    for m in MODEL_ORDER:
        for aug, dsets in subgroups_for(m):
            # span the mean line only over the COMPLETE cells that contribute to
            # it — not the full reserved block (which may include partial/"running"
            # cells the line would misleadingly cross).
            complete = [ds for ds in dsets
                        if (m, aug, ds) in by and len(by[(m, aug, ds)]) >= EXPECT]
            mean = pooled([(m, aug, ds) for ds in dsets])
            if mean is None or not complete:
                continue
            cxs = [centers[(m, aug, ds)] for ds in complete]
            xa, xb = min(cxs) - BOX_W / 2, max(cxs) + BOX_W / 2
            ax.plot([xa, xb], [mean, mean], ":", color="#8a8a8a", lw=2.2, zorder=6)
            ax.text(xb + 0.06, mean, f"{mean:.3f}", va="center", ha="left",
                    fontsize=8.5, color="#8a8a8a", zorder=7)

    # ---- MuLAN paper reference lines (Table 1, mutation-based split, S1102) ----
    # Same device as the ddg_scaling scatterplot: dashed-grey horizontals at the
    # published MuLAN S1102 PCC as a target, drawn across the whole panel. Paper
    # Table 1 reports both PLM backbones on S1102 (mutation-based 10-fold CV):
    # MuLAN-Ankh-large 0.868, MuLAN-ESM2-3B 0.854. Labels stack (Ankh above its
    # line, ESM2 below) so the two close values don't collide at the right edge.
    for name, pcc, va, off in [("MuLAN-Ankh", 0.868, "bottom", 0.0015),
                               ("MuLAN-ESM2", 0.854, "top", -0.0015)]:
        ax.axhline(pcc, ls="--", lw=1.3, color="#8a8a8a", alpha=0.9, zorder=1)
        ax.text(0.997, pcc + off, f"{name} S1102 (paper) {pcc:.3f}",
                transform=ax.get_yaxis_transform(), va=va, ha="right",
                fontsize=9, style="italic", color="#6f6f6f", zorder=2)

    ax.set_ylabel("Per-fold Pearson r  (10-fold CV)", fontsize=12.5)
    # data-driven y-range with padding so no fold point clips (top ESM C 6B fold
    # is ~0.932); recomputes as new folds land.
    allv = np.concatenate([v for v in by.values() if len(v) > 0])
    ax.set_ylim(float(allv.min()) - 0.013, float(allv.max()) + 0.013)
    ax.set_xlim(-0.7, xmax + 0.7)

    # ---- nested x axis: inner=dataset code, mid=aug, outer=model ----
    inner_ticks, inner_labs = [], []
    for m in MODEL_ORDER:
        for aug, dsets in subgroups_for(m):
            for ds in dsets:
                inner_ticks.append(centers[(m, aug, ds)])
                inner_labs.append(DS_ABBR[ds])
    ax.set_xticks(inner_ticks)
    ax.set_xticklabels(inner_labs, fontsize=7.5, rotation=90, color="#666666")
    ax.tick_params(axis="x", length=0, pad=2)

    for m in MODEL_ORDER:
        # aug (middle) labels under each subgroup
        for aug, dsets in subgroups_for(m):
            x0, x1 = sub_span[(m, aug)]
            mid = (x0 + x1) / 2
            ax.annotate(AUG_LABEL[aug], xy=(mid, 0), xycoords=("data", "axes fraction"),
                        xytext=(0, -22), textcoords="offset points",
                        ha="center", va="top", fontsize=9.5, color="#333333")
        # model (outer) label + bracket under the block
        x0, x1 = model_span[m]
        mid = (x0 + x1) / 2
        ax.annotate(MODEL_ABBR[m], xy=(mid, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -40), textcoords="offset points",
                    ha="center", va="top", fontsize=12, fontweight="bold")
        pstr, dim = MODEL_META[m]
        ax.annotate(f"{pstr} · {dim}-d", xy=(mid, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -56), textcoords="offset points",
                    ha="center", va="top", fontsize=8.5, color="#666666")
        ax.annotate("", xy=(x0 + 0.08, -0.175), xytext=(x1 - 0.08, -0.175),
                    xycoords=("data", "axes fraction"), textcoords=("data", "axes fraction"),
                    arrowprops=dict(arrowstyle="-", color="#9a9a9a", lw=1.0))
        # divider before each model block (except the first)
        if m != MODEL_ORDER[0]:
            ax.axvline(x0 - MODEL_GAP / 2, color="#cfcfcf", lw=1.2, zorder=0)

    # ---- legends: benchmark color (top strip) + mean-line key (in-axes) ----
    # only key the datasets actually present (300ep is S1102-only -> a 1-item key,
    # not a misleading 4-color strip).
    _present_ds = [d for d in ["S1102", "S1131", "S4169", "S2003"]
                   if any(k[2] == d and len(v) for k, v in by.items())]
    ds_handles = [Patch(facecolor=cm.dataset_color(d), edgecolor=cm.dataset_color(d),
                        alpha=0.55, label=d) for d in _present_ds]
    leg1 = fig.legend(handles=ds_handles, loc="upper center", bbox_to_anchor=(0.5, 0.96),
                      ncol=len(_present_ds), fontsize=10, frameon=False, title="benchmark (color)",
                      title_fontsize=10, columnspacing=1.6, handletextpad=0.5)
    mean_handles = [
        Line2D([0], [0], marker="D", color="#8c8c8c", linestyle="none",
               markerfacecolor=(0.55, 0.55, 0.55, 0.10), markeredgecolor="#8c8c8c",
               markeredgewidth=0.8, markersize=10,
               label="benchmark mean ± 95% CI (diamond)"),
        Line2D([0], [0], color="#8a8a8a", lw=2.2, linestyle=":",
               label="base / +aug subgroup mean"),
        Line2D([0], [0], color="#7d7d7d", lw=1.8, label="median (box)"),
        Line2D([0], [0], color="#8a8a8a", lw=1.3, linestyle="--",
               label="MuLAN paper target (S1102, mut-split)"),
    ]
    ax.legend(handles=mean_handles, loc="lower left", fontsize=9, frameon=True,
              facecolor="white", edgecolor="#d9d9d9", borderpad=0.5, labelspacing=0.3)

    _title = ("MuLAN ΔΔG variability — PLM × augmentation (S1102 per-fold Pearson r)"
              if _IS_300EP
              else "MuLAN ΔΔG variability — PLM × augmentation × benchmark (per-fold Pearson r)")
    fig.suptitle(_title, fontsize=15, fontweight="bold", y=1.035)
    fig.text(0.5, 1.0, _SERIES_CFG["subtitle"], ha="center", fontsize=10.5,
             style="italic", color="#555555")
    # data-driven status note so the caption stays honest as sweeps land
    partial = sorted(f"{MODEL_ABBR[m]} {ds}" for (m, aug, ds), v in by.items()
                     if 0 < len(v) < EXPECT)
    partial_note = ("partial cells faded: " + ", ".join(partial) + "." if partial
                    else "all present cells complete (10-fold).")
    if _IS_300EP:
        # narrow figure (few models) -> the caption MUST wrap or bbox_inches="tight"
        # expands the canvas to a single long line's width and squashes the plot.
        _queued = ("base+aug complete (10/10) for 6 PLMs; ESM3 is base-only (its +aug arm re-runs). "
                   if _SERIES == "300ep_balanced"
                   else "\"running\" cells = +aug and saprot/esm2 base still queued. ")
        cap = (
            "Box = q25/median/q75 + 1.5·IQR whiskers (grey = median); dots = the 10 CV folds; "
            "light-grey diamond = mean ± 95% CI of the mean (JMP-style).\n"
            "Dark-grey number = each S1102 box mean. Dashed grey = MuLAN paper Table 1 "
            "(S1102, mut-split): Ankh 0.868, ESM2 0.854.\n"
            "S1102-only at 300ep; " + _queued
            + partial_note + " Model label = encoder params · dim; see RESULTS.md §10.")
    else:
        cap = (
            "Box = q25/median/q75 + 1.5·IQR whiskers (grey line = median); dots = individual CV folds (color = benchmark). "
            "Light-grey diamond in each box = that benchmark's mean (horizontal line) ± 95% CI of the mean (top/bottom vertices, JMP-style); "
            "medium-grey dotted line = the +aug (left) / base (right) subgroup mean (pooled over complete 10-fold cells); "
            "dark-grey number left of each S1102 box spells out its mean. "
            "Tier-1 aug benchmarks are now landing (SaProt +aug S1131 done; "
            "S4169/S2003 +aug in flight); other models' aug cells are still S1102-only. "
            "Dashed grey horizontals = MuLAN paper Table 1 targets (mutation-based 10-fold CV, S1102): "
            "MuLAN-Ankh 0.868, MuLAN-ESM2 0.854.\nModel label shows encoder params · embedding dim. "
            "empty cells marked \"running\" are unrun — benchmark/+aug sweeps are only "
            "run for some PLMs (Ankh3-L/-XL and AIDO are S1102-only; ESM C benchmarks land per Linux snapshot). "
            + partial_note + " 10-fold protocol per PLM; see RESULTS.md §10.")

    fig.text(0.5, -0.14, cap, ha="center", fontsize=8.6, color="#777777")

    for ext in ("png", "svg"):
        out = os.path.join(HERE, f"{_SERIES_CFG['out']}.{ext}")
        fig.savefig(out, dpi=170, bbox_inches="tight")
        print("wrote", out)
    plt.close(fig)


if __name__ == "__main__":
    plot()
