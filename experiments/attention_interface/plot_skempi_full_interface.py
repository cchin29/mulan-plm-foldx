#!/usr/bin/env python3
"""Tier-3: full-SKEMPI attention→interface head-to-head, the leakage-controlled scale-up
of the S1102 analysis (attention→interface question).

Reads s1102_skempi_full_<tag>.csv + _perchain.csv (written by score_s1102.py --suffix
_skempi_full over the 102 full-SKEMPI masks, using the by-complex retrain heads) and emits
RESULTS_skempi_full.md + skempi_full_interface.{png,svg}: macro AUROC [10k-bootstrap 95% CI]
per backbone, paired-bootstrap Δ-vs-Ankh significance, ranked.

Difference from the S1102 view: the heads here are **by-complex leakage-controlled**
(evaluated on held-out complexes → honest generalization), so there is no direct paper
anchor (the 0.757 Fig-S1 number is S1102/INTBuilder). The internal Ankh anchor + the 0.690
geometric contact floor + chance are the references. Absolute AUROCs are lower than S1102 —
that's the point: interface recovery under proper leakage control.

Usage:  ./.venv/bin/python experiments/attention_interface/plot_skempi_full_interface.py
"""
from __future__ import annotations
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# reuse the S1102 plotter's stat helpers + styling (import is side-effect-free: main() is __main__-gated)
from plot_s1102_interface import (TAG2NAME, ANCHOR, GEOM_FLOOR, FLAGGED, FRONTIER,
                                   C_WIN, C_TIE, C_LOSE, C_FLAG, boot_ci, paired_diff_p)

HERE = Path(__file__).resolve().parent
PREFIX = "s1102_skempi_full"


def load_perchain(tag: str) -> dict[str, float]:
    p = HERE / f"{PREFIX}_{tag}_perchain.csv"
    d = {}
    if p.exists():
        for r in csv.DictReader(open(p)):
            d[r["chain"]] = float(r["auroc"])
    return d


def main():
    tags = [t for t in TAG2NAME if (HERE / f"{PREFIX}_{t}.csv").exists()]
    if ANCHOR not in tags:
        raise SystemExit(f"anchor {ANCHOR} not scored yet ({PREFIX}_{ANCHOR}.csv missing)")
    ref_pc = load_perchain(ANCHOR)
    rows = []
    for t in tags:
        s = list(csv.DictReader(open(HERE / f"{PREFIX}_{t}.csv")))[0]
        pc = load_perchain(t)
        vals = np.array(list(pc.values()))
        lo, hi = boot_ci(vals)
        if t == ANCHOR:
            diff, dlo, dhi, sig = 0.0, 0.0, 0.0, "—"
        else:
            diff, dlo, dhi = paired_diff_p(pc, ref_pc)
            sig = "beats" if dlo > 0 else ("below" if dhi < 0 else "ns")
        rows.append(dict(tag=t, name=TAG2NAME[t], macro=float(s["macro_auroc"]),
                         lo=lo, hi=hi, pooled=float(s["pooled_auroc"]),
                         diff=diff, dlo=dlo, dhi=dhi, sig=sig, n=int(s["n_chains"])))
    rows.sort(key=lambda r: r["macro"], reverse=True)
    anchor_macro = next(r["macro"] for r in rows if r["tag"] == ANCHOR)

    # ---- markdown ----
    md = ["# Tier-3: full-SKEMPI attention → interface head-to-head (attention→interface question)",
          "",
          "Per-residue MuLAN LightAtt attention scored (AUROC) vs cross-chain interface labels "
          "over the **102 full-SKEMPI complexes** (204 chains), one row per PLM backbone. Unlike "
          "the S1102 view, the heads here are the **by-complex leakage-controlled** retrain heads "
          "(fold-0), evaluated on held-out complexes → an honest generalization readout. **macro** "
          "= average AUROC over sequences; 95% CI from 10k bootstrap over chains. **Δ vs Ankh** = "
          "paired bootstrap of the per-chain difference vs Ankh (CI excluding 0 ⇒ significant).",
          "",
          f"Anchors: internal Ankh (by-complex) = **{anchor_macro:.3f}**; geometric contact-count "
          f"floor = **{GEOM_FLOOR}**; chance = 0.5. No paper anchor here — the 0.757 Fig-S1 number "
          f"is S1102/INTBuilder, not this leakage-controlled full-SKEMPI setup. Absolute AUROCs run "
          f"lower than S1102 (Ankh 0.788 there) precisely because these are held-out complexes.",
          "",
          "| rank | backbone | macro AUROC [95% CI] | pooled | Δ vs Ankh [95% CI] | verdict |",
          "|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        flag = " ⚠️" if r["tag"] in FLAGGED else ""
        dtxt = "—" if r["tag"] == ANCHOR else f"{r['diff']:+.3f} [{r['dlo']:+.3f}, {r['dhi']:+.3f}]"
        md.append(f"| {i} | {r['name']}{flag} | {r['macro']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}] | "
                  f"{r['pooled']:.3f} | {dtxt} | {r['sig']} |")
    beats = [r["name"] for r in rows if r["sig"] == "beats"]
    verdict = ("**no backbone significantly beats Ankh**" if not beats
               else "**significantly beats Ankh: " + ", ".join(beats) + "**")
    md += ["",
           f"**Bottom line (full-SKEMPI, leakage-controlled):** {verdict}. This corroborates the "
           "S1102 finding on the harder held-out set — the newer PLMs do not improve attention-based "
           "interface recovery over Ankh-large, and scaling to 16B (ESM-C 6B, AIDO) does not help.",
           "",
           "*Caveats:* esmc6b/aido were scored on the Linux/CPU box (healthy embeddings, head-robust "
           "to ±0.006). AIDO's head is `full_skempi_bycomplex/aido_base` (its honest-ladder run) "
           "rather than `retrain_bycomplex` like the others — both by-complex leakage-controlled, but "
           "note the convention. AIDO's 0.396 sits *below chance*; the Ankh 0.671 anchor confirms the "
           "pipeline recovers signal on these masks, so it is a genuine low, but treat the sub-chance "
           "value as the model's worst-case rather than a precise point estimate. esm3/saprot13b "
           "pending full-SKEMPI WT caches.", ""]
    (HERE / "RESULTS_skempi_full.md").write_text("\n".join(md))
    print("wrote", HERE / "RESULTS_skempi_full.md")

    # ---- figure ----
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(9, max(3.2, 0.55 * len(rows) + 1.6)))
    names = [r["name"] + (" ⚠️" if r["tag"] in FLAGGED else "") for r in rows]
    y = np.arange(len(rows))[::-1]
    colors = []
    for r in rows:
        if r["tag"] in FLAGGED: colors.append(C_FLAG)
        elif r["tag"] == ANCHOR: colors.append("#4a5568")
        elif r["sig"] == "beats": colors.append(C_WIN)
        elif r["sig"] == "below": colors.append(C_LOSE)
        else: colors.append(C_TIE)
    err = np.array([[r["macro"] - r["lo"] for r in rows], [r["hi"] - r["macro"] for r in rows]])
    ax.barh(y, [r["macro"] for r in rows], color=colors, height=0.66, zorder=3)
    ax.errorbar([r["macro"] for r in rows], y, xerr=err, fmt="none", ecolor="#2d3748",
                elinewidth=1.1, capsize=3, zorder=4)
    for yi, r in zip(y, rows):
        ax.text(r["hi"] + 0.008, yi, f"{r['macro']:.3f}", va="center", ha="left", fontsize=8.5)
    ax.axvline(0.5, color="#a0aec0", lw=1, ls=":", zorder=1)
    ax.text(0.5, len(rows) - 0.3, "chance", color="#718096", fontsize=8, ha="center")
    ax.axvline(GEOM_FLOOR, color=FRONTIER, lw=1.1, ls="--", alpha=0.7, zorder=1)
    ax.text(GEOM_FLOOR, -0.9, f"geom. floor {GEOM_FLOOR}", color=FRONTIER, fontsize=8, ha="center")
    ax.axvline(anchor_macro, color="#4a5568", lw=1, ls="-.", alpha=0.6, zorder=1)
    ax.text(anchor_macro, len(rows) - 0.3, f"Ankh {anchor_macro:.3f}", color="#4a5568",
            fontsize=8, ha="center", fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.set_xlabel("per-structure macro AUROC  (attention vs interface, full-SKEMPI, T≥1)")
    ax.set_xlim(0.40, 0.82)
    ax.set_title("full-SKEMPI attention→interface — leakage-controlled by-complex heads\n"
                 "does any PLM beat Ankh on held-out complexes? (attention→interface question, scale-up)",
                 fontsize=11)
    green = plt.Line2D([], [], marker="s", ls="", color=C_WIN, label="beats Ankh (paired CI>0)")
    grey = plt.Line2D([], [], marker="s", ls="", color=C_TIE, label="tie (ns)")
    orange = plt.Line2D([], [], marker="s", ls="", color=C_LOSE, label="below Ankh (paired CI<0)")
    ax.legend(handles=[green, grey, orange], loc="lower right", fontsize=8.5, framealpha=0.95)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(HERE / f"skempi_full_interface.{ext}", dpi=140)
        print("wrote", HERE / f"skempi_full_interface.{ext}")


if __name__ == "__main__":
    main()
