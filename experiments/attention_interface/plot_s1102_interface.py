#!/usr/bin/env python3
"""Tier-2: S1102 attention->interface head-to-head across PLM backbones.

Reads the per-backbone summaries (s1102_<tag>.csv) and per-chain AUROCs
(s1102_<tag>_perchain.csv) written by score_s1102.py, and produces:
  * RESULTS_s1102.md  -- ranked table: macro AUROC [bootstrap 95% CI], pooled,
    Δ vs Ankh-large, and a paired-bootstrap significance test vs Ankh.
  * s1102_interface.{png,svg} -- ranked bar chart of macro AUROC with CI whiskers,
    reference lines (paper Ankh 0.757, our Ankh anchor, geometric floor 0.690, chance).

Answers the attention→interface question: "do the tested PLMs improve interaction-site prediction
over Ankh-large?" The absolute level differs from the paper (our heavy-atom-8A labels
+ fold-0 balanced heads vs the paper's INTBuilder labels + released full-train head;
Ankh reproduces at 0.788 vs the paper's 0.757). The valid comparison is the RANKING
across backbones under one consistent protocol, with Ankh-large as the internal anchor.

Usage:  ./.venv/bin/python experiments/attention_interface/plot_s1102_interface.py
"""
from __future__ import annotations
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RNG = np.random.default_rng(42)
NBOOT = 10000

TAG2NAME = {
    "ankh": "Ankh-large", "esm2": "ESM2-3B", "saprot": "SaProt", "saprot13b": "SaProt-1.3B",
    "prostt5": "ProstT5", "ankh3_large": "Ankh3-large", "ankh3_xl": "Ankh3-xl",
    "esmc600m": "ESM-C 600M", "esmc6b": "ESM-C 6B", "aido": "AIDO-16B", "esm3": "ESM3-1.4B",
}
ANCHOR = "ankh"                 # internal reference backbone
PAPER_ANKH = 0.757             # Fig S1 average AUROC (INTBuilder labels)
GEOM_FLOOR = 0.690            # skempi_interface cross-chain Cα-8Å contact-count AUROC
# esm3: provenance-VERIFIED (head trained on these exact SDK embeddings) -> 0.424 is real,
# not a bug. ESM3 carries massive-activation outliers (emb std ~325 vs ~0.04 elsewhere); its
# attention partially tracks outlier-magnitude positions (Spearman +0.22) not interfaces,
# though the ddG head copes (PCC ~0.82). Genuine attention/ddG dissociation; scale outlier.
FLAGGED = {"esm3"}

FRONTIER = "#7b5aa6"
C_WIN, C_TIE, C_LOSE, C_FLAG = "#2f855a", "#b8c2cc", "#c05621", "#a0aec0"


def load_perchain(tag: str) -> dict[str, float]:
    p = HERE / f"s1102_{tag}_perchain.csv"
    d = {}
    if p.exists():
        for r in csv.DictReader(open(p)):
            d[r["chain"]] = float(r["auroc"])
    return d


def boot_ci(vals: np.ndarray) -> tuple[float, float]:
    idx = RNG.integers(0, len(vals), size=(NBOOT, len(vals)))
    means = vals[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_diff_p(x: dict[str, float], ref: dict[str, float]) -> tuple[float, float, float]:
    """Paired bootstrap of mean(x)-mean(ref) over shared chains. Returns
    (diff, ci_lo, ci_hi). CI excluding 0 => significant at ~0.05."""
    common = sorted(set(x) & set(ref))
    dx = np.array([x[c] - ref[c] for c in common])
    idx = RNG.integers(0, len(dx), size=(NBOOT, len(dx)))
    dd = dx[idx].mean(axis=1)
    return float(dx.mean()), float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))


def main():
    summ = {}
    for r in csv.DictReader(open(HERE / "s1102_ankh.csv")):  # header schema
        break
    tags = [t for t in TAG2NAME if (HERE / f"s1102_{t}.csv").exists()]
    rows = []
    ref_pc = load_perchain(ANCHOR)
    for t in tags:
        s = list(csv.DictReader(open(HERE / f"s1102_{t}.csv")))[0]
        pc = load_perchain(t)
        vals = np.array(list(pc.values()))
        lo, hi = boot_ci(vals)
        macro = float(s["macro_auroc"])
        pooled = float(s["pooled_auroc"])
        if t == ANCHOR:
            diff, dlo, dhi, sig = 0.0, 0.0, 0.0, "—"
        else:
            diff, dlo, dhi = paired_diff_p(pc, ref_pc)
            sig = "beats" if dlo > 0 else ("below" if dhi < 0 else "ns")
        rows.append(dict(tag=t, name=TAG2NAME[t], macro=macro, lo=lo, hi=hi,
                         pooled=pooled, diff=diff, dlo=dlo, dhi=dhi, sig=sig))
    rows.sort(key=lambda r: r["macro"], reverse=True)
    anchor_macro = next(r["macro"] for r in rows if r["tag"] == ANCHOR)

    # ---- markdown ----
    md = ["# Tier-2: S1102 attention → interface head-to-head (attention→interface question)",
          "",
          "Per-residue MuLAN LightAtt attention scored (AUROC) against cross-chain interface "
          "labels on the 220 S1102 wild-type chains, one row per PLM backbone (fold-0 balanced "
          "head). **macro** = average AUROC over sequences (the paper's Fig S1 statistic); "
          "95% CI from 10k bootstrap over chains. **Δ vs Ankh** = paired bootstrap of the "
          "per-chain AUROC difference vs Ankh-large (CI excluding 0 ⇒ significant).",
          "",
          f"Anchors: paper Fig S1 Ankh-large = **{PAPER_ANKH}** (INTBuilder labels); our Ankh "
          f"reproduces at **{anchor_macro:.3f}** on heavy-atom-8Å labels; geometric "
          f"contact-count floor = **{GEOM_FLOOR}**.",
          "",
          "| rank | backbone | macro AUROC [95% CI] | pooled | Δ vs Ankh [95% CI] | verdict |",
          "|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        flag = " ⚠️" if r["tag"] in FLAGGED else ""
        dtxt = "—" if r["tag"] == ANCHOR else f"{r['diff']:+.3f} [{r['dlo']:+.3f}, {r['dhi']:+.3f}]"
        md.append(f"| {i} | {r['name']}{flag} | {r['macro']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}] | "
                  f"{r['pooled']:.3f} | {dtxt} | {r['sig']} |")
    md += ["",
           "⚠️ **ESM3**: provenance-verified — the esm3 head trained on these exact "
           "`scratch/embeddings_esm3` SDK embeddings, so 0.424 is **real, not a bug**. ESM3's "
           "embeddings carry massive-activation outliers (per-chain std ~325 vs ~0.04 for the "
           "other PLMs); its attention partially tracks outlier-*magnitude* positions "
           "(Spearman(att, ‖emb‖) ≈ +0.22) rather than interface residues, even though the ddG "
           "regression head still works (CV PCC ~0.82). A genuine attention↔ddG dissociation; "
           "kept out of the headline as an embedding-scale outlier, not a fair interface signal.",
           "",
           "**Bottom line:** the newer PLMs do **not** robustly beat Ankh-large on attention-"
           "based interaction-site prediction. Only the structure-aware SaProt and ESM-C 600M "
           "match/edge it, and the margin is within noise (see Δ-vs-Ankh CIs). Pure-sequence "
           "successors (ESM2-3B, ProstT5, Ankh3) score clearly lower. **Scaling to 16B does not "
           "help:** ESM-C 6B (0.603) and AIDO-16B (0.619) both land significantly below Ankh — and "
           "ESM-C 6B is ~0.19 *below* its own 600M sibling. (esmc6b/aido from the Linux/CPU box; "
           "healthy embeddings, head-robust to ±0.006 — not scale outliers like esm3.)",
           "",
           "*Fold robustness:* a fold-5 spot-check reproduces the ranking — top cluster SaProt "
           "0.795 / ESM-C 600M 0.799 / Ankh 0.784 (still a 3-way tie), all successors below "
           "(SaProt-1.3B 0.748, Ankh3-xl 0.659, ProstT5 0.652, Ankh3-large 0.623, ESM2-3B 0.574). "
           "No backbone beats Ankh in either fold; the fold-0 numbers above are not an artifact.", ""]
    (HERE / "RESULTS_s1102.md").write_text("\n".join(md))
    print("wrote", HERE / "RESULTS_s1102.md")

    # ---- figure ----
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(9, 5.2))
    names = [r["name"] + (" ⚠️" if r["tag"] in FLAGGED else "") for r in rows]
    y = np.arange(len(rows))[::-1]
    anchor_macro = next(r["macro"] for r in rows if r["tag"] == ANCHOR)
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
    ax.axvline(PAPER_ANKH, color="#dd6b20", lw=1.3, zorder=2)
    ax.text(PAPER_ANKH, len(rows) - 0.3, f"paper Ankh {PAPER_ANKH}", color="#dd6b20",
            fontsize=8, ha="center", fontweight="bold")
    ax.axvline(anchor_macro, color="#4a5568", lw=1, ls="-.", alpha=0.6, zorder=1)
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.set_xlabel("per-structure macro AUROC  (attention vs interface, S1102, T≥1)")
    ax.set_xlim(0.40, 0.86)
    ax.set_title("Do the tested PLMs improve attention-based interface prediction?\n"
                 "MuLAN attention→interface AUROC across backbones (fold-0 heads, heavy-atom-8Å labels)",
                 fontsize=11)
    green = plt.Line2D([], [], marker="s", ls="", color=C_WIN, label="beats Ankh (paired CI>0)")
    grey = plt.Line2D([], [], marker="s", ls="", color=C_TIE, label="tie (ns)")
    orange = plt.Line2D([], [], marker="s", ls="", color=C_LOSE, label="below Ankh (paired CI<0)")
    ax.legend(handles=[green, grey, orange], loc="lower right", fontsize=8.5, framealpha=0.95)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(HERE / f"s1102_interface.{ext}", dpi=140)
        print("wrote", HERE / f"s1102_interface.{ext}")


if __name__ == "__main__":
    main()
