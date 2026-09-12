#!/usr/bin/env python3
"""Aggregate per-FOLD test Pearson r into the tidy long CSV that feeds the JMP-style
variability chart (plot_variability.py). Columns: model, aug (base | aug), dataset,
fold, pcc — all individual fold values kept (the chart plots them + overlays box plots).

SERIES-AWARE via the MULAN_PLOT_SERIES env var (same three protocols plot_variability.py
and plot_ddg_scaling.py render). Pick the series; the script picks the source and the
output CSV:

  50ep (default)  -> benchmarks_folds.csv
      Curated MAP below (multi-dataset: S1102 + the S1131/S4169/S2003 benchmark sweeps),
      scanning scratch/results/cv10_* and bench_* dirs. Edit MAP to add a cell.

  300ep_balanced  -> benchmarks_folds_300ep_balanced.csv   (balanced seed-42 equal-fold CV)
  300ep           -> benchmarks_folds_300ep.csv            (paper-faithful rng.integers CV)
      AUTO-DISCOVERED (no MAP) from scratch/results/embedding_sweep[_balanced]/<tag>[_aug]/.
      S1102-only. A `<tag>` dir is the base arm, a `<tag>_aug` dir the Tier-1 aug arm;
      the tag -> display name comes from TAG2NAME. New models appear on the plot the
      moment their folds land — just add the tag to TAG2NAME here (and the model to
      plot_variability's _ORDER_* to actually draw it). Non-model dirs (e.g. the
      `<tag>_foldxmlp` FoldX arms) have no TAG2NAME entry and are skipped + listed.

Partial cells (<10 folds) are emitted too — the plot renders them as an "n/10" cell,
mirroring the 50ep behavior. Re-run after new folds land:

    python aggregate_folds.py                                   # 50ep
    MULAN_PLOT_SERIES=300ep_balanced python aggregate_folds.py  # balanced
    MULAN_PLOT_SERIES=300ep          python aggregate_folds.py  # rng.integers

Deps: none (stdlib).
"""
import glob
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.normpath(os.path.join(HERE, "..", "scratch", "results"))

SERIES = os.environ.get("MULAN_PLOT_SERIES", "50ep").lower()

# Auto-discovered series (300ep*): (results subdir under scratch/results, output CSV).
# The 50ep series is MAP-driven instead (see MAP + build_50ep below).
AUTO_SERIES = {
    "300ep_balanced": ("embedding_sweep_balanced", "benchmarks_folds_300ep_balanced.csv"),
    "300ep":          ("embedding_sweep",          "benchmarks_folds_300ep.csv"),
}

# tag -> display name for the auto-discovered series. Same display names the 50ep MAP keys
# and plot_variability._ORDER_* lists use, so a model reads identically across all series.
# Add a row here to teach the aggregator a new PLM tag.
TAG2NAME = {
    "ankh": "Ankh-large",
    "prostt5": "ProstT5",
    "esm2": "ESM2-3B",
    "saprot": "SaProt (WT-3Di)",
    "saprot13b": "SaProt-1.3B (AFDB)",
    "esmc600m": "ESM C 600M",
    "esmc6b": "ESM C 6B",
    "ankh3_large": "Ankh3-large",
    "ankh3_xl": "Ankh3-xl",
    "aido": "AIDO.Protein-16B",
    "esm3": "ESM3-open (1.4B)",
}

# model -> aug -> {dataset: result-dir}. Extend to add a model / dataset / aug cell.
MAP = {
    "ESM C 6B": {
        # benchmark dirs auto-populate once the Linux box finishes them; if the
        # bench_cv_driver writes a different tag than "esmc6b", edit these names.
        "base": {"S1102": "cv10_esmc6b", "S1131": "bench_S1131_esmc6b",
                 "S4169": "bench_S4169_esmc6b", "S2003": "bench_S2003_esmc6b"},
        # +aug benchmarks running on the Linux box (esmc6b_aug_chain.sh): S1131
        # complete (10/10); S2003 in flight, S4169 queued (long pole). Partial
        # cells render faded; empty ones ("running") until their folds land.
        "aug":  {"S1102": "cv10_aug_esmc6b", "S1131": "bench_S1131_esmc6b_aug",
                 "S4169": "bench_S4169_esmc6b_aug", "S2003": "bench_S2003_esmc6b_aug"},
    },
    # ESM C 600M (esm-SDK, 1152-d) — S1102-only. Uses the SAME cv10 protocol/splits
    # as the other cv10_* cells (50ep/patience10, cv10_splits) so it is apples-to-
    # apples on the plot; embeddings pre-cached in scratch/embeddings_esmc600m.
    # (A separate 300ep/patience30 run under embedding_sweep/esmc600m feeds the
    # embedding_sweep summary — NOT this plot.) No aug / no benchmark sweeps planned.
    "ESM C 600M": {
        "base": {"S1102": "cv10_esmc600m"},
        "aug":  {"S1102": "cv10_aug_esmc600m"},   # Tier-1 aug (reverse+identity), 50ep/patience10
    },
    "SaProt (WT-3Di)": {
        "base": {"S1102": "cv10_saprot", "S1131": "bench_S1131_saprot",
                 "S4169": "bench_S4169_saprot", "S2003": "bench_S2003_saprot"},
        # aug benchmarks landing via mac_queue2 (S1131 done; S4169 partial; S2003 running)
        "aug":  {"S1102": "cv10_saprot_aug", "S1131": "bench_S1131_saprot_aug",
                 "S4169": "bench_S4169_saprot_aug", "S2003": "bench_S2003_saprot_aug"},
    },
    "Ankh-large": {
        "base": {"S1102": "cv10_ankh", "S1131": "bench_S1131_ankh",
                 "S4169": "bench_S4169_ankh", "S2003": "bench_S2003_ankh"},
        "aug":  {"S1102": "cv10_aug_ankh"},
    },
    # Ankh3 (NLU-prefix): S1102 base+aug only — benchmark sweeps not planned, so
    # their S1131/S4169/S2003 cells stay empty (render "running").
    "Ankh3-xl": {
        "base": {"S1102": "cv10_ankh3_xl_nlu"},
        "aug":  {"S1102": "cv10_ankh3_xl_aug_nlu"},
    },
    "Ankh3-large": {
        "base": {"S1102": "cv10_ankh3_large_nlu"},
        "aug":  {"S1102": "cv10_ankh3_large_aug_nlu"},
    },
    # SaProt-1.3B (AFDB_OMG_NCBI checkpoint): S1102 base+aug only — no
    # S1131/S4169/S2003 sweeps planned, so those cells stay empty ("running").
    "SaProt-1.3B (AFDB)": {
        "base": {"S1102": "cv10_saprot13b"},
        "aug":  {"S1102": "cv10_saprot13b_aug"},
    },
    # AIDO.Protein-16B: S1102 base + aug (aug ran on Linux 2026-07-08, 0.836 ±
    # 0.057). No benchmark sweeps planned, so it gets an S1102-only base+aug block.
    "AIDO.Protein-16B": {
        "base": {"S1102": "cv10_aido"},
        "aug":  {"S1102": "cv10_aug_aido"},
    },
    "ESM2-3B": {
        "base": {"S1102": "cv10_esm2", "S1131": "bench_S1131_esm2",
                 "S4169": "bench_S4169_esm2", "S2003": "bench_S2003_esm2"},
        "aug":  {"S1102": "cv10_esm2_aug"},
    },
    "ProstT5": {
        "base": {"S1102": "cv10_prostt5", "S1131": "bench_S1131_prostt5",
                 "S4169": "bench_S4169_prostt5", "S2003": "bench_S2003_prostt5"},
        # aug benchmarks landing via mac_queue2 (S1131 done; S4169 partial; S2003 running)
        "aug":  {"S1102": "cv10_aug_prostt5", "S1131": "bench_S1131_prostt5_aug",
                 "S4169": "bench_S4169_prostt5_aug", "S2003": "bench_S2003_prostt5_aug"},
    },
}


def folds(dirpath):
    # Keyed by fold id (not appended) so a stray/duplicate fold_* dir can't inflate a
    # cell's fold count or skew its mean — a dup is warned and ignored, first wins.
    out = {}
    patt = os.path.join(dirpath, "fold_*", "training_run", "all_results.json")
    for f in sorted(glob.glob(patt)):
        fold = int(os.path.basename(os.path.dirname(os.path.dirname(f))).split("_")[1])
        try:
            with open(f) as fh:
                pcc = json.load(fh)["test_pcc"]
        except (KeyError, json.JSONDecodeError, OSError):
            continue
        if fold in out:
            print(f"  WARN: duplicate fold_{fold} for {dirpath!r} ({f}) — ignoring dup")
            continue
        out[fold] = pcc
    return sorted(out.items())


def _flag(fs):
    if len(fs) == 0:
        return "  (no rows -> 'running')"
    return "" if len(fs) == 10 else "  <-- PARTIAL"


def _stat(fs):
    pccs = [p for _, p in fs]
    return f"  mean={st.mean(pccs):.4f} pstdev={st.pstdev(pccs):.4f}" if pccs else ""


def build_50ep():
    """Curated multi-dataset 50ep series from the MAP (S1102 + benchmark sweeps)."""
    rows, report = ["model,aug,dataset,fold,pcc"], []
    for model, augmap in MAP.items():
        for aug, dsmap in augmap.items():
            for ds, d in dsmap.items():
                fs = folds(os.path.join(RESULTS, d))
                for fold, pcc in fs:
                    rows.append(f"{model},{aug},{ds},{fold},{pcc:.6f}")
                report.append(f"  {model:18s} {aug:4s} {ds:6s} {len(fs):2d} folds  ({d}){_flag(fs)}")
    return rows, report, os.path.join(HERE, "benchmarks_folds.csv")


def build_auto(subdir, outname):
    """Auto-discover a 300ep* series: every <tag>[_aug] dir under scratch/results/<subdir>,
    S1102-only, tag -> display name via TAG2NAME. Prints mean/pstdev per cell so the
    matching ddg_scaling_data_*.csv row (pcc, pcc_std = pstdev) is trivial to fill."""
    root = os.path.join(RESULTS, subdir)
    if not os.path.isdir(root):
        sys.exit(f"no such results dir: {root}")
    rows, report, skipped = ["model,aug,dataset,fold,pcc"], [], []
    for d in sorted(os.listdir(root)):
        full = os.path.join(root, d)
        if not os.path.isdir(full) or not glob.glob(os.path.join(full, "fold_*")):
            continue  # not a fold-bearing sweep dir
        aug = "aug" if d.endswith("_aug") else "base"
        base_tag = d[:-4] if d.endswith("_aug") else d
        name = TAG2NAME.get(base_tag)
        if name is None:
            skipped.append(d)  # e.g. <tag>_foldxmlp FoldX arms — not a plotted PLM
            continue
        fs = folds(full)
        for fold, pcc in fs:
            rows.append(f"{name},{aug},S1102,{fold},{pcc:.6f}")
        report.append(f"  {name:18s} {aug:4s} {len(fs):2d} folds  ({d}){_flag(fs)}{_stat(fs)}")
    if skipped:
        report.append(f"  skipped (no TAG2NAME entry — not a plotted model): {skipped}")
    return rows, report, os.path.join(HERE, outname)


if SERIES == "50ep":
    rows, report, OUT = build_50ep()
elif SERIES in AUTO_SERIES:
    rows, report, OUT = build_auto(*AUTO_SERIES[SERIES])
else:
    sys.exit(f"unknown MULAN_PLOT_SERIES={SERIES!r} — use 50ep | 300ep | 300ep_balanced")

with open(OUT, "w") as fh:
    fh.write("\n".join(rows) + "\n")
print(f"[{SERIES}] wrote {OUT} ({len(rows) - 1} fold rows)")
print("\n".join(report))
