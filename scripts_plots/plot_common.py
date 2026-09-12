#!/usr/bin/env python3
"""Shared theme + helpers for the MuLAN figure scripts.

Both `plot_ddg_scaling.py` (performance/cost vs PLM scale) and
`plot_benchmarks.py` (benchmark-generalization heatmap + faceted bars) import
from here so the whole figure set shares one visual identity — fonts, the
platform color families (Linux-blue / MPS-magenta), the per-model palette, and
the CSV-loading convention. Change a color or a font once, here, and every
figure updates.

No plotting happens in this module; it only defines style + small helpers.
Deps: matplotlib, pandas.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---- global rcParams (the "reference figure" look) -------------------------
# Applied by apply_theme(); call it once at the top of each figure's plot().
RC = {
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
}


def apply_theme():
    """Set the shared rcParams. Idempotent; call at the start of plot()."""
    plt.rcParams.update(RC)


# ---- platform color families (unchanged from the original scaling plot) -----
# Marker color keys on WHERE A RUN'S CV10 TRAINED (run_platform), not where its
# embeddings were generated: Linux 40c CPU box -> blue, Apple M4 Pro/MPS -> magenta.
COLORS = {
    "seq_cv10_linux": "#16386f",   # dark navy — sequence-only CV (primary)
    "aug_cv10_linux": "#5b9bd5",   # medium blue — + Tier-1 augmentation
    "seq_cv10_mps": "#8E1B6B",     # dark magenta — sequence-only CV (primary)
    "aug_cv10_mps": "#E39BC8",     # light magenta — + Tier-1 augmentation
    "reference": "#8a8a8a",        # grey dashed — paper target
    "pending": "#b0b0b0",          # grey — planned/not-yet-run
    "lineage": "#8f8f8f",          # neutral grey — family-lineage connector
}


def platform_color(series_key, run_platform):
    """Pick the seq/aug color for a row, keyed on WHERE ITS CV10 RAN
    (`run_platform`: mps | linux | ...). Falls back to the linux (original)
    shade for anything unrecognized."""
    plat = "mps" if str(run_platform).strip().lower() == "mps" else "linux"
    return COLORS[f"{series_key}_{plat}"]


# ---- per-model palette (benchmark figures) ---------------------------------
# One stable color per PLM so a model reads as the same hue across every
# benchmark facet. Keyed by the exact `model` string used in the CSVs.
MODEL_COLORS = {
    "ESM C 6B": "#16386f",           # navy — current best PLM
    "SaProt (WT-3Di)": "#2e7d5b",    # green — structure-aware
    "Ankh-large": "#5b9bd5",         # blue — the incumbent baseline
    "ESM2-3B": "#c07b3f",            # amber — older-generation ESM
    "ProstT5": "#8E1B6B",            # magenta — the other baseline
    "Ankh3-large": "#9aa3b2",        # slate — held on benchmarks
    "Ankh3-xl": "#6b7280",           # dark slate — held on benchmarks
    "AIDO.Protein-16B": "#b0563f",   # brick — dropped on benchmarks
}
MODEL_FALLBACK = "#7f7f7f"


def model_color(model):
    return MODEL_COLORS.get(str(model), MODEL_FALLBACK)


# ---- per-benchmark (dataset) palette ---------------------------------------
# Qualitative, print-friendly. Used when a figure encodes DATASET by color
# (e.g. the variability chart grouped model -> aug -> benchmark).
DATASET_COLORS = {
    "S1102": "#2c6fbb",   # blue   — anchor SKEMPI single-mut
    "S1131": "#2e8b74",   # teal   — GeoPPI interface
    "S4169": "#d98a2b",   # orange — GeoPPI all (single-chain)
    "S2003": "#9b4f9e",   # purple — non-Ala single
}
DATASET_FALLBACK = "#7f7f7f"


def dataset_color(ds):
    return DATASET_COLORS.get(str(ds), DATASET_FALLBACK)


# ---- cell/bar states for partly-filled matrices ----------------------------
# Drives how a not-yet-measured (or deliberately skipped) model×dataset cell
# renders, so the figure degrades gracefully as sweeps land: flip a CSV row's
# `status` from pending -> measured (and fill pcc) and the cell fills in.
STATUS_STYLE = {
    "measured": dict(face=None,       hatch=None,  label=None,     text="#222222"),
    "pending":  dict(face="#ededed",  hatch="///", label="running", text="#7a7a7a"),
    "held":     dict(face="#f4f4f4",  hatch=None,  label="held",    text="#9a9a9a"),
    "dropped":  dict(face="#fafafa",  hatch="xx",  label="—",       text="#b4b4b4"),
}


# ---- small formatting helpers (shared) -------------------------------------
def fmt_min(x):
    """1.97 -> '2', 69.7 -> '69.7', 37.0 -> '37'."""
    return f"{x:g}"


def plat_short(p):
    """Abbreviate a platform tag, e.g. 'Linux 40c CPU / 62G' -> 'Linux 40c'."""
    if p is None or (isinstance(p, float) and pd.isna(p)):
        return ""
    return " ".join(str(p).split()[:2])


def load_csv(path, numeric_cols):
    """Read a '#'-commented CSV and coerce the given columns to numeric
    (blank -> NaN), the shared loader convention for these figures."""
    df = pd.read_csv(path, comment="#")
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
