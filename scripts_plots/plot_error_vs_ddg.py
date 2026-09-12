#!/usr/bin/env python3
"""Scatterplot matrix: per-model mispredict error (signed = true - pred) vs true ddG,
on the balanced 300/30 OOF folds. Shows the shared shrink-to-mean blind spot across models."""
import os
import glob
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = str(Path(__file__).resolve().parents[1])
SPLIT = f"{ROOT}/scratch/foldx_s1102/splits_balanced_foldxdec"
RES   = f"{ROOT}/scratch/results/embedding_sweep_balanced"
INC   = f"{ROOT}/scratch/incoming_esmc6b_20260714"

# base-arm prediction path per model
PATHS = {
    "ESM2-3B":   f"{RES}/esm2/fold_{{f}}/training_run/test_predictions.tsv",
    "ESM C 600M":f"{RES}/esmc600m/fold_{{f}}/training_run/test_predictions.tsv",
    "ProstT5":   f"{RES}/prostt5/fold_{{f}}/training_run/test_predictions.tsv",
    "SaProt":    f"{RES}/saprot/fold_{{f}}/training_run/test_predictions.tsv",
    "Ankh-large":f"{ROOT}/scratch/results/cv10_ankh_converge/fold_{{f}}/training_run/test_predictions.tsv",
    "ESM C 6B":  f"{INC}/results/base/S1102/fold_{{f}}/training_run/test_predictions.tsv",
}

# truth
truth = {}
for f in range(10):
    for ln in open(f"{SPLIT}/fold_{f}/S1102_filtered_test.tsv"):
        p = ln.rstrip("\n").split("\t")
        if len(p) < 4: continue
        truth[(p[0].split("_")[0], p[2])] = float(p[3])

preds = {m: {} for m in PATHS}
for m, tmpl in PATHS.items():
    for f in range(10):
        fp = tmpl.format(f=f)
        if not os.path.exists(fp): continue
        for ln in open(fp):
            p = ln.rstrip("\n").split("\t")
            if len(p) < 4: continue
            preds[m][(p[0].split("_")[0], p[2])] = float(p[3])

keys = [k for k in truth if all(k in preds[m] for m in PATHS)]
print("aligned points:", len(keys))
y_true = np.array([truth[k] for k in keys])

# named hotspots to highlight
HOT = {("1MAH","WA276R"),("2PCC","EA290A"),("1PPF","LB18W"),("1MAH","YA69N"),
       ("2O3B","DB75N"),("2O3B","DB75E"),("1PPF","LB18Y"),("1BRS","RA81Q"),
       ("1KTZ","SB28L"),("1R0R","AB10R")}
hot_idx = np.array([k in HOT for k in keys])

models = list(PATHS)
fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), sharex=True, sharey=True)
axes = axes.ravel()
lim = 9
for ax, m in zip(axes, models):
    err = np.array([truth[k] - preds[m][k] for k in keys])  # signed error, + = under-predicted
    # global |err| vs |true| correlation (the shrink signature)
    r = np.corrcoef(np.abs(err), np.abs(y_true))[0, 1]
    # slope of pred~true (shrink: <1)
    slope = np.polyfit(y_true, np.array([preds[m][k] for k in keys]), 1)[0]
    ax.axhline(0, color="#888", lw=0.8, zorder=1)
    # guide: perfect model has err=0; a mean-predictor has err = true - mean(true)
    xs = np.linspace(-lim, lim, 50)
    ax.plot(xs, xs - y_true.mean(), ls=":", color="#c44", lw=1.0, zorder=1,
            label="mean-predictor")
    ax.scatter(y_true[~hot_idx], err[~hot_idx], s=9, alpha=0.35,
               color="#2b6cb0", edgecolors="none", zorder=2)
    ax.scatter(y_true[hot_idx], err[hot_idx], s=42, color="#e53e3e",
               edgecolors="k", linewidths=0.5, zorder=3, label="named hotspots")
    ax.set_title(f"{m}", fontsize=12, fontweight="bold")
    ax.text(0.03, 0.97, f"slope(pred~true)={slope:.2f}\ncorr(|err|,|ΔΔG|)={r:.2f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="#ccc", alpha=0.85))
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.grid(True, alpha=0.15)

for ax in axes[3:]:
    ax.set_xlabel("true ΔΔG (kcal/mol)", fontsize=10)
for ax in (axes[0], axes[3]):
    ax.set_ylabel("mispredict error  (true − pred)", fontsize=10)
axes[2].legend(loc="lower right", fontsize=8, framealpha=0.9)

fig.suptitle("Per-model mispredict error vs true ΔΔG — S1102 balanced OOF (base arm, 1100 muts)\n"
             "positive error = under-prediction; the funnel opening toward ±extremes = shared shrink-to-mean",
             fontsize=13, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.955])
out = os.environ.get("OUT", f"{ROOT}/scripts_plots/error_vs_ddg_matrix.png")
fig.savefig(out, dpi=140)
print("wrote", out)
