#!/usr/bin/env python3
"""The headline figure: honest-split ladder — base collapses, FoldX holds, gap widens.
Reads scripts_plots/results_matrix_ps.csv (leaky -> by-complex -> clustered)."""
import csv, numpy as np
from pathlib import Path

import benchmarks as _bench   # data/benchmarks/frontier.tsv, protocol-tagged
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

ROOT = str(Path(__file__).resolve().parents[1])
rows=list(csv.DictReader(open(f"{ROOT}/scripts_plots/results_matrix_ps.csv")))
def val(split,arm_kind):
    d={}
    for r in rows:
        if r["split"]==split and r["arm"]==arm_kind and r["ps_spearman_T10"]:
            d[r["model"]]=float(r["ps_spearman_T10"])
    return d
CORE=["Ankh-large","ProstT5","SaProt","ESM2-3B","ESM-C 600M","ESM-C 6B"]
# `split` values as results_matrix.py writes them today. The three rungs are all S1102_filtered
# (1100 mutations) — this figure is the early ladder, not the full-SKEMPI one, so the 2026-08-04
# FoldX coverage step (87.8% -> 99.2%, single-point full-SKEMPI only) does not touch it. The names
# below were "leaky (S1102)" / "by-complex" / "clustered" until results_matrix.py prefixed the
# rungs with their dataset; unprefixed they silently matched nothing and every lookup raised.
splits=["S1102 leaky (per-mut CV)","S1102 by-complex","S1102 clustered"]
xlab=["leaky\n(S1102)","by-complex\n(whole PDB)","clustered\n(mmseqs ≤60%)"]

base={s:val(s,"base") for s in splits}
fxs ={s:val(s,"fx_scalar") for s in splits}
fxm ={s:val(s,"fx_mlp") for s in splits}
# leaky FoldX not in this CSV -> use the rescore_perstructure leaky foldxmlp mean (~0.54) as anchor
leaky_fx_mean=0.54

fig,(axA,axB)=plt.subplots(1,2,figsize=(15.5,6.2),gridspec_kw={"width_ratios":[0.9,1.3]})

# ---- Panel A: ladder ----
x=[0,1,2]
# individual base lines (light)
for m in CORE:
    ys=[base[s].get(m,np.nan) for s in splits]
    axA.plot(x,ys,color="#2b6cb0",alpha=0.25,lw=1,marker="o",ms=3,zorder=1)
# mean base / FoldX
base_mean=[np.nanmean([base[s].get(m,np.nan) for m in CORE]) for s in splits]
fxs_mean =[leaky_fx_mean]+[np.nanmean([fxs[s].get(m,np.nan) for m in CORE]) for s in splits[1:]]
axA.plot(x,base_mean,color="#2b6cb0",lw=3.5,marker="o",ms=9,label="base (PLM only) — mean",zorder=3)
axA.plot(x,fxs_mean,color="#dd6b20",lw=3.5,marker="s",ms=9,label="+ FoldX (scalar) — mean",zorder=3)
# fill the widening gap
axA.fill_between(x,base_mean,fxs_mean,color="#dd6b20",alpha=0.10,zorder=0)
for xi,b,f in zip(x,base_mean,fxs_mean):
    axA.annotate(f"Δ{f-b:+.2f}",(xi,(b+f)/2),ha="center",va="center",fontsize=9,color="#9c4221",
                 bbox=dict(boxstyle="round,pad=0.15",fc="white",ec="#dd6b20",alpha=0.85))
# ⚠ METRIC MISMATCH. Both panels' axes are PER-STRUCTURE Spearman, but ProtBFF reports the
# clustered split *pooled* only -- 0.477 pooled Spearman, 0.514 pooled Pearson. plot_ppS_scaling.py
# and data/benchmarks/README.md both state that the clustered panel therefore has no per-structure
# comparator. This line is drawn anyway, so the annotation now names the metric instead of leaving
# a bare "~0.48" on a per-structure axis for a reader to mistake. Whether to keep the line at all
# is open -- see OPEN_QUESTIONS.md, "Reading the comparator figures".
_PROTBFF_POOLED = _bench.comparators("pooled_spearman", "clustered_id60", only=["ProtBFF"])["ProtBFF"]
axA.axhline(_PROTBFF_POOLED,ls=":",color="#666",lw=1.2)
axA.annotate(f"ProtBFF clustered anchor {_PROTBFF_POOLED:.3f} (POOLED Spearman, not per-structure)",
             (0.55,_PROTBFF_POOLED+0.005),ha="left",va="bottom",fontsize=8.5,color="#666")
axA.set_xticks(x); axA.set_xticklabels(xlab,fontsize=10)
axA.set_ylabel("per-structure Spearman ρ  (T≥10)",fontsize=11)
axA.set_ylim(-0.02,0.62); axA.grid(True,alpha=0.15)
axA.set_title("A · Base collapses out-of-family; FoldX holds → the gap WIDENS",fontsize=12,fontweight="bold")
axA.legend(loc="upper right",fontsize=9.5,framealpha=0.95)
axA.annotate("base: 0.39→0.33→0.20\n(ProstT5 → 0.02)",(0.02,0.05),xycoords="axes fraction",
             fontsize=8.5,color="#2b6cb0")

# ---- Panel B: clustered ρ = base + FoldX-lift (stacked); scalar (top) vs MLP (bottom) per PLM ----
from matplotlib.patches import Patch
plms=["ESM-C 6B","ESM-C 600M","ESM2-3B","SaProt","Ankh-large","ProstT5"]
_CLUST=splits[2]   # same rung name as Panel A's third point; do not restate the literal
cb=val(_CLUST,"base"); cs=val(_CLUST,"fx_scalar"); cm=val(_CLUST,"fx_mlp")
y=np.arange(len(plms)); h=0.38
GREY="#b8c2cc"; ORA="#dd6b20"; BLU="#3182ce"
for i,p in enumerate(plms):
    b,fs,fm=cb[p],cs[p],cm[p]
    # scalar bar (upper), MLP bar (lower): base segment (grey) + lift segment (colored) → total = final ρ
    axB.barh(y[i]+h/2,b,h,color=GREY,edgecolor="white",lw=0.5)
    axB.barh(y[i]+h/2,fs-b,h,left=b,color=ORA,edgecolor="white",lw=0.5)
    axB.barh(y[i]-h/2,b,h,color=GREY,edgecolor="white",lw=0.5)
    axB.barh(y[i]-h/2,fm-b,h,left=b,color=BLU,edgecolor="white",lw=0.5)
    # lift value centered in the colored segment only if it's wide enough; final ρ (+lift) at the tip
    if fs-b>=0.05: axB.annotate(f"+{fs-b:.2f}",(b+(fs-b)/2,y[i]+h/2),va="center",ha="center",fontsize=7.5,color="white",fontweight="bold")
    if fm-b>=0.05: axB.annotate(f"+{fm-b:.2f}",(b+(fm-b)/2,y[i]-h/2),va="center",ha="center",fontsize=7.5,color="white",fontweight="bold")
    axB.annotate(f"{fs:.2f} (+{fs-b:.2f})",(fs+0.008,y[i]+h/2),va="center",ha="left",fontsize=7.8,color="#9c4221",fontweight="bold")
    axB.annotate(f"{fm:.2f} (+{fm-b:.2f})",(fm+0.008,y[i]-h/2),va="center",ha="left",fontsize=7.8,color="#2c5282",fontweight="bold")
# base value labelled once (guide), on ESM-C 6B's grey base
axB.annotate("base",(cb["ESM-C 6B"]/2,y[0]+h/2),va="center",ha="center",fontsize=7,color="#3d4852")
axB.axvline(_PROTBFF_POOLED,ls=":",color="#666",lw=1.2)
axB.annotate(f"ProtBFF {_PROTBFF_POOLED:.3f} (pooled)",xy=(_PROTBFF_POOLED,0.965),xycoords=("data","axes fraction"),
             ha="center",va="top",fontsize=8,color="#666",
             bbox=dict(boxstyle="round,pad=0.2",fc="white",ec="none",alpha=0.9))
axB.set_yticks(y); axB.set_yticklabels(plms,fontsize=10); axB.invert_yaxis()
axB.set_xlabel("per-structure Spearman ρ (clustered)  =  base  ⊕  FoldX lift",fontsize=11)
axB.set_xlim(0,0.70); axB.grid(True,axis="x",alpha=0.15)
axB.set_title("B · Clustered ρ decomposed: base (grey) + FoldX lift → final;\nscalar (top) vs 12-term MLP (bottom)",fontsize=11.5,fontweight="bold")
axB.legend(handles=[Patch(fc=GREY,label="base ρ (PLM only)"),Patch(fc=ORA,label="+ FoldX scalar lift"),
                    Patch(fc=BLU,label="+ FoldX 12-term MLP lift")],loc="lower right",fontsize=9,framealpha=0.95)

fig.suptitle("MuLAN + FoldX on the leakage-controlled SKEMPI ladder — per-structure Spearman",
             fontsize=13,y=0.995)
fig.tight_layout(rect=[0,0,1,0.96])
out=f"{ROOT}/scripts_plots/generalization_ladder.png"
fig.savefig(out,dpi=140); print("wrote",out)
