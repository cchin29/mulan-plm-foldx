#!/usr/bin/env python3
"""Bootstrap 95% CIs over complexes for per-structure Spearman (T>=10), plus paired contrasts.
Complements rescore.py: the per-structure mean is over only ~24 complexes, so arm gaps need CIs.
Dependency-free (numpy only); resamples the complex set (cluster bootstrap), seed-fixed."""
import os
import glob
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = str(Path(__file__).resolve().parents[2])
SPLIT=f"{ROOT}/scratch/foldx_s1102/splits_balanced_foldxdec"
RES=f"{ROOT}/scratch/results/embedding_sweep_balanced"
INC=f"{ROOT}/scratch/incoming_esmc6b_20260714"
T=10; B=10000; SEED=0

PLMS=["esm2","esmc600m","prostt5","saprot","ankh","esmc6b"]
def base_path(p):
    if p=="ankh": return f"{ROOT}/scratch/results/cv10_ankh_converge/fold_{{f}}/training_run/test_predictions.tsv"
    if p=="esmc6b": return f"{INC}/results/base/S1102/fold_{{f}}/training_run/test_predictions.tsv"
    return f"{RES}/{p}/fold_{{f}}/training_run/test_predictions.tsv"
def aug_path(p):
    if p=="esmc6b": return f"{INC}/results/aug/S1102/fold_{{f}}/training_run/test_predictions.tsv"
    return f"{RES}/{p}_aug/fold_{{f}}/training_run/test_predictions.tsv"
def fx_path(p):
    if p=="esmc6b": return f"{INC}/data/S1102/foldx/cv10_esmc6b_balanced/foldx_mlp/fold_{{f}}/training_run/test_predictions.tsv"
    return f"{RES}/{p}_foldxmlp/fold_{{f}}/training_run/test_predictions.tsv"
PATHS={}
for p in PLMS:
    PATHS[f"{p}_base"]=base_path(p); PATHS[f"{p}_aug"]=aug_path(p); PATHS[f"{p}_foldxmlp"]=fx_path(p)
PATHS["mint_complex"]=f"{ROOT}/scratch/results/mint_a2/mint/fold_{{f}}/training_run/test_predictions.tsv"
PATHS["mint_mono"]=f"{ROOT}/scratch/results/mint_mono/mint_mono/fold_{{f}}/training_run/test_predictions.tsv"

truth={}; comp={}
for f in range(10):
    for ln in open(f"{SPLIT}/fold_{f}/S1102_filtered_test.tsv"):
        a=ln.rstrip("\n").split("\t")
        if len(a)<4: continue
        pdb=a[0].split("_")[0]; truth[(pdb,a[2])]=float(a[3]); comp[(pdb,a[2])]=pdb
def load(tmpl):
    d={}
    for f in range(10):
        fp=tmpl.format(f=f)
        if not os.path.exists(fp): return None
        for ln in open(fp):
            a=ln.rstrip("\n").split("\t")
            if len(a)>=4: d[(a[0].split("_")[0],a[2])]=float(a[3])
    return d
preds={k:load(v) for k,v in PATHS.items()}
# consensus
keys6=[k for k in truth if all(k in preds[f"{p}_base"] for p in PLMS)]
for arm in ["base","aug","foldxmlp"]:
    d={}
    for k in truth:
        vals=[preds[f"{p}_{arm}"].get(k) for p in PLMS]
        if all(v is not None for v in vals): d[k]=float(np.mean(vals))
    preds[f"consensus_{arm}"]=d
ARMS=list(preds)

def spearman(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    if len(x)<3 or x.std()==0 or y.std()==0: return np.nan
    rx=np.argsort(np.argsort(x)).astype(float); ry=np.argsort(np.argsort(y)).astype(float)
    # average ranks for ties
    def avgrank(v):
        order=np.argsort(v,kind="mergesort"); r=np.empty(len(v)); r[order]=np.arange(len(v))
        # handle ties -> mean rank
        vals=v[order]; i=0
        while i<len(v):
            j=i
            while j+1<len(v) and vals[j+1]==vals[i]: j+=1
            if j>i:
                m=(r[order[i:j+1]]).mean(); r[order[i:j+1]]=m
            i=j+1
        return r
    rx=avgrank(x); ry=avgrank(y)
    return float(np.corrcoef(rx,ry)[0,1])

# per-complex spearman per arm over complexes with >=T muts
comps=sorted(set(comp.values()))
comp_keys={c:[k for k in truth if comp[k]==c] for c in comps}
qual=[c for c in comps if len(comp_keys[c])>=T]
percx={}  # arm -> {complex: spearman}
for arm in ARMS:
    dd={}
    for c in qual:
        ks=[k for k in comp_keys[c] if k in preds[arm]]
        if len(ks)<T: continue
        s=spearman([truth[k] for k in ks],[preds[arm][k] for k in ks])
        if not np.isnan(s): dd[c]=s
    percx[arm]=dd
common=sorted(set.intersection(*[set(percx[a].keys()) for a in ARMS]))
print(f"qualifying complexes (>= {T} muts): {len(qual)}; common across arms: {len(common)}")
M={a:np.array([percx[a][c] for c in common]) for a in ARMS}
nC=len(common)

rng=np.random.default_rng(SEED)
BOOT=rng.integers(0,nC,size=(B,nC))  # shared resample indices -> paired
def ci(vec):
    bs=vec[BOOT].mean(axis=1)
    return vec.mean(), np.percentile(bs,2.5), np.percentile(bs,97.5)

out=[]; w=lambda s: (out.append(s), print(s))
w("\n==== per-structure Spearman (T>=10) with 95% cluster-bootstrap CI ====")
w(f"{'arm':>20} {'mean':>7}  [{'2.5%':>6},{'97.5%':>6}]")
order=sorted(ARMS,key=lambda a:-M[a].mean())
for a in order:
    m,lo,hi=ci(M[a]); w(f"{a:>20} {m:7.3f}  [{lo:6.3f},{hi:6.3f}]")

def paired(a,b):
    d=M[a]-M[b]; bs=d[BOOT].mean(axis=1)
    return d.mean(), np.percentile(bs,2.5), np.percentile(bs,97.5), float((bs>0).mean())
w("\n==== paired contrasts (Δ = A − B, shared complex resample) ====")
contrasts=[(f"{p}_foldxmlp",f"{p}_base") for p in PLMS] + [
    ("consensus_foldxmlp","consensus_base"),
    ("consensus_foldxmlp","esmc6b_foldxmlp"),   # top ensemble vs top single
    ("esmc6b_foldxmlp","esmc6b_base"),
    ("mint_complex","consensus_base"),
    ("mint_complex","mint_mono"),
    ("esmc6b_base","mint_complex"),
]
w(f"{'contrast':>34} {'dmean':>7}  [{'2.5%':>6},{'97.5%':>6}]  P(>0)")
for a,b in contrasts:
    dm,lo,hi,pg=paired(a,b); w(f"{a+' - '+b:>34} {dm:7.3f}  [{lo:6.3f},{hi:6.3f}]  {pg:.3f}")

open(f"{ROOT}/experiments/rescore_perstructure/bootstrap_ci.txt","w").write("\n".join(out))
print("\nwrote bootstrap_ci.txt")
