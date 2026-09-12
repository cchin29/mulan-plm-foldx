#!/usr/bin/env python3
"""Cross-arm mispredicted-ddG analysis on the 300ep/patience30 BALANCED splitter.
Arms: base, aug, foldx-phase2 (foldxmlp).  PLMs with all 3 balanced arms: esm2, esmc600m,
prostt5, saprot, esmc6b.  Pools the 10 disjoint OOF folds -> full S1102 (1100) and hunts
hotspots across ALL folds."""
import os
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = str(Path(__file__).resolve().parents[1])
SPLIT = f"{ROOT}/scratch/foldx_s1102/splits_balanced_foldxdec"
RES   = f"{ROOT}/scratch/results/embedding_sweep_balanced"
INC   = f"{ROOT}/scratch/incoming_esmc6b_20260714"
NFOLD = 10

# path template per (plm,arm) -> file with {f}. embedding_sweep PLMs share one layout; esmc6b differs.
def std_paths(plm):
    return {"base":  f"{RES}/{plm}/fold_{{f}}/training_run/test_predictions.tsv",
            "aug":   f"{RES}/{plm}_aug/fold_{{f}}/training_run/test_predictions.tsv",
            "foldx": f"{RES}/{plm}_foldxmlp/fold_{{f}}/training_run/test_predictions.tsv"}
PATHS = {p: std_paths(p) for p in ["esm2","esmc600m","prostt5","saprot"]}
# Ankh: balanced base preds live in cv10_ankh_converge (fork-default/balanced splitter, matches
# benchmarks_folds_300ep_balanced.csv fold0=0.719983 to the digit); aug/foldx in the sweep dir.
PATHS["ankh"] = {
    "base":  f"{ROOT}/scratch/results/cv10_ankh_converge/fold_{{f}}/training_run/test_predictions.tsv",
    "aug":   f"{RES}/ankh_aug/fold_{{f}}/training_run/test_predictions.tsv",
    "foldx": f"{RES}/ankh_foldxmlp/fold_{{f}}/training_run/test_predictions.tsv"}
PATHS["esmc6b"] = {
    "base":  f"{INC}/results/base/S1102/fold_{{f}}/training_run/test_predictions.tsv",
    "aug":   f"{INC}/results/aug/S1102/fold_{{f}}/training_run/test_predictions.tsv",
    "foldx": f"{INC}/data/S1102/foldx/cv10_esmc6b_balanced/foldx_mlp/fold_{{f}}/training_run/test_predictions.tsv"}
PLMS = list(PATHS); ARMS = ["base","aug","foldx"]

def mut_pos(m): c=m[1:-1]; return int(c[1:]), c[0]
def key(pdb,mut): return (pdb,mut)

# truth + fold + 12 foldx terms
truth={}
for f in range(NFOLD):
    for ln in open(f"{SPLIT}/fold_{f}/S1102_filtered_test.tsv"):
        p=ln.rstrip("\n").split("\t")
        if len(p)<4: continue
        pdb=p[0].split("_")[0]; mut=p[2]; pos,ch=mut_pos(mut)
        truth[key(pdb,mut)]=dict(true=float(p[3]),fold=f,chain=ch,pos=pos,wt=mut[0],aa=mut[-1],
                                 foldx=[float(x) for x in p[4:16]] if len(p)>=16 else None,pdb=pdb)
print(f"pooled truth rows: {len(truth)}")

preds=defaultdict(dict); missing=defaultdict(int)
for plm in PLMS:
    for arm in ARMS:
        for f in range(NFOLD):
            fp=PATHS[plm][arm].format(f=f)
            if not os.path.exists(fp): missing[(plm,arm)]+=1; continue
            for ln in open(fp):
                p=ln.rstrip("\n").split("\t")
                if len(p)<4: continue
                preds[(plm,arm)][key(p[0].split("_")[0],p[2])]=float(p[3])
for k,v in missing.items():
    if v: print("MISSING folds",k,v)
for plm in PLMS:
    print("  "+plm.ljust(10)+" "+" ".join(f"{a}={len(preds[(plm,a)])}" for a in ARMS))

keys=[k for k in truth if all(k in preds[(plm,a)] for plm in PLMS for a in ARMS)]
print(f"\naligned keys across {len(PLMS)} PLMs x 3 arms: {len(keys)}")

def pcc(a,b):
    a,b=np.array(a),np.array(b)
    return float(np.corrcoef(a,b)[0,1]) if len(a)>2 and a.std()>0 and b.std()>0 else float('nan')

resid={(plm,a):{k:truth[k]['true']-preds[(plm,a)][k] for k in keys} for plm in PLMS for a in ARMS}
arm_abserr={a:{k:np.mean([abs(resid[(plm,a)][k]) for plm in PLMS]) for k in keys} for a in ARMS}
arm_signed={a:{k:np.mean([resid[(plm,a)][k] for plm in PLMS]) for k in keys} for a in ARMS}
grand={k:np.mean([abs(resid[(plm,a)][k]) for plm in PLMS for a in ARMS]) for k in keys}
grand_signed={k:np.mean([resid[(plm,a)][k] for plm in PLMS for a in ARMS]) for k in keys}

out=[]
def w(s=""): out.append(s); print(s)

w("\n======== POOLED OOF PCC (all 1100) per PLM x arm ========")
w(f"{'PLM':>9} | {'base':>7} {'aug':>7} {'foldx':>7} | dPCC(foldx-base)")
for plm in PLMS:
    r=[pcc([truth[k]['true'] for k in keys],[preds[(plm,a)][k] for k in keys]) for a in ARMS]
    w(f"{plm:>9} | {r[0]:7.3f} {r[1]:7.3f} {r[2]:7.3f} |  {r[2]-r[0]:+.3f}")
w("  --- 4/5-PLM consensus prediction (mean pred) ---")
for a in ARMS:
    mp=[np.mean([preds[(plm,a)][k] for plm in PLMS]) for k in keys]
    w(f"    {a:>6}: {pcc([truth[k]['true'] for k in keys],mp):.3f}")

w("\n======== PER-FOLD PCC BY ARM (mean of PLMs) ========")
w(f"{'fold':>4} | {'base':>7} {'aug':>7} {'foldx':>7} | n")
fold_pcc=defaultdict(dict)
for f in range(NFOLD):
    fk=[k for k in keys if truth[k]['fold']==f]; row=[]
    for a in ARMS:
        vals=[pcc([truth[k]['true'] for k in fk],[preds[(plm,a)][k] for k in fk]) for plm in PLMS]
        fold_pcc[f][a]=np.nanmean(vals); row.append(np.nanmean(vals))
    w(f"{f:>4} | {row[0]:7.3f} {row[1]:7.3f} {row[2]:7.3f} | {len(fk)}")
order=sorted(range(NFOLD),key=lambda f:fold_pcc[f]['base'])
w("weakest->strongest (base): "+", ".join(f"f{f}={fold_pcc[f]['base']:.3f}" for f in order))

w("\n======== TOP 25 CROSS-MODEL HOTSPOTS (grand mean |err|) ========")
w(f"{'pdb':>5} {'mut':>7} {'fold':>4} {'true':>7} | {'base':>6} {'aug':>6} {'foldx':>6} | {'|err|':>6} {'foldx<base':>10}")
ranked=sorted(keys,key=lambda k:-grand[k])
for k in ranked[:25]:
    pdb,mut=k; t=truth[k]['true']
    pb,pa,pf=[np.mean([preds[(plm,a)][k] for plm in PLMS]) for a in ARMS]
    w(f"{pdb:>5} {mut:>7} {truth[k]['fold']:>4} {t:>7.2f} | {pb:6.2f} {pa:6.2f} {pf:6.2f} | {grand[k]:6.2f} {str(abs(t-pf)<abs(t-pb)):>10}")

top40=ranked[:40]
w("\n======== WORST-40 BY FOLD ========")
fc=defaultdict(int)
for k in top40: fc[truth[k]['fold']]+=1
w("  "+"  ".join(f"f{f}:{fc[f]}" for f in range(NFOLD)))

w("\n======== HOTSPOT CONCENTRATION BY COMPLEX (top-40) ========")
byc=defaultdict(list)
for k in top40: byc[k[0]].append(k[1])
for pdb,ms in sorted(byc.items(),key=lambda x:-len(x[1])):
    if len(ms)>=2: w(f"  {pdb}: {len(ms)}  {ms}")

w("\n======== HARD COMPLEXES (mean grand|err|, >=4 muts) ========")
ce=defaultdict(list); cf=defaultdict(set)
for k in keys: ce[k[0]].append(grand[k]); cf[k[0]].add(truth[k]['fold'])
for pdb,me,n,fs in sorted([(p,np.mean(v),len(v),sorted(cf[p])) for p,v in ce.items() if len(v)>=4],key=lambda x:-x[1])[:15]:
    w(f"  {pdb}: mean|err|={me:.2f} n={n} folds={fs}")

w("\n======== ARM-vs-ARM & CROSS-PLM RESIDUAL CORR ========")
for i in range(3):
    for j in range(i+1,3):
        a=[arm_signed[ARMS[i]][k] for k in keys]; b=[arm_signed[ARMS[j]][k] for k in keys]
        w(f"  {ARMS[i]:>6} vs {ARMS[j]:>6}: r={pcc(a,b):.3f}")
w("  cross-PLM (base):")
for i in range(len(PLMS)):
    for j in range(i+1,len(PLMS)):
        a=[resid[(PLMS[i],'base')][k] for k in keys]; b=[resid[(PLMS[j],'base')][k] for k in keys]
        w(f"    {PLMS[i]:>9} vs {PLMS[j]:>9}: r={pcc(a,b):.3f}")

w("\n======== FOLDX-PHASE2 ON HARDEST (worst-N by base |err|) ========")
br=sorted(keys,key=lambda k:-arm_abserr['base'][k])
for N in (25,50,100,200):
    sub=br[:N]
    mb,ma,mf=[np.mean([arm_abserr[a][k] for k in sub]) for a in ARMS]
    fw=np.mean([arm_abserr['foldx'][k]<arm_abserr['base'][k] for k in sub])
    aw=np.mean([arm_abserr['aug'][k]<arm_abserr['base'][k] for k in sub])
    w(f"  worst-{N:>3}: MAE base={mb:.2f} aug={ma:.2f} foldx={mf:.2f} | foldx<base {fw*100:3.0f}% aug<base {aw*100:3.0f}%")

w("\n======== DIRECTION & TYPE (worst-40) ========")
under=sum(1 for k in top40 if grand_signed[k]>0)
w(f"  underpredicted {under}/40 ; mean|true| worst40={np.mean([abs(truth[k]['true']) for k in top40]):.2f} vs global {np.mean([abs(truth[k]['true']) for k in keys]):.2f}")
def ala(k): return truth[k]['aa']=='A'
def big(k): return truth[k]['wt'] in set('GASCTDNP') and truth[k]['aa'] in set('WYFRKH')
w(f"  worst40: ->Ala {np.mean([ala(k) for k in top40])*100:.0f}% gain-of-bulk {np.mean([big(k) for k in top40])*100:.0f}%")
w(f"  all    : ->Ala {np.mean([ala(k) for k in keys])*100:.0f}% gain-of-bulk {np.mean([big(k) for k in keys])*100:.0f}%")

# FoldX-term vs base residual
TERMN=["Interaction(total)","BB_Hbond","SC_Hbond","VdW","Electrostatics","Solv_Polar","Solv_Hphob","VdW_clash","ent_SC","ent_MC","tors_clash","bb_clash"]
resb={k:truth[k]['true']-np.mean([preds[(plm,'base')][k] for plm in PLMS]) for k in keys}
w("\n======== FoldX-term vs BASE consensus residual (global / worst-100) ========")
w100=sorted(keys,key=lambda k:-abs(resb[k]))[:100]
for j,nm in enumerate(TERMN):
    if truth[keys[0]]['foldx'] is None: break
    g=pcc([truth[k]['foldx'][j] for k in keys],[resb[k] for k in keys])
    h=pcc([truth[k]['foldx'][j] for k in w100],[resb[k] for k in w100])
    w(f"    {nm:>18}: {g:6.3f} / {h:6.3f}")

json.dump({"n_aligned":len(keys),"plms":PLMS,
  "pooled_pcc":{plm:{a:pcc([truth[k]['true'] for k in keys],[preds[(plm,a)][k] for k in keys]) for a in ARMS} for plm in PLMS},
  "fold_pcc":{str(f):fold_pcc[f] for f in range(NFOLD)},
  "top40":[{"pdb":k[0],"mut":k[1],"fold":truth[k]['fold'],"true":truth[k]['true'],
            "base":float(np.mean([preds[(p,'base')][k] for p in PLMS])),
            "aug":float(np.mean([preds[(p,'aug')][k] for p in PLMS])),
            "foldx":float(np.mean([preds[(p,'foldx')][k] for p in PLMS])),
            "grand_abserr":float(grand[k])} for k in ranked[:40]]},
  open(f"{ROOT}/scratch/outputs/hotspots_table.json","w"),indent=1)
open(f"{ROOT}/scratch/outputs/hotspot_report.txt","w").write("\n".join(out))
print("\nwrote json+report")
