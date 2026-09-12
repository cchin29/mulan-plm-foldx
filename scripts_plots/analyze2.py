#!/usr/bin/env python3
"""Pass 2: pooled PCC per arm, FoldX-term diagnostics on residuals, complex names."""
import os
import csv
import numpy as np
from pathlib import Path
from collections import defaultdict
ROOT = str(Path(__file__).resolve().parents[1])
SPLIT=f"{ROOT}/scratch/foldx_s1102/splits_balanced_foldxdec"
RES=f"{ROOT}/scratch/results/embedding_sweep_balanced"
PLMS=["esm2","esmc600m","prostt5","saprot"]; ARMS={"base":"{plm}","aug":"{plm}_aug","foldx":"{plm}_foldxmlp"}
def mp(m): c=m[1:-1]; return int(c[1:])
truth={}
for f in range(10):
    for ln in open(f"{SPLIT}/fold_{f}/S1102_filtered_test.tsv"):
        p=ln.rstrip("\n").split("\t")
        if len(p)<16: continue
        pdb=p[0].split("_")[0]; mut=p[2]
        truth[(pdb,mut)]=dict(true=float(p[3]),fold=f,terms=[float(x) for x in p[4:16]])
preds=defaultdict(dict)
for plm in PLMS:
    for arm,t in ARMS.items():
        for f in range(10):
            fp=f"{RES}/{t.format(plm=plm)}/fold_{f}/training_run/test_predictions.tsv"
            if not os.path.exists(fp): continue
            for ln in open(fp):
                p=ln.rstrip("\n").split("\t")
                if len(p)<4: continue
                preds[(plm,arm)][(p[0].split("_")[0],p[2])]=float(p[3])
keys=[k for k in truth if all(k in preds[(plm,arm)] for plm in PLMS for arm in ARMS)]
def pcc(a,b):
    a,b=np.array(a),np.array(b)
    return float(np.corrcoef(a,b)[0,1]) if len(a)>2 and a.std()>0 and b.std()>0 else float('nan')

out=[]
def w(s=""): out.append(s); print(s)

# pooled PCC per arm (all 1100 OOF, per PLM then averaged) + overall mean-of-PLM prediction
w("======== POOLED OOF PCC (all 1100) ========")
w(f"{'arm':>6} | " + " ".join(f"{p:>9}" for p in PLMS) + " |  4-PLM-mean-pred")
for arm in ARMS:
    per=[]
    for plm in PLMS:
        per.append(pcc([truth[k]['true'] for k in keys],[preds[(plm,arm)][k] for k in keys]))
    tvals=[truth[k]['true'] for k in keys]
    mpred=[np.mean([preds[(plm,arm)][k] for plm in PLMS]) for k in keys]
    w(f"{arm:>6} | " + " ".join(f"{x:9.3f}" for x in per) + f" |  {pcc(tvals,mpred):.3f}")

# consensus residual (base) & FoldX-term correlations
res_base={k: truth[k]['true']-np.mean([preds[(plm,'base')][k] for plm in PLMS]) for k in keys}
TERMN=["Interaction(total)","BB_Hbond","SC_Hbond","VdW","Electrostatics","Solv_Polar","Solv_Hphob","VdW_clash","ent_SC","ent_MC","tors_clash","bb_clash"]
w("\n======== FoldX-term vs BASE consensus residual (does physics flag the misses?) ========")
w("  corr(term, residual)  [+ = term tracks under-prediction]   global / worst-100")
base_rank=sorted(keys,key=lambda k:-abs(res_base[k])); w100=base_rank[:100]
for j,name in enumerate(TERMN):
    g=pcc([truth[k]['terms'][j] for k in keys],[res_base[k] for k in keys])
    h=pcc([truth[k]['terms'][j] for k in w100],[res_base[k] for k in w100])
    w(f"    {name:>18}: {g:6.3f} / {h:6.3f}")

# complex names
name={}
for row in csv.reader(open(f"{ROOT}/scratch/skempi_v2.csv"),delimiter=';'):
    if not row or row[0].startswith('#'):
        header=row; continue
    pdb=row[0].split("_")[0]
    if pdb not in name and len(row)>13:
        name[pdb]=(row[12].strip(),row[13].strip())
HARD=["2O3B","1MAH","2PCC","1CSE","1AK4","2WPT","1JTG","2J0T","1FCC","1BRS","1KTZ","2FTL","1PPF","1R0R","2G2U"]
w("\n======== HARD-COMPLEX IDENTITIES ========")
for pdb in HARD:
    p1,p2=name.get(pdb,("?","?")); w(f"  {pdb}: {p1}  ::  {p2}")
open(f"{ROOT}/scratch/outputs/hotspot_report2.txt","w").write("\n".join(out))
