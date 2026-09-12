#!/usr/bin/env python3
"""Is FoldX stochastic, or does it just depend on batch context?

`foldx_antisymmetry.py --control` re-runs one mutation ALONE and finds the stored value reproduced
exactly only ~25% of the time. That is ambiguous between two very different causes:

  (a) BuildModel is stochastic -- side-chain sampling that differs run to run;
  (b) BuildModel is deterministic but context-dependent -- the canonical run passed every mutation
      of a complex in one `individual_list.txt`, and a single-mutation re-run is a different call.

The distinction matters. Under (a) the FoldX channel carries irreducible noise and always did, which
is a property of every FoldX number in this repository. Under (b) the stored values are perfectly
reproducible and only the single-mutation control was measuring the wrong thing, which would mean
the noise term in the variance decomposition is overstated.

This re-runs the ORIGINAL batch: the whole `individual_list.txt` for a complex, in one BuildModel
call, exactly as `build_ddg.py` did, then AnalyseComplex over all of it. Same command, same input,
same batch. Any deviation left is stochasticity with nothing else it could be.

Usage:  ./.venv/bin/python experiments/beyond_foldx/foldx_batch_control.py [--complexes N]
Output: experiments/beyond_foldx/foldx_batch_control.csv
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FX = os.environ.get("FOLDX_BIN", os.path.join(ROOT, "..", "foldx5_MacSilicon_0", "foldx5.1_20261231"))
WORK = os.path.join(ROOT, "scratch", "foldx_skempi_full", "work")
RESULTS = os.path.join(ROOT, "scratch", "foldx_skempi_full", "results")
SKEMPI = os.path.join(ROOT, "scratch", "skempi_v2.csv")
PROBE = os.path.join(ROOT, "scratch", "foldx_batchctl")
OUT = os.path.join(HERE, "foldx_batch_control.csv")
TERM = "Interaction Energy"


def run(cmd, cwd, log):
    with open(log, "a") as fh:
        subprocess.run(cmd, shell=True, check=False, cwd=cwd, stdout=fh, stderr=fh)


def parse_interaction(path):
    lines = open(path).readlines()
    hdr = next(l for l in lines if l.startswith("Pdb\t")).rstrip("\n").split("\t")
    data = next(l for l in lines if l.split("\t")[0].endswith(".pdb")).rstrip("\n").split("\t")
    return dict(zip(hdr, data))


def groups():
    out = {}
    with open(SKEMPI, encoding="latin-1") as fh:
        for r in csv.DictReader(fh, delimiter=";"):
            p = r["#Pdb"].split("_")
            if len(p) >= 3:
                out.setdefault(p[0], f"{p[1]},{p[2]}")
    return out


def stored(pdb):
    """i (1-based, individual_list order) -> stored Interaction Energy."""
    p = os.path.join(RESULTS, f"{pdb}.json")
    if not os.path.exists(p):
        return {}
    d = json.load(open(p))["muts"]
    il = os.path.join(WORK, pdb, "individual_list.txt")
    muts = [l.strip().rstrip(";") for l in open(il) if l.strip()]
    byc = {v.get("cleaned", k): v for k, v in d.items()}
    return {i: byc[m][TERM] for i, m in enumerate(muts, 1)
            if m in byc and TERM in byc[m]}


def one_complex(args):
    pdb, garg = args
    src = os.path.join(WORK, pdb)
    st = stored(pdb)
    if not st:
        return []
    wd = os.path.join(PROBE, pdb)
    os.makedirs(wd, exist_ok=True)
    log = os.path.join(wd, "foldx.log")
    shutil.copy(os.path.join(src, f"{pdb}_Repair.pdb"), os.path.join(wd, f"{pdb}_Repair.pdb"))
    shutil.copy(os.path.join(src, "individual_list.txt"), os.path.join(wd, "individual_list.txt"))
    n = sum(1 for _ in open(os.path.join(wd, "individual_list.txt")))
    # identical to build_ddg.py's call
    run(f'"{FX}" --command=BuildModel --pdb={pdb}_Repair.pdb '
        f'--mutant-file=individual_list.txt --numberOfRuns=1', wd, log)
    mut = [f"{pdb}_Repair_{i}.pdb" for i in range(1, n + 1)]
    wt = [f"WT_{pdb}_Repair_{i}.pdb" for i in range(1, n + 1)]
    if not all(os.path.exists(os.path.join(wd, p)) for p in mut + wt):
        return []
    with open(os.path.join(wd, "pdblist.txt"), "w") as fh:
        fh.write("\n".join(mut + wt) + "\n")
    run(f'"{FX}" --command=AnalyseComplex --pdb-list=pdblist.txt '
        f'--analyseComplexChains={garg}', wd, log)
    rows = []
    for i in range(1, n + 1):
        mf = os.path.join(wd, f"Interaction_{pdb}_Repair_{i}_AC.fxout")
        wf = os.path.join(wd, f"Interaction_WT_{pdb}_Repair_{i}_AC.fxout")
        if not (os.path.exists(mf) and os.path.exists(wf)) or i not in st:
            continue
        again = round(float(parse_interaction(mf)[TERM]) - float(parse_interaction(wf)[TERM]), 4)
        rows.append(dict(pdb=pdb, i=i, stored=st[i], rerun=again, dev=round(again - st[i], 4)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--complexes", type=int, default=25)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    g = groups()
    pdbs = [os.path.basename(d) for d in sorted(glob.glob(os.path.join(WORK, "*")))
            if os.path.exists(os.path.join(d, "individual_list.txt")) and os.path.basename(d) in g]
    # stride, so the sample spans the set rather than its alphabetical head
    step = max(len(pdbs) / args.complexes, 1)
    sel = [pdbs[int(k * step)] for k in range(min(args.complexes, len(pdbs)))]
    print(f"batch-re-running {len(sel)} complexes with {args.jobs} workers", flush=True)
    os.makedirs(PROBE, exist_ok=True)
    rows = []
    with Pool(args.jobs) as pool:
        for r in pool.imap_unordered(one_complex, [(p, g[p]) for p in sel]):
            rows.extend(r)
            print(f"  {len(rows)} mutations recorded", flush=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["pdb", "i", "stored", "rerun", "dev"])
        w.writeheader()
        w.writerows(rows)
    import numpy as np
    d = np.array([r["dev"] for r in rows])
    print(f"\n[wrote] {OUT}  ({len(rows)} rows)")
    print(f"exact reproductions: {int((np.abs(d) < 1e-9).sum())}/{len(d)} "
          f"({(np.abs(d) < 1e-9).mean():.0%})")
    print(f"deviation: mean {d.mean():+.4f}  sd {d.std():.4f}  max|dev| {np.abs(d).max():.3f}")
    print("\nIf this reproduces exactly, the single-mutation control was measuring batch context,")
    print("not stochasticity, and the noise term in the variance decomposition is overstated.")


if __name__ == "__main__":
    main()
