#!/usr/bin/env python3
"""Is FoldX antisymmetric under mutation reversal?

Executes docs/history/runbooks/RUN_FOLDX_ANTISYMMETRY_PROBE.md. Tier-1 augmentation builds each reverse row by NEGATING
the FoldX terms. That is exact for the experimental label (thermodynamics) but only an assumption
for a ddG computed from a modelled mutant structure. If it is false, every augmented FoldX row
carries a fabricated value, which would degrade the arms reading those columns and leave the base
arm untouched -- the sign split RESULTS.md 24 reports, under a repairable cause rather than an
intrinsic one.

Protocol per sampled forward mutation <wt><chain><pos><mt> on complex <PDB>, index i:

  1. `<PDB>_Repair_<i>.pdb` -- the mutant structure BuildModel already produced -- is the input.
  2. BuildModel the REVERSE mutation <mt><chain><pos><wt> on it.
  3. AnalyseComplex on the resulting pair, with the SAME chain groups the forward run used
     (SKEMPI `#Pdb` = PDB_<g1>_<g2>).
  4. ddG_reverse = IE(reverted) - IE(mutant re-minimised), mirroring the forward convention
     ddG_forward = IE(mutant) - IE(WT re-minimised) exactly.

RepairPDB is deliberately NOT re-run on the mutant. That keeps the reverse calculation closer to
the forward geometry, biasing the result TOWARD antisymmetry -- so a null is weak evidence and a
positive is strong. Stated up front so the direction of the bias is not chosen after the fact.

Usage:
    ./.venv/bin/python experiments/beyond_foldx/foldx_antisymmetry.py --smoke      # 1 complex
    ./.venv/bin/python experiments/beyond_foldx/foldx_antisymmetry.py              # full sample
Output: experiments/beyond_foldx/foldx_antisymmetry.csv (resumable -- completed rows are skipped).
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

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ROOT = os.path.dirname(ROOT)
FX = os.environ.get("FOLDX_BIN", os.path.join(ROOT, "..", "foldx5_MacSilicon_0", "foldx5.1_20261231"))
WORK = os.path.join(ROOT, "scratch", "foldx_skempi_full", "work")
RESULTS = os.path.join(ROOT, "scratch", "foldx_skempi_full", "results")
SKEMPI = os.path.join(ROOT, "scratch", "skempi_v2.csv")
AUG = os.path.join(ROOT, "scratch/foldx_skempi_full/clustered_all/splits_cath_foldx_aug/fold_{f}/skempi_all_train.tsv")
PROBE = os.path.join(ROOT, "scratch", "foldx_antisym")     # scratch: gitignored, safe to wipe
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "foldx_antisymmetry.csv")

TERMS = ["Interaction Energy", "Backbone Hbond", "Sidechain Hbond", "Van der Waals",
         "Electrostatics", "Solvation Polar", "Solvation Hydrophobic",
         "Van der Waals clashes", "entropy sidechain", "entropy mainchain",
         "torsional clash", "backbone clash"]
COLS = ["pdb", "i", "mut_fwd", "mut_rev", "ddg_fwd", "ddg_rev", "sum",
        "chain", "pos", "wt", "mt", "groups", "in_aug", "n_res", "status"]


def run(cmd: str, cwd: str, log: str) -> None:
    with open(log, "a") as fh:
        subprocess.run(cmd, shell=True, check=False, cwd=cwd, stdout=fh, stderr=fh)


def parse_interaction(path: str) -> dict:
    with open(path) as fh:
        lines = fh.readlines()
    hdr = next(l for l in lines if l.startswith("Pdb\t")).rstrip("\n").split("\t")
    data = next(l for l in lines if l.split("\t")[0].endswith(".pdb")).rstrip("\n").split("\t")
    row = dict(zip(hdr, data))
    return {t: float(row[t]) for t in TERMS if t in row}


def skempi_groups() -> dict:
    """pdb -> 'g1,g2', the AnalyseComplex chain groups, from SKEMPI's `#Pdb` column."""
    out = {}
    with open(SKEMPI, encoding="latin-1") as fh:
        for r in csv.DictReader(fh, delimiter=";"):
            parts = r["#Pdb"].split("_")
            if len(parts) >= 3:
                out.setdefault(parts[0], f"{parts[1]},{parts[2]}")
    return out


def augmented_rows() -> set:
    """(pdb, mut) present in an augmented train split -- the rows whose reverse was fabricated."""
    aug = set()
    for f in range(3):
        p = AUG.format(f=f)
        if not os.path.exists(p):
            continue
        for ln in open(p):
            a = ln.rstrip("\n").split("\t")
            if len(a) >= 5:
                aug.add((a[0].split(".")[0], a[2]))
    return aug


def reverse_mut(m: str):
    """TC156A -> AC156T. Returns (rev, chain, pos, wt, mt) or None if not single-point."""
    if len(m) < 4 or "," in m:
        return None
    wt, chain, mt, pos = m[0], m[1], m[-1], m[2:-1]
    if not pos or not pos[0].isdigit() and not (pos[0] == "-" and pos[1:].isdigit()):
        return None
    return f"{mt}{chain}{pos}{wt}", chain, pos, wt, mt


def n_residues(pdb_path: str) -> int:
    """Interface-bearing complex size, as a covariate: bigger complexes relax more under BuildModel."""
    seen = set()
    try:
        for ln in open(pdb_path):
            if ln.startswith("ATOM"):
                seen.add((ln[21], ln[22:27]))
    except OSError:
        return 0
    return len(seen)


def build_worklist(smoke: bool, aug_only: bool = False) -> list:
    """Every single-point mutation with a retained mutant structure.

    Not restricted to the augmented rows: antisymmetry is a property of FoldX, not of which rows
    happened to be augmented, so the whole retained set estimates it far better (1744 vs 201) and
    the augmented rows remain a stratum inside it via `in_aug`. Restricting up front would have
    thrown away 88% of the evidence for no gain in relevance.
    """
    aug = augmented_rows()
    groups = skempi_groups()
    work = []
    for d in sorted(glob.glob(os.path.join(WORK, "*"))):
        pdb = os.path.basename(d)
        il = os.path.join(d, "individual_list.txt")
        if not os.path.exists(il) or pdb not in groups:
            continue
        nres = n_residues(os.path.join(d, f"{pdb}_Repair.pdb"))
        muts = [l.strip().rstrip(";") for l in open(il) if l.strip()]
        for i, m in enumerate(muts, 1):
            in_aug = (pdb, m) in aug
            if aug_only and not in_aug:
                continue
            if not os.path.exists(os.path.join(d, f"{pdb}_Repair_{i}.pdb")):
                continue
            r = reverse_mut(m)
            if r:
                work.append((pdb, i, m) + r + (groups[pdb], int(in_aug), nres))
        if smoke and work:
            return work[:2]
    return work


def forward_ddg(pdb: str, mut: str) -> float:
    """The stored forward ddG, so the comparison uses the value the augmentation negated."""
    p = os.path.join(RESULTS, f"{pdb}.json")
    if not os.path.exists(p):
        return float("nan")
    d = json.load(open(p))["muts"]
    # keys are the role-mutation strings; `cleaned` carries the PDB-chain form we matched on
    for k, v in d.items():
        if k == mut or v.get("cleaned") == mut:
            return float(v.get("Interaction Energy", float("nan")))
    return float("nan")


def probe_one(pdb, i, mut, rev, chain, pos, wt, mt, garg, mode="reverse") -> tuple:
    """BuildModel one mutation and return (ddg, status).

    mode='reverse'  -- the test: the REVERSE mutation on the retained mutant structure.
    mode='control'  -- the repeatability control: the SAME FORWARD mutation on the same repaired
                       wild type the stored value came from. Any deviation there is FoldX run-to-run
                       noise, not hysteresis, and it is the only way to tell how much of the
                       reverse-test spread is a real asymmetry. Without it a noisy-but-antisymmetric
                       FoldX and a quiet-but-asymmetric one produce the same sd(fwd+rev).
    """
    if mode == "control":
        src = os.path.join(WORK, pdb, f"{pdb}_Repair.pdb")
        wd = os.path.join(PROBE, f"ctl_{pdb}_{i}")
        stem = f"{pdb}_Repair"
        apply_mut = mut
    else:
        src = os.path.join(WORK, pdb, f"{pdb}_Repair_{i}.pdb")
        wd = os.path.join(PROBE, f"{pdb}_{i}")
        stem = f"{pdb}_Repair_{i}"
        apply_mut = rev
    os.makedirs(wd, exist_ok=True)
    log = os.path.join(wd, "foldx.log")
    shutil.copy(src, os.path.join(wd, f"{stem}.pdb"))
    # FoldX resolves its molecules/ directory relative to the binary, so nothing else to stage.
    with open(os.path.join(wd, "individual_list.txt"), "w") as fh:
        fh.write(f"{apply_mut};\n")
    run(f'"{FX}" --command=BuildModel --pdb={stem}.pdb '
        f'--mutant-file=individual_list.txt --numberOfRuns=1', wd, log)
    m_pdb, w_pdb = f"{stem}_1.pdb", f"WT_{stem}_1.pdb"
    if not all(os.path.exists(os.path.join(wd, p)) for p in (m_pdb, w_pdb)):
        return float("nan"), "build_failed"
    with open(os.path.join(wd, "pdblist.txt"), "w") as fh:
        fh.write(f"{m_pdb}\n{w_pdb}\n")
    run(f'"{FX}" --command=AnalyseComplex --pdb-list=pdblist.txt '
        f'--analyseComplexChains={garg}', wd, log)
    mf = os.path.join(wd, f"Interaction_{stem}_1_AC.fxout")
    wf = os.path.join(wd, f"Interaction_WT_{stem}_1_AC.fxout")
    if not (os.path.exists(mf) and os.path.exists(wf)):
        return float("nan"), "analyse_failed"
    return round(parse_interaction(mf)["Interaction Energy"]
                 - parse_interaction(wf)["Interaction Energy"], 4), "ok"


def _task_ctl(t):
    """Repeatability control: re-run the forward mutation and compare with the stored value."""
    pdb, i, mut, rev, chain, pos, wt, mt, garg, in_aug, nres = t
    fwd = forward_ddg(pdb, mut)
    again, status = probe_one(pdb, i, mut, rev, chain, pos, wt, mt, garg, mode="control")
    return dict(pdb=pdb, i=i, mut_fwd=mut, mut_rev="(control: forward re-run)",
                ddg_fwd=fwd, ddg_rev=again,
                sum=(again - fwd) if fwd == fwd and again == again else "",
                chain=chain, pos=pos, wt=wt, mt=mt, groups=garg,
                in_aug=in_aug, n_res=nres, status=status)


def _task(t):
    """One mutation, in its own directory. FoldX has no shared state between runs, so the only
    thing that must not collide is the working directory -- keyed by (pdb, index) already."""
    pdb, i, mut, rev, chain, pos, wt, mt, garg, in_aug, nres = t
    fwd = forward_ddg(pdb, mut)
    rv, status = probe_one(pdb, i, mut, rev, chain, pos, wt, mt, garg)
    return dict(pdb=pdb, i=i, mut_fwd=mut, mut_rev=rev, ddg_fwd=fwd, ddg_rev=rv,
                sum=(fwd + rv) if fwd == fwd and rv == rv else "",
                chain=chain, pos=pos, wt=wt, mt=mt, groups=garg,
                in_aug=in_aug, n_res=nres, status=status)


def _flush(rows):
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="two mutations from the first complex")
    ap.add_argument("--aug-only", action="store_true",
                    help="restrict to rows in an augmented train split (the 2026-08-05 pilot)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--control", action="store_true",
                    help="repeatability control: re-run the FORWARD mutation, compare with stored")
    ap.add_argument("--jobs", type=int, default=8,
                    help="parallel FoldX processes; each runs in its own directory")
    args = ap.parse_args()
    if not os.path.exists(FX):
        sys.exit(f"FoldX binary not found at {FX}; set FOLDX_BIN")
    os.makedirs(PROBE, exist_ok=True)

    global OUT
    if args.control:
        OUT = OUT.replace("foldx_antisymmetry.csv", "foldx_repeatability.csv")
    work = build_worklist(args.smoke, aug_only=args.aug_only)
    if args.limit and args.limit < len(work):
        # Even stride, not a prefix. The worklist is ordered by complex, so a prefix would sample
        # only the alphabetically-first complexes -- fine for a smoke test, wrong for the control,
        # whose whole purpose is a noise floor representative of the set the test ran on.
        step = len(work) / args.limit
        work = [work[int(k * step)] for k in range(args.limit)]

    # Resume: rows already recorded are kept verbatim and dropped from the worklist. The older
    # serial rows lack in_aug/n_res, so backfill those two rather than recomputing the FoldX call.
    rows, done = [], set()
    if os.path.exists(OUT):
        meta = {(w_[0], str(w_[1])): (w_[9], w_[10]) for w_ in work}
        with open(OUT) as fh:
            for r in csv.DictReader(fh):
                k = (r["pdb"], r["i"])
                if not r.get("in_aug") and k in meta:
                    r["in_aug"], r["n_res"] = meta[k]
                rows.append({c: r.get(c, "") for c in COLS})
                done.add(k)
    todo = [w_ for w_ in work if (w_[0], str(w_[1])) not in done]
    print(f"worklist {len(work)}  done {len(done)}  to run {len(todo)}  jobs {args.jobs}",
          flush=True)

    if todo:
        with Pool(args.jobs) as pool:
            fn = _task_ctl if args.control else _task
            for n, r in enumerate(pool.imap_unordered(fn, todo, chunksize=1), 1):
                rows.append(r)
                if n % 10 == 0 or n == len(todo):
                    _flush(rows)
                    print(f"  [{n}/{len(todo)}] {r['pdb']} {r['mut_fwd']} "
                          f"fwd {r['ddg_fwd']:+.3f} rev {r['ddg_rev']:+.3f} {r['status']}",
                          flush=True)
    _flush(rows)
    print(f"[wrote] {OUT}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
