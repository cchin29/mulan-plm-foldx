#!/usr/bin/env python3
"""Phase 2 (docs/history/PLAN_RETRAIN_BYCOMPLEX.md §7): homology-clustered split for S1102.

By-complex (Phase 1) still leaks by homology — the honest, frontier-comparable generalization
test quarantines whole sequence families. This builds the 2c recipe: cluster the 220 S1102
interface chains at <=60% identity (mmseqs `easy-cluster`, the CD-HIT-equivalent ProtBFF uses),
then single-linkage the 110 complexes into **families** (two complexes are the same family if
they share any chain cluster) and GroupKFold on families — or, if a mega-family dominates,
fall back to leave-one-family-out (LOFO), exactly as the plan's coverage warning anticipates.

Default = ASSESS ONLY (no split files written): cluster, report family count + size
distribution + a feasibility read (can 3 balanced folds survive, or is it LOFO?). This is the
§7 "assess cluster count first" gate. Pass --emit {kfold,lofo} to write splits afterwards.

    ./.venv/bin/python experiments/retrain_split/cluster_split.py                 # assess @60%
    ./.venv/bin/python experiments/retrain_split/cluster_split.py --min-seq-id 0.3   # stricter-merge sensitivity
    ./.venv/bin/python experiments/retrain_split/cluster_split.py --emit kfold --folds 3

mmseqs writes only under scratch/ (gitignored). Base 4-col split emission mirrors
build_splits_bycomplex.py; the FoldX (foldxdec / foldx) variants are produced afterwards by
merge_foldx_decomposed.py / merge_foldx.py --src <out dir>, same as Phase 1.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import build_splits_bycomplex as B  # reuse read_rows/group_by_complex/carve_val/write_split  # noqa: E402

DATA = os.path.join(ROOT, "experiments/embedding_sweep/data/S1102_filtered.tsv")
FASTA = os.path.join(ROOT, "experiments/embedding_sweep/data/wt_sequences.fasta")


class UF:
    def __init__(self, items):
        self.p = {x: x for x in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def run_mmseqs(min_seq_id, cov, workdir):
    # mmseqs mangles paths containing spaces, so run it entirely inside a space-free temp dir,
    # then copy the cluster TSV back to
    # `workdir` (repo scratch/, gitignored) for provenance via Python (handles spaces).
    os.makedirs(workdir, exist_ok=True)
    safe = tempfile.mkdtemp(prefix="mmseqs_s1102_")
    assert " " not in safe, f"temp dir has a space: {safe}"
    fa = os.path.join(safe, "input.fasta")
    shutil.copy(FASTA, fa)
    prefix = os.path.join(safe, "clust")
    tmp = os.path.join(safe, "tmp")
    cmd = [
        "mmseqs", "easy-cluster", fa, prefix, tmp,
        "--min-seq-id", str(min_seq_id), "-c", str(cov), "--cov-mode", "0",
    ]
    print(f"[mmseqs] easy-cluster --min-seq-id {min_seq_id} -c {cov}  (in {safe})")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-2000:] + r.stderr[-2000:])
        raise SystemExit(f"mmseqs failed ({r.returncode})")
    tsv = prefix + "_cluster.tsv"
    saved = os.path.join(workdir, "cluster.tsv")
    shutil.copy(tsv, saved)
    chain2rep = {}
    with open(saved) as fh:
        for line in fh:
            rep, member = line.rstrip("\n").split("\t")
            chain2rep[member] = rep
    shutil.rmtree(safe, ignore_errors=True)
    print(f"[mmseqs] cluster.tsv -> {saved}")
    return chain2rep


def build_families(groups, chain2rep):
    """Single-linkage complexes that share a chain cluster. Returns {complex: family_id}."""
    complexes = list(groups)
    # complex -> set of chain-cluster reps for its two chains
    comp_chains = {}
    for c, rows in groups.items():
        chains = set()
        for r in rows:
            chains.add(chain2rep.get(r[0], r[0]))
            chains.add(chain2rep.get(r[1], r[1]))
        comp_chains[c] = chains
    # cluster-rep -> complexes containing it
    rep2comps = defaultdict(list)
    for c, chains in comp_chains.items():
        for ch in chains:
            rep2comps[ch].append(c)
    uf = UF(complexes)
    for ch, comps in rep2comps.items():
        for c in comps[1:]:
            uf.union(comps[0], c)
    fam = {c: uf.find(c) for c in complexes}
    # relabel families 0..n-1 by descending mut count for readability
    fam_muts = defaultdict(int)
    for c, f in fam.items():
        fam_muts[f] += len(groups[c])
    order = sorted(set(fam.values()), key=lambda f: -fam_muts[f])
    relabel = {f: i for i, f in enumerate(order)}
    return {c: relabel[f] for c, f in fam.items()}


def report(groups, fam):
    total = sum(len(v) for v in groups.values())
    fams = defaultdict(list)
    for c, f in fam.items():
        fams[f].append(c)
    print(f"\n[assess] {len(fams)} families over {len(groups)} complexes / {total} muts")
    print(f"{'fam':>4} {'complexes':>10} {'muts':>6} {'%muts':>6} {'>=10mut':>8}  example complexes")
    big_frac = 0.0
    for f in sorted(fams, key=lambda f: -sum(len(groups[c]) for c in fams[f])):
        comps = fams[f]
        muts = sum(len(groups[c]) for c in comps)
        big = sum(1 for c in comps if len(groups[c]) >= 10)
        frac = muts / total
        big_frac = max(big_frac, frac) if f == 0 else big_frac
        ex = ",".join(sorted(comps)[:5]) + ("…" if len(comps) > 5 else "")
        print(f"{f:>4} {len(comps):>10} {muts:>6} {100*frac:>5.1f}% {big:>8}  {ex}")
    # feasibility read
    biggest = max(sum(len(groups[c]) for c in fams[f]) for f in fams)
    print(f"\n[feasibility] largest family = {100*biggest/total:.1f}% of all muts; "
          f"{len(fams)} families total.")
    if len(fams) < 3:
        print("  -> FEWER THAN 3 families: GroupKFold(3) impossible. Use leave-one-family-out (LOFO).")
    elif biggest > 0.45 * total:
        print("  -> a mega-family dominates (>45% muts): 3 balanced folds won't survive. "
              "Recommend LOFO on the largest family, or K = #families with per-fold reporting.")
    else:
        print("  -> 3-fold GroupKFold on families looks feasible (assess balance in the emitted splits).")
    return fams


def emit_kfold(groups, fam, out_dir, n_folds, val_frac, seed):
    """GroupKFold by family: greedy assign whole families (largest first) to lightest fold."""
    fams = defaultdict(list)
    for c, f in fam.items():
        fams[f].append(c)
    order = sorted(fams, key=lambda f: (-sum(len(groups[c]) for c in fams[f]), f))
    load = [0] * n_folds
    fold_of_fam = {}
    for f in order:
        i = min(range(n_folds), key=lambda k: (load[k], k))
        fold_of_fam[f] = i
        load[i] += sum(len(groups[c]) for c in fams[f])
    _write_folds(groups, fam, fams, fold_of_fam, n_folds, out_dir, val_frac, seed)


def emit_lofo(groups, fam, out_dir, val_frac, seed):
    """Leave-one-family-out: hold out family 0 (largest = protease-inhibitor) as the single test
    set; train/val carved by whole family from the rest (no family split across train/val)."""
    import random
    fams = defaultdict(list)
    for c, f in fam.items():
        fams[f].append(c)
    test_fams = [0]
    pool_fams = [f for f in fams if f != 0]
    rng = random.Random(seed)
    val_fams = _carve_val_families(pool_fams, fams, groups, val_frac, rng)
    train_fams = set(pool_fams) - val_fams
    split_c = {
        "train": [c for f in train_fams for c in fams[f]],
        "val": [c for f in val_fams for c in fams[f]],
        "test": [c for f in test_fams for c in fams[f]],
    }
    for name, comps in split_c.items():
        rows = [r for c in sorted(comps) for r in groups[c]]
        B.write_split(os.path.join(out_dir, "fold_0", f"{B.BASENAME}_{name}.tsv"), rows)
    nr = {k: sum(len(groups[c]) for c in v) for k, v in split_c.items()}
    print(f"[lofo] fold_0: test=family0 {len(split_c['test'])}c/{nr['test']}r  "
          f"train {len(split_c['train'])}c/{len(train_fams)}fam  "
          f"val {len(split_c['val'])}c/{len(val_fams)}fam -> {out_dir}")


def _carve_val_families(train_fams, fams, groups, val_frac, rng):
    """Pick whole FAMILIES from the train pool for val (~val_frac of train muts), so a family is
    never split across train/val. Iterate SMALLEST family first (rng only breaks size ties) so a
    mega-family (e.g. the 593-mut protease family) is never pulled into val and never starves
    train — the loop reaches the val target on small families and stops before the big ones."""
    fam_sz = {f: sum(len(groups[c]) for c in fams[f]) for f in train_fams}
    total = sum(fam_sz.values())
    target = val_frac * total
    ordered = sorted(train_fams, key=lambda f: (fam_sz[f], rng.random()))
    val, val_rows = [], 0
    for f in ordered:
        if val_rows >= target and val:
            break
        if len(val) == len(ordered) - 1:        # keep >=1 family for train
            break
        val.append(f)
        val_rows += fam_sz[f]
    return set(val)


def _write_folds(groups, fam, fams, fold_of_fam, n_folds, out_dir, val_frac, seed):
    import random
    for i in range(n_folds):
        test_fams = [f for f, k in fold_of_fam.items() if k == i]
        pool_fams = [f for f, k in fold_of_fam.items() if k != i]
        rng = random.Random(seed + i)
        val_fams = _carve_val_families(pool_fams, fams, groups, val_frac, rng)
        train_fams = set(pool_fams) - val_fams
        split_c = {
            "train": [c for f in train_fams for c in fams[f]],
            "val": [c for f in val_fams for c in fams[f]],
            "test": [c for f in test_fams for c in fams[f]],
        }
        for name, comps in split_c.items():
            rows = [r for c in sorted(comps) for r in groups[c]]
            B.write_split(os.path.join(out_dir, f"fold_{i}", f"{B.BASENAME}_{name}.tsv"), rows)
        nr = {k: sum(len(groups[c]) for c in v) for k, v in split_c.items()}
        print(f"[emit] fold_{i}: train {nr['train']}r/{len(split_c['train'])}c/"
              f"{len(train_fams)}fam  val {nr['val']}r/{len(split_c['val'])}c/{len(val_fams)}fam  "
              f"test {nr['test']}r/{len(split_c['test'])}c/{len(test_fams)}fam")
    print(f"[emit] {n_folds}-fold family-grouped splits (family-whole train/val/test) -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-seq-id", type=float, default=0.6)
    ap.add_argument("--cov", type=float, default=0.8)
    ap.add_argument("--emit", choices=["kfold", "lofo"], default=None)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    pct = int(round(a.min_seq_id * 100))
    workdir = a.workdir or os.path.join(ROOT, f"scratch/clust_mmseqs_id{pct}")
    rows = B.read_rows(DATA)
    groups = B.group_by_complex(rows)
    chain2rep = run_mmseqs(a.min_seq_id, a.cov, workdir)
    nclust = len(set(chain2rep.values()))
    print(f"[mmseqs] {len(chain2rep)} chains -> {nclust} chain clusters @ id>={a.min_seq_id} cov>={a.cov}")
    fam = build_families(groups, chain2rep)
    report(groups, fam)

    if a.emit:
        out = a.out or os.path.join(ROOT, f"scratch/splits_clustered_id{pct}_{a.emit}")
        if a.emit == "kfold":
            emit_kfold(groups, fam, out, a.folds, a.val_frac, a.seed)
        else:
            emit_lofo(groups, fam, out, a.val_frac, a.seed)


if __name__ == "__main__":
    main()
