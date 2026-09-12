#!/usr/bin/env python3
"""Build 3-fold, by-complex CV splits for the leakage-controlled retrain (§2a).

The paper/balanced 10-fold CV splits *by mutation*, so the same complex sits in both
train and test. This builder partitions by **whole complex** (PDB id) so train/val/test
within a fold never share a complex — the honest generalization split.

Method (§2a, "size-balanced by-complex"):
  * group the pooled 1100 rows by complex id = col1.split("_")[0] (110 complexes, 1:1);
  * greedy: assign complexes largest-first to the currently lightest fold (ties -> lowest
    fold index) -> 3 size-balanced folds;
  * rotate: each fold is the held-out TEST once, the other two are TRAIN, and a by-complex
    VAL slice (~15% of *training complexes'* muts, whole complexes only) is carved from the
    train pool with a fixed seed for early stopping.

Emits the base 4-col format (s1 s2 mut label) the trainer consumes, identical to
experiments/embedding_sweep/splits/*. The 12-term FoldX (foldxdec) variant is produced
separately by scratch/foldx_s1102/merge_foldx_decomposed.py --src <this out dir>, which
re-standardizes each term train-only on THESE folds (leakage-free).

Usage:
    python experiments/retrain_split/build_splits_bycomplex.py \
        [--data experiments/embedding_sweep/data/S1102_filtered.tsv] \
        [--out scratch/splits_bycomplex_seed42] [--folds 3] [--val-frac 0.15] [--seed 42]
"""
import argparse
import os
import random
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BASENAME = "S1102_filtered"


def complex_of(row):
    """Complex id = PDB code, e.g. '1A22_A' -> '1A22'."""
    return row[0].split("_")[0]


def read_rows(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 4:
                rows.append(p)
    return rows


def group_by_complex(rows):
    g = defaultdict(list)
    for r in rows:
        g[complex_of(r)].append(r)
    return g


def assign_folds(groups, n_folds):
    """Greedy largest-first bin-packing into n_folds by mutation count.

    Ties in complex size broken by complex id (deterministic); ties in fold load broken
    by lowest fold index. Returns {complex_id: fold_index}.
    """
    order = sorted(groups, key=lambda c: (-len(groups[c]), c))
    load = [0] * n_folds
    assign = {}
    for c in order:
        f = min(range(n_folds), key=lambda i: (load[i], i))
        assign[c] = f
        load[f] += len(groups[c])
    return assign


def carve_val(train_complexes, groups, val_frac, rng):
    """Pick whole complexes from the train pool for VAL until ~val_frac of train muts.

    Shuffle deterministically, then accumulate complexes until the val row budget is met.
    Guarantees >=1 val complex and never empties the train pool.
    """
    total = sum(len(groups[c]) for c in train_complexes)
    target = val_frac * total
    shuffled = sorted(train_complexes)          # stable base order
    rng.shuffle(shuffled)
    val, val_rows = [], 0
    for c in shuffled:
        if val_rows >= target and val:
            break
        if len(val) == len(shuffled) - 1:       # keep >=1 complex for train
            break
        val.append(c)
        val_rows += len(groups[c])
    return set(val)


def write_split(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        for r in rows:
            fh.write("\t".join(r) + "\n")


def build(data, out_dir, n_folds, val_frac, seed):
    rows = read_rows(data)
    groups = group_by_complex(rows)
    assert sum(len(v) for v in groups.values()) == len(rows)
    fold_of = assign_folds(groups, n_folds)

    print(f"[build] {len(rows)} rows / {len(groups)} complexes -> {n_folds} folds")
    for i in range(n_folds):
        comps = [c for c, f in fold_of.items() if f == i]
        muts = sum(len(groups[c]) for c in comps)
        big = sum(1 for c in comps if len(groups[c]) >= 10)
        print(f"  fold {i}: {muts} muts | {len(comps)} complexes | {big} complexes >=10 muts")

    for i in range(n_folds):
        test_c = {c for c, f in fold_of.items() if f == i}
        train_pool = [c for c, f in fold_of.items() if f != i]
        rng = random.Random(seed + i)
        val_c = carve_val(train_pool, groups, val_frac, rng)
        train_c = set(train_pool) - val_c

        splits = {"train": train_c, "val": val_c, "test": test_c}
        for name, cset in splits.items():
            out = [r for c in sorted(cset) for r in groups[c]]
            write_split(os.path.join(out_dir, f"fold_{i}", f"{BASENAME}_{name}.tsv"), out)
        nrows = {k: sum(len(groups[c]) for c in v) for k, v in splits.items()}
        print(f"  wrote fold_{i}: train {nrows['train']}r/{len(train_c)}c  "
              f"val {nrows['val']}r/{len(val_c)}c  test {nrows['test']}r/{len(test_c)}c")
    print(f"[build] done -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "experiments/embedding_sweep/data/S1102_filtered.tsv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "scratch/splits_bycomplex_seed42"))
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    build(a.data, a.out, a.folds, a.val_frac, a.seed)


if __name__ == "__main__":
    main()
