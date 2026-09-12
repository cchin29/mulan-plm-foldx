#!/usr/bin/env python3
"""Unit tests for build_splits_bycomplex.py (§2). Builds into a temp dir and asserts:
  (i)   no complex appears in >1 of {train,val,test} within any fold (the whole point);
  (ii)  union of the test folds == the full 1100-row dataset (exact partition);
  (iii) every row's 4 base columns are preserved verbatim;
  (iv)  folds are size-balanced (muts within a small tolerance) and each test fold has
        >=1 complex with >=10 muts (per-structure metrics need it).

Run:  python experiments/retrain_split/test_build_splits.py
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import build_splits_bycomplex as B  # noqa: E402

DATA = os.path.join(ROOT, "experiments/embedding_sweep/data/S1102_filtered.tsv")
N_FOLDS = 3


def _read(path):
    return B.read_rows(path)


def main():
    src_rows = _read(DATA)
    src_keys = {(r[0], r[1], r[2]) for r in src_rows}
    assert len(src_keys) == len(src_rows), "duplicate keys in source dataset"

    with tempfile.TemporaryDirectory() as tmp:
        B.build(DATA, tmp, N_FOLDS, 0.15, 42)

        all_test = []
        fold_muts = []
        for i in range(N_FOLDS):
            fd = os.path.join(tmp, f"fold_{i}")
            splits = {s: _read(os.path.join(fd, f"{B.BASENAME}_{s}.tsv")) for s in ("train", "val", "test")}
            comp = {s: {B.complex_of(r) for r in rows} for s, rows in splits.items()}

            # (i) pairwise-disjoint complexes across train/val/test
            assert not (comp["train"] & comp["test"]), f"fold {i}: train/test share a complex"
            assert not (comp["train"] & comp["val"]), f"fold {i}: train/val share a complex"
            assert not (comp["val"] & comp["test"]), f"fold {i}: val/test share a complex"
            assert comp["val"], f"fold {i}: empty val"
            assert comp["train"], f"fold {i}: empty train"

            # (iv-b) per-structure feasibility
            from collections import Counter
            tc = Counter(B.complex_of(r) for r in splits["test"])
            assert sum(1 for c in tc if tc[c] >= 10) >= 1, f"fold {i}: no test complex >=10 muts"

            all_test.extend(splits["test"])
            fold_muts.append(len(splits["test"]))

            # every fold: train+val+test == whole dataset (each complex assigned once)
            union = splits["train"] + splits["val"] + splits["test"]
            assert len(union) == len(src_rows), f"fold {i}: train+val+test != full dataset ({len(union)})"

        # (ii) test folds exactly partition the dataset
        test_keys = [(r[0], r[1], r[2]) for r in all_test]
        assert len(test_keys) == len(src_rows), f"test union size {len(test_keys)} != {len(src_rows)}"
        assert set(test_keys) == src_keys, "test union != source keys"
        assert len(set(test_keys)) == len(test_keys), "a row appears in >1 test fold"

        # (iii) base columns preserved verbatim
        src_by_key = {(r[0], r[1], r[2]): r[3] for r in src_rows}
        for r in all_test:
            assert src_by_key[(r[0], r[1], r[2])] == r[3], f"label mangled for {r[:3]}"

        # (iv-a) size balance: spread <= 2% of mean
        spread = max(fold_muts) - min(fold_muts)
        assert spread <= 0.02 * (len(src_rows) / N_FOLDS), f"folds unbalanced: {fold_muts}"

    print(f"OK — all tests passed ({len(src_rows)} rows, {N_FOLDS} folds, test sizes {fold_muts})")


if __name__ == "__main__":
    main()
