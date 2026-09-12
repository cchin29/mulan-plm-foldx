# Per-fold results

`all_results.json` and `test_predictions.tsv` for every trained fold, so every reported metric can
be recomputed without retraining anything. **666 folds, 30.2 MB.**

## Scope

A fold is here if its tier's split is **obtainable** from this repository — eight ship under
[`../data/splits/`](../data/splits/), and CATH builds from a recipe in the ninth. Results for a
split nobody can get are not useful to anyone, which is what excludes the `cv10_*` and `bench_*`
families and the exploratory runs.

| tier | folds | arms | split |
|---|---:|---:|---|
| `full_skempi` | 84 | 28 | `splits_skempi_full_clustered_id60_kfold` |
| `full_skempi_bycomplex` | 81 | 27 | `splits_skempi_full_bycomplex_seed42` |
| `full_skempi_bycomplex_all` | 81 | 27 | `splits_skempi_full_bycomplex_all_seed42` |
| `full_skempi_cath` | 27 | 27 | `splits_skempi_full_cath_kfold` |
| `full_skempi_clustered_all` | 81 | 27 | `splits_skempi_full_clustered_all_id60_kfold` |
| `full_skempi_mp_bycomplex` | 63 | 21 | `splits_skempi_full_mp_bycomplex_seed42` |
| `full_skempi_mp_clustered` | 63 | 21 | `splits_skempi_full_mp_clustered_id60_kfold` |
| `retrain_bycomplex` | 93 | 31 | `splits_bycomplex_seed42` |
| `retrain_clustered` | 93 | 31 | `splits_clustered_id60_kfold` |

`full_skempi_cath` is one fold per arm because the CATH superfamily hold-out is a single hold-out
rather than a k-fold; every other tier is three.

**`full_skempi_cath` is also the one tier whose split is not shipped**, and its predictions differ
in shape because of it. That split joins onto a third-party partition
([`../NOTICE`](../NOTICE)), so it is built locally from a recipe. Since our test rows are the exact
complement of train ∪ val within the 5778 rows this tier keeps — 23 of the 5801 have no CATH
label and fall in no partition — keyed predictions would disclose the partition
anyway — so this tier's `test_predictions.tsv` is a **single unkeyed column**, one value per line,
in the row order of `fold_0/skempi_all_test.tsv`. Build the split, then join by position;
[`../data/splits/splits_skempi_full_cath_kfold/README.md`](../data/splits/splits_skempi_full_cath_kfold/README.md)
has the build, the verify and a worked example. Every other tier ships predictions keyed by
`(chain1, chain2, mutation)`.

## Layout

```
results/<tier>/<model>_<arm>/fold_<k>/
    all_results.json        metrics as written by the training run
    test_predictions.tsv    per-row predictions -- the thing to rescore from
```

The working tree writes these one level deeper, under `fold_<k>/training_run/`. That level is
flattened here, and the shape above is the contract every path in
[`../docs/REPRODUCE.md`](../docs/REPRODUCE.md) is quoted against.

## Reading the numbers

`all_results.json` carries platform-dependent floating point: the same fold trained on MPS and on
CUDA does not produce bit-identical values. Compare within a platform, or recompute from
`test_predictions.tsv` with `mulan.metrics` — which is the reason to ship the predictions and not
only the metrics.

`test_pcc` is a pooled Pearson over the fold's test rows. It is **not** the conclusion metric
anywhere in [`../docs/RESULTS.md`](../docs/RESULTS.md), which reports per-structure Spearman at
T ≥ 10 with folds pooled before grouping by complex. The two are far apart and move independently;
[`../docs/SPLITS_AND_METRICS.md`](../docs/SPLITS_AND_METRICS.md) decomposes the difference.

## Coverage lineages, and why they are not here

The FoldX channel's coverage rose over the life of this work in two steps, from 28% to ~99%, and
each step moved the arms that read it. The single-point and combined row sets differ after the
first step — 28% / 87.8% / 99.2% and 28% / 90.8% / 99.1% respectively — so quote a figure with its row
set, from [`../docs/RESULTS.md`](../docs/RESULTS.md) §22. The working tree keeps the superseded runs under
`__cov_pre`, `__cov87`, `__cov99_partial` and `__badkey` suffixes so a changed number can be traced
to the change that caused it.

Those 275 archived folds are **not published**. Every fold here is a current-lineage run, so a
directory in this tree can be read as the arm it names, with no suffix semantics to learn and no
way to quote a superseded number by mistake. The provenance the archives support is told in prose
where it belongs: [`../docs/RESULTS.md`](../docs/RESULTS.md) §22 covers the coverage steps and what
each one moved.

## What is not reproducible from this directory

The augmented tiers. `full_skempi_cath_aug` and `full_skempi_clustered_all_aug` read splits that
are not published under `data/splits/`, so they fall outside the scope rule above and their folds
are absent, and unlike CATH there is no recipe for them here. That absence is not a loss: most augmented **FoldX** arms on record were trained on a
channel carrying a constant offset. **All 28 affected runs have since been re-run** — six on
2026-08-06 and the remaining 22 on 08-08 — so [`../docs/RESULTS.md`](../docs/RESULTS.md) §24 now
reports every augmented FoldX arm on the repaired lineage, and the lineage column it used to carry
is gone. Five of the six losses it reported turned out to be the defect rather than the
augmentation. The augmented base arms are unaffected, reading only the
4-column split; §24 reports their result as one positive measurement against four failed
replications.

The augmented splits rebuild from the canonical ones with
`experiments/full_skempi_seqonly/merge_foldx_aug.py`, which reconstructs the standardisation
constants and aborts if they fail to reproduce the committed channel.
