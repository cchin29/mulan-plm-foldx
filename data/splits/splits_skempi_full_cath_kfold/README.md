# CATH-superfamily hold-out

This split is **not** redistributed here. Every other split under `data/splits/` is our own
partitioning and ships in full; this one joins our rows onto the static `cath_fold` column
published with [USP-ddG](https://github.com/ak422/USP-ddG), following the CATH-superfamily
protocol introduced by CATH-ddG. The partition is theirs, that repository ships no LICENSE file,
and its terms cannot be established from it — so the recipe ships and the artifact does not.
Provenance and citations are in [`../../../NOTICE`](../../../NOTICE).

## Build

```bash
git clone https://github.com/ak422/USP-ddG /path/to/USP-ddG

python experiments/full_skempi_seqonly/build_split_cath.py \
    --usp /path/to/USP-ddG/data/SKEMPI2/skempi_v2.csv
```

The other input — our SKEMPI 2.0 curation, `data/skempi_full/{single,multi}_point.tsv` — ships
here, so nothing else is needed. Runtime is about a second.

## Verify

```bash
( cd data/splits/splits_skempi_full_cath_kfold && shasum -a 256 -c MANIFEST.sha256 )
```

The manifest's paths are relative to that directory, hence the subshell — it leaves the working
directory at the repo root, which is where the commands below expect to be. On Linux the tool is
usually `sha256sum -c` rather than `shasum -a 256 -c`. `build_split_cath.py` also takes `--out`;
building elsewhere leaves `MANIFEST.sha256` behind in the repo, so copy it across or pass the
repo-relative path to the checker.

Six files, and all six must pass. That is what makes the rebuild usable as evidence: the manifest
records the exact split every CATH number in [`../../../docs/RESULTS.md`](../../../docs/RESULTS.md)
was computed on, so a passing check means a locally built split is the same one, not merely a
similar one.

Expected shape:

| file | rows |
|---|---:|
| `fold_0/skempi_all_train.tsv` | 4213 |
| `fold_0/skempi_all_val.tsv` | 878 |
| `fold_0/skempi_all_test.tsv` | 687 |
| `test_single.tsv` | 421 |
| `test_multiple.tsv` | 266 |

4213 + 878 + 687 = **5778, not 5801**. `build_split_cath.py` drops rows whose PDB has no
`cath_fold` label in USP-ddG's table, and it reports the count as it goes. So this tier runs on 23
fewer rows than every other tier here, and the test rows are the complement of train ∪ val within
those 5778 rather than within 5801.

`JOIN_REPORT.txt` is written alongside them and records the coverage audit: 687 of USP's 813 val
rows survive this repository's SKEMPI extraction (84.5%), over 50 of their 53 val complexes.

One fold, not three — the CATH hold-out is a single hold-out. The val rows are a seeded 10% carve
of the *train* complexes, used for early stopping only; they are ours, not part of the third-party
partition, and never overlap test.

## Scoring the shipped predictions against it

`results/full_skempi_cath/` ships predictions as a **single unkeyed column**, one value per line,
in the row order of `fold_0/skempi_all_test.tsv`. Every other tier ships them keyed by
`(chain1, chain2, mutation)`; this one cannot, because our test rows are the complement of
train ∪ val within the same 5801, so keyed predictions would disclose the whole partition and the
withholding above would be cosmetic.

Once the split is built, join by position. Run from the repo root:

```python
import collections
from scipy.stats import spearmanr

SPLIT = "data/splits/splits_skempi_full_cath_kfold/fold_0/skempi_all_test.tsv"
PRED  = "results/full_skempi_cath/esmc6b_foldx_scalar/fold_0/test_predictions.tsv"

# split rows are headerless TSV: chain1, chain2, mutation, ddG(kcal/mol)
rows = [l.rstrip("\n").split("\t") for l in open(SPLIT)]
pred = [float(l) for l in open(PRED)]
assert len(pred) == len(rows) == 687

by = collections.defaultdict(lambda: ([], []))
for (chain1, _chain2, _mut, ddg), p in zip(rows, pred):
    complex_key = chain1.split("_")[0]          # e.g. 1AK4.A.D_A -> 1AK4.A.D
    by[complex_key][0].append(p)
    by[complex_key][1].append(float(ddg))

rhos = [spearmanr(p, t).statistic for p, t in by.values() if len(t) >= 10]
print(len(rhos), sum(rhos) / len(rhos))        # -> 13 0.4378...
```

The reported figure is the **mean** of the per-complex Spearmans over the 13 complexes carrying at
least 10 mutations, not the median (0.4208) and not a pooled correlation. `mulan.metrics.per_structure`
applies the same definition and additionally drops complexes with zero variance in either vector;
none of the 13 here are affected. That 0.4378 is the 0.438 in the top-level README.

## Determinism

The builder's output is byte-reproducible: same inputs, same bytes, independent of `PYTHONHASHSEED`.
That was not true before 2026-08-05 — the val rows were ordered by set iteration, so the same seed
produced a byte-different val file run to run. Only the order was affected, never the row set. The
runs under `results/` predate the fix and were trained against the same val rows in a different
order; train and test were deterministic throughout and are unaffected.
