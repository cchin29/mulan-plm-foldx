# Splits and metrics

How this fork evaluates, why it differs from the upstream paper, and how to read the numbers against
the published ΔΔG frontier.

## The short version

Upstream MuLAN reports **random 10-fold cross-validation** over mutations. That splits *mutations*,
not *complexes* — so the same protein complex appears in training and test, and a model can score
well by memorising a complex rather than learning the effect of a mutation. Pooled Pearson under
that protocol runs around 0.85–0.88.

Every current method in this area is instead benchmarked under a homology-controlled hold-out and
scored **per structure**. Moving to that protocol costs far more than any modelling choice in this
project gained: on the same trained model, the metric change alone takes ~0.88 pooled Pearson to
~0.49 per-structure Spearman, and the split change takes it from ~0.49 to ~0.37. Those two steps are
the single most important thing to understand about the numbers in this repository.

The legacy 10-fold protocol is retained — it is the only way to compare against the published
upstream number — but it is **not** the protocol any headline claim is made on.

## The four protocols

Ordered by strictness. Each holds out a progressively larger unit, so progressively less of the test
set is "already seen" in some form.

| Protocol | Held-out unit | Folds | What it controls for |
|---|---|---|---|
| `legacy_cv10` | individual mutations | 10 | **nothing** — same complex in train and test. Paper-reproduction only. |
| `bycomplex` | whole complexes | 3 | memorising a complex's binding energetics from its other mutations. |
| `clustered_id60` | sequence families (CD-HIT / mmseqs ≤60% identity, coverage 0.8, single-linkage over shared interface chains) | 3 | memorising a *homolog* of the test complex. |
| `cath` | CATH superfamilies | 1 | structural homology, the strictest available. Uses USP-ddG's partition verbatim — which is not ours to redistribute, so this is the one split built locally rather than shipped: [`../data/splits/splits_skempi_full_cath_kfold/README.md`](../data/splits/splits_skempi_full_cath_kfold/README.md). |

Two details that matter for comparability:

**By-complex is easier than it sounds.** A by-complex hold-out still leaves close homologs of the
test complex in training. USP-ddG measure ~88.7% of their by-complex test set as "easy" by this
criterion. Our by-complex numbers are therefore **not** comparable to a frontier CATH number even
though the metric is identical.

**The CATH split is not ours.** `build_split_cath.py` joins the static `cath_fold` column that
USP-ddG ship in their SKEMPI-2 table onto our rows, by complex (the label is constant per complex).
That gives the literal held-out superfamilies they report on, so a row from this repository drops
directly into their Table 1. It is a single fixed partition, not cross-validation — which also means
it has no fold spread to measure variance against, and small per-arm differences on it should not be
over-read.

Splits are also built over single-point, multi-point, and combined mutation sets, since the frontier
papers differ in which they report.

> **The by-complex unit is the interface, not the PDB code.** SKEMPI defines `2C5D`, `3SE3` and
> `3SE4` under two chain pairings each, and the splitter holds out `code.g1.g2`. Four folds
> therefore put one PDB code on both sides — `3SE3` in `bycomplex_all` folds 0 and 2 and in
> `mp_bycomplex` fold 1, `2C5D` in `mp_bycomplex` fold 0 — and in two of those the identical
> mutation string appears in train and test under different interface labels with different ΔΔG.
> That is 2 test rows in roughly 1900 per fold and it moves no reported number, but if you
> re-partition these rows yourself, key on the PDB code. `mulan/metrics/core.py`'s `pdb_of` uses
> the same interface-level key, so those pairings are averaged as two independent structures in
> the per-structure metric.

## The metric

The headline is **per-structure Spearman correlation with T ≥ 10 mutations per complex** — group the
test predictions by complex, compute Spearman within each complex that has at least 10 mutations,
then average. This is USP-ddG's "Per-PPI SpearmanR" and the standard in this literature. On a
multi-fold tier the folds' test predictions are **pooled before grouping by complex**, so each
complex is scored once on all of its test rows and the tier yields one number, not a mean over
folds; where a fold-averaged variant is reported it is labelled as such.

Why per-structure rather than pooled: a pooled correlation over all test mutations is dominated by
*between*-complex variance — complexes differ enormously in their ΔΔG range, so a model that ranks
complexes correctly but ranks mutations within a complex at chance still scores well. Per-structure
correlation asks the question a protein engineer actually has: *given this interface, which mutation
should I make?* The argument is set out in `experiments/METRIC_RATIONALE.md`.

Also computed, for comparability with whichever numbers a given paper reports: pooled Pearson and
Spearman, RMSE, MAE, AUROC for destabilizing and strongly-destabilizing calls (positive class:
measured ΔΔG > 0, and measured ΔΔG ≥ `mulan.metrics.STRONG` = 2.0 kcal/mol; the predicted ΔΔG is
the score), precision/recall at the top 50, per-structure correlation at T ≥ 5, and
cluster-bootstrap confidence intervals resampling over complexes.

The scoring core is `mulan/metrics`; every scorer imports it rather than reimplementing it.

### Which tier is the default

Which truth a scorer scores against is an **evaluation tier** — a protocol applied to a dataset.
`mulan.metrics.get_tier()` with no argument returns the CATH-superfamily hold-out.

This is deliberate and it is a change. The tier used to be two module-level globals initialised
to the *leaky* 10-fold splits, which every scorer then reassigned. The default behaviour of the
metric core was therefore the one protocol no claim rests on, and a scorer that forgot to
reassign would score against the wrong truth while reporting a perfectly normal-looking number.

The leaky tier is still available — reproducing the upstream paper's headline requires it — but
only by naming it:

```bash
python experiments/rescore_perstructure/rescore.py --list-tiers
python experiments/rescore_perstructure/rescore.py --tier legacy_cv10
```

Running it with no `--tier` is now an error rather than a silent fallback. The second command
shows how a tier is named; it does not run from a clone, because `rescore.py` reads working-tree
paths under `scratch/` that are not shipped — [`REPRODUCE.md`](REPRODUCE.md) gives the route that
does.

## Composition of the benchmark subsets

Both arguments above — that a by-complex hold-out is weaker than it sounds, and that the metric
averages over a minority of the complexes — are claims about the shape of SKEMPI rather than about
any model. They are easier to check than to believe from a table.

`scripts_plots/plot_treemaps.py` draws each subset four times, holding the mutations fixed and
changing only what counts as a group. Going down the list is the ladder above: the stricter the
split, the coarser the unit it has to hold out whole, and the fewer independent units the benchmark
turns out to contain.

| view | one tile is | |
|---|---|---|
| `scale` | one complex | [figure](../scripts_plots/treemaps/treemap_scale.png) |
| `clusters` | one ≤60% sequence family | [figure](../scripts_plots/treemaps/treemap_clusters.png) |
| `cath` | one CATH superfamily group | [figure](../scripts_plots/treemaps/treemap_cath.png) |
| `coverage` | one complex, coloured by whether it clears T ≥ 10 | [figure](../scripts_plots/treemaps/treemap_coverage.png) |

**The data is extremely long-tailed.** In full SKEMPI single-point, 103 of 315 complexes carry a
single mutation, while the largest ten carry 35% of the set. S1102 is more concentrated still: its
largest ten carry 66%.

**Clustering collapses it much further than the complex count suggests.** 315 single-point
complexes fall into 93 families, and one protease/inhibitor family holds 40% of all mutations.
S1102's 111 complexes fall into 36, of which the largest holds 54%. That collapse is the reason a
by-complex hold-out leaves a homolog of the test complex in training.

**CATH is finer, not coarser** — 159 families against 93 for single-point — because single-linkage
merges aggressively through shared inhibitor chains while the both-partners rule keeps distinct
antigen/antibody pairings apart. It controls a different axis rather than simply a stricter one.

**The metric keeps most of the mutations and a minority of the complexes.** At T ≥ 10, single +
multi scores 121 of 337 complexes holding 5106 of 5801 mutations; S1102 scores 24 of 111. The
complex is the unit being averaged, so the effective sample size is the smaller number.

```bash
python scripts_plots/plot_treemaps.py --view all --out scripts_plots/treemaps --plots --html
```

Every input is in this repository — `data/splits/`, `data/splits/clusters_id60/`,
`experiments/cath_leakage/cath_families.json`, `examples/S1102.tsv` — and the counts written to
`scripts_plots/treemaps/TREEMAPS.txt` are the ones quoted above. `--html` adds a hoverable page per
view, which is where family membership is legible; a static tile has no room for it. `--html`
needs `plotly`, which is in no extra — without it the command still writes the PNGs and the text
summary, and prints `[html] plotly not installed`. A view whose
inputs are absent is named and skipped rather than quietly drawn smaller: the combined set has no
`clusters` panel, because single- and multi-point were clustered in separate mmseqs runs and
merging two runs is not the same partition as clustering the union.

The CATH grouping here is our own SIFTS derivation
(`experiments/cath_leakage/cath_leakage.py`, both-partners rule over `pdb_chain_cath_uniprot`).
It is **not** the fixed partition the `cath` split uses — that one is USP-ddG's, as above.

## Reading our numbers against the frontier

Comparator numbers are transcribed from the source papers into
`experiments/BENCHMARK_MATRIX.md`; a consolidated table lives in `experiments/METRIC_RATIONALE.md`.

Three things to check before comparing any two numbers:

1. **Same split family?** CATH-superfamily, CD-HIT ≤60% (ProtBFF's sequence-identity criterion;
   the clustering here is built with MMseqs2 to the same threshold), and by-complex are three
   different difficulties. Only same-family comparisons mean anything.
2. **Same metric?** Per-structure and pooled correlations are not interchangeable, and some papers
   report only one. At least one frontier comparator publishes pooled-only numbers on the clustered
   split, so no per-structure comparison is possible there at all.
3. **Same mutation set?** Single-point, multi-point and combined sets give different numbers on the
   same model.

The peer group for this work is the **sequence/PLM** row of those tables, not the structure-based
methods — MuLAN is a frozen language model plus a light attention head, with FoldX entering only as
a score channel. The interesting comparison is whether that combination reaches methods that consume
structure directly.

> **This document quotes no results of its own.** [`RESULTS.md`](RESULTS.md) and the figure CSVs
> are the source of truth and are updated together; the per-fold predictions under
> [`../results/`](../results/README.md) make every figure there recomputable. The approximate ladder
> figures above (~0.88, ~0.49, ~0.37) are inherited from `../experiments/METRIC_RATIONALE.md`
> (ESM-C 6B) rather than re-derived against the final rescore, and carry that caveat. The
> dataset-shape counts are a different kind of number, recomputed from the shipped splits by
> `scripts_plots/plot_treemaps.py`.

## Where the code lives

| Stage | Files |
|---|---|
| Dataset construction | `experiments/full_skempi_seqonly/build_skempi_full.py`, `build_multipoint.py` |
| Split builders | `experiments/retrain_split/build_splits_bycomplex.py`, `cluster_split.py`; `experiments/full_skempi_seqonly/build_split_{bycomplex,clustered,clustered_all,cath,multipoint}.py` |
| Legacy 10-fold | `mulan/data.py::split_data` (`split_method="random"` reproduces the upstream partition; `"balanced"` gives equal-size folds), driven by `experiments/embedding_sweep/gen_splits.py` |
| Tier configuration | `experiments/*/config_*.sh` |
| Runner | `experiments/retrain_split/run_bycomplex.sh` — shared by every tier; `ARMS="base foldx foldx_scalar"` |
| Scoring | `experiments/rescore_perstructure/rescore.py` (core) and the `score_*.py` wrappers in `experiments/full_skempi_seqonly/` |
| Aggregation | `scripts_plots/results_matrix.py` |

`split_data` is an *offline* splitter — it writes fold TSVs, and `mulan-train` consumes
pre-materialized files. It is never called during training.
