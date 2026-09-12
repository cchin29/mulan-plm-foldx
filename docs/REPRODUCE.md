# Reproducing the results

> **Status: the verification path is complete; the retraining path is an outline.** Every reported
> metric can be recomputed from what ships — §"Verifying without retraining" is exact and has been
> run end to end. The stage-by-stage commands for *regenerating* embeddings and retraining are
> sketched rather than scripted; what follows is enough to judge whether a given result is
> reproducible from what is published here.

## What ships, and what is rebuilt locally

| | Ships in the repo | Why |
|---|---|---|
| **Split definitions** | ✅ `data/splits/` | Small, and the single most reusable artifact — our exact leakage-controlled partitions, so our numbers are checkable and a new method can be evaluated on the same hold-outs. **Except CATH**, which is built locally: see the row below. |
| **CATH-superfamily split** | 🔧 build | Not ours to redistribute — it joins our rows onto USP-ddG's `cath_fold` column, and that repository ships no licence. `data/splits/splits_skempi_full_cath_kfold/` carries the recipe and a `MANIFEST.sha256`; clone USP-ddG, run `build_split_cath.py --usp <csv>`, and check the manifest. Byte-reproducible, about a second. |
| **SKEMPI curation** | ✅ `data/skempi_full/` | Our 5801 curated rows, unpartitioned — what the CATH builder joins against. |
| **FoldX ΔΔG results** | ✅ `data/foldx/` | Per-complex JSON. Regenerating them needs a FoldX licence — free for academic and non-profit use, paid for commercial — and CPU-weeks. |
| **Per-fold results** | ✅ `results/` | `all_results.json` + `test_predictions.tsv` for 666 trained folds — what makes every reported metric recomputable without retraining. Scope is the nine tiers whose splits are obtainable above (eight shipped, CATH built); superseded coverage lineages are excluded. Layout and per-tier counts: [`results/README.md`](../results/README.md). |
| **Comparator tables** | ✅ `data/benchmarks/` | Frontier numbers transcribed from the source papers, 40 rows, one per `(method, metric, protocol)`. Read via `scripts_plots/benchmarks.py`. Some plotting scripts still carry their own hardcoded copies, which have drifted; the shipped table is the one to trust. |
| **FoldX-annotated splits** | ❌ | The 5-column scalar and 16-column decomposed TSVs the FoldX arms train on. Rebuildable from `data/foldx/` + `data/splits/` with `mulan/foldx/merge.py`, but the full-SKEMPI mergers still resolve working-tree paths, so that is not yet a one-command step. The S1102 pair, `experiments/foldx_s1102/merge_foldx{,_decomposed}.py`, runs against shipped inputs in a second (`EMBEDDING_SETUP.md` §4a) — but the §18 runs and the `retrain_*` tiers were built from FoldX results that are not the shipped store, so their exact splits are not rebuildable here. |
| **PLM embeddings** | ❌ | Tens of gigabytes, and regenerable from public model weights. |
| **Model checkpoints** | ❌ | Large; the upstream MuLAN checkpoints are the authors' artifacts (see [NOTICE](../NOTICE)). |
| **FoldX binary, PDB inputs** | ❌ | Licensed / externally distributed. |

The consequence this is built for: **verifying every number without a GPU, a FoldX licence, or a
large download** — the per-fold predictions and the splits are enough, and compute is needed only
to retrain.

The ✅ rows are what make that promise good, and they divide by cost rather than by size. The
FoldX energies are the hardest to *obtain* — a licence and CPU-weeks — while the 666 folds under
`results/` are the most expensive to *regenerate*: GPU-weeks, plus the per-backbone environments
this document devotes a section to. Both are small on disk. Everything the ❌ rows withhold is
either regenerable from public weights or not ours to redistribute.

## The stages

```
  SKEMPI 2.0 (external)
        │
        ├─► dataset construction ──────► single / multi / combined mutation tables + WT FASTA
        │
        ├─► split construction ────────► data/splits/splits_*_{bycomplex,clustered,cath}_*/fold_*/
        │
        ├─► FoldX compute ─────────────► data/foldx/  (needs FoldX; skip — results ship)
        │        RepairPDB → BuildModel → AnalyseComplex
        │
        ├─► PLM embedding generation ──► local cache, one directory per model
        │        (needs the PLM weights; several models need their own environment)
        │
        ├─► merge ─────────────────────► split TSVs carrying the score channel
        │        4-col base · 5-col scalar · 16-col decomposed
        │
        ├─► training ──────────────────► results/<tier>/<model>_<arm>/fold_<k>/
        │        arms: base · foldx_scalar · foldx
        │
        └─► scoring ───────────────────► per-structure Spearman, AUROC, pooled P/S, bootstrap CI
```

## Environments

The models do not share one dependency set — this is a real constraint, not an oversight:

| Extra | For | Conflict |
|---|---|---|
| base | ESM-2, Ankh v1/v3, ProstT5, SaProt, ProtBERT | `transformers>=4.27`, no upper bound; validated only on 4.27–4.44 |
| `requirements-esmc.txt` (not an extra) | ESM-C, ESM3 **embedding generation** | `esm` pins `transformers<4.48.2`; training runs 4.44.x — **separate environment, by design** |
| `[mint]` | MINT | needs the MINT source and checkpoint out of band |
| `[struct3di]` | ProstT5 structure mode | `mini3di` |
| `[struct]` | per-residue structural context | [foldenv](https://github.com/cchin29/foldenv) |
| `[plots]` | figures | matplotlib |

See [`EMBEDDING_SETUP.md`](EMBEDDING_SETUP.md) for which generator produces which cache, and
[`MPS_COMPATIBILITY.md`](MPS_COMPATIBILITY.md) for what does and does not run on Apple Silicon.

## Verifying without retraining

1. Read `results/<tier>/<model>_<arm>/fold_<k>/test_predictions.tsv`.
2. Join against the matching truth column in `data/splits/<protocol>/fold_<k>/*_test.tsv`. The
   tier and split directory names differ; [`results/README.md`](../results/README.md) maps each of
   the nine tiers to its split. Rows join on `(chain1, chain2, mutation)` and the two files are in
   the same order, so a mismatch in row count means the wrong pairing rather than missing data.
3. Score with `mulan.metrics`, which needs only numpy and scipy — `pip install numpy scipy` is
   enough, torch is not imported unless a model-side name is touched:

   ```python
   from mulan import metrics
   mean_pearson, mean_spearman, n_scored, n_degenerate = metrics.per_structure(pred, true, complex_key, T=10)
   ```

   `complex_key` is `chain1.split("_")[0]`; `n_degenerate` counts complexes that reached T but
   were skipped because predictions or labels had zero variance. On a multi-fold tier,
   concatenate the folds' predictions and truths first and call `per_structure` once: the
   published numbers pool folds before grouping by complex, and a mean of per-fold results is a
   different number.
   The headline is the **mean** per-structure Spearman at
   T ≥ 10 — see [`SPLITS_AND_METRICS.md`](SPLITS_AND_METRICS.md) for why, and
   [`../data/splits/splits_skempi_full_cath_kfold/README.md`](../data/splits/splits_skempi_full_cath_kfold/README.md)
   for a complete runnable example that lands on a published number.

> **The `score_*.py` wrappers and `experiments/rescore_perstructure/rescore.py` do not read this
> layout.** They are the working-tree scorers: they resolve predictions under `scratch/results/…/
> `fold_<k>/training_run/` and splits under `scratch/splits_*`, neither of which exists in a clone,
> and `rescore.py` additionally asserts the legacy S1102 fold shape. They ship as a record of how
> the numbers were produced, not as a route to reproduce them. The three steps above are that
> route, and `mulan.metrics` is the same code those scripts call.

**The CATH tier differs on both counts**, because its split is built rather than shipped. Build it
first (one command, see the 🔧 row above), and note that `results/full_skempi_cath/` ships
predictions as a single unkeyed column joined **by position** to `fold_0/skempi_all_test.tsv`, not
by `(chain1, chain2, mutation)`. Keyed rows would disclose the partition that is being withheld.
[`data/splits/splits_skempi_full_cath_kfold/README.md`](../data/splits/splits_skempi_full_cath_kfold/README.md)
carries a worked example that lands on the headline 0.4378.

Metric values in `all_results.json` are those written by the training run itself and carry
platform-dependent floating-point differences: the same fold trained on MPS and on CUDA will not
produce bit-identical numbers. Compare within a platform, or recompute from the predictions.

**One reported number is outside this path: FoldX alone.** The 0.383 that
[`../README.md`](../README.md) sets beside the headline is scored from the FoldX channel itself,
not from any model's predictions, so it needs the annotated splits in the ❌ row above rather than
anything under `results/`. Every *model* number in [`RESULTS.md`](RESULTS.md) is reproducible by
the three steps above, except the augmented (`+aug`) cells of §24, whose folds do not ship
(`results/README.md` says why); the FoldX-alone comparator currently is not either, and closing that gap means
shipping the merged channel splits or making the merger runnable against `data/`.
