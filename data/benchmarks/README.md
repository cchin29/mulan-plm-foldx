# Comparator numbers

Frontier results transcribed from the source papers, so this repository's figures read them from
one place instead of carrying them inline. Previously the same numbers lived hardcoded in five
scripts, which is how the defect below survived.

`frontier.tsv` — one row per `(method, metric, protocol)`:

| column | meaning |
|---|---|
| `method` | the comparator, as its paper names it |
| `metric` | `ps_spearman_T10` · `pooled_spearman` · `pooled_pearson` · `auroc` |
| `protocol` | **which hold-out and which mutation set the number was measured on** |
| `value` | as printed in the source |
| `source` | paper and table |

## The `protocol` column

A comparator number is comparable to a MuLAN arm **only when the protocol matches**. The values
here are not interchangeable across rows, and treating them as a flat `{method: value}` map is
exactly the mistake that prompted this file.

| `protocol` | what it means |
|---|---|
| `bycomplex_all` | whole-PDB hold-out, **one model trained over single + multi-point together** |
| `cath` | CATH-superfamily hold-out (USP-ddG's) |
| `clustered_id60` | sequence-clustered ≤60% identity |
| `bycomplex` | whole-PDB hold-out, AUROC tables |

**Every method in this file is single+multi.** Not one frontier comparator here trains on
single-point alone, so every MuLAN rung set against them must be a `-ALL` rung. ProtBFF was recorded
as single-point in `BENCHMARK_MATRIX.md` until 2026-08-05, which is what had kept the two clustered
panels of `ppS_scaling` on the single-point rung; its ≈6,631-mutation set is single **and** multiple,
and SKEMPI2 has no such number of single-point entries.

### The defect this fixes

`plot_ppS_scaling.py` and `plot_ppS_generalization.py` carried the *same five numbers*
(RDE-Network, DiffAffinity, Prompt-DDG, CATH-ddG, BA-DDG) attached to *different* rungs.
`plot_ppS_scaling.py` was right and said so in a comment — those methods train one model over
single + multi, so they belong on the combined `-ALL` rung, and it warns "**do NOT move them back
onto `fullSK bycomplex-SP`**". `plot_ppS_generalization.py` did precisely that, plotting them
against MuLAN bars trained on single-point only.

Neither script was wrong about its own numbers; they disagreed about which rung the numbers
described, and nothing could detect that because each held its own copy. With the protocol
recorded beside the value, attaching a `bycomplex_all` number to a single-point rung is a visible
mismatch rather than an invisible one. It stayed visible, and documented, until 2026-09-12, when
`plot_ppS_generalization.py` moved its clustered and by-complex panels to the `-ALL` rungs; the two
scripts now agree on the rung as well as the numbers.

## Deliberately excluded

- **GearBind by-complex 0.525** — *pooled* Spearman, not per-structure. Placing it beside
  per-structure values would flatter it against a different metric.
- **Anything from the clustered tier as a per-structure comparator** — ProtBFF reports one whole-set
  correlation on that split and no per-structure figure, so the clustered per-structure panel has no
  comparator at all. An empty band is the correct output there; the comparable number lives on the
  fold-averaged companion panel.

## Provenance of the `FoldX` rows

Two rows here are FoldX baselines as a frontier paper reports them, drawn on `ppS_scaling` beside
our own measured line rather than instead of it:

| row | reported | ours, same split |
|---|--:|--:|
| `FoldX pooled_spearman clustered_id60` (ProtBFF Table 2) | 0.294 | **0.521** fold-avg |
| `FoldX ps_spearman_T10 bycomplex_all` (Prompt-DDG Table 1) | 0.369 | **0.430** |

The column header says *split*, not *split & metric*, deliberately: the ProtBFF row's key reads
`pooled_spearman` while the source reports that split fold-averaged (see the header comment in
`frontier.tsv`), so the metrics are matched only to within the ~0.05 the two readings differ by.

Ours come from `experiments/rescore_perstructure/foldx_alone_baseline_allplm_cov99.csv`. They are
not interchangeable with these, and the gap matters to any claim built on either: this pipeline
runs RepairPDB before AnalyseComplex and drops complexes it cannot repair, and the mutation sets
differ. **That the by-complex pair differ by 0.06 while the clustered pair differ by 0.23** is the
useful part — it locates the divergence in the clustered protocol rather than in the FoldX pipeline
at large. Treat these rows as comparators, never as a check on our numbers.

**A FoldX row must never enter a frontier band.** `comparators()` returns every method for a
`(metric, protocol)` pair, so any call feeding a band needs an explicit `only=`. Adding the
by-complex FoldX row would otherwise have dropped that band's floor from DiffAffinity 0.397 to
0.369 and added FoldX to the comparator set, with nothing to signal it.

## Provenance and licence

These are figures transcribed from published tables, attributed per row. They are third-party
results, not ours, and are reproduced for comparison. See [`../../NOTICE`](../../NOTICE).

MuLAN's own numbers are **not** here — they come from the per-fold results and are regenerated,
never transcribed. Mixing measured and transcribed values in one file is how a stale copy of a
own result starts looking like a citation.
