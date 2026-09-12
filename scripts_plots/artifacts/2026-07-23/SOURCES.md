# Sources & provenance — 2026-07-23 leakage / split artifacts

> **SUPERSEDED.** This index records what the 2026-07-23 artifacts were built from, on that date.
> The CATH per-structure ρ of 0.411 it names was later retracted; the corrected value is 0.438
> (`docs/RESULTS.md` §25). Read it as provenance, not as a source of numbers.

This directory holds the corrected **leakage_explainer.html** and **split_matrix.html**
(plus the sibling benchmark/metric artifacts). This file is a **pointer index**, not a
copy of the underlying docs: the sources below live in the repo and evolve, so they are
referenced by path + last-modified time rather than duplicated here (copies go stale).

- **Snapshot / access time (all pointers verified):** `2026-07-23T09:52:14Z`
- **Repo root for all paths below:** the repository root (i.e. `<repo>/…`)

## Artifacts in this directory

Two classes of file live here. **Snapshot-unique** HTML has no other on-disk source, so the
copy here *is* the source of record. **Mirror-of-live** files are byte-copies of a file that
still evolves under `mulan/experiments/`; they are pinned point-in-time snapshots and will
drift from the live source — check the sha256 against the live path before trusting them.

| file | class | sha256 | snapshot mtime (UTC) | source of record / live path |
|---|---|---|---|---|
| `leakage_explainer.html` | snapshot-unique | `c923f70336e8e2719c7e0bc9e604eaedda92768aa7a5468aa62cff7c167749dd` | 2026-07-23T09:43:24Z | **authored directly in HTML** — no other source |
| `split_matrix.html` | snapshot-unique | `9bba0a1a07322029b604c2219605bcf4108128d48309c1c325af63004d285a0c` | 2026-07-23T09:43:25Z | **authored directly in HTML** — no other source |
| `benchmark_matrix.html` | mirror-of-live | `929e598f5fef8eb2197a65e0cff371a99572ace902205342895ce366f1c22897` | 2026-07-23T09:45:27Z | live: `experiments/benchmark_matrix.html` |
| `metric_rationale.html` | mirror-of-live | `548e2b722f0e26adfe6e7ccec0814fd4e9c48274dc8a3d48afa41b3479267d68` | 2026-07-23T09:45:27Z | live: `experiments/metric_rationale.html` |
| `gen_benchmark_matrix_data.py` | mirror-of-live | `49f5248d5f0715b6dc709313905964bdcf6c0e6498acaaa2aa10ae2d1623d53c` | 2026-07-23T09:45:27Z | live: `experiments/gen_benchmark_matrix_data.py` (regenerates the matrix arrays) |

> **2026-09-09:** a SUPERSEDED banner was inserted at the top of the four HTML files above, so
> they no longer hash to the 2026-07-23 values in the table. As they sit here now:
> `leakage_explainer.html` `cf8048f5387a3ab693fea46cc3ebbe069c0d2e099f9c5494fbc8ac8ddc3b4f30`,
> `split_matrix.html` `3a601684c1cbefc3f092f77d0c918a5ef5344b9d6c866be2221e925a7af4498b`,
> `benchmark_matrix.html` `54b34ea95751d73adc3d670b8a095ccf0bf1c6af1a6d8fbcf7e1bd1b803f88f1`,
> `metric_rationale.html` `064aa7053e68ac434864248edc1b4cd50e4a44515f6e068b33c1d5f0ba161b2c`.
> Remove the inserted `<div ...>SUPERSEDED ...</div>` line to recover the 2026-07-23 bytes; in
> the two files that have no `<body>` tag, also remove the line break the insertion added after
> `</style>`.

> **Why no companion `.md` for the two snapshot-unique HTMLs:** the leakage explainer and
> split matrix were built directly in HTML — there is no markdown draft that was translated.
> The HTML *is* the source of record.

## Companion markdowns — referenced, not copied

These were briefly copied into this directory (2026-07-23) and then **removed on purpose**:
they are living docs under `mulan/experiments/` that keep evolving, so a copy here would go
stale. Pull the live file from the path below; the sha256 / mtime pin the version current at
snapshot time (`2026-07-23T09:55Z`).

| companion (live path under `mulan/`) | pairs with | sha256 (at snapshot) | mtime (UTC) | bytes |
|---|---|---|---|---|
| `experiments/BENCHMARK_MATRIX.md` | `benchmark_matrix.html` | `eccee1648a818b6e1f13de20f3c0c9431d80d7bcac97e0e6272c878b3ecd948b` | 2026-07-23T08:54:04Z | 18047 |
| `experiments/METRIC_RATIONALE.md` | `metric_rationale.html` | `cf3f13791f8b16fed270a1d019946bf1a02d573a1d31c8cbfbcca4039e436aef` | 2026-07-21T22:06:27Z | 8629 |
| `experiments/REVIEW_gaps_and_landscape.md` | leakage/metric-inflation critique (feeds `leakage_explainer.html`, `metric_rationale.html` §) | `5951c91c1c1a4d48c17ffdb1461b509e5730eed9e25de326ec5f04d6439a58ae` | 2026-07-20T08:35:59Z | 24006 |

## Underlying analysis & data (pointers, not copies)

| path (under `mulan/`) | provides | last modified (UTC) | bytes |
|---|---|---|---|
| `experiments/paper_fairness/results/LEAKAGE_AUDIT.md` | SKEMPI leakage + provenance audit → explainer's leakage tiers | 2026-07-10T09:14:36Z | 2373 |
| `experiments/retrain_split/SUMMARY.md` | by-complex retrain, per-structure Spearman (T≥10) | 2026-07-20T08:29:09Z | 14054 |
| `experiments/retrain_split/SUMMARY_clustered.md` | CD-HIT ≤60% clustered retrain, per-structure Spearman | 2026-07-20T08:29:09Z | 14344 |
| `experiments/full_skempi_seqonly/SUMMARY_cath.md` | CATH-superfamily hold-out (USP-ddG's split) retrain → the 0.411 | 2026-07-23T07:57:56Z | 2252 |
| `scripts_plots/results_matrix_ps.csv` | consolidated per-structure Spearman across all splits/models (the companion-table numbers) | 2026-07-23T07:58:29Z | 13148 |
| `experiments/retrain_split/PLAN_RETRAIN_BYCOMPLEX.md` | split-ladder plan / gates | 2026-07-20T08:30:15Z | 12353 |
| `experiments/retrain_split/cluster_split.py` | CD-HIT/mmseqs ≤60% clustering + single-linkage `build_families` | 2026-07-16T01:04:32Z | 11857 |
| `mulan/data.py` (`split_data`) | rng-42 per-mutation CV10 fold assignment | — | — |
| `experiments/cath_leakage/cath_families.json` | CATH superfamily families via SIFTS (both-partners rule); §2c treemap | 2026-07-19T12:13:57Z | 73061 |
| `data/splits/clusters_id60/s1102.tsv` | S1102 mmseqs ≤60% chain clusters | 2026-07-16T01:04:44Z | 3080 |
| `data/splits/clusters_id60/skempi_full_single.tsv` | full-SKEMPI single-point chain clusters; §2b treemap | 2026-07-17T00:05:05Z | 15126 |
| `data/splits/clusters_id60/skempi_full_multi.tsv` | full-SKEMPI multi-point chain clusters; §2b treemap | 2026-07-17T21:52:13Z | 7100 |

> The three cluster tables were under `scratch/clust_mmseqs_*_id60/cluster.tsv` when this snapshot
> was taken, and are now tracked at the paths above, byte-identical. `scripts_plots/plot_treemaps.py`
> regenerates the §2/§2b/§2c treemaps from them; the versions in this directory were hand-authored
> HTML with no generator.

_(The two sibling-artifact markdown sources, `experiments/BENCHMARK_MATRIX.md` and
`experiments/METRIC_RATIONALE.md`, are listed with checksums under **Companion markdowns** above.)_

## Key rendered numbers → where they come from

- **CATH-superfamily (leakage-controlled headline, since retracted — see the note at the top):**
  MuLAN+FoldX **0.411** (full-SKEMPI CATH-all,
  ESM-C 6B + FoldX scalar) vs USP-ddG **0.493** / CATH-ddG **0.494** — `results_matrix_ps.csv`,
  `SUMMARY_cath.md`. MuLAN sits just below on per-structure ρ, competitive on CATH AUROC.
- **Down-the-ladder S1102 base ρ:** 0.489 (leaky) → 0.452 (by-complex) → 0.370 (clustered) —
  `results_matrix_ps.csv`.
- **Clustered tier = CD-HIT ≤60% sequence identity** (MuLAN's retrain split). **iDist ≤0.03
  interface dedup** (PPIRef/PPIformer) is a **leakage diagnostic, not a split** — different
  operation, kept as a reference row in `split_matrix.html` only.
- **Residual-leakage %s** computed from MuLAN's on-disk split inputs
  (S1102_filtered = 1100; skempi_full single/multi = 4165 / 1636).

## Verification note

The copies in this directory are the verified master: each checksum in the table was confirmed
with `sha256sum` on 2026-07-23. The four HTML files changed on 2026-09-09 (banner); their current
checksums are in the note under the table.
