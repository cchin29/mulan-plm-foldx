# Full-SKEMPI ESM-C 6B — base vs FoldX (homology-clustered ≤60% id, 3-fold)

> **⛔ Superseded 2026-07-30 — every FoldX-arm figure below is the 28%-coverage lineage.**
> The FoldX channel these arms read covered ~28% of complexes; it is now 99.2% single-point, and
> the merger that produced the intermediate 87.8% state was itself retracted. The base arm is
> unaffected — it never reads the channel — which is why `docs/RESULTS.md` §23 quotes the same
> 0.193 base value and a scalar arm of **0.417** against the **0.251** below.
> Current numbers: `docs/RESULTS.md` §23 and `scripts_plots/results_matrix_ps.csv`.
> Diagnosis of the coverage defect: `SP_FOLDX_COVERAGE_AUDIT.md`. Kept unedited as the record of
> what the run reported at the time.

_RDE-curated single-point set: 4165 muts, per-fold [1673, 1246, 1246], 315 complexes
(96 with ≥10 muts). All three arms share **identical test keys** → apples-to-apples.
Run finished 2026-07-18 on the GPU box; early-stopping at epochs 39–56._

## Pooled + per-structure metrics

| arm | n | pearson | spearman | rmse | ps_pearson_T10 | ps_spearman_T10 | auroc_destab | auroc_strong | prec@50 |
|---|---|---|---|---|---|---|---|---|---|
| esmc6b_base | 4165 | 0.400 | 0.249 | 1.683 | 0.230 | 0.193 | 0.619 | 0.633 | 0.540 |
| esmc6b_foldx (12-term MLP) | 4165 | 0.342 | 0.298 | 1.688 | 0.263 | 0.205 | 0.607 | 0.697 | 0.500 |
| esmc6b_foldx_scalar | 4165 | 0.407 | 0.323 | 1.701 | 0.272 | **0.251** | 0.646 | 0.688 | 0.580 |

## Gate A — FoldX per-structure Spearman lift over base (T≥10)

- **foldx_scalar: Δ = +0.057** (0.193 → 0.251)
- foldx (12-term MLP): Δ = +0.012 (0.193 → 0.205)

Both positive → FoldX physics adds signal under the leakage-controlled clustered split; the **scalar
ΔΔG term dominates the 12-term MLP**, echoing the S1102 result. Caveat: full-SKEMPI FoldX
coverage is only 28% of complexes, so the FoldX arms lean on a covered subset even though
the *scored* test keys match base.

## Per-fold pooled Spearman (fold_0 = protease-inhibitor mega-family)

| arm | fold_0 | fold_1 | fold_2 |
|---|---|---|---|
| base | 0.383 | 0.184 | 0.221 |
| foldx | 0.464 | 0.192 | 0.186 |
| foldx_scalar | 0.454 | 0.278 | 0.258 |

_Scored via `scratch/score_full_skempi_esmc6b.py` (reuses `rescore.py` core)._
