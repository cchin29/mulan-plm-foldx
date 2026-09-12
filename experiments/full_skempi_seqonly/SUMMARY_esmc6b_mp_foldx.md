# Multi-point ESM-C 6B + FoldX — leakage-controlled-split summary

_ESM-C 6B, full-SKEMPI **multi-point** subset (≥2 simultaneous mutations), 3-fold CV,
300ep/patience-30, cached embeddings. Two leakage-controlled splits: **clustered id60** (headline —
no ≥60%-identity complex shared train/test) and **by-complex** (seed 42). Scored
2026-07-18 via `scratch/score_mp_esmc6b.py` (reuses `rescore.py`; per-structure Spearman
over complexes with T≥10 mutations). n = 1636 mutations / 146 complexes / 43 scorable
structures. **98.7% FoldX coverage** (vs 28% for single-point full-SKEMPI) — a cleaner
physics test._

## Per-structure Spearman T≥10 (the conclusion metric)

| split | base | foldx (12-term MLP) | foldx_scalar | Δ MLP | Δ scalar |
|-------|-----:|--------------------:|-------------:|------:|---------:|
| **clustered (headline)** | 0.137 | **0.320** | 0.270 | **+0.182** | +0.132 |
| by-complex | 0.374 | **0.423** | 0.391 | **+0.049** | +0.016 |

Pooled (context only): clustered pearson base 0.288 → foldx 0.545 → scalar 0.447;
by-complex 0.595 → 0.657 → 0.543.

## Two conclusions

1. **FoldX physics generalizes — and the lift WIDENS under the harder split.** MLP-FoldX
   per-structure lift grows by-complex **+0.049 → clustered +0.182**; scalar likewise
   **+0.016 → +0.132**. Gate A **PASS** on both splits, both arms. This is the "physics
   generalizes" signature (lift increases exactly where sequence homology can't leak).

2. **On multi-point the 12-term MLP BEATS the scalar — an inversion of the single-point
   story.** clustered: MLP 0.320 > scalar 0.270 (+0.050); by-complex: 0.423 > 0.391
   (+0.032). Contrast single-point ESM-C (S1102 + full-SKEMPI single), where the scalar
   Interaction-Energy ≥ the 12-term MLP. **Coherent mechanism:** a single mutation's
   effect is largely captured by one scalar ΔΔG; two+ simultaneous mutations have
   cross-term / per-component structure the scalar collapses but the 12-term
   decomposition retains.

## Single-point cross-check (S1102 300ep CV10) — the inversion, confirmed both ways

Ran the matching **single-point** three-way (S1102, 300ep, 10-fold; `foldx_esmc6b_chain.sh`
+ GPU-offloaded `foldx_mlp` folds; `scratch/foldx_s1102/cv10_esmc6b/summary_mlp.json`),
pooled test PCC:

| arm | PCC | paired Δ vs base | vs scalar |
|-----|----:|-----------------:|----------:|
| baseline | 0.8653 | — | — |
| foldx **scalar** | **0.8710** | +0.0057 (wins 8/10) | — |
| foldx **MLP** | 0.8705 | +0.0052 (wins 7/10) | **−0.0006 (tie)** |

On single mutations scalar ≈ MLP (dead heat, scalar marginally ahead) — the 12 extra
terms add nothing. On multi mutations MLP beat scalar by ~+0.05 (above). Same PLM, same
FoldX pipeline, only mutation multiplicity differs → the decomposition earns its keep
**only** when mutations co-occur. The inversion is real, not a fluke of one split.

## Framing

_FoldX physics helps ESM-C under leakage-controlled splits everywhere, and more so as the split gets
harder. For single mutations the strongest PLM already captures what a scalar ΔΔG adds and
the decomposition is redundant; for multi-mutations the 12-term decomposition carries extra
signal the scalar cannot, and the full MLP wins._

Data: `scratch/results/full_skempi_mp_{clustered,bycomplex}/esmc6b_{base,foldx,foldx_scalar}/`
(json+tsv pulled to Linux; checkpoints left on GPU). Rerun: `python scratch/score_mp_esmc6b.py both`.
