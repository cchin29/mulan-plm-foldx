# Benchmark matrix — dataset & split coverage at a glance

_Papers first (frontier + the original MuLAN paper); this work's runs expanded by embedding × arm at
the bottom. All MuLAN numbers are **per-structure Spearman ρ (T≥10)**._

**The `CATH` column is the primary comparison. `byCplx` and the `CD-HIT` columns are reference**,
carried so this work can be set against papers that report no CATH number. That ordering is not
only about leakage control: CATH is the one tier where MuLAN's FoldX channel and the frontier's
FoldX are *verified* to agree — 0.006 apart at 100% coverage on both sides, **on AUROC**. On
per-structure Spearman the same tier's two FoldX rows are 0.047 apart (0.430 published against
0.383 here, on 813 rows against 687), so the shared baseline holds for the metric that was checked
and not for the one the tables are written in. The by-complex tiers differ by 0.06 and the
clustered tier by 0.218 — quote those as reference, never as a headline.

**Split key (introduced by):** `CV10` leaky per-mutation 10-fold (classical/MuLAN lineage) · `byCplx`
whole-PDB-out 3-fold (**RDE-Network**, 2023) · `iDist≤.03` interface dedup (**PPIRef/PPIformer**, 2024) ·
`CD-HIT ≤60%` sequence identity (**ProtBFF**, 2025; MuLAN uses this) · `CATH` superfamily hold-out TM<0.6
(**CATH-ddG**, 2025; reused by USP-ddG).
_MuLAN clusters by **sequence identity**, not by **iDist ≤0.03 interface deduplication** (PPIRef/PPIformer) —
a different criterion (see ††)._ **Value tags:** bare = per-structure/per-PPI Spearman ·
`ᵖ` pooled Spearman · `ᵖᶜ` pooled Pearson · `ᴿ` re-evaluated on CATH by
USP-ddG / CATH-ddG (not the method's own paper) · `ˢ` secondary source (Prompt-DDG Table 1, arXiv
2405.10348) · `ᴮ` BA-DDG Table 1 (arXiv 2410.09543) · `ᶜ` CATH-ddG Suppl. Table S2 ·
`●` uses split, value not in hand · `—` not used. OOD columns = sets shared by ≥2 papers; one-offs in **Other**.

## Papers (frontier + original MuLAN)

_Green = per-structure Spearman ρ (T≥10). Clustered tier is split by criterion+metric: **iDist≤0.03**
(interface dedup, PPIformer per-protein Sp) vs **CD-HIT≤60%** (sequence identity, MuLAN & ProtBFF) shown
as a **whole-set** Spearman & Pearson — ProtBFF reports no per-structure figure on this split, so the
MuLAN table below carries both the fold-averaged and the pooled form of it. Per-structure Spearman on a
sequence-clustered split is not reported by any frontier paper, so it has no comparator._

| Method — Date | SKEMPI | S/M | SARS | HER2 | AbBiB | Other | CV10ᵖᶜ | byCplx | CATH | iDist≤.03 | **CD-HIT Sp** | CD-HIT Ps |
|---|---|:--:|:--:|:--:|:--:|---|:--:|:--:|:--:|:--:|:--:|:--:|
| RDE-Network — 2023 (ICLR) | full | S+M | — | — | — | — | — | 0.401ˢ | 0.288ᴿ | — | — | — |
| DiffAffinity — 2023 (NeurIPS) | full | S+M | — | — | — | — | — | 0.397ˢ | 0.249ᴿ | — | — | — |
| PPIformer — 2024 (ICLR) | full | S+M | ● | — | — | staphylokinase | — | — | 0.216ᴿ | 0.44 | — | — |
| Prompt-DDG — 2024 (ICML) | full | S+M | — | — | — | — | — | 0.426ˢ | 0.303ᴿ | — | — | — |
| GearBind — 2024 (Nat. Commun.) | full | S+M | ● | ● | — | CR3022 | — | 0.525ᵖ | — | — | — | — |
| BA-DDG — 2025 (ICLR) | full | S+M | — | — | ● | — | — | 0.513ᴮ | 0.402ᴿ | — | — | — |
| CATH-ddG — 2025 (Bioinf./ISMB) | full | S+M | — | ● | — | — | — | 0.460ᶜ | **0.494** | — | — | — |
| USP-ddG — 2025-11 (bioRxiv) | full | S+M | ● | ● | ● | — | — | — | **0.493** | — | — | — |
| ProSST (base) — 2024 (NeurIPS) | full | S+M | — | — | — | — | — | — | — | — | 0.354ᵖ | 0.428ᵖ |
| ProMIM — 2024 (arXiv) | full | S+M | ● | — | — | — | — | — | — | — | 0.464ᵖ | 0.486ᵖ |
| ProtBFF — 2025-12 (bioRxiv) | full | S+M¶ | ● | — | — | — | — | — | — | — | **0.477ᵖ** | **0.514ᵖ** |
| **MuLAN+Ankh — 2024-08 (bioRxiv)** | **S1102** | **S**‡ | — | — | — | — | **≈0.85ᵖᶜ** | — | — | — | — | — |
| **MuLAN SKEMPI best — this work** | **full** | **S**§§ | — | — | — | — | — | **0.467** | **0.438** | — | **0.509** | **0.548** |
| **MuLAN S1102 best — this work** | **S1102** | **S** | — | — | — | — | **0.892** | **0.552** | — | — | **0.516** | **0.648** |

ˢ by-complex per-structure Spearman from **Prompt-DDG Table 1** (arXiv 2405.10348, HTML) — the standard
RDE 3-fold split, secondary source (the primary RDE/DiffAffinity tables are bitmaps). Same table: FoldX
0.369, DDGPred 0.341, RDE-Linear 0.263.
ᴮ **BA-DDG 0.513** from **BA-DDG Table 1** (arXiv 2410.09543, LaTeX source) — per-structure Spearman,
3-fold by-complex (per-structure Pearson 0.545). The same table confirms the ˢ values exactly
(RDE 0.4010 / DiffAffinity 0.3970 / Prompt-DDG 0.4257). The "prior SoTA 0.4324" BA-DDG beats is
**Surface-VQMAE** (0.4324) / ProMIM (0.4310) — not GearBind.
**GearBind byCplx `0.525ᵖ`** is **pooled** Spearman on its own **split-by-complex 5-fold CV**
(GearBind Table 1, doi:10.1038/s41467-024-51563-8; GearBind 0.498 / GearBind+P 0.525 / Ensemble 0.643,
Pearson 0.659/0.676/0.729). GearBind reports **no per-structure** number, so `ᵖ` flags it as not
directly comparable to the per-structure byCplx cells.
ᶜ **CATH-ddG 0.460** (by-complex, per-PPI Spearman, all mutations) from its own **Supplementary Table S2**
(btaf228, RDE 3-fold split; the same table re-confirms RDE 0.4010 / DiffAffinity 0.3970 / Prompt-DDG 0.4257).
**USP-ddG reports no by-complex number by design** — it evaluates only on CATH (arguing by-complex is
the leaky baseline), so its byCplx cell is `—`. All CATH cells are now verified against **USP-ddG
Table 1** (v2 PDF, primary): USP-ddG 0.4927 · CATH-ddG 0.4940 · BA-DDG 0.4021 · RDE 0.2880 · DiffAffinity 0.2494.
ᴿ CATH cells are USP-ddG / CATH-ddG **re-evaluations** of these baselines on the CATH hold-out
(`../data/benchmarks/frontier.tsv`, from USP-ddG Table 1 + CATH-ddG's verified reproduction) — not the
method's own paper. **CATH-ddG 0.494 / USP-ddG 0.493** (bold) are their own results.
†† PPIformer's split (`iclr24`) uses **iDist ≤0.03 interface deduplication** (PPIRef) — an *interface*-
similarity criterion, **not** MuLAN's ≤60% *sequence*-identity clustering, so it's bucketed under `clust`
only loosely. The per-protein Spearman **0.44** is from **CATH-ddG Table 3** ("PPIFORMER data splitting", btaf228).
¶ Marked `S` here until 2026-08-05, which was wrong, and it was the only frontier row so marked.
ProtBFF's set is SKEMPI2 minus the 10 largest complexes ≈ **6,631 mutations** — and
`../docs/history/PLAN_FULL_SKEMPI.md` §2, which is where that convention and that count are
recorded, states it over **single and multiple** mutations. SKEMPI2 does not hold 6,631 single-point
entries, so `S` cannot be right. The marking mattered: it is the only thing that had kept the
clustered panels of `ppS_scaling` on the single-point rung while every other panel used single+multi.

§§ Marked `S+M` here until 2026-08-05, which was wrong: every cell in this row comes from the
single-point full-SKEMPI table below. MuLAN's combined single+multi rungs exist and are scored, but they
are a separate tier — the frontier rows in this table are `S+M`, so reading this row against them
overstates the match. Cells are the best arm per column, refreshed at 99.2% FoldX coverage (they read
0.354 / 0.411 / 0.324 / 0.407 at the 28% lineage).

‡ MuLAN single-point runs; sibling **S1400** covers multi-point. `≈0.85ᵖᶜ` = the leaky-regime pooled
Pearson MuLAN historically reported — the inflated metric (see `METRIC_RATIONALE.md`).

## MuLAN — full-SKEMPI (this work, 2026-07)

_`byCplx` & `CATH` = **per-structure Spearman** (single-point; 96 / 13 complexes). The three `CD-HIT`
columns are all the CD-HIT ≤60% clustered split (4,165 single-point muts), the tier that matches ProtBFF
(0.477 / 0.514) — **fold-avg** correlates within each test fold and averages the three, **pooled**
correlates once over the union. Both are carried because ProtBFF's own column has been read both ways
here, and the two differ by up to 0.05 on a given arm; the ProtBFF verdict happens to be the same under
either. Best per column **bold**. Regenerate with `gen_benchmark_matrix_data.py` (pooled, byCplx, CATH)
and `score_sp_all.py clustered` (fold-avg, via `results_matrix_ps.csv`)._

| Embedding | arm | byCplx (ps-Sp) | CATH (ps-Sp) | CD-HIT Sp (fold-avg) | CD-HIT Sp (pooled) | CD-HIT Ps (pooled) |
|---|---|:--:|:--:|:--:|:--:|:--:|
| ESM-C 6B | base | 0.310 | 0.360 | 0.263 | 0.249 | 0.400 |
| ESM-C 6B | + FoldX scalar | 0.451 | **0.438** | 0.469 | 0.472 | **0.548** |
| ESM-C 6B | + FoldX MLP | 0.457 | 0.431 | 0.504 | 0.484 | 0.522 |
| ESM-C 600M | base | 0.227 | 0.117 | 0.247 | 0.282 | 0.406 |
| ESM-C 600M | + FoldX scalar | 0.434 | 0.351 | 0.321 | 0.270 | 0.241 |
| ESM-C 600M | + FoldX MLP | **0.467** | 0.370 | 0.475 | 0.480 | 0.484 |
| SaProt | base | 0.120 | 0.229 | 0.054 | 0.033 | 0.081 |
| SaProt | + FoldX scalar | 0.411 | 0.398 | 0.410 | 0.386 | 0.409 |
| SaProt | + FoldX MLP | 0.435 | 0.397 | 0.492 | 0.483 | 0.484 |
| ProstT5 | base | 0.194 | 0.070 | 0.071 | 0.032 | 0.086 |
| ProstT5 | + FoldX scalar | 0.452 | 0.415 | 0.453 | 0.430 | 0.439 |
| ProstT5 | + FoldX MLP | 0.464 | 0.398 | 0.492 | 0.480 | 0.458 |
| ESM2-3B | base | 0.261 | 0.243 | 0.213 | 0.231 | 0.227 |
| ESM2-3B | + FoldX scalar | 0.459 | 0.375 | 0.452 | 0.456 | 0.490 |
| ESM2-3B | + FoldX MLP | 0.445 | 0.352 | 0.501 | 0.488 | 0.489 |
| Ankh-large | base | 0.284 | 0.331 | 0.189 | 0.186 | 0.260 |
| Ankh-large | + FoldX scalar | 0.435 | 0.421 | 0.462 | 0.466 | 0.482 |
| Ankh-large | + FoldX MLP | 0.444 | 0.387 | **0.505** | **0.509** | 0.529 |
| Ankh3-large | base | 0.203 | 0.148 | 0.108 | 0.093 | 0.146 |
| Ankh3-large | + FoldX scalar | 0.436 | 0.355 | 0.428 | 0.411 | 0.440 |
| Ankh3-large | + FoldX MLP | 0.449 | 0.392 | 0.476 | 0.466 | 0.463 |
| Ankh3-xl | base | 0.268 | 0.266 | 0.225 | 0.247 | 0.376 |
| Ankh3-xl | + FoldX scalar | 0.463 | 0.414 | 0.456 | 0.449 | 0.475 |
| Ankh3-xl | + FoldX MLP | 0.454 | 0.414 | 0.490 | 0.482 | 0.494 |
| AIDO-16B | base | 0.180 | 0.248 | 0.145 | 0.182 | 0.233 |
| AIDO-16B | + FoldX scalar | 0.447 | 0.417 | 0.442 | 0.432 | 0.456 |
| AIDO-16B | + FoldX MLP | 0.435 | 0.377 | 0.490 | 0.487 | 0.497 |

## MuLAN — S1102 ladder (this work, 2026-07)

_`CV10`/`byCplx` = **per-structure Spearman** (leaky → by-complex); `CD-HIT Sp/Ps` = **pooled** Spearman &
Pearson on the S1102 CD-HIT ≤60% split. **NB: S1102 is a curated subset from the original MuLAN project and an easier benchmark — these CD-HIT pooled
numbers (~0.5–0.65) are NOT dataset-matched to ProtBFF; the full-SKEMPI table above (0.509/0.548) is.**
`CV10 (P)` = leaky pooled Pearson. Best per column **bold**._

| Embedding | arm | CV10 (P) | CV10 (ps-Sp) | byCplx (ps-Sp) | CD-HIT Sp (pooled) | CD-HIT Ps (pooled) |
|---|---|:--:|:--:|:--:|:--:|:--:|
| ESM-C 6B | base | 0.886 | 0.489 | 0.452 | 0.448 | 0.642 |
| ESM-C 6B | + FoldX scalar | **0.892** | **0.579** | 0.546 | 0.516 | **0.648** |
| ESM-C 6B | + FoldX MLP | 0.890 | 0.564 | 0.551 | 0.469 | 0.609 |
| ESM-C 600M | base | 0.841 | 0.325 | 0.283 | 0.465 | 0.433 |
| ESM-C 600M | + FoldX scalar | — | — | 0.528 | 0.513 | 0.430 |
| ESM-C 600M | + FoldX MLP | 0.858 | 0.508 | 0.494 | 0.462 | 0.570 |
| SaProt | base | 0.856 | 0.377 | 0.325 | 0.304 | 0.312 |
| SaProt | + FoldX scalar | — | — | **0.552** | 0.415 | 0.384 |
| SaProt | + FoldX MLP | 0.867 | 0.546 | 0.491 | 0.474 | 0.504 |
| ProstT5 | base | 0.829 | 0.412 | 0.296 | 0.078 | 0.210 |
| ProstT5 | + FoldX scalar | — | — | 0.549 | 0.402 | 0.457 |
| ProstT5 | + FoldX MLP | 0.842 | 0.568 | 0.453 | 0.430 | 0.510 |
| ESM2-3B | base | 0.838 | 0.354 | 0.325 | 0.362 | 0.415 |
| ESM2-3B | + FoldX scalar | — | — | 0.480 | 0.514 | 0.522 |
| ESM2-3B | + FoldX MLP | 0.857 | 0.564 | 0.474 | **0.528** | 0.549 |
| Ankh-large | base | 0.844 | 0.387 | 0.289 | 0.311 | 0.422 |
| Ankh-large | + FoldX scalar | — | — | 0.495 | 0.442 | 0.513 |
| Ankh-large | + FoldX MLP | 0.860 | 0.521 | 0.482 | 0.438 | 0.530 |
| Ankh3-large | base | — | — | 0.281 | 0.193 | 0.305 |
| Ankh3-large | + FoldX scalar | — | — | 0.523 | 0.358 | 0.402 |
| Ankh3-large | + FoldX MLP | — | — | 0.436 | 0.420 | 0.467 |
| Ankh3-xl | base | — | — | 0.349 | 0.317 | 0.387 |
| Ankh3-xl | + FoldX scalar | — | — | 0.540 | 0.430 | 0.459 |
| Ankh3-xl | + FoldX MLP | — | — | 0.457 | 0.447 | 0.509 |
| MINT (multi-chain) | base | 0.856 | — | 0.316 | 0.167 | 0.245 |

## Read it in three passes

1. **SKEMPI + S/M (papers table):** every frontier row is `full v2 · S+M`; the original **MuLAN** row is
   the lone `S1102 · single`; MuLAN full-SKEMPI (below) is the only MuLAN configuration at `full · S+M`.
2. **Metrics differ by column:** `byCplx` & `CATH` are per-structure Spearman (frontier-comparable);
   `CD-HIT Sp/Ps` are whole-set, carried **both fold-averaged and pooled** (ProtBFF reports one number
   on this split and it has been read as each; the MuLAN verdict is the same either way); `iDist≤.03` is PPIformer's own
   interface-dedup split. MuLAN-original sits alone under `CV10` with a **pooled Pearson ≈0.85** (the
   inflated metric).
3. **Two clustered comparisons, don't conflate them:** the sequence-clustered (CD-HIT ≤60%) tier is only
   frontier-comparable to **ProtBFF**, and only on a whole-set correlation — never per-structure.
   Dataset-matched is the **combined single+multi** clustered rung (5,801 muts), not the single-point one
   — ProtBFF trains over single and multiple (see ¶). There MuLAN+FoldX is **0.561 P / 0.549 S pooled**,
   **0.547 S fold-averaged**, vs ProtBFF **0.514 / 0.477** → **MuLAN above, on both readings of the
   metric.** This line said "MuLAN below" off **0.407 P / 0.324 S** until 2026-08-05; those were
   28%-FoldX-coverage numbers on the single-point rung. The S1102 CD-HIT numbers (~0.65 P / 0.52 S)
   remain higher only because S1102 is an easy subset — not dataset-matched to ProtBFF.

   **Do not quote this comparison — it is the one rung where the baselines do not match.** Our
   unsupervised FoldX-alone here is **0.521 fold-averaged**, already above ProtBFF's reported 0.477
   before any PLM is involved, while ProtBFF's Table 2 puts FoldX at **0.294**. That 0.218 gap was
   traced on 2026-08-05 (`rescore_perstructure/FOLDX_ALONE_BASELINE_RESULT.md`, "Distance between this
   FoldX and the frontier's") and it is **not** a stronger FoldX pipeline: on the CATH tier, the one at
   100% coverage on both sides, our FoldX and the literature's agree to **0.006**. On the two
   by-complex tiers the gap is a consistent +0.06 from two independent tables — a coverage advantage on
   the full set. The clustered rung is 3.6× that, and the excess sits in exactly the components a
   lower-coverage FoldX loses first: multi-point rows and whole-complex ordering.

   Because **ProtBFF is the same idea as MuLAN's FoldX arms** — biophysics into a frozen PLM — its
   headline inherits its FoldX baseline. The lift over each method's own FoldX runs the other way:
   ProtBFF **+0.183**, MuLAN **+0.025**. So 0.547 vs 0.477 is at least as consistent with better FoldX
   coverage as with a better model. **Quote the by-complex tier instead**, where the baselines differ
   by 0.06 rather than 0.218 and MuLAN's +0.049 lift sits mid-field — above RDE-Network (+0.032),
   level with Prompt-DDG (+0.057), below CATH-ddG (+0.091) and BA-DDG (+0.144).

## The CATH column — the primary comparison

**CATH is the tier to lead on. The other two are reference tiers**, carried for papers that report
no CATH number. Two reasons, and the second was only established on 2026-08-05:

1. It is the most leakage-controlled hold-out here — superfamily-level, TM < 0.6 to train.
2. **It is the only tier where this FoldX and the frontier's are verified to agree.** Both sides
   are at 100% channel coverage on its 687 mutations, and the two FoldX numbers differ by 0.006
   (AUROC 0.760 vs 0.754). Every comparison on this tier therefore rests on a shared baseline. On
   the by-complex tiers the baselines differ by 0.06, and on the clustered tier by 0.218 — see §3
   of "Read it in three passes".

Per-structure Spearman on the CATH-superfamily hold-out, all-muts, 687 mutations over 13 complexes
at T ≥ 10. **FoldX alone is placed in the ranking**, because a reader cannot otherwise tell how
much of any row was free:

| # | method | CATH ps-Sp |
|--:|---|--:|
| 1 | CATH-ddG | 0.494 |
| 2 | USP-ddG | 0.493 |
| 3 | flex-ddG | 0.454 |
| 4 | **MuLAN + FoldX** (ESM-C 6B, scalar) | **0.438** |
| 5 | FoldX | 0.430 |
| 6 | BA-DDG | 0.402 |
| 7 | **FoldX alone** (unsupervised) | **0.383** |
| 8 | Prompt-DDG | 0.303 |
| 9 | RDE-Network | 0.288 |
| 10 | DiffAffinity | 0.249 |
| 11 | PPIformer | 0.216 |

→ MuLAN + FoldX (frozen sequence PLM, no structure at inference) lands **4th of eleven — above
BA-DDG, the interface-aware transformers (Prompt-DDG, PPIformer) and the pretraining GNNs (RDE,
DiffAffinity)** — and below the two relaxed-structure SOTA models and flex-ddG. The row immediately
beneath it is the published FoldX baseline at 0.430, 0.008 away and on a different row set, so this
is a place on the leaderboard rather than a margin over the field's physics.

→ **Two readings, and both belong in the same table.** MuLAN's lift over the physics it is handed
is **+0.055** [+0.001, +0.118] — the only one of the eighteen CATH arms clearing the cost-free
comparator with a CI excluding zero. And **FoldX alone at 0.383 already outranks four of the seven
published models**, which is a finding about the benchmark as much as about any method.

This section read `MuLAN+FoldX 0.411` until 2026-08-05 — the 28%-FoldX-coverage lineage.

## AUROC (sign-of-effect) — companion

The frontier's classification metric (destabilizing vs not). **Clean comparison = CATH** (both all-muts),
where MuLAN ranks **2nd, above CATH-ddG** — far more competitive than its per-structure Spearman.

| Method | CATH (all-muts) | by-complex |
|---|:--:|:--:|
| USP-ddG | **0.802** | — |
| **MuLAN + FoldX (ESM-C 6B)** | **0.791** | 0.732§ |
| CATH-ddG | 0.781 | 0.776 |
| flex-ddG | 0.764 | — |
| FoldX | 0.754 | 0.658 |
| RDE-Network | 0.745 | 0.745 |
| BA-DDG | 0.732 | 0.773 |
| DiffAffinity | 0.625 | 0.744 |
| Prompt-DDG | — | 0.757 |
| ProMIM | — | 0.760 |

§ MuLAN by-complex **0.732** is now **all-muts** (pools MuLAN's separate single-point 0.726 + multi-point
0.772 runs — two models, vs the frontier's one), so composition-matched to the frontier column. Still
below the DL frontier (0.74–0.78), above FoldX — MuLAN's AUROC edge shows on the harder **CATH** tier
(0.791, 2nd), not the easier by-complex. AUROC = MuLAN `auroc_destab`; CATH from USP-ddG Table 1,
by-complex from BA-DDG Table 1 / CATH-ddG Suppl. S2. ProtBFF & GearBind report no AUROC. **Precision@k is
MuLAN-only** — it lives in `METRIC_RATIONALE.md`, not here. Regenerate MuLAN AUROC (incl. SP/MP/all-muts
and per-arm) via `gen_benchmark_matrix_data.py`.

<details><summary><b>MuLAN AUROC by embedding × arm</b> (best CATH <b>0.806</b> = Ankh-large base; best by-complex all-muts <b>0.760</b> = Ankh3-xl + FoldX MLP)</summary>

| Embedding | arm | CATH | by-complex (all-muts) |
|---|---|:--:|:--:|
| ESM-C 6B | base | 0.787 | 0.691 |
| ESM-C 6B | + FoldX scalar | 0.791 | 0.732 |
| ESM-C 6B | + FoldX MLP | 0.748 | 0.743 |
| ESM-C 600M | base | 0.732 | 0.663 |
| ESM-C 600M | + FoldX scalar | 0.779 | 0.721 |
| ESM-C 600M | + FoldX MLP | 0.777 | 0.719 |
| SaProt | base | 0.716 | 0.588 |
| SaProt | + FoldX scalar | 0.774 | 0.674 |
| SaProt | + FoldX MLP | 0.750 | 0.679 |
| ProstT5 | base | 0.700 | 0.629 |
| ProstT5 | + FoldX scalar | 0.766 | 0.714 |
| ProstT5 | + FoldX MLP | 0.745 | 0.754 |
| ESM2-3B | base | 0.745 | 0.658 |
| ESM2-3B | + FoldX scalar | 0.786 | 0.725 |
| ESM2-3B | + FoldX MLP | 0.736 | 0.730 |
| Ankh-large | base | **0.806** | 0.680 |
| Ankh-large | + FoldX scalar | 0.805 | 0.740 |
| Ankh-large | + FoldX MLP | 0.748 | 0.755 |
| Ankh3-large | base | 0.724 | 0.663 |
| Ankh3-large | + FoldX scalar | 0.791 | 0.720 |
| Ankh3-large | + FoldX MLP | 0.718 | 0.732 |
| Ankh3-xl | base | 0.781 | 0.682 |
| Ankh3-xl | + FoldX scalar | 0.791 | 0.743 |
| Ankh3-xl | + FoldX MLP | 0.750 | **0.760** |
| AIDO-16B | base | 0.680 | 0.654 |
| AIDO-16B | + FoldX scalar | 0.780 | 0.720 |
| AIDO-16B | + FoldX MLP | 0.754 | 0.721 |

</details>

## Caveats to state

- **OOD gap:** no MuLAN row touches SARS-CoV-2 / HER2 / AbBiBench — the one axis still unmatched.
- **`byCplx` across datasets:** frontier by-complex per-structure ≈ RDE 0.401 / DiffAffinity 0.397 /
  Prompt-DDG 0.426 / **CATH-ddG 0.460** / **BA-DDG 0.513** (the structure-based ceiling). MuLAN
  **full-SKEMPI** by-complex (single-point) is **0.467** — inside the band, above CATH-ddG 0.460 and
  below the BA-DDG ceiling; it read 0.354 ("just under this band") at the 28% FoldX coverage lineage;
  MuLAN **S1102** by-complex is 0.55, inflated by the easy subset. (GearBind's 0.525 is *pooled*
  Spearman on a 5-fold by-complex split — not per-structure, so not in the same comparison.)
- **CD-HIT ≤60% leaderboard (from ProtBFF Table 2 + our full-SKEMPI single+multi rung):**
  Pearson — **MuLAN+FoldX 0.561** (ESM-C 6B + scalar) > ProtBFF **0.514** > ProMIM 0.486 >
  ProSST-base 0.428 > ESM2-base 0.194. Spearman — **MuLAN+FoldX 0.549 pooled / 0.547 fold-averaged**
  (ESM-C 6B + MLP) > ProtBFF **0.477**, so the ordering does not turn on which reading of the column is
  right. Both sides are single+multi (see ¶); the sets are **6,631 vs 5,801** muts — close, not
  identical, and ours is the more heavily filtered.
  **This leaderboard is not quotable as a win.** The FoldX baselines differ by 0.218 (ours 0.521,
  ProtBFF Table 2 **0.294**) and ProtBFF is a FoldX-into-PLM method, so its headline carries that
  handicap forward. Lift over each method's own FoldX: ProtBFF +0.183, MuLAN +0.025. See §3 of "Read it
  in three passes", and quote the by-complex tier instead.
  MuLAN read 0.407 P / 0.324 S here until 2026-08-05 — 28%-coverage FoldX numbers on the single-point
  rung, three lineages stale.
  Regenerate MuLAN's pooled numbers with `experiments/gen_benchmark_matrix_data.py` when new folds land.
- **iDist ≤0.03 ≠ CD-HIT ≤60%:** PPIformer's 0.44 is on an *interface*-deduplicated split, a different
  criterion — kept in its own column.
- **`ᴿ` CATH cells are re-evaluations** by USP-ddG / CATH-ddG, not the baselines' own papers.
- **The ProtBFF comparison must use the full-SKEMPI 0.548 P / 0.509 S numbers**, not the S1102
  clustered ones — those are not dataset-matched.

---

_MuLAN values from `scripts_plots/results_matrix_ps.csv`, `experiments/full_skempi_seqonly/results_cath.csv`,
and per-structure Spearman computed from the full-SKEMPI SP prediction files (`scratch/results/full_skempi{,_bycomplex}`).
Paper cells verified from the primary tables: BA-DDG Table 1 (byCplx), USP-ddG Table 1 + CATH-ddG
(CATH), Prompt-DDG Table 1 (RDE/DiffAffinity/Prompt-DDG byCplx), ProtBFF Table 2 (clustered)._

## References (DOIs)

- **SKEMPI v2** — Jankauskaitė et al., *Bioinformatics* 2019 · doi:10.1093/bioinformatics/bty635
- **MuLAN** — bioRxiv 2024 · doi:10.1101/2024.08.24.609515
- **RDE-Network** — Luo et al., ICLR 2023 · bioRxiv doi:10.1101/2023.02.28.530137 · OpenReview _X9Yl1K2mD
- **DiffAffinity** — Liu et al., NeurIPS 2023 · arXiv:2310.19849
- **PPIformer** — Bushuiev et al., ICLR 2024 · arXiv:2310.18515
- **Prompt-DDG** — Wu et al., ICML 2024 · arXiv:2405.10348
- **GearBind** — Cai et al., *Nat. Commun.* 2024 · doi:10.1038/s41467-024-51563-8 · arXiv:2304.08818 (note: 10.1038/s41467-024-49798-6 is a *different* paper, FSFP)
- **BA-DDG** — ICLR 2025 · arXiv:2410.09543
- **CATH-ddG** — Yu et al., *Bioinformatics* 2025 (ISMB) · doi:10.1093/bioinformatics/btaf228
- **USP-ddG** — Yu et al., *Bioinformatics* 2026 (ISMB) · doi:10.1093/bioinformatics/btag249
- **ProtBFF** — bioRxiv 2025 · doi:10.64898/2025.12.23.696257
- **ProSST** — Li et al., NeurIPS 2024 · bioRxiv doi:10.1101/2024.04.15.589672
- **ProMIM** — Mo et al., 2024 · arXiv:2405.17802

**Split provenance:** by-complex 3-fold — RDE-Network (doi:10.1101/2023.02.28.530137); iDist ≤0.03 interface dedup —
PPIRef/PPIformer (arXiv:2310.18515, Zenodo 10.5281/zenodo.12789167); CD-HIT ≤60% sequence identity —
ProtBFF (doi:10.64898/2025.12.23.696257), clustering tool CD-HIT (Fu et al. 2012, doi:10.1093/bioinformatics/bts565);
CATH-superfamily hold-out — CATH-ddG (doi:10.1093/bioinformatics/btaf228, reused by USP-ddG).

_Also cited: Surface-VQMAE (BA-DDG's prior per-structure SoTA, 0.432) and ProMIM (0.431); FoldX 5 —
foldxsuite.crg.eu._
