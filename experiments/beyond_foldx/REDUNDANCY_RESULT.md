# Redundancy between the base pathway and the FoldX scalar — result

_Computed 2026-08-04. Script: `redundancy.py`. Data: `redundancy.csv` (covered rows, primary),
`redundancy_allrows.csv` (ESM-C 6B, all rows). No training — predictions already on disk._

Queue item 2 of `README.md`. The question was whether the base pathway is redundant with the
FoldX channel, and what ceiling that redundancy places on any rule combining the two.

## Method

All quantities are computed **within complex** on ranks (T ≥ 10, folds pooled), matching ppS.
Conventions are inherited by import from `experiments/rescore_perstructure/foldx_alone_baseline.py`,
so `rho_base` reproduces `results_matrix_ps.csv` and `rho_fx` reproduces FoldX-alone.

`base~fx` is Spearman(base prediction, FoldX scalar). The base arm never receives the FoldX
number, so any agreement is independent convergence.

`part_base` is Spearman(base, truth | FoldX) — the PLM's contribution after removing everything
FoldX already explains.

`blend2` is a **global linear combiner** of the two rank signals, fitted leave-one-complex-out:
features and target are within-complex normalised ranks, the fit uses every row outside the
held-out complex, and ppS is recomputed on the complex held out. That is the analogue of what
training does — one rule for all complexes — and it self-validates: a single-feature FoldX fit is
affine in rank(FoldX) within each complex, so it must return FoldX-alone exactly. It does, to
1e-6, on all 44 (tier, model) cells; the script raises `[BUG ]` otherwise.

A per-complex leave-one-out fit was tried first and **rejected**: at n = 10 each point has enough
leverage over its own held-out prediction to scramble the within-complex ordering, and the
FoldX-only reference returned 0.217 against a true 0.418. The bias is not shared equally by the
one- and two-feature fits, so differences taken across it are uninterpretable.

## Result

Covered rows. `gain` = blend2 − FoldX alone; ✱ = 95% CI excludes 0 (cluster bootstrap over
complexes, B = 10,000, seed 0).

| tier | model | base | FoldX | base~fx | part_base | blend2 | gain |
|---|---|---|---|---|---|---|---|
| clustered-SP | ESM-C 6B | 0.193 | 0.418 | 0.165 | 0.130 | 0.427 | +0.009 |
| | Ankh3-xl | 0.147 | 0.418 | 0.128 | 0.099 | 0.420 | +0.002 |
| | ESM-C 600M | 0.149 | 0.418 | 0.157 | 0.096 | 0.418 | +0.000 |
| | Ankh-large | 0.145 | 0.418 | 0.177 | 0.078 | 0.414 | −0.004 |
| | ESM2-3B | 0.124 | 0.418 | 0.152 | 0.072 | 0.418 | −0.000 |
| | AIDO-16B | 0.122 | 0.418 | 0.156 | 0.060 | 0.418 | −0.000 |
| | ProstT5 | 0.052 | 0.418 | 0.044 | 0.028 | 0.418 | +0.000 |
| bycomplex-SP | ESM-C 6B | 0.311 | 0.418 | 0.244 | 0.243 | 0.453 | **+0.034 ✱** |
| | Ankh-large | 0.285 | 0.418 | 0.231 | 0.220 | 0.447 | **+0.029 ✱** |
| | ESM2-3B | 0.262 | 0.418 | 0.200 | 0.215 | 0.444 | +0.026 |
| | Ankh3-xl | 0.266 | 0.418 | 0.216 | 0.204 | 0.437 | +0.019 |
| | ESM-C 600M | 0.227 | 0.418 | 0.235 | 0.163 | 0.425 | +0.007 |
| MP-bycomplex | ESM-C 6B | 0.379 | 0.393 | 0.253 | 0.282 | 0.443 | +0.049 |
| MP-clustered | ESM-C 6B | 0.146 | 0.393 | 0.168 | 0.053 | 0.385 | −0.008 |
| CATH-all | ESM-C 6B | 0.360 | 0.383 | 0.285 | 0.263 | 0.429 | +0.046 |

Full 44-cell table in `redundancy.csv`.

## Findings

**1. The two channels are not redundant.** `base~fx` runs 0.04–0.37, mostly 0.15–0.27. The base
pathway and the physics score largely disagree within complex, which rules out the simplest
explanation for the flat FoldX arms — that the PLM is reproducing FoldX from sequence and the
combination therefore has nothing to add. It is not; the combination has plenty to add in
principle.

**2. Non-redundant PLM signal exists, and it is homology-dependent.** `part_base` for ESM-C 6B
is 0.243 on by-complex-SP and 0.130 on clustered-SP; for Ankh-large, 0.220 → 0.078; for ESM2-3B,
0.215 → 0.072. Same rows, different partition. Whatever survives FoldX in the base pathway is
roughly halved when homologs leave the training set — the same signature the base arms show
directly.

**3. Under homology control, an optimal blend of the two signals returns nothing.** On
clustered-SP the gain is +0.009 at best (ESM-C 6B) and indistinguishable from zero for all ten
backbones; MP-clustered is the same. This is the sharpest statement the analysis supports: it is
not that MuLAN's head fails to exploit the base pathway, it is that **the base pathway's output
contains no usable complement to FoldX once homology is controlled**, for any backbone, under any
weighting.

**4. On by-complex the head is already at the linear optimum.** Compare the observed arms against
the blend on bycomplex-SP:

| | FoldX alone | blend2 (optimal) | observed fx_scalar | observed fx_mlp |
|---|---|---|---|---|
| ESM-C 6B | 0.418 | 0.453 | 0.451 | 0.457 |
| Ankh-large | 0.418 | 0.447 | 0.435 | 0.444 |
| Ankh3-xl | 0.418 | 0.437 | 0.463 | 0.454 |

The trained scalar arm lands within 0.002 of the best achievable linear blend for ESM-C 6B. There
is no combiner improvement to be had on this tier — the ceiling is the information, not the head.

**5. The 12-term arm's edge on clustered-SP is physics, not PLM.** On clustered-SP the optimal
blend of base + FoldX *scalar* reaches 0.427 at best, yet observed `fx_mlp` arms sit at 0.443
(ProstT5), 0.446 (ESM-C 6B), 0.439 (Ankh3-large). Those exceed what any combination of the base
pathway with the FoldX scalar can produce, so the excess must come from the **twelve decomposed
FoldX terms** rather than from the backbone. That is consistent with `fx_mlp` beating `fx_scalar`
on clustered-SP for all nine backbones while the backbones themselves differ by 0.14 in base.

> **Measured directly in `BLEND12_RESULT.md` (2026-08-04).** Confirmed with two corrections. A
> linear blend of the twelve terms with no PLM reaches **0.434**, above the 0.427 base+scalar
> ceiling, and the nine observed `fx_mlp` arms span 0.422–0.446 around it — they land on the
> no-PLM physics line rather than separating from it. But the decomposition's own contribution is
> **+0.016 [−0.007, +0.038]**, not resolvable at 95 complexes, and it vanishes entirely (+0.001)
> if the terms are rank-encoded instead of z-encoded: what it carries is term *magnitude*, which
> is what additivity in kcal/mol predicts.

## Augmentation on the combined clustered tier — 2026-08-05

Data: `redundancy_clustered_all.csv`, from `redundancy.py --tier clustered-ALL`. The tier map gained
`fullSK clustered-ALL` and `fullSK clustered-ALL+aug`, which read the **same** three test TSVs —
augmentation touches train only, and the files are byte-identical by md5 — so aug-vs-plain is a
paired contrast on one test set. `redundancy.csv` predates those two tiers and holds the 44 cells
of the table above.

Ankh-large is the only backbone with a base arm on the augmented tier.

| | rho_base | base~fx | part_base | blend2 | gain |
|---|---|---|---|---|---|
| clustered-ALL | 0.110 | 0.162 | 0.033 | 0.434 | +0.003 [−0.003, +0.009] |
| clustered-ALL **+aug** | 0.166 | 0.181 | **0.081** | 0.433 | +0.002 [−0.006, +0.010] |
| paired Δ | +0.057 | +0.019 | **+0.047 [+0.004, +0.093] ✱** | | |

119 complexes, covered rows, which is why `rho_base` differs in the third decimal from the 0.109 /
0.165 that `results_matrix_ps.csv` reports over 121 complexes and all rows. The ppS contrast quoted
below is the 121-complex form; the `part_base` and `base~fx` contrasts are the covered-row form.

**Augmentation adds non-redundant signal that no combination rule can spend.** `part_base` roughly
doubles and clears zero paired, while `base~fx` does not move — the augmented base arm is not
converging on FoldX, it is carrying more of what FoldX lacks. The optimally weighted blend of the
two returns +0.003 before and +0.002 after. This is finding 3 reached from the opposite direction:
there, a fixed pathway whose complement to physics could not be weighted into a gain; here, the
complement is *increased* by half again and the gain is unchanged.

The trained arms are worse than unchanged. Ankh-large's FoldX arms fall 0.418 → 0.353 (scalar,
Δ −0.065 [−0.097, −0.034]) and 0.410 → 0.368 (12-term, Δ −0.042 [−0.080, −0.001]) while its base
arm rises 0.109 → 0.165 (Δ +0.056 [+0.010, +0.104]). Significant in both directions inside one
backbone on one tier. "No usable signal" predicts flat, not down.

**Both FoldX figures above were measured on the defective augmented channel** (`RESULTS.md` §24).
Re-run against the repaired splits on 2026-08-06 they become −0.024 [−0.047, −0.002] and
−0.001 [−0.031, +0.033]: the scalar arm still loses, at a third of the recorded size, and the
12-term arm does not lose at all. The paragraph below asks for a mechanism behind a decline that is
now much of the way to being accounted for by the channel offset. Read what follows as the analysis
that was done against the pre-repair numbers, not as a standing open question of that size.

The mechanism for the decline is open. Reverse augmentation negates the label and all twelve FoldX
terms together (verified in the split TSVs, with unaugmented standardisation constants preserved),
which constrains the model toward an odd-symmetric response to the FoldX features. The data are
asymmetric — the FoldX→ΔΔG slope is +1.16 on stabilizing rows against +0.83 on destabilizing, with
a +1.22 kcal/mol intercept — but the slope asymmetry is **+0.33 [−0.15, +0.73]** under a cluster
bootstrap over the 238 training complexes and does not clear zero, so that explanation is not
supported at this sample size. The intercept is rank-invariant and cannot affect ppS on its own.

## The FoldX-alone baseline has moved, and the 07-27 contrasts are stale

`FOLDX_ALONE_BASELINE_RESULT.md` reports FoldX-alone at **0.363** on the SP tiers and states it
"will not move — it does not depend on training". That is true with respect to training but not
with respect to coverage, and coverage has since gone **87.8% → 99.2%**. Recomputed on the
current splits, FoldX-alone is **0.418** (all rows and covered rows agree to three decimals at
99.2%; the 07-27 covered-only figure was 0.395). MP tiers: 0.393. CATH: 0.383, unchanged at 100%
coverage.

Every paired contrast in that document was therefore computed against a baseline **0.055 too
low** on the SP tiers. The direction is unfavourable: the 18 recorded wins over FoldX-alone are
overstated, and the count will fall when regenerated. The script takes ~40 s:

    python experiments/rescore_perstructure/foldx_alone_baseline.py --root . --model <list>

This should be rerun before any of that document's numbers are quoted again, and before item 4 is
scheduled — item 4's premise is converting ESM-C 6B's CATH-single +0.127 into a result, and that
figure is a contrast against the old baseline.

## Consequences for the queue

**Item 1 (residual learning) keeps its value but changes its target.** The bound measured here
applies to combining the *existing base prediction* with FoldX; a model trained on
ΔΔG_exp − ΔΔG_FoldX sees embeddings, not the base scalar, so it can in principle surface signal
the base head never did. But finding 4 removes the by-complex motivation — the head is already at
the linear optimum there — and finding 3 says that on clustered tiers the base pathway's output
is empty. The experiment is now a test of whether the *embeddings* hold clustered-tier signal that
the current head discards, which is a narrower and more honest framing than "make pass-through
free".

**Item 3 (A1 interface cross-attention) gains priority.** Findings 1–3 locate the failure in what
the pathway extracts, not in how it is combined. A1 is the only queued change that alters
extraction.

**A cheap follow-up is now well posed.** Rerun `blend2` with the twelve decomposed FoldX terms in
place of the scalar. That splits finding 5 cleanly: it measures how much of the `fx_mlp` arms'
advantage is the decomposition alone, with no PLM involved, and it needs only the 16-column split
TSVs already on disk.
