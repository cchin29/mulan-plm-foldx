# The FoldX-alone baseline — result

_Computed 2026-07-27, regenerated 2026-08-04. Script:
`experiments/rescore_perstructure/foldx_alone_baseline.py`. **Current data:
`foldx_alone_baseline_allplm_cov99.csv`.** The 2026-07-27 files — `foldx_alone_baseline.csv`,
`foldx_alone_baseline_allplm.csv`, `foldx_alone_baseline.html` and `foldx_alone_baseline.png` —
are the pre-fix 87.8%-coverage lineage and are not current; see `README.md` in this directory._

> **v3 — regenerated 2026-08-04 at 99.2% FoldX coverage.** The baseline itself moved. Everything
> below v3 is the 87.8%-coverage computation and is **superseded**; it is retained as the lineage
> record, with its data in `foldx_alone_baseline_allplm.csv`. Current data:
> **`foldx_alone_baseline_allplm_cov99.csv`**. Read §"v3" first.
>
> **v2 — swept all 10 backbones.** The Ankh-only picture below was too pessimistic as a statement
> about *the method*. Ankh-large genuinely does not clear the FoldX baseline; **ESM-C 6B does**,
> and convincingly. The multi-backbone result is in §"All ten backbones" — read that first.

# v3 — regenerated at 99.2% coverage

The v1/v2 caveat "FoldX-alone will not move — it does not depend on training" is true of training
and false of **coverage**, which has since gone 87.8% → 99.2% on the single-point tiers. The
baseline moved with it, and the MuLAN arms moved too (the channel is fed the same improved
FoldX values), so both sides of every contrast changed.

| tier | FoldX alone, v2 | FoldX alone, v3 | coverage |
|---|---|---|---|
| clustered-SP / by-complex-SP | 0.363 | **0.418** | 87.8% → 99.2% |
| MP-clustered / MP-by-complex | 0.393 | 0.393 | 98.7%, unchanged |
| CATH-all | 0.383 | 0.383 | 100%, unchanged |
| CATH-single | 0.398 | 0.398 | 100%, unchanged |
| S1102 | 0.443 | 0.442 | 100%, unchanged |

Only the SP tiers moved. The +0.055 shift there is the single largest correction in this document.

## Sweep totals

| | v2 (87.8%) | v3 (99.2%) |
|---|---|---|
| significantly **beat** FoldX alone | 18 | **19** |
| significantly **lose** to FoldX alone | 61 | 61 |
| indistinguishable | 147 | 152 |
| total contrasts | 226 | 232 |

The win count did **not** fall, contrary to what the higher baseline alone would predict — the
FoldX arms gained roughly as much as the baseline did. What changed is the composition.

## What moved

**The clustered tiers stopped being empty.** v2 recorded one win in 56 clustered contrasts; v3
records **three in 76**, and all three are `fx_mlp`: ESM-C 6B +0.031 [+0.001, +0.061], ProstT5
+0.030 [+0.005, +0.053], Ankh3-large +0.028 [+0.004, +0.052]. That every clustered win carries
the 12-term decomposition and none carries the scalar is followed up in
`experiments/beyond_foldx/BLEND12_RESULT.md`.

**ESM-C 6B's flagship result shrank but survived.** CATH-single `fx_mlp` was +0.127 [+0.048,
+0.216]; it is now **+0.097 [+0.017, +0.194]**. CATH-all's win moved from `fx_mlp` +0.085 to
`fx_scalar` +0.055 [+0.001, +0.118], with the MLP arm at +0.048 and no longer significant.

**Ankh-large now has zero wins**, down from one. The direct answer to the question about that
backbone is unchanged in direction and slightly stronger: 0 beats, 5 losses, 22 indistinguishable.

**No `base` arm beats FoldX alone — still 0, now in 78 contrasts.** This is the one v2 statement
that survives the regeneration untouched.

| backbone | beats | loses | n.s. | best FoldX-arm contrast (v3) |
|---|---|---|---|---|
| **ESM-C 6B** | **6** | 4 | 17 | S1102 by-complex MLP +0.109 [+0.016, +0.206] |
| ProstT5 | 4 | 7 | 16 | S1102 by-complex scalar +0.107 [+0.037, +0.182] |
| Ankh3-xl | 3 | 6 | 18 | by-complex-SP scalar +0.049 [+0.019, +0.080] |
| ESM2-3B | 2 | 3 | 16 | by-complex-SP scalar +0.046 [+0.015, +0.078] |
| Ankh3-large | 1 | 10 | 16 | clustered-SP MLP +0.028 [+0.004, +0.052] |
| ESM-C 600M | 1 | 6 | 14 | by-complex-SP MLP +0.053 [+0.020, +0.087] |
| AIDO-16B | 1 | 8 | 12 | by-complex-SP scalar +0.034 [+0.005, +0.062] |
| SaProt-1.3B | 1 | 1 | 5 | S1102 by-complex scalar +0.141 [+0.047, +0.238] |
| Ankh-large | 0 | 5 | 22 | CATH-single MLP +0.084 [−0.019, +0.203] *(n.s.)* |
| SaProt-650M | 0 | 11 | 16 | S1102 by-complex scalar +0.110 [−0.002, +0.225] *(n.s.)* |

## ESM-C 6B, every tier (v3)

| tier | FoldX alone | base | Δ | + scalar | Δ | + 12-term MLP | Δ |
|---|---|---|---|---|---|---|---|
| CATH-single | 0.398 | 0.377 | −0.021 | 0.456 | +0.058 | **0.495** | **+0.097 ✱** |
| CATH-multiple | 0.540 | 0.469 | −0.071 | 0.526 | −0.014 | 0.509 | −0.030 |
| CATH-all | 0.383 | 0.360 | −0.023 | **0.438** | **+0.055 ✱** | 0.431 | +0.048 |
| by-complex-SP | 0.418 | 0.310 | **−0.106 ✱** | **0.451** | **+0.039 ✱** | **0.457** | **+0.044 ✱** |
| clustered-SP | 0.418 | 0.193 | **−0.223 ✱** | 0.417 | +0.001 | **0.446** | **+0.031 ✱** |
| MP-by-complex | 0.393 | 0.374 | −0.014 | 0.391 | +0.008 | 0.423 | +0.042 |
| MP-clustered | 0.393 | 0.137 | **−0.247 ✱** | 0.270 | **−0.105 ✱** | 0.320 | −0.058 |
| S1102 by-complex | 0.442 | 0.452 | +0.009 | 0.546 | +0.104 | **0.551** | **+0.109 ✱** |
| S1102 clustered | 0.442 | 0.370 | −0.073 | 0.460 | +0.017 | 0.384 | −0.058 |

The frontier-comparison caveat from v2 stands and tightens: the CATH figure to quote beside
USP-ddG 0.493 / CATH-ddG 0.494 is now **0.438** (scalar) or 0.431 (MLP), not 0.468, against a
FoldX-alone row of 0.383.

---

# v1 / v2 — superseded (87.8% coverage)

_Retained as the lineage record. The figures below were computed against a FoldX-alone baseline
of 0.363 on the SP tiers, which is 0.055 too low._

## The missing comparator

Every MuLAN+FoldX number reported so far was compared against **MuLAN base**. It was never
compared against **FoldX by itself** — the unsupervised physics score used directly as a
predictor. That is the comparator the published SKEMPI frontier uses (RDE-Network, Prompt-DDG,
DiffAffinity, CATH-ddG, USP-ddG all report a FoldX baseline column), and it is the free
alternative to training anything at all.

Because the FoldX ΔΔG is already sitting in column 5 of every `splits_*_foldx` TSV next to the
experimental label in column 4, this cost nothing to compute.

## Method

Identical to `experiments/rescore_perstructure/rescore.py`, deliberately:

- complex key = `chain1_id.split("_")[0]`
- per-structure Spearman: mean within-complex ρ over complexes with **n ≥ 10**, skipping
  complexes with zero variance in prediction or truth
- test folds are **pooled** before grouping by complex
- 95% CI = **cluster bootstrap over complexes** (resample the complex set, B = 10,000, seed 0)
- MuLAN-vs-FoldX contrasts are **paired** on the shared complex set

**Validation:** every `MuLAN base` value reproduces `results_matrix_ps.csv` to 4 decimals
(clustered-SP 0.1417, by-complex-SP 0.2843, MP-clustered 0.0914, MP-by-complex 0.3402,
CATH-all 0.3308, CATH-single 0.3211, CATH-multi 0.4444, S1102 by-complex 0.2888, S1102 clustered
0.1265). The FoldX-alone column therefore comes off the same machinery, not a parallel
implementation.

Two internal consistency checks also pass: tiers that are different *partitions of the same rows*
(clustered-SP vs by-complex-SP; MP-clustered vs MP-by-complex; S1102 clustered vs by-complex)
return **identical** FoldX-alone values, as they must — FoldX-alone does not depend on how the
rows are split.

Results-dir → tier mapping was resolved by per-fold test-row signature, not by name:
`full_skempi` 1673/1246/1246 = clustered-SP · `full_skempi_bycomplex` 1389/1388/1388 =
by-complex-SP · `full_skempi_mp_*` 546/545/545 · `full_skempi_cath` 687.

## Result

PLM = Ankh-large. Per-structure Spearman ρ (T≥10). **Δ** is the paired contrast vs FoldX alone;
**bold ✱** = 95% CI excludes 0.

| Split tier | cplx | FoldX alone | base | Δ | +FoldX scalar | Δ | +FoldX MLP | Δ |
|---|---|---|---|---|---|---|---|---|
| CATH-single | 11 | 0.398 | 0.321 | −0.077 | 0.420 | +0.021 | **0.455** | +0.056 |
| CATH-multiple | 5 | 0.540 | 0.444 | −0.095 | 0.499 | −0.041 | **0.589** | +0.049 |
| CATH-all | 13 | 0.383 | 0.331 | −0.052 | 0.382 | −0.001 | 0.387 | +0.004 |
| by-complex-SP | 96 | 0.363 | 0.284 | **−0.079 ✱** | **0.409** | **+0.046 ✱** | 0.403 | +0.040 |
| clustered-SP | 96 | 0.363 | 0.142 | **−0.221 ✱** | 0.354 | −0.008 | 0.359 | −0.003 |
| MP-by-complex | 42 | 0.393 | 0.340 | −0.041 | 0.397 | +0.018 | 0.391 | +0.014 |
| MP-clustered | 42 | 0.393 | 0.091 | **−0.294 ✱** | 0.331 | **−0.067 ✱** | 0.355 | −0.033 |
| S1102 by-complex | 24 | 0.443 | 0.289 | −0.154 | **0.495** | +0.052 | 0.482 | +0.040 |
| S1102 clustered | 24 | 0.443 | 0.127 | **−0.316 ✱** | 0.426 | −0.016 | 0.433 | −0.010 |

### Three findings

**1. FoldX alone is remarkably stable at ρ ≈ 0.36–0.44 across every tier — including the ones
where MuLAN collapses.** It does not care whether the test complex is out-of-cluster or
out-of-superfamily, because it is physics computed from the test structure. That stability is
what makes it such a demanding baseline under leakage control.

**2. MuLAN base never beats FoldX alone — on any tier.** It is significantly *worse* on four of
nine, by as much as −0.32 (S1102 clustered) and −0.29 (MP-clustered).

**3. Of the 27 MuLAN-vs-FoldX contrasts, exactly one is significantly positive.** By-complex-SP
with the FoldX scalar: **+0.046 [+0.008, +0.085]**. Two are significantly *negative*
(MP-clustered +scalar −0.067; and every `base` arm noted above). The remaining contrasts are
statistically indistinguishable from zero.

## Implications for the current slide claims

The reported framing — *"the FoldX channel triples Ankh-large's clustered-split ρ from 0.13 to
0.43"* — is arithmetically true and **materially misleading**. The channel is not recovering
generalization; it is handing the model a number that already scores 0.36 on its own, and the
model is passing it through. Stated against the correct comparator:

> On leakage-controlled splits, MuLAN + the FoldX channel performs **the same as reporting the
> FoldX score directly**, on 8 of 9 tiers. The single reliable gain is +0.046 on by-complex-SP.

This also affects the frontier comparison. Placing "MuLAN+FoldX 0.468" beside USP-ddG 0.493 and
CATH-ddG 0.494 implies MuLAN is a near-frontier learned model. The FoldX-alone row (0.383–0.398
on that same CATH set) has to sit next to it, or the table overstates what was built.

### The best-supported positive claim

**CATH-superfamily, single-point, +FoldX 12-term MLP: ρ = 0.455 vs FoldX alone 0.398, Δ = +0.056
[−0.016, +0.129].** That is the largest and most consistent positive effect in the table (the MLP
arm is ≥ FoldX-alone on 5 of 9 tiers and is the best arm on both CATH sub-tiers) — but its CI
includes zero on **11 complexes**, so it is a hypothesis with a promising point estimate, not a
result. Powering it up is now the most valuable single experiment available: more CATH folds,
more seeds.

## Consequences for the augmentation plan

1. **The success criterion changes.** "≥ +0.03 over MuLAN-base" is the wrong bar — MuLAN-base
   loses to a free baseline. The bar is **improvement over FoldX alone**.
2. **The detectable-effect problem is now quantified.** The paired CI half-widths are ≈ ±0.04 on
   the 96-complex tiers and ≈ ±0.07 to ±0.16 on the 5–24-complex tiers. A +0.03 target is
   **inside the noise on every tier except the two 96-complex ones**. Any augmentation experiment
   run on the CATH tier as currently folded cannot resolve its own target effect.
3. **It sharpens what augmentation has to do.** FoldX-as-a-feature is exhausted — the model
   already extracts essentially all of it. If FoldX-as-labels is to be worth anything, it must
   buy something the channel cannot: transferable structure across *complexes*, which is an
   argument for the non-SKEMPI Source B, not for saturating complexes already in training.
4. **A cheap ablation just became interesting:** MuLAN base trained *with* the pseudo-labels but
   evaluated *without* the FoldX channel. If that closes the base-vs-FoldX gap, augmentation has
   distilled the physics into the sequence pathway — which would be a genuinely useful result
   (a FoldX-free model at FoldX-level accuracy) and is not a claim the channel can make.

## Caveats

- **CATH is one fold.** 13 complexes at T≥10 (11 single / 5 multi), and 1JTG alone contributes
  194 of 687 mutations. Every CATH number in this table, including the encouraging +0.056, is
  fragile to that composition.
- The by-complex and clustered rungs were **mid-rerun** as of 2026-07-26 (unified single+multi +
  expanded FoldX coverage). FoldX-alone will not move — it does not depend on training — but the
  MuLAN columns will, so the contrasts must be recomputed when the rebuild lands. The script
  takes ~40 s; re-run it.
- Coverage: uncovered mutations carry FoldX = 0 (per-fold standardised mean), 87.8 % coverage on
  the SP tiers. Scored on covered rows only, FoldX-alone rises 0.363 → **0.395** there, which
  makes the comparison *harsher*, not kinder. Both numbers are in the CSV.
- Ankh-large only. ESM-C 6B is the stronger ΔΔG backbone and may clear FoldX-alone by more; the
  script takes `--model esmc6b` and should be run for it next.

---

# All ten backbones

The script now takes a comma-separated `--model`. Sweeping every backbone with predictions on
disk — esmc6b, ankh, ankh3_xl, ankh3_large, saprot, saprot13b, prostt5, esmc600m, esm2, aido —
gives **226 paired contrasts against FoldX alone**:

| | count |
|---|---|
| significantly **beat** FoldX alone | **18** |
| significantly **lose** to FoldX alone | 61 |
| indistinguishable | 147 |

Two facts hold across the whole sweep:

- **No `base` arm beats FoldX alone anywhere — 0 wins in 79 contrasts.** Without the channel, no
  frozen-PLM MuLAN configuration reaches an unsupervised physics score under leakage control.
- **All 18 wins carry the FoldX channel**, and they concentrate on the **CATH** and **by-complex**
  tiers. The **clustered** tiers produce **one win in 56 contrasts** (ankh3_xl + MLP, +0.034).

## Per backbone

| backbone | beats | loses | n.s. | best FoldX-arm contrast |
|---|---|---|---|---|
| **ESM-C 6B** | **5** | 3 | 19 | CATH-single MLP **+0.127 [+0.048, +0.216]** |
| Ankh3-xl | 4 | 6 | 17 | by-complex-SP scalar +0.071 [+0.023, +0.116] |
| ProstT5 | 4 | 7 | 16 | S1102 by-complex scalar +0.107 [+0.038, +0.182] |
| SaProt-650M | 2 | 11 | 10 | CATH-single MLP +0.060 [+0.003, +0.116] |
| SaProt-1.3B | 1 | 1 | 5 | S1102 by-complex scalar +0.141 [+0.047, +0.238] |
| **Ankh-large** | **1** | 5 | 21 | CATH-single MLP +0.056 [−0.016, +0.129] *(n.s.)* |
| Ankh3-large | 1 | 11 | 15 | by-complex-SP scalar +0.041 [+0.001, +0.083] |
| ESM2-3B | 0 | 3 | 16 | — |
| ESM-C 600M | 0 | 6 | 15 | — |
| AIDO-16B | 0 | 8 | 13 | — |

## ESM-C 6B, every tier

| tier | FoldX alone | base | + scalar | Δ | + 12-term MLP | Δ |
|---|---|---|---|---|---|---|
| CATH-single | 0.398 | 0.377 | 0.446 | +0.047 | **0.525** | **+0.127 ✱** |
| CATH-multiple | 0.540 | 0.469 | 0.556 | +0.016 | 0.562 | +0.022 |
| CATH-all | 0.383 | 0.360 | **0.468** | **+0.085 ✱** | **0.468** | +0.085 |
| by-complex-SP | 0.363 | 0.310 | **0.421** | **+0.058 ✱** | 0.416 | **+0.053 ✱** |
| clustered-SP | 0.363 | 0.193 | 0.373 | +0.010 | 0.378 | +0.015 |
| MP-by-complex | 0.393 | 0.374 | 0.391 | +0.008 | 0.423 | +0.042 |
| MP-clustered | 0.393 | 0.137 | 0.270 | **−0.105 ✱** | 0.320 | −0.058 |
| S1102 by-complex | 0.443 | 0.452 | 0.546 | +0.104 | **0.551** | **+0.109 ✱** |
| S1102 clustered | 0.443 | 0.370 | 0.460 | +0.017 | 0.384 | −0.058 |

### Implications

**1. The frontier comparison survives — for ESM-C 6B, and only with the baseline shown.** The
`0.468` CATH figure already on the benchmark matrix is the ESM-C 6B FoldX arm, and it **does**
significantly beat FoldX alone (+0.085 [+0.009, +0.167]). Placing it beside USP-ddG 0.493 /
CATH-ddG 0.494 is defensible — provided the FoldX-alone row (0.383) sits in the same table.
Without that row the reader cannot tell that ~0.38 of the 0.468 was free.

**2. Ankh-large does not clear the physics baseline, and the answer is a negative one:**
Ankh-large + the FoldX channel does **not** reliably beat FoldX alone (1 win, 5 losses, 21 n.s.;
its best result, CATH-single +0.056, has a CI crossing zero). **ESM-C 6B is the configuration that
clears the bar.** Any claim resting on Ankh-large as the FoldX backbone needs this table beside it.

**3. The 12-term MLP earns its place — on the strong backbone.** ESM-C 6B's largest and only
CATH-single win is the MLP arm (+0.127 vs the scalar's non-significant +0.047). This is consistent
with the earlier "MLP is the upside on larger sets" finding, and it is the first time the
decomposition has been shown to buy something a *free* alternative doesn't already provide.

**4. The clustered tiers are where the method genuinely fails.** One win in 56 contrasts, and on
MP-clustered several arms are significantly *worse* than FoldX alone (ESM-C 6B scalar −0.105,
SaProt MLP −0.198, Ankh3-large MLP −0.096). When homology is fully controlled, the learned model
does not merely stop adding — it actively degrades the physics signal it is given. **This, not the
CATH tier, is the finding the augmentation work should target.**

**5. FoldX-alone is a fixed, model-independent reference.** It does not move when the by-complex
and clustered MuLAN reruns land, so the contrast column can be regenerated in ~40 s:
`python experiments/rescore_perstructure/foldx_alone_baseline.py --root . --model <list>`.

## Distance between this FoldX and the frontier's

Four tiers publish a FoldX number that can be set against this one. The gap is not uniform, and
that is what identifies its cause.

| tier | metric | here | published | gap | n | source |
|---|---|--:|--:|--:|--:|---|
| CATH-superfamily | AUROC | 0.760 | 0.754 | **+0.006** | 687 | USP-ddG Table 1 |
| by-complex | per-structure Sp | 0.430 | 0.369 | +0.061 | 5801 | Prompt-DDG Table 1 |
| by-complex | AUROC | 0.717 | 0.658 | +0.059 | 5801 | BA-DDG Table 1 |
| clustered id60 | pooled Sp | 0.512 | 0.294 | **+0.218** | 5801 | ProtBFF Table 2 |

**The CATH row rules out a stronger pipeline.** That tier is 687 mutations at 100% FoldX coverage
on both sides, and the two agree to 0.006. Whatever produces the other gaps is not RepairPDB, not
the FoldX version, and not the choice of `Interaction Energy` as the scalar.

**The two by-complex rows are one effect, measured twice.** +0.061 and +0.059, from two independent
tables, on two different metrics, over the same 5,801 rows. A consistent gap of that size on the
full set and none on the fully-covered subset is what a coverage advantage looks like: single-point
coverage here went 28% → 87.8% → 99.2%, and the last of those steps alone moved FoldX-alone ppS
0.363 → 0.418.

**The clustered row is 3.6× the by-complex rows, so it is something else on top.** Decomposing the
pooled statistic on that rung:

| component | Spearman |
|---|--:|
| single-point rows only | 0.437 |
| multi-point rows only | 0.572 |
| between-complex (complex means, n=337) | 0.596 |
| within-complex (both mean-centred) | 0.497 |

Pooling over the combined rung rewards two things the per-structure metric discards — getting the
multi-point rows right, and ordering whole complexes against each other. Those are the first two
things a lower-coverage FoldX loses, because coverage fails by whole structure and multi-point
mutations are the hardest to map. Degrading this data the way that failure mode actually works
reproduces the published value: multi-point rows made unusable **and** ~40% of complexes made
unusable gives 0.287, against ProtBFF's 0.294. Row-wise random dropout does not — it needs ~40%
coverage to get there, which no published pipeline would report.

This is inference from our side of the comparison only. ProtBFF's FoldX predictions are not
available, so the mechanism is consistent with the evidence rather than demonstrated.

Every figure in this section — the cross-tier table, the decomposition and both simulations —
regenerates with `python experiments/rescore_perstructure/foldx_gap_analysis.py --root .` (needs
`scratch/`; ~10 s).

## Lift over the FoldX each method was built on

The gap above matters because **ProtBFF is the same idea as the FoldX arms here** — biophysical
features scaled into a frozen PLM. Its headline therefore inherits its FoldX baseline, and the
comparable quantity is not the headline but the lift over it.

| tier | method | its FoldX | its result | lift |
|---|---|--:|--:|--:|
| clustered id60 (fold-avg Sp) | ProtBFF | 0.294 | 0.477 | **+0.183** |
| | MuLAN + FoldX | 0.521 | 0.547 | **+0.025** |
| by-complex (per-structure Sp) | RDE-Network | 0.369 | 0.401 | +0.032 |
| | Prompt-DDG | 0.369 | 0.426 | +0.057 |
| | MuLAN + FoldX | 0.430 | 0.479 | +0.049 |
| | CATH-ddG | 0.369 | 0.460 | +0.091 |
| | BA-DDG | 0.369 | 0.513 | +0.144 |

**On the clustered tier the headline comparison is not safe.** 0.547 against 0.477 is a 0.070 win
sitting on a 0.218 difference in the baseline both methods start from; it is at least as consistent
with better FoldX coverage as with a better model. The lift column says the opposite of the
headline — ProtBFF extracts +0.183 from its physics channel where this work extracts +0.025.

**On the by-complex tier the comparison is safe, and it is the one to quote.** The baselines differ
by 0.061 rather than 0.218, and MuLAN's +0.049 sits mid-field: above RDE-Network, level with
Prompt-DDG, below CATH-ddG and BA-DDG. That is the defensible statement about what the channel is
worth, and it is a more modest one than the clustered tier appears to support.
