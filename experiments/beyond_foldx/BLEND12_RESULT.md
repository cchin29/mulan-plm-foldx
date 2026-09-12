# The 12-term decomposition versus the FoldX total — result

_Computed 2026-08-04. Script: `blend12.py`. Data: `blend12.csv` (rank features),
`blend12_z.csv` (z features). No training._

Follow-up to `REDUNDANCY_RESULT.md` finding 5. That finding inferred, from the fact that observed
`fx_mlp` arms on clustered-SP exceed the 0.427 ceiling of any base + FoldX-*scalar* blend, that
the excess had to come from the twelve decomposed terms. This measures it directly.

## Method

Three global leave-one-complex-out linear blends over the same 4131 covered rows, same ppS
convention, same self-validation as `redundancy.py`:

| | features | meaning |
|---|---|---|
| `blend_1` | FoldX total | must equal FoldX alone exactly — validation |
| `blend_12` | 12 decomposed terms | the decomposition on its own, **no PLM** |
| `blend_13` | 12 terms + base prediction | decomposition plus backbone |

`gain_12 = blend_12 − blend_1` and `gain_plm = blend_13 − blend_12`. `blend_1` reproduced FoldX
alone to 1e-6 on all 44 cells under both encodings; the script raises `[BUG ]` otherwise. The
scalar and decomposed splits were also cross-checked to agree on `ddG_experimental` row by row.

Features are within-complex normalised. Energy terms are additive in kcal/mol, so ranking each
term separately discards information the total retains — both encodings are therefore reported,
and the difference between them turns out to be the finding.

## Result — the decomposition needs magnitudes, and buys little either way

SP tiers, identical for every backbone since no PLM enters:

| encoding | FoldX alone | blend_12 | gain_12 | 95% CI |
|---|---|---|---|---|
| within-complex rank | 0.418 | 0.419 | +0.001 | [−0.016, +0.017] |
| within-complex z | 0.418 | 0.434 | +0.016 | [−0.007, +0.038] |

Rank-encoding each term destroys the decomposition entirely — a linear rule over the twelve
term *orderings* is worth nothing over the total. Z-encoding preserves magnitude and recovers
+0.016. Whatever the decomposition carries is in the term magnitudes, which is what additivity in
kcal/mol predicts. Neither figure is significant at 95 complexes, and on CATH the decomposition
is negative (−0.038, 13 complexes).

## The trained arms sit exactly on the no-PLM physics line

Clustered-SP, z encoding. `lin[12]` = 0.434 involves **no backbone at all**:

| backbone | lin[12] | lin[12+base] | observed fx_scalar | observed fx_mlp |
|---|---|---|---|---|
| ESM-C 6B | 0.434 | 0.455 | 0.417 | 0.446 |
| ProstT5 | 0.434 | 0.433 | 0.410 | 0.444 |
| Ankh3-large | 0.434 | 0.435 | 0.388 | 0.439 |
| ESM2-3B | 0.434 | 0.439 | 0.395 | 0.432 |
| Ankh3-xl | 0.434 | 0.445 | 0.411 | 0.431 |
| AIDO-16B | 0.434 | 0.434 | 0.397 | 0.428 |
| Ankh-large | 0.434 | 0.439 | 0.406 | 0.427 |
| SaProt | 0.434 | 0.434 | 0.392 | 0.427 |
| ESM-C 600M | 0.434 | 0.436 | 0.355 | 0.422 |

Two things read off this table.

**Every `fx_scalar` arm is at or below FoldX alone (0.418).** All nine, range 0.355–0.417. On the
leakage-controlled tier the trained scalar arm is never better than reporting the FoldX number
directly, and for ESM-C 600M it is 0.063 worse.

**Every `fx_mlp` arm lands on the no-PLM line.** The nine observed values span 0.422–0.446 around
`lin[12]` = 0.434, achieved with no backbone. The spread across nine backbones (0.024) is smaller
than the gap between the weakest backbone's two arms. Finding 5's attribution is confirmed: the
`fx_mlp` advantage over `fx_scalar` on clustered-SP is the decomposition, not the model — but
with the correction that the decomposition's own contribution (+0.016) is not statistically
resolvable, so what the table really shows is nine trained models converging on a physics
baseline rather than separating from it.

## What the backbone adds on top of full physics

`gain_plm`, the base prediction added to all twelve terms (z encoding; ✱ = CI excludes 0):

| tier | ESM-C 6B | Ankh-large | ESM2-3B | best other |
|---|---|---|---|---|
| clustered-SP | +0.021 | +0.005 | +0.005 | +0.011 (Ankh3-xl) |
| by-complex-SP | **+0.041 ✱** | **+0.038 ✱** | **+0.036 ✱** | +0.021 (Ankh3-xl) |
| MP-clustered | −0.010 | +0.004 | — | +0.000 |
| MP-by-complex | +0.058 | **+0.069 ✱** | — | +0.050 (Ankh3-xl) |
| CATH-all | +0.041 | **+0.070 ✱** | −0.000 | +0.031 (SaProt) |

This reproduces `REDUNDANCY_RESULT.md` finding 3 against a stronger physics reference: with the
full decomposition already in the model, the backbone adds nothing resolvable on either clustered
tier, for any of the nine. The by-complex gains survive.

## Consequences

**Item 1 (residual learning) should target the 12-term residual, not the scalar residual.** The
scalar arms are pinned at FoldX-alone on clustered-SP; `blend_12` is the honest physics reference
and it is +0.016 higher. Training against ΔΔG_exp − ΔΔG_FoldX would credit the model for the
+0.016 that a linear reweighting already provides.

**The `fx_mlp` vs `fx_scalar` comparison in RESULTS.md needs a caveat.** The MLP arm beating the
scalar arm on clustered-SP for all nine backbones reads as an architectural result; it is a
physics result, and the same gap is obtained with the backbone removed.

**Nothing here is resolvable at 95 complexes.** `gain_12` +0.016 [−0.007, +0.038] and `gain_plm`
+0.021 [−0.001, +0.043] are both point estimates straddling zero. The power problem named in
queue item 4 is the binding constraint on this line of analysis too, not only on the CATH tier.
