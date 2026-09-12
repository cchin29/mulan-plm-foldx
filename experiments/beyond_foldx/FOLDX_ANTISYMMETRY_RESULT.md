# FoldX antisymmetry under mutation reversal, and the augmented channel's construction

Executes `docs/history/runbooks/RUN_FOLDX_ANTISYMMETRY_PROBE.md`. Asks whether the
augmented FoldX column — built by negating the forward value — carries what FoldX would actually
produce for the reverse mutation.

**Two independent defects, needing different responses.** One is a deterministic construction error
in the standardisation, exactly repairable with no FoldX runs — **repaired 2026-08-05, §1.2**. The
other is FoldX's own path dependence, repairable only by computing the reverse rows, and still open.
They compound.

Regenerate: `foldx_aug_offset.py` (§1; reads committed files only, and now reports on the preserved
`*_aug__negz` splits since the live ones are repaired); `foldx_antisymmetry.py` then
`foldx_antisymmetry_report.py` (§2–§4); `foldx_antisymmetry.py --control` then
`foldx_repeatability_report.py`, and `foldx_batch_control.py` (§2.1). Everything but §1 needs
`scratch/` and a FoldX binary.

## 1. Negation and standardisation do not commute

Until the repair, `merge_foldx_aug.py` negated the **already-standardised** vector —
`[-x for x in vec]` — justified as "each term is already train-centered at ~0 by the canonical
standardization, so negation adds negligible offset".

Train-centring makes the *standardised* values have mean zero. That is true and it is not the
question. The question is whether negating the standardised value equals standardising the negated
value:

```
z(x)  = (x - mu)/sigma      the pipeline's channel for a forward row
z(-x) = (-x - mu)/sigma     what a reverse row should carry
-z(x) = (mu - x)/sigma      what the augmentation writes

-z(x) - z(-x) = 2*mu/sigma  a constant offset on every reverse row
```

`mu` is the **raw** train mean of Interaction Energy. It is not near zero — most mutations are
destabilising. Recovering the constants by regressing the committed channel on the raw value it came
from:

| tier / fold | mu | sigma | offset | vs the channel's own sd |
|---|--:|--:|--:|--:|
| clustered-ALL fold_0 | +0.990 | 1.813 | **+1.092 sd** | 1.57× |
| clustered-ALL fold_1 | +1.145 | 2.052 | **+1.116 sd** | 1.27× |
| clustered-ALL fold_2 | +0.831 | 1.891 | **+0.879 sd** | 0.96× |
| CATH fold_0 | +0.926 | 1.955 | **+0.947 sd** | 1.06× |

(The constants here are the exact ones the repair uses, recovered from unclipped rows only — §1.2.
An earlier pass of this table fitted all rows including the clipped ones and read 2–10% high on
`sigma`; the offsets it gave, +0.82 to +1.08 sd, differ from the exact ones by at most 0.06 sd.)

Applied to **51% of the training set** — every synthesized row. Verified exact against the forward
row in all 6309 cases checked, zero exceptions.

**The label is handled correctly, which isolates the defect.** Labels are raw kcal/mol (plain-train
mean +1.071, range −9.55 to +11.36), so `-float(label)` is exact, and the augmented label mean
collapses to −0.001 as adding −y for every y requires. The sharpest statement of the bug is that
**the label is negated in raw units and the channel in standardised units, and only one of those is
right.** The two raw means are nearly identical — label +1.071, FoldX +0.986, both encoding "most
mutations destabilise" — so had the channel been negated in raw units and re-standardised with the
same constants, it would have paralleled the label exactly — which is what the repair does.

**The direction compounds.** A reverse row's negated label claims *stabilising* while its channel is
shifted toward *destabilising* by about one full channel width. That is an anti-correlated
corruption on half the training data, and it reaches only the arms that read the channel — the base
arm has none. **It predicts §24's sign split on its own**, with no appeal to anything below.

## 1.1 The identity anchors carry the same error

An identity (wt→wt) row's true FoldX vector is raw **zero** — no mutation, no energy change — whose
standardised form is `-mu/sigma`, not 0. The augmentation wrote 0, which says "an average mutation",
and which is also the value the canonical merge writes for rows FoldX never scored. So the anchors
were both misplaced by half the reverse-row offset and indistinguishable from missing data.

## 1.2 Repair

`merge_foldx_aug.py` now writes each synthesized row's channel as the standardised form of the
correct raw value:

| row type | true raw value | channel written |
|---|---|---|
| forward | x | `clip((x - mu)/sigma)` — copied verbatim from the canonical split |
| reverse | −x | `clip(-z(x) - 2*mu/sigma)`, which equals `clip((-x - mu)/sigma)` |
| identity | 0 | `clip(-mu/sigma)` |
| forward uncovered | — | 0 on both rows; FoldX never scored it, so its reverse is missing too |

No FoldX runs are required. The canonical merges do not persist `mu` and `sigma`, so they are
reconstructed per fold per term by regressing the committed channel on the raw FoldX terms in
`--results`. That relation is exactly linear away from the clip bounds, so fitting it on unclipped
rows is reconstruction rather than estimation, and the build **aborts** unless the recovered constants
reproduce every joinable committed value to within `--tol` (default 2e-5 — the split stores five
decimals, so this is rounding). Recovery uses the 40–50% of train rows whose `(pdb, mut)` key resolves
directly against the raw JSONs; the canonical merge joins through a chain and residue-number remap
this script deliberately does not reproduce, and two clean points determine a line. The constants are
written to `standardization.json` beside the split.

Where the forward row saturated the ±4 clip its raw value is unrecoverable, so the reverse row uses
the bound — correct in sign and at least as extreme as written. That is 332 of 4213 CATH reverse rows
and 697 of 9675 clustered-ALL ones, and it affects only the saturated terms within those rows.

## 1.3 Verification of the rebuilt splits

- **5959 reverse rows** reproduce `z(-x)` recomputed independently from the raw FoldX JSONs, max
  deviation **1e-5** — the file's own rounding. (Checked on every reverse row whose forward partner
  both resolves against the raw JSONs and is unclipped.)
- All **125** reverse rows whose forward row FoldX never scored remain all-zero.
- All **688** identity rows carry `-mu/sigma` exactly.
- Forward train rows and the whole of val/test remain **byte-identical** to the canonical split, so
  the paired aug-vs-plain comparison still runs on one test set.
- Columns 1–4 are unchanged from the pre-repair build, so the derived 4-column base splits and the
  augmented fasta need no rebuild — and the base arm, which reads only those columns, was never
  affected by this defect.

The pre-repair splits are kept as `*_aug__negz` beside the rebuilt ones.

**The augmented FoldX arms must be re-run.** Every existing one trained on the offset channel, which
makes those results uninterpretable rather than negative.

## 2. FoldX is not antisymmetric, and the failure is directional

1744 single-point mutations over 112 complexes — every one with a retained mutant structure, not
just the 201 inside an augmented split. Zero failed runs. BuildModel the reverse mutation on the
structure BuildModel already produced, AnalyseComplex with the same chain groups, compare with the
stored forward value.

| statistic | value | 95% CI | under antisymmetry |
|---|--:|--:|--:|
| regression slope, −rev on fwd | **0.742** | [0.633, 0.866] | 1 |
| corr(fwd, −rev) Pearson | 0.840 | [0.761, 0.916] | +1 |
| corr(fwd, −rev) Spearman | 0.862 | | +1 |
| mean(fwd + rev) | **+0.170** | [+0.118, +0.224] | 0 |
| sd(fwd + rev) | 0.796 | [0.562, 1.018] | 0 |
| sd(ddG), for scale | 1.464 | | |

**The slope is the interpretable one.** Reverting a mutation recovers **74%** of what the forward
move spent, and the interval excludes 1. That is the signature of the structure relaxing around the
mutant: going back does not return what coming forward cost. `mean|ddG|` is 0.963 forward against
0.857 reverse, the same fact in absolute terms.

Per-row, the negation is wrong by a median of **20%** of the value it claims, and **31% of rows are
off by more than half their own magnitude** (525 of 1668 with a non-zero forward value).

## 2.1 Controls: FoldX is deterministic, but its ΔΔG depends on batch composition

`sd(fwd+rev)` alone cannot separate genuine path dependence from a FoldX that simply does not
repeat. Two controls settle it, and the second changed what the first appeared to mean.

**Single-mutation re-run** (501 mutations, 92 complexes): re-running the *same forward mutation* on
the *same repaired wild type* reproduces the stored value exactly only **25%** of the time, sd
**0.480**, with excursions to 5.9. Read alone that would say FoldX is heavily stochastic.

**Batch re-run** (358 mutations, 25 complexes): replaying the *original* call — the complex's whole
`individual_list.txt` in one BuildModel, exactly as `build_ddg.py` issued it — reproduces
**358 of 358 exactly, sd 0.0000**.

So FoldX is **deterministic**, and the single-mutation spread is not noise: **a mutation's ΔΔG
depends on which other mutations shared its batch.** FoldX carries state across the entries of a
mutant file, so ΔΔG is not a function of (structure, mutation) alone.

**This is a reproducibility property of every FoldX number in this repository**, independent of
augmentation. An independent reproduction that submits mutations one at a time, or in different
groupings, will not recover the committed channel values — it must replay the same batches. It also
means the ±0.48 spread is systematic rather than random, so it cannot be averaged away.

**Re-analysis with the confound removed.** The reverse rows were computed as single-mutation runs
while the stored forward values came from batches, so the two sat in different contexts. The
single-mutation control supplies a forward value in the *same* context as the reverse, and on the
501 rows carrying both:

| forward value used | slope −rev ~ fwd | mean(fwd+rev) | corr |
|---|--:|--:|--:|
| stored batch run (context-mismatched) | 0.834 [0.713, 0.922] | +0.160 [+0.103, +0.221] | 0.905 |
| **re-run in matched context** | **0.776 [0.644, 0.910]** | **+0.173 [+0.104, +0.243]** | 0.886 |

**The confound was slightly hiding the effect, not creating it.** Matched, the slope falls further
below 1 and the mean sum rises. Both intervals exclude their antisymmetric values under either
treatment, so the §2 conclusion stands on the controlled comparison and does not depend on the
batch-context artefact.

## 3. What drives it

**Magnitude, overwhelmingly.** The error grows about sixfold across terciles of |ΔΔG|, and it grows
in the mean as well as the spread — so it is a bias, not just scatter:

| tercile | n | mean abs ddG | mean(fwd+rev) | sd(fwd+rev) |
|---|--:|--:|--:|--:|
| small | 582 | 0.069 | −0.006 | 0.216 |
| medium | 581 | 0.547 | +0.087 | 0.445 |
| large | 581 | 2.274 | **+0.429** | **1.246** |

**The largest FoldX values are the ones whose negation is most wrong** — and those are the rows
carrying the most signal, so the damage is concentrated where it costs most.

**The small tercile is not evidence of anything.** Per §2.1 the batch-context artefact accounts for
98% of its spread (sd 0.214 of 0.216), against 25% in the medium tercile and 39% in the large. Below
about |ΔΔG| 0.1 the reversal test cannot see past the artefact, so the trend above should be read
from the medium and large rows only — which is also where the augmented channel's signal lives.

Secondary structure in the error, by mean |fwd+rev|:

| contrast | worse | better |
|---|--:|--:|
| reversal removes volume (>40 Å³) | **0.643** | restores volume 0.302 · neutral 0.266 |
| glycine either side | **0.613** | proline 0.168 · cysteine 0.239 |
| charge change | 0.407 | no charge change 0.277 |
| non-alanine substitution | 0.405 | alanine scan 0.261 |
| small complexes (≤558 res) | 0.393 | medium 0.286 · large 0.291 |

All are consistent with one mechanism: the more the forward mutation let the structure relax, the
less reverting recovers. Removing volume on reversal means the forward move *added* volume and the
neighbourhood opened around it. Glycine's error is a heavy tail rather than a shift — median 0.046
against a mean of 0.613 — which fits backbone flexibility producing occasional large rearrangements
rather than a uniform penalty.

**The rows actually augmented are the easier ones.** Mean |error| is 0.204 on the 201 rows inside an
augmented split against 0.340 on the 1543 outside it. So the pilot restricted to augmented rows
would have understated the effect, and the augmentation as built is less corrupted by hysteresis
than the full set implies — which is a point in favour of §1 being the dominant defect of the two.

## 4. The pre-registered decision, applied as written

The runbook fixed the rule before the numbers existed, precisely so the reading could not be chosen
afterwards. It lands **in between**: correlation 0.840 is neither ≥0.95 nor ≤0.8, and sd(fwd+rev)
0.796 is below sd(ddG) 1.464.

**So this does not by itself narrow §24's conclusion, and it is not being used to.** The rule
names the ESM-C 6B augmented base arm (`docs/history/runbooks/RUN_AUG_BASE_ESMC6B_GPU.md`) as the decisive experiment in
this branch, and that remains true.

What the probe does establish, independent of that rule:

- **Exact antisymmetry is rejected.** The slope's interval excludes 1 and the mean-sum's excludes 0.
  The augmentation's stated premise — "FoldX binding ΔΔG is antisymmetric term-for-term" — is false
  as an equality, and holds only as an approximation that degrades with effect size.
- **§1 is a separate, larger, and exactly repairable defect** which the pre-registered rule was
  never about, and which alone predicts the observed sign split.

**Order of work.** §1 is fixed and the splits are rebuilt (§1.2–§1.3). What that buys is the ability
to ask the question again: every augmented FoldX result on record was trained on a channel offset by
more than its own spread, so none of them measure augmentation, and **re-running the augmented arms
is now the next step**. Only if the sign split survives that does §2's hysteresis need addressing,
and the remedy there is to compute reverse rows with FoldX rather than derive them, at roughly this
probe's cost scaled to the augmented set. The ESM-C 6B *base* arm (`docs/history/runbooks/RUN_AUG_BASE_ESMC6B_GPU.md`)
never read the channel, so it is unaffected by the repair and remains decisive as written.
