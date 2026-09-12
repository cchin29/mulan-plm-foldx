# Critical review — draft v1 (FoldX arc + augmentation plan)

_2026-07-27. Adversarial read of `FOLDX_STRUCTURE_ARC_AND_AUGMENTATION_PLAN.md`, with three claims
re-derived from the repo rather than taken from the existing write-ups. Ordered by whether the
finding **changes a conclusion**, **weakens the evidence**, or **is a fixable defect**._

---

# Tier 1 — findings that change a conclusion

## 1.1 ⚠️ The missing FoldX-alone baseline

**What the draft claims (§4.6):** under CD-HIT-clustered splits, Ankh-large base per-structure
ρ = 0.13 and the FoldX channel "more than triples it to 0.43" — presented as *the headline of the
whole FoldX thread*.

**What is never asked:** how well does **FoldX by itself**, unsupervised, score on those same test
sets? The draft even notes (§4.5) that the entire published frontier uses FoldX exactly this way —
as a baseline predictor — and then omits the comparison.

**Computed here** from the split files (col 4 = FoldX scalar, col 3 = experimental label), same
per-structure Spearman, same T≥10 rule, fold 0:

| Test set | MuLAN base | +FoldX scalar | +FoldX MLP | **FoldX alone** | MuLAN's real increment |
|---|---|---|---|---|---|
| CATH-superfamily, single | 0.321 | 0.420 | **0.455** | **0.398** | **+0.057** |
| CATH-superfamily, multi | 0.444 | 0.499 | **0.589** | **0.540** | **+0.049** |
| CATH-superfamily, all | 0.331 | 0.382 | 0.387 | **0.383** | **+0.004** |
| clustered (SP, full-SKEMPI) | 0.142 | 0.354 | 0.360 | **0.350** | **+0.010** |
| clustered (MP) | 0.091 | 0.331 | 0.355 | **0.369** | **−0.014** |
| by-complex (MP) | 0.340 | 0.397 | 0.391 | **0.437** | **−0.040** |

_(FoldX-alone computed on the same fold-0 test TSVs; covered rows only where coverage < 100 %. The
CATH row is the cleanest — `splits_cath_foldx/fold_0/skempi_all_test.tsv`, 99.9 % coverage. The
recomputation reproduces the `results_matrix_ps.csv` complex counts exactly, 11 / 13, which
validates the comparison.)_

**Consequence.** The "+0.21 to +0.31 lift" is almost entirely *the model learning to read the
number we handed it*. Measured against the correct comparator, MuLAN contributes **+0.05 at best**
(CATH single/multi) and **≈0 or negative on three of six tiers**. On MP-clustered and MP-by-complex,
**the better move is to throw MuLAN away and report FoldX**.

This does not make the work uninteresting — "+0.05 over an unsupervised physics baseline on an
out-of-superfamily split" is a legitimate, frontier-comparable claim, and it is *how the frontier
reports*. But it is a different claim from the one in the draft, and a reviewer
will ask for this table in the first five minutes.

**Actions:**
- Add FoldX-alone as a row in every §4.6 table and in the benchmark matrix, before it is
  presented. It is a 20-line script; the data is already on disk.
- Restate the headline as *"MuLAN adds +0.05 per-structure ρ over the FoldX baseline on the
  out-of-superfamily split"* — and be explicit that on the multi-point tiers it currently does not.
- **This also invalidates the plan's success criteria (§12).** "ΔΔG ρ rises ≥ +0.03 over
  MuLAN-base" is the wrong bar; the bar is **improvement over FoldX-alone**, because that is the
  free alternative.

## 1.2 The Task-2 hypothesis premise

**The draft's mechanism (§11):** FoldX interface-saturation augmentation should sharpen the
attention map "because it densely supervises the interface region that currently gets attention
only incidentally."

**Measured:** on 15 sampled S1102 complexes, **99.1 % (233/235) of SKEMPI single-point mutations
already sit on a cross-chain 5 Å interface residue** (most complexes 100 %; 3SGB 177/177).

So the training signal is *already* ~100 % interface-localised. Interface-saturation augmentation
does **not** change the interface-vs-bulk balance — it cannot, there is no bulk to displace. The
stated mechanism is void.

**What augmentation actually changes** is *within-interface coverage*: SKEMPI is wildly
non-uniform — a handful of saturated protease/inhibitor complexes (3BT1 240 muts, 1R0R 191, 3SGB
191, 1PPF 190) alongside a long tail of complexes with **1–13 mutations each**. Saturation moves a
1-mutation complex to ~750.

**And the sign of that effect is genuinely ambiguous.** SKEMPI's positions are hotspot-biased —
experimentalists mutate residues that matter. That bias may be *exactly what makes the attention
map peak sharply*. Uniform interface coverage could just as easily **flatten** it. The draft states
a one-directional prediction where the honest position is a two-sided one.

**Actions:**
- Rewrite the hypothesis: *"augmentation redistributes training mass from hotspot-biased sampling
  to uniform interface coverage; whether that sharpens or blurs the attention map is the
  experiment."* A blur is a real, reportable result.
- The measurement that decides it should be run *before* any training: interface-AUROC of the
  existing Ankh-large head restricted to hotspot vs non-hotspot interface residues.

## 1.3 Source A ranking vs the measured failure mode

The failure the plan exists to fix is **cross-complex / cross-family**: base ρ falls 0.39 → 0.28
→ 0.14 as splits go leaky → by-complex → clustered. That is a *between-complex* generalization
failure.

**Source A (saturation on the training complexes) adds zero new complexes.** It densifies mutations
inside complexes the model already sees. It attacks the *within*-complex axis while the deficit is
on the *between*-complex axis.

There is a defensible counter-argument — per-structure Spearman *is* a within-complex ranking
metric, so a transferable within-complex ranking function learned from denser sampling could carry
to unseen complexes. But the draft **assumes** this rather than arguing it, and then ranks the one
source that genuinely adds structural diversity (**Source B**, non-SKEMPI complexes) as *secondary*
and schedules it for week 3, i.e. probably never.

**Action:** either argue the within-complex transfer mechanism explicitly, or promote Source B —
and note that the cheapest decisive experiment is the **contrast** between them (more mutations on
old complexes vs. same budget of mutations on new complexes). That contrast is a better paper than
either arm alone, and the draft buries it in week 3 as an afterthought.

## 1.4 Stage-A degeneracy

**§10 says:** Stage A pre-trains on FoldX pseudo-labels; §11's matrix includes
"Ankh-large + FoldX channel + augmentation."

In that cell, during Stage A, **the training target is FoldX ΔΔG and the `add_scores` channel input
is FoldX ΔΔG — the same number.** The head can drive the loss to ~0 by learning the identity map
from the channel slot, learning nothing from the embeddings. Stage A becomes a no-op at best and a
head-corrupting initialisation at worst, and **nothing in the logs would look wrong** — the loss
curve would look excellent.

**Action:** Stage A must **disable the FoldX channel** (train the pure sequence pathway to imitate
FoldX), then Stage B re-enables it for fine-tuning on real labels. State this explicitly; it is the
kind of detail that silently invalidates a month of runs. The 12-term MLP arm has the same problem
in a milder form (term 1 *is* the target).

---

# Tier 2 — evidential weaknesses

## 2.1 Confidence intervals, seed replication, and the 13-complex headline

- The CATH tier is **one fold**. Its per-structure Spearman is a mean over **13 complexes**
  (11 for single-point).
- Those 13 are heavily concentrated: **1JTG contributes 194 of 687 mutations (28 %)** and scores
  ρ = +0.549 for FoldX alone. Drop 1JTG and the tier mean moves materially.
- The per-complex ρ spread on that fold is enormous: **+0.06 (2PCC, 3SZK) to +0.76 (1EMV)**.
- Part I elsewhere reports fold-level std of **±0.03–0.05**. Differences of that size are quoted
  throughout §4.6 and §12 **without a single interval**.

**Action:** bootstrap CIs over complexes (the frontier does this), and report ≥3 seeds. Until then
no delta below ~0.05 should be asserted as real. This directly undercuts §12's "+0.03" criterion —
see 2.3.

## 2.2 Two metrics and two split regimes, presented adjacently

§4.4 reports **pooled Pearson on the leaky S1102 CV10** (+0.0093 / +0.0166); §4.6 reports
**per-structure Spearman on leakage-controlled splits** (+0.06 to +0.31). The draft caveats this
once, but a reader skimming slides will read "+0.017 → +0.30" as the same quantity growing. The
metric change alone is worth −0.39 (the project's own Metric-Rationale artifact: 0.881 → 0.489).

**Action:** never put the two on one slide without an explicit "different metric, different split"
band, and consider reporting §4.4 in per-structure Spearman too so the arc is measured in one unit.

## 2.3 The success threshold and the noise floor

"+0.03 per-structure ρ" (§12) has no power analysis behind it. Given 11–31 complexes at T≥10 and a
per-complex ρ spread of 0.06–0.76, the standard error of the mean is plausibly ~0.06–0.08 — i.e.
**the stated success criterion may be smaller than one standard error.**

**Action:** compute the bootstrap SE of per-structure ρ on each tier *first* (a 30-minute job on
existing predictions), then set the threshold at ≥2 SE. If that turns out to be +0.12, the plan
needs to know that before spending a week of GPU.

## 2.4 Provisional rows in the §4.6 table

The draft carries the caveat, but it is under-weighted: the by-complex and clustered rungs were
mid-rerun (unified single+multi + expanded FoldX coverage) as of 2026-07-26. The coverage fix
previously moved CATH FoldX numbers by **+0.04 to +0.12 PCC** — i.e. larger than every effect the
plan is designed to detect. Building the augmentation matrix on top of numbers that are about to
move risks re-running everything.

**Action:** gate week 1 on the rerun landing, or explicitly accept that the baseline column will be
refreshed and budget for it.

---

# Tier 3 — cost model defects

## 3.1 FoldX per-mutation cost

It was derived from mtimes of `*_Repair_N.pdb` — **BuildModel output only**. The pipeline also runs
**`AnalyseComplex` on two chain groups for every mutant plus the WT**. That work is not in the
measured interval. True end-to-end cost is plausibly **1.5–2× higher (~12–15 s/mutation)**, which
scales every FoldX line in §8 by the same factor.

**Action:** time one complex end-to-end with `time` before committing to the 50 k target. Cheap,
and it moves the 50 k FoldX budget from 8 h to 13–16 h on 14 cores — still fine, but the number
should be right.

## 3.2 Embedding throughput and storage

- **13 s/embedding** comes from dividing a 21.9 h wall-clock span by 6,422 files. Any idle time,
  interruption, or overnight gap is baked in. It is an *upper bound*, not a throughput measurement.
- **"~5–10× on the 3090"** is invented. Nothing in the repo measures Ankh-large embedding
  throughput on that GPU. The entire week-1/week-2 schedule rests on it.
- **No disk-space check was performed** on either the Mac or the GPU box, yet the plan proposes
  33–65 GB of new embeddings (and the alternative 200 k tier would need 260 GB).

**Action:** measure Ankh-large throughput on the 3090 with a 200-sequence timing run, and `df -h`
both machines, before the 50 k target is committed to anything.

## 3.3 FoldX failure rate on chosen mutations

All existing coverage statistics are for **SKEMPI-derived** mutations. Saturation introduces
substitutions SKEMPI never contains at those positions — →Pro and →Gly at structured positions,
small→large at buried interface positions. FoldX BuildModel failure/instability rates on that
distribution are **unknown and unbudgeted**, and §4.4 already documents that FoldX's magnitudes are
least reliable exactly there (clash overshoot).

**Action:** add a 500-mutation pilot measuring failure rate and the ΔΔG distribution (watch for a
fat destabilising tail that would dominate the pseudo-label loss), and consider clipping or
rank-transforming labels.

## 3.4 The `train.py` checkpoint claim

The draft says `scripts/train.py` "has no checkpoint-loading path (`mulan.load_pretrained` is used
for inference only)." Reading the file: `load_pretrained` **is** in the training path (it
initialises the model when the argument names a registered model), and `save_model_ckpt` already
writes `{state_dict, config}`. So `--init-from` is a **~5-line change reusing existing
machinery**, not a new capability. The conclusion (small change) survives; the justification given
for it is wrong and should be corrected before anyone quotes it.

---

# Tier 4 — omissions

| # | Gap | Why it matters |
|---|---|---|
| 4.1 | **No reproduction anchor in Part I.** Part I never states how the local Ankh-large numbers line up against the *published* MuLAN result. | Part II §11 calls Ankh-large "the MuLAN paper's own backbone, so the comparison lands against the published model" — that claim is unsupported inside the document. |
| 4.2 | **Term 1 is collinear with terms 2–12.** FoldX's Interaction Energy is a weighted sum of the components, so the 12-term MLP input is rank-deficient by construction. | Unremarked. It is benign for an MLP but it is the reason "MLP ⊃ scalar" and it should be stated, not left for a reviewer to notice. |
| 4.3 | **Multi-point augmentation is unaddressed** beyond one risk-table line — yet the MP tiers are the weakest (base ρ 0.091) and `augment.py` explicitly skips multi-point reversal. | The tier with the most headroom gets the least plan. |
| 4.4 | **No abandonment criterion.** §12 lists five outcomes, all of which lead onward. There is no rule for "FoldX-alone beats every MuLAN arm, stop." | Given 1.1, that outcome is live on three tiers *today*. |
| 4.5 | **Everything is still SKEMPI.** Part II never evaluates outside the database the training data comes from, so "generalization" remains in-distribution w.r.t. SKEMPI's own composition. **AbBiBench** is the available external set. | A cross-database result would be worth more than another SKEMPI tier, and the "data extension" framing is compatible with it. |
| 4.6 | **Nothing on packaging for release.** The plan has an appendix file map but no packaging step, no environment spec, no data manifest. | Publishable code and data need all three; an appendix file map is not a substitute. |
| 4.7 | **`results_all` symlink fragility.** Coverage depends on a symlink farm assembled by `build_results_all.py` across five campaign dirs. Nothing in the plan verifies it before the augmentation campaign adds a sixth. | This exact class of bug already cost 28 → 88 % coverage once. |

---

# Next steps, in order

1. **Compute and publish the FoldX-alone row on every tier** (§1.1). Half a day. It reframes the
   story and it is the first thing a reviewer asks for. Nothing else should start first.
2. **Bootstrap CIs on the existing predictions** (§2.3). Sets an honest detection threshold and
   tells whether the planned experiment can even resolve its target effect.
3. **Fix the Stage A degeneracy in the design** (§1.4) and the Task-2 hypothesis wording (§1.2)
   before any compute is spent.
4. **Measure, don't estimate:** FoldX end-to-end per mutation, Ankh-large throughput on the 3090,
   free disk on both machines (§3.1–3.2).
5. **Then** run the 10 k pilot — but on the **Source A vs Source B contrast**, not Source A alone,
   because that contrast is the actual scientific question (§1.3).

**One-line summary of the review:** the retrospective is solid but is missing the one comparator
that determines what it means; the plan is well-costed but its primary source attacks the wrong
axis, its interface hypothesis rests on a premise that measurement contradicts, and its pre-training
stage as written can be solved trivially by the model.
