# Why the headline metric is per-structure Spearman, not cv10 pooled Pearson

_Rationale for reporting **per-structure Spearman ρ (T≥10)** as MuLAN's headline ΔΔG metric instead of
the historical **10-fold-CV pooled Pearson (PCC)**. §"Slide skeleton" at the end is the draft for the
later deck. Naming: "leakage-controlled" / "clustered" / "generalization ladder"._

---

## The two metrics

**cv10 / pooled Pearson (the old headline).** 10-fold CV over *mutations*; pool every out-of-fold
prediction across all complexes into one pile; compute a single Pearson correlation vs measured ΔΔG.
ESM-C 6B base on balanced S1102 = **0.881** (mean per-fold PCC, `results_matrix.csv`). This is the
SOTA-looking number MuLAN historically quoted.

**Per-structure Spearman, T≥10 (the new headline).** For each complex separately, rank-correlate its
*own* mutations' predicted vs measured ΔΔG, then average across complexes; score only complexes with
≥10 measured mutations (fewer than that, a within-complex rank correlation is noise). This is RDE's
`per_complex_corr` convention — our implementation matches theirs to <1e-9. Same ESM-C 6B base, leaky =
**0.489**.

Same model, same data: **0.881 vs 0.489.** That gap is the reason for the switch.

---

## Why pooled Pearson overstates — three compounding effects

1. **Bulk domination.** SKEMPI ΔΔG is mostly near-neutral mutations with a thin large-effect tail. A
   model that predicts near-zero with the right sign captures the bulk and scores high PCC while failing on
   the large-effect and fine-ranking cases that matter for design. Pearson rewards a shrunk predictor.

2. **Between-complex variance (the big one).** Pooling ~300 complexes means most variance in the pile is
   *between* complexes (complex A averages +2 kcal/mol, complex B −1). A model that captures only those
   per-complex offsets — gross features like interface size, or just "I've seen this complex" — gets a
   high pooled Pearson even with zero within-complex ranking skill. Pooling hands out credit for the
   easy between-complex spread.

3. **Linear-calibration sensitivity.** Pearson measures linear fit: a few large-|ΔΔG| outliers swing it,
   and it rewards absolute kcal/mol calibration. But a ΔΔG tool doesn't need calibration — it needs to
   *rank* candidates. Spearman (rank) is robust to monotonic miscalibration and to the heavy tail.

---

## Why per-structure Spearman is the right headline

The real design question is *within one complex*: given this interface, which candidate mutation most
improves (or least harms) binding? That is exactly a per-structure ranking question, and per-structure
Spearman measures exactly it — stripping out the between-complex freebie (effect 2) and the bulk
(effect 1). It is also the **field standard**: RDE-Network, PPIformer, DiffAffinity, Prompt-DDG,
CATH-ddG, USP-ddG ("Per-PPI"), and ProtBFF all headline this metric — so reporting it is what makes
MuLAN *comparable*, not a number no one else uses.

---

## The metric cut and the split cut are coupled

They compound; separating them is clarifying. ESM-C 6B base:

| | Pearson | pooled Spearman | per-structure Spearman |
|---|---|---|---|
| **leaky** (per-mut 10-fold) | **0.881** (mean per-fold) | — | 0.489 |
| **clustered** (≤60%) | 0.642 (pooled) | 0.448 | 0.370 |

- **Metric cut** (top row, left→right): pooled PCC → per-structure Spearman drops **0.88 → 0.49** at a
  fixed split.
- **Split cut** (right column, top→bottom): leaky → clustered drops **0.49 → 0.37** at a fixed metric.

Both are real; the metric cut is the larger single one. And they are not independent: the between-complex
signal that inflates pooled Pearson is *exactly where homology leakage lives* — a leaky model recognizes
"this resembles a training complex" and reproduces its mean/scale, which pooled Pearson rewards. So
**pooled Pearson on a leaky split compounds both effects, while per-structure Spearman on a
leakage-controlled (clustered) split controls for both — a bulk-robust metric on a homology-controlled split.**

The masking is visible directly. At the clustered split, ESM-C 6B base still shows **pooled PCC 0.642**
(looks fine) while its **per-structure Spearman is 0.370**, and ProstT5 base collapses to **0.020**. If
we kept pooled Pearson as headline, the "base PLMs memorize family identity and collapse out-of-family"
story — and the FoldX rescue — would be invisible. Per-structure Spearman is what makes the
generalization ladder legible.

---

## AUROC — sign-of-effect

**Definition.** Area under the ROC curve. Give each mutation a binary label — *destabilizing* (measured
ΔΔG > 0, weakens binding) vs stabilizing/neutral — and rank all mutations by predicted ΔΔG. AUROC is the
probability a randomly chosen destabilizing mutation is ranked above a randomly chosen non-destabilizing
one: **0.5 = chance, 1.0 = perfect**, threshold-free.

**How we compute it.** Label by the sign of measured ΔΔG, score by predicted ΔΔG, take the Mann-Whitney
AUROC (our `auroc_destab`). This is the RDE / USP-ddG convention, so it lines up directly with the
frontier's "AUROC". It scores **direction** — "weaken or strengthen binding?" — not magnitude, and needs
no per-complex sample-size threshold (unlike per-structure Spearman).

**Why it matters.** It's robust where the correlation metrics aren't: no T≥10 cutoff, and not inflated by
the near-zero bulk or between-complex spread. It's the **triage metric** (which mutations to make vs
avoid) and degrades gracefully with the split (~0.85 leaky → high-0.7s clustered). **MuLAN's standout
result:** on the strict CATH split MuLAN+FoldX AUROC **0.791 ranks 2nd** (above CATH-ddG 0.781, under
USP-ddG 0.802) — while its per-structure Spearman (0.438) is mid-pack. So the model calls sign-of-effect
near the top even when exact within-complex ranking is middling. Caveats: AUROC lives in a narrow
~0.6–0.8 band (read ordering, not gaps), and it judges *direction only* — a model can score high AUROC
yet still mispredict the *magnitude* of large-effect mutations (the tail RMSE catches).

## Other metrics we keep

**Precision@k** — of the top-*k* mutations flagged, the fraction that are genuinely large-|ΔΔG| (the
retrieval view of the design task); no T≥10 threshold needed. **Pooled Pearson/Spearman** survive in
exactly one place (frontier §1b) because ProtBFF reports pooled, so a like-for-like line-up needs it —
and there MuLAN+FoldX's pooled 0.648 / 0.516 sits at/above ProtBFF's 0.514 / 0.477 (S1102; on the
dataset-matched full-SKEMPI it is below — see BENCHMARK_MATRIX).

**Rule of thumb:** lead with per-structure; cite pooled only when comparing with methods that reported pooled.

**One line to say out loud:** the leakage-controlled evaluation is *supposed* to produce smaller numbers
than the 0.88 people remember. That drop is the evaluation getting stricter, not the model getting worse.

---

## Slide skeleton (for the later deck)

**Title:** "Why our headline number went from 0.88 to 0.49 (and why that's the leakage-controlled one)"

**One-line takeaway (top):** Same model, same data — pooled Pearson rewards the easy between-complex
spread; per-structure Spearman measures the design question (rank mutations *within* a complex).

**Center visual — the 2×3 decomposition table** (ESM-C 6B base):

| | Pearson | pooled Spearman | per-structure Spearman |
|---|---|---|---|
| leaky | 0.881 (mean per-fold) | — | 0.489 |
| clustered | 0.642 (pooled) | 0.448 | 0.370 |

with two arrows annotated: horizontal "metric candor: 0.88→0.49" and vertical "split candor:
0.49→0.37".

**Left callout — 3 reasons pooled Pearson inflates:** bulk domination · between-complex variance ·
outlier/calibration sensitivity. (Between-complex variance = where leakage hides.)

**Right callout — why per-structure Spearman:** matches the real use-case (rank a complex's mutations) ·
field standard (RDE / PPIformer / CATH-ddG / USP-ddG / ProtBFF) · exposes the base-PLM collapse that
pooled PCC masks.

**Footer / speaker note:** "Smaller numbers = stricter metric, not a worse model. We still report
AUROC + precision@k for triage, and pooled only to line up against ProtBFF."

_Sources: `scripts_plots/results_matrix.csv` (leaky cv10 PCC), `experiments/retrain_split/results_clustered.csv`
(pooled + per-structure, clustered), `scripts_plots/results_matrix_ps.csv` (per-structure ladder). RDE
`per_complex_corr` — doi:10.1101/2023.02.28.530137. Metric-inflation critique in
`REVIEW_gaps_and_landscape.md` §2.2._
