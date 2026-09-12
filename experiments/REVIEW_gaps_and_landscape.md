# MuLAN ΔΔG work — critical review: gaps, directions, and redundancy vs. the literature

_Written 2026-07-15. Scope: the S1102 mispredict / cross-fold analysis
(`scripts_plots/S1102_CROSSFOLD_MISPREDICT.md`, `FOLDX_SUMMARY.md`), the A2/MINT arm
(`docs/history/PLAN_MINT_A2.md`), and the queued A1/C1 levers
(`docs/history/PLAN_A1_INTERFACE_XATTN.md`). This is a self-critique + landscape check, not a
results doc — every "action" here is a proposal, nothing is run._

---

## 0. TL;DR

- **The defensible core of this work is a *diagnostic*, not a new predictor:** it localizes the
  extreme-ΔΔG tail failure to the **head + MSE loss** (not embedding partner-blindness), shows
  orthogonality is **cross-modality (physics vs. PLM), not cross-PLM**, and gives a **FoldX
  scalar-vs-MLP-by-dataset-size** rule. Those are not in the SOTA papers.
- **The single biggest gap:** the CV split is **per-mutation random**, which leaks whole complexes
  between train and test. Every PCC and every hotspot statement is conditioned on that. Published
  work shows leave-one-complex-out collapses ddG predictors (TopNetTree → PCC ≈ 0.17), so our
  absolute numbers are **not comparable to the frontier** and the "generalizes beyond SKEMPI" claim
  is **untested**.
- **Redundancy:** the *phenomenology* (PLMs shrink large ΔΔG; structure/physics helps hotspots;
  complex-context embeddings on SKEMPI) is **largely published**. The *method* directions (A1
  interface-attention, B physics channel) **overlap heavily** with RDE-Network / PPIformer /
  Prompt-DDG / GearBind. The *mechanistic diagnostic* is the novel wedge — **but only survives as a
  contribution under a non-leaky split.**

---

## 1. What is genuinely solid (keep this framing)

The contribution is analytical. Specifically:

1. **Tail-bottleneck localization.** The MINT complex-vs-mono ablation (`docs/history/PLAN_MINT_A2.md` §8a)
   isolates that giving the model cross-chain *information* (+0.035 bulk) does **not** de-shrink the
   tail — so the limiter is the pooled head + MSE loss, not the embedding. That is a clean negative
   result the "we beat SOTA by X" papers don't make.
2. **Orthogonality is cross-modality, not cross-PLM.** Residual corr PLM↔PLM ≈ 0.78–0.92 vs.
   FoldX↔PLM ≈ 0.46–0.55, holding even for structure-aware SaProt — quantified per-mutation.
3. **FoldX scalar vs. 12-term MLP tracks dataset size** across S1102/S1131/S2003/S4169 (RESULTS
   §20.1): scalar the safe default, MLP the upside on ≥2k-mut sets.
4. **A reproducible per-mutation hotspot taxonomy** (complexes, mechanisms, the error-vs-ΔΔG funnel).

These stand on their own **as diagnostics of a frozen-PLM + light-attention head**. They do not, on
their own, constitute a competitive predictor.

---

## 2. Gaps / what's missing

> **Status update 2026-07-19 — the leakage/split gap (§2.1) is now CLOSED.** The full
> leakage-controlled ladder has been run: by-complex and CD-HIT ≤60% clustered
> (`experiments/retrain_split/`, `scripts_plots/results_matrix_ps.csv`) **and** the strictest
> **CATH-superfamily hold-out** on USP-ddG's literal 813-mutation test set
> (`experiments/full_skempi_seqonly/{SUMMARY_cath.md,results_cath*.csv}`). Headline: base PLM ρ collapses
> out-of-family (0.39→0.20 clustered; 0.02 for ProstT5), the FoldX arm holds, and at the CATH tier
> MuLAN + FoldX-scalar (ESM-C 6B) reaches overall ρ **0.584** / per-PPI **0.411** — mid-pack, above the
> pretraining GNNs, ~0.03–0.07 below the relaxed-structure SOTA. The comparator figures behind that
> placement were never confirmed against the source tables and should not be quoted from here.
> The forward-looking "not yet computed" language below (§2.1, §4, §5) predates this run and is retained
> for context.

### 2.1 ⚠ CV split leakage — the load-bearing gap
The balanced/paper 10-fold CV splits **by mutation**, so the same complex appears in train and test.
Consequences:
- **Inflated numbers.** 0.85–0.89 pooled PCC is an in-distribution (seen-complex) figure. The field
  reports much lower under clustered splits: a leave-one-complex-out test drops TopNetTree to
  **PCC ≈ 0.17** [[GNN-PPI pretraining]](https://arxiv.org/pdf/2008.12473); out-of-superfamily
  robustness is the explicit target of [[CATH-ddG]](https://doi.org/10.1093/bioinformatics/btaf228).
- **Shrinkage is partly a CV artifact.** "Distributional bias" under LOO/clustered CV *induces*
  regression-to-the-mean independent of the model
  [[distributional-bias analysis]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11177965/). So the
  error-vs-ΔΔG funnel (`error_vs_ddg_matrix.png`) cannot be cleanly attributed to head+loss until
  it is reproduced under a non-leaky split.
- **The generalization claim is untested.** "Will bite on antibody design / de novo binders" is
  asserted, not measured.

**Action (highest value):** rerun the core analysis under **leave-one-complex-out** and/or
**CATH-superfamily-clustered** folds. Expect a large PCC drop; the *interesting* question is whether
the arm ranking (FoldX > aug; MINT bulk lift; the hotspot set) survives.

### 2.2 Metrics are bulk-dominated
PCC rewards a shrunk predictor and is dominated by the easy near-zero bulk. The field standard is
**per-structure Spearman** + **ranking/retrieval** (e.g. precision@k on |ΔΔG| ≥ 2) — which is also
what the design use-case needs. worst-40 MAE and shrinkage slope were good additions; adopt
per-complex Spearman and top-k retrieval to be decision-relevant *and* comparable to published
methods (which report exactly these).

### 2.3 No anti-symmetry / self-consistency test
ΔΔG(A→B) ≈ −ΔΔG(B→A) is the axis the published bias literature centers on
[[Quantification of biases, Brief. Bioinform. 2024]](https://academic.oup.com/bib/article/25/1/bbad491/7513597).
The reverse-mutation augmentation data already exists — measuring the model's anti-symmetry is
nearly free and directly connects this work to that framing.

### 2.4 No SOTA structure baseline in the comparison
Everything is measured against *base PLM* and *FoldX*. "FoldX doubles hotspot rank corr" is a low
bar: RDE-Network, PPIformer, Prompt-DDG, GearBind already do interface-aware ddG far better. Without
one of them on the same split, the A1/B "add interface/physics" thesis is untethered from the actual
frontier.

### 2.5 Label-noise floor unaudited
Extreme hotspots (1MAH WA276R = 8.81 kcal/mol) may be noisy, multi-condition, or aggregated SKEMPI
entries. Chasing them without an experimental-uncertainty floor risks fitting label error. A
provenance/temperature audit of the worst-40 would set a realistic ceiling.

### 2.6 flex-ddG / FEP proposed everywhere, run nowhere
The docs repeatedly nominate flex-ddG/FEP as the magnitude fix. A **10-hotspot flex-ddG spot check**
would validate (or kill) the "structure has the magnitude, FoldX just can't emit it" claim that
motivates B1 — cheap relative to its narrative weight.

### 2.7 Frozen PLM only
No fine-tuning / LoRA of the PLM on the task. The tail may need task adaptation a frozen embedding +
tiny head cannot provide. Worth at least one LoRA arm as a ceiling probe.

### 2.8 Benchmark scope — S1102 is the *classical-lineage* subset, not the frontier set
Our whole analysis is **S1102** (1,102 **single**-point mutations / ~110 binary complexes). That is a
curated subset from the *classical structure-feature* ML lineage (mCSM-PPI2, BindProfX, MuPIPR, iSEE,
GeoPPI, DLA) — the one MuLAN benchmarks against (siblings S1131/S2003/S4169 single-point, S1400
multi-point). **The modern DL frontier does not use S1102 at all.** RDE / DiffAffinity / Prompt-DDG /
PPIformer / GearBind / BA-DDG / CATH-ddG / USP-ddG / ProtBFF all evaluate on the **full curated SKEMPI
v2** (~7,085 mutations / ~345 complexes), report **single *and* multiple** mutations separately, use
homology-aware splits (§4a), and add **antibody-antigen / SARS-CoV-2 OOD** sets (AB-Bind, AbBiBench
CR6261, trastuzumab-HER2, RBD-ACE2/antibody DMS). Consequences:
- Our numbers are comparable to the *classical* leaderboard (MuLAN's turf), **not** to the frontier —
  different dataset size, single-point only, easier splits, no antibody OOD.
- Two axes are **entirely absent** from our work and first-class in the frontier: **multiple-point
  mutations** (epistasis — USP-ddG's headline hard case) and **antibody-antigen generalization** (the
  design-relevant OOD, and where MuLAN's own paper already reports a drop on S2003's antibody classes).
- This compounds §2.1/§4a: matching the frontier needs not just a harder *split* but the full *dataset*
  + multi-point + an OOD set. Scoped in `retrain_split` Phase 3 + "§ expansion effort" below.

---

## 3. Redundancy vs. bioRxiv / published work

**Candid verdict: phenomenology mostly known; the MuLAN-specific diagnostic is not; the method
fixes overlap heavily with existing structure-based ddG models.**

### Already published (redundant as novelty)
| claim in this work | prior art |
|---|---|
| PLMs underpredict / narrow-range on large ΔΔG; biased toward destabilizing | [Brief. Bioinform. 2024 (biases)](https://academic.oup.com/bib/article/25/1/bbad491/7513597); narrow-range T5/LSTM obs. |
| Regression-to-mean under CV | [distributional-bias LOO-CV](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11177965/) |
| Interface/structure-aware beats sequence-only on ddG hotspots | [RDE-PPI (ICLR 2023)](https://github.com/luost26/RDE-PPI); [PPIformer (ICLR 2024)](https://github.com/anton-bushuiev/PPIformer); [Prompt-DDG (2024)](https://arxiv.org/abs/2405.10348); GearBind; [3D-ΔΔG](https://onlinelibrary.wiley.com/doi/10.1002/prot.26837) |
| Complex-context (multi-chain) embeddings on SKEMPI ΔΔG | [MINT (Nat. Commun. 2026)](https://www.nature.com/articles/s41467-025-67971-3) — MINT's own SKEMPI result |
| FoldX physics + ML fusion | many (dual-channel / feature-fusion ddG models) |
| **FoldX biophysical features scaled into embeddings + cross-embedding attention + anti-symmetry** (≈ our **A1+B combined**) | **[ProtBFF (bioRxiv 2025-12-23, doi:10.64898/2025.12.23.696257)](https://www.biorxiv.org/content/10.64898/2025.12.23.696257)** — encoder-agnostic plug-in on ESM2/ESM3/ProSST |
| **SKEMPI homology leakage inflates by-complex CV** (our §2.1) | **ProtBFF** (CD-HIT clustering: 335→253 clusters at 99%, 136 at 60%); **[USP-ddG (bioRxiv 2025-11-09)](https://www.biorxiv.org/content/10.1101/2025.11.09.687124)** (by-complex CV is **88.7% "easy"**, TM-score ≥0.6 to train); **[CATH-ddG (Bioinformatics 2025)](https://doi.org/10.1093/bioinformatics/btaf228)** |

**Implication for A1/A2/B as *methods* — now sharper (Dec-2025 lit).** **ProtBFF is essentially our
A1+B, published**: it computes FoldX biophysical scores (interface, burial, dihedral, SASA, lDDT),
scales per-residue embeddings by them, fuses WT/mutant via **cross-embedding attention**, and averages
forward/reverse for **anti-symmetry** — encoder-agnostic on frozen PLMs. So building A1/B as novel
methods is largely pre-empted; the pragmatic move is to **reproduce ProtBFF's channel on the MuLAN
head**, not reinvent it. A2 (MINT on MuLAN) partly reproduces MINT's own SKEMPI result; the *negative*
"doesn't route through this head" is the only new part. See the frontier survey that preceded `experiments/BENCHMARK_MATRIX.md` (a private note, not shipped)
and `.../papers/` for full notes + DOIs.

### Not obviously done before (the defensible wedge)
- **Mechanistic attribution** that cross-chain *information* is not the bottleneck — the *head + MSE
  loss* is — via the paired MINT complex/mono ablation + the arm×PLM residual decomposition.
- **Scalar-vs-MLP dataset-size rule** across four SKEMPI benchmarks.
- **Per-mutation hotspot taxonomy + cross-modality orthogonality** quantified on identical folds.

These are "why does the PLM-head approach hit a wall, and what is/ isn't orthogonal" contributions —
publishable as analysis, **conditional on a non-leaky split** (else the absolute numbers aren't a
fair comparison to SOTA that reports LOCO/per-structure metrics).

**The leakage-controlled-split frontier is ~0.51 Pearson / ~0.48 Spearman** (ProtBFF's best, ProSST+ProtBFF, at
CD-HIT 60%), *not* the 0.85 our leaky per-mutation CV shows — and **by-complex CV is *still* not leakage-free**
(USP-ddG: 88.7% "easy" by TM-score; ProtBFF: 253 clusters even at 99% id). So the defensible bar is the
**CATH-superfamily / CD-HIT ≤60%** split, and our diagnostic wedge only counts if measured there.

---

## 4. Directions, ranked by value

1. **Homology-controlled CV — and by-complex is not enough.** Reframes everything; the leakage-controlled
   generalization test; makes numbers comparable to the frontier. **Do this first** — it conditions
   every claim. **Update (Dec-2025 lit):** leave-one-*complex*-out still leaks by homology — USP-ddG
   shows 88.7% of by-complex test mutations are "easy" (TM-score ≥0.6 to train); ProtBFF shows 253
   CD-HIT clusters even at 99% identity. So the real target is a **CATH-superfamily hold-out**
   (CATH-ddG's 813-mutation set) or **CD-HIT ≤60%** clustering — treat plain by-complex as a first
   waypoint, not the destination.
2. **Benchmark PPIformer / RDE / Prompt-DDG on the same split** — as baselines *and* as the
   "complex-aware channel" the docs keep proposing. PPIformer's pretrained interface representation
   is exactly that channel, already built — likely a stronger B/A1 than FoldX or a hand-rolled
   cross-attention head.
3. **Per-structure Spearman + top-k retrieval metrics** (precision@k on |ΔΔG| ≥ 2). Field-standard
   and decision-relevant.
4. **Antibody–antigen holdout** (SKEMPI AB subset / AB-Bind / the 608 set) — test the asserted
   generalization directly; the data-volume/diversity question is itself live
   [[Nat. Comp. Sci. 2025]](https://www.nature.com/articles/s43588-025-00823-8).
5. **Anti-symmetry evaluation** — cheap, ties to the published bias framing (§2.3).
6. **flex-ddG spot-check** on ~10 hotspots — validate the magnitude-ceiling claim (§2.6).
7. **Multi-task with mega-scale monomer stability data** (e.g. mega-scale ΔG, ProteinGym) — orders
   of magnitude more labels to learn magnitude / de-shrink the tail, then transfer to PPI ΔΔG.
8. **Label-noise floor** — SKEMPI provenance audit of the worst-40 (§2.5).

**Sequencing note.** #1 and #3 are pure re-analysis on existing predictions/splits (cheap, no new
training) and should precede any further method work — if the arm ranking does not survive a clustered
split, the A1/B/C effort should be re-scoped before spending compute.

---

## 4a. Standard evaluation protocol (what the frontier uses — adopt for comparability)

There is a **de facto benchmark** on SKEMPI v2, set by Luo et al. (RDE-Network, ICLR 2023) and used by
essentially every method since (Prompt-DDG, DiffAffinity, GearBind, Boltzmann-aligned inverse folding,
DSSA-PPI, Twin Peaks). Matching it is the fastest route to comparability.

- **Split — 3-fold, by structure/complex.** The curated set (~7,085 mutations / ~348 complexes) is
  split so **each complex appears in exactly one fold** (2 train/val, 1 test; every point tested once)
  — leave-complex-out packaged as 3-fold, with *published fold indices* everyone reuses. This is
  **not** our per-mutation 10-fold, and our S1102/S1131/S4169 come from the *older* biophysics-ML
  lineage (mCSM-PPI2 / SAAMBE), so neither our split nor our subsets are frontier-comparable as-is.
- **Metrics — per-structure is the headline.** Overall Pearson / Spearman / RMSE / MAE / **AUROC**
  (sign-of-effect classification), **plus per-structure Pearson & Spearman** (correlation computed
  *within* each complex, averaged across complexes) — the latter is what the field leads on and where
  methods separate. Per-structure requires complexes with enough mutations (our 59 singletons can't
  contribute; another reason to move to the curated set).
- **Generalization frontier — homology-clustered / de-leaked.** PPIformer (PPIRef de-dup, cross-family),
  CATH-ddG (out-of-superfamily), and ["Revealing data leakage in PPI benchmarks"](https://arxiv.org/pdf/2404.10457)
  argue even the standard 3-fold under-controls homology (our 37-complex serine-protease/inhibitor family
  would still straddle folds).

**Recommendation (report both):**
- **Comparable:** adopt the **RDE 3-fold by-complex split verbatim** on the curated SKEMPI v2 set and
  lead with **per-structure Spearman** (+ overall Pearson / RMSE / AUROC). Then a claim reads "FoldX-MLP
  adds +X per-structure Spearman vs. RDE/PPIformer," which the field can place.
- **Defensible:** add a **CATH-/Foldseek-clustered** split (stricter than the 3-fold) as a second
  evaluation; the gap between the two quantifies how much PCC was homology memorization.
- **Adopt AUROC now** — sign classification sidesteps the magnitude-shrinkage and label-noise problems
  and is design-relevant.
- **Cheapest first step (no retrain):** re-score *existing* OOF predictions grouped by complex →
  per-structure Spearman + AUROC, to see whether the arm ranking / hotspot story survives the field's
  metrics before committing to the RDE-split retrain. **Implemented in
  `experiments/rescore_perstructure/` (§ below).** Caveat: this changes the **metric**, not the
  **split** — the predictions are still from the leaky per-mutation CV, so these per-structure numbers
  are an upper bound, not the generalization figure.

  **✅ DONE (2026-07-16) — and it paid off (metrics validated <1e-9 vs the RDE-PPI reference):**
  - **Pooled ρ was ~half leverage:** overall Spearman 0.72–0.81 → per-structure **0.31–0.59** (every
    arm −0.22 to −0.42), confirming most of the headline PCC was the 3 mega-scanned complexes, not
    within-complex skill.
  - **FoldX-MLP is the top lever by a wide margin — ~10× its pooled lift:** per-structure **+0.08 to
    +0.21 ρ** over the matched base PLM, **significant on all 6 PLMs** (paired cluster-bootstrap, CIs
    exclude 0). Pooled PCC had been *understating* FoldX; the frontier metric strengthens it.
  - **MINT is the *worst* arm per-structure** (0.31, Δ=−0.14 vs consensus base, P≈0), its bulk +0.035
    exposed as cross-complex leverage; the per-structure context lift is **not** significant (+0.025,
    P=0.73). Sharpens the A2 negative.
  - **Retrieval survives** (AUROC 0.85–0.91, precision@50 ≈ 1.0) — the failure is magnitude /
    within-complex ranking, not top-destabilizer flagging.
  - **Implication:** the arm ranking not only survives the field's metric, it *sharpens* (FoldX up,
    MINT down) — so the A1/B/C effort is worth continuing, and the by-complex retrain (§4a #1) is the
    right next spend; FoldX's relative edge is predicted to widen further under it.

---

## 4b. Expansion effort — what it takes to match the recent papers

Matching the frontier means four things at once (see §2.8): **full curated SKEMPI v2** (~7,085 muts /
~345 complexes, not S1102's 1,102) · **single *and* multiple** mutations · **homology-aware split**
(§4a) · **≥1 antibody-antigen / DMS OOD** set. Broken down by cost, and using MuLAN's sequence-only
nature to sequence the work cheap→expensive:

**A. Cheap / mostly plumbing (reuse what exists):**
- **Splits** — already on disk: RDE by-complex code, PPIRef `iclr24` folds, and CD-HIT/CATH recipes.
  Apply to the full set; no modeling. (days)
- **Metrics/eval** — the `rescore_perstructure` harness + RDE reference already cover per-structure /
  AUROC / single-vs-multiple. (done)
- **Multiple-point mutations** — MuLAN is sequence-based and handles them natively (S1400 already in the
  paper); just include the multi-mutation rows. (days)
- **Base/aug PLM embeddings on full SKEMPI** — one-time embedding-gen for ~7,085 mutant seqs + ~690 WT
  chains per PLM. Sub-6B (Ankh/ESM2/ESMC600m): hours each, cached. ESM C 6B: GPU-heavy (they did S1102
  6B on GPU; ~6–7× the muts) — a few GPU-days. **This is the main *base-arm* cost, and it's tractable.**

**B. Expensive / the real bottlenecks:**
- **FoldX at benchmark scale (for the FoldX/ProtBFF-style arm)** — RepairPDB ×345 complexes + BuildModel
  + AnalyseComplex ×7,085 muts (12 terms each). Embarrassingly parallel but **days–weeks of CPU**, plus
  the SKEMPI2 PDB set. This is table-stakes for the frontier (ProtBFF/USP-ddG both do it) and the single
  biggest new compute.
- **Antibody/multi-chain chain-mapping** — S1102 explicitly *filtered out* the multi-chain role groups
  (1AHW/1DVF-style `H_L_antigen`); the full set is full of them. The role→interacting-chain-pair mapping
  for `AnalyseComplex` (and the interface mask, if A1) needs real work on antibody complexes. (weeks,
  fiddly)
- **Antibody-antigen / DMS OOD sets** — AB-Bind, AbBiBench CR6261, trastuzumab-HER2, SARS-CoV-2 RBD DMS:
  sequences are easy (base arm = embeddings + labels), but the **FoldX arm needs their structures +
  FoldX ΔΔG** → more of bottleneck B. (days–weeks per set)

**Pragmatic staging (leverages MuLAN's sequence-only edge):**
1. **Sequence-only base arms on full SKEMPI v2, single+multi, CD-HIT/CATH split, per-structure+AUROC.**
   Embeddings + reuse splits/metrics — **~1–2 weeks, no FoldX.** This is the *leakage-controlled, publishable first
   deliverable* and the *fair* comparison for a sequence-only method (vs ESM-1v / MSA-Transformer /
   Tranception / MINT, not the structure SOTA).
2. **Add the FoldX/biophysical channel** (= ProtBFF-style) on the full set → competes with the
   structure-integrated SOTA. **+ weeks (FoldX-at-scale + antibody chain-mapping).**
3. **Add one antibody OOD set** (AbBiBench or RBD-DMS) for the design-relevant generalization claim.

**Bottom line:** the *sequence-only, full-SKEMPI, leakage-controlled-split* comparison (step 1) is achievable in
~1–2 weeks and is the right first move — it's cheap *because* MuLAN needs no per-mutant structure, and
it's the defensible frame for a sequence method. Everything that touches **FoldX or antibody structures**
(steps 2–3) is a multi-week, compute- and plumbing-heavy effort — and it's exactly where the frontier's
structure-based methods have a head start, so weigh whether MuLAN competes there or stays in its
sequence-only lane and reports the step-1 comparison plus the diagnostic.

---

## 5. Bottom line for the project

- **As a method paper**, much of A1/A2/B is redundant with the interface-aware ddG frontier and would
  be judged against LOCO/per-structure numbers this work has not yet computed.
- **As a diagnostic study** — "what limits a frozen-PLM + light-attention head on PPI ΔΔG, and what is
  genuinely orthogonal to it" — there is a real, defensible wedge, *provided* the split-leakage caveat
  is addressed and the metrics move to per-structure / ranking.
- **The cheapest high-value move is re-analysis, not more arms:** clustered-CV + per-structure metrics
  first; they may change which of A1/C1/B is worth building.

---

## Sources
- Quantification of biases in ΔΔG predictions — Brief. Bioinform. 2024: https://academic.oup.com/bib/article/25/1/bbad491/7513597
- Distributional bias compromises LOO-CV: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11177965/
- Data volume/diversity for antibody–antigen ΔΔG — Nat. Comp. Sci. 2025: https://www.nature.com/articles/s43588-025-00823-8
- GNN pre-training for PPI ΔΔG (TopNetTree LOCO ≈ 0.17): https://arxiv.org/pdf/2008.12473
- PPIformer (ICLR 2024): https://github.com/anton-bushuiev/PPIformer · https://arxiv.org/abs/2310.18515
- RDE-PPI / Rotamer Density Estimator (ICLR 2023): https://github.com/luost26/RDE-PPI
- Prompt-DDG / microenvironment-aware hierarchical prompt (2024): https://arxiv.org/abs/2405.10348
- CATH-ddG (out-of-superfamily robustness, Bioinformatics 2025): https://doi.org/10.1093/bioinformatics/btaf228
- 3D-ΔΔG dual-channel structure model: https://onlinelibrary.wiley.com/doi/10.1002/prot.26837
- MINT (Nat. Commun. 2026): https://www.nature.com/articles/s41467-025-67971-3
