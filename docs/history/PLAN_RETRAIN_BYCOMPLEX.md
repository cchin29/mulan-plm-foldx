# Leakage-controlled retrain — by-complex and RDE-comparable splits

> **Archived — executed.** Both phases ran. The by-complex and homology-clustered results ship as
> `experiments/retrain_split/SUMMARY.md`, `SUMMARY_clustered.md` and their two CSVs, and the
> drivers named here — `run_bycomplex.sh`, `cluster_split.py`, `config_{bycomplex,clustered}.sh` —
> are in that directory and cite this document by section.

_Written 2026-07-16. Implementation plan. Turns the frontier-metric
re-score (`experiments/rescore_perstructure/`, a *metric* change on leaky predictions) into the leakage-controlled
**split** change: retrain on **by-complex** folds so train/test never share a complex. Companion to
`REVIEW_gaps_and_landscape.md` §4a (#1) and the reference splits in `../reference_ddg_splits/`._

## 0. Why / what changes

The current 10-fold CV splits **by mutation**, so the same complex sits in train and test — the
per-structure re-score showed the pooled ρ 0.85 is ~half that leakage (within-complex ρ only
0.31–0.59, still an upper bound). This plan produces the **generalization** number: retrain with
**whole complexes** held out. **Scope decision: focus base + FoldX-MLP; deprioritize
Tier-1 aug** (aug was the weak lever pooled *and* per-structure — run it only if time permits, last).

**Central hypothesis to test:** under a non-leaky split every arm drops, but **FoldX-MLP's relative
lift *widens*** — its ΔΔG is complex-independent physics that doesn't need to have seen the complex,
whereas the base PLM's edge came partly from memorizing it. If FoldX's per-structure advantage grows
(or the base collapses toward it), that's strong evidence the physics channel is the part that
generalizes. This is the pre-registered prediction.

---

## 1. Isolation — all new dirs, nothing existing is touched

| purpose | path | note |
|---|---|---|
| this plan + split builder + eval | `experiments/retrain_split/` | new dir |
| by-complex split files | `scratch/splits_bycomplex_seed42/fold_{0,1,2}/` | new; mirrors `splits_balanced_foldxdec` format |
| RDE-partition split files | `scratch/splits_rde_iclr24/fold_{0,1,2}/` | S1102 mapped onto RDE folds |
| retrain results | `scratch/results/retrain_bycomplex/<plm>_{base,foldx}/fold_{0,1,2}/` | **NOT** `embedding_sweep_balanced/` |
| config/driver | `experiments/retrain_split/config_bycomplex.sh` | 3-fold, isolated `ES_RESULTS_DIR` |

`scratch/` is gitignored, so no collision with the tracked balanced anchors. **Reuse read-only:** the
cached PLM embeddings (per-label `.pt`) and the cached FoldX 12-term vectors — *only the split
membership changes*, so **no embeddings or FoldX are regenerated** (this is what makes it cheap).

---

## 2. Splits (measured on S1102 — real numbers)

**All 110 S1102 complexes are single-PDB and map 1:1 to a complex id** (`0` multi-chain role groups,
per `PLAN_A1.md` §2). Build **3-fold, partitioned by whole complex** (3 folds ≈ RDE's protocol and
gives ~8 qualifying complexes per test fold for per-structure metrics — 10-fold by-complex would leave
too few).

**2a. Primary — size-balanced by-complex (our data, cleanest).** Greedy assign complexes (largest
first) to the lightest fold. Verified balance:

| fold | muts | complexes | complexes ≥10 muts |
|---|---|---|---|
| 0 | 367 | 37 | 8 |
| 1 | 367 | 37 | 8 |
| 2 | 366 | 36 | 8 |

Rotate: each fold is the held-out **test** once; the other two folds are **train**, with a
**by-complex val slice** carved from them (hold ~15% of *training complexes* for early stopping — never
split a complex across train/val). Seed fixed (42). This is directly comparable to the existing
balanced numbers (same 1100 rows, same PLMs, **only** the split changes) → the drop = the leakage.

> **⚠ By-complex is a *waypoint*, not the leakage-controlled split (Dec-2025 lit).** Leave-one-complex-out still
> leaks by homology: USP-ddG reports **88.7% of by-complex test mutations are "easy"** (TM-score ≥0.6
> to a training structure), and ProtBFF finds **253 CD-HIT clusters even at 99% identity**. So 2a below
> quantifies the leakage *relative to our current per-mutation CV*, but the frontier generalization number needs
> the family-level splits in §7 (now promoted to Phase 2, do soon after 2a). See
> the frontier survey that preceded `experiments/BENCHMARK_MATRIX.md` (a private note, not shipped).

**2b. Comparable — PPIformer `skempi2_iclr24_split` (family-stratified, NOT random-by-complex).**
Corrected from an earlier draft that called this "RDE-partition": the JSON is PPIformer/PPIRef's split
and is **homology/family-stratified** — verified: fold-0 holds **41 of the protease-inhibitor family
and 3 antibodies**, fold-1 holds **25 antibodies and 0 proteases**, fold-2 is misc, `test` neither.
Whole structural families are quarantined per fold. All 110 S1102 complexes map to it, distributed
fold-0 60 complexes / 719 muts, fold-1 22 / 122, fold-2 23 / 232, test 5 / 27. Using it gives a genuine
**out-of-family** test (e.g. test fold-0 = "train on antibodies+misc, predict protease-inhibitor ΔΔG")
*and* the PPIformer partition structure. **Caveats on the S1102 subset:** coverage is uneven (fold-0
dominates; folds 1/2 have few ≥10-mut complexes → noisy per-structure), so this is the stricter,
comparability-structured secondary — 2a stays the well-powered primary, and it belongs conceptually
with the Phase-2 clustered splits, not the "comparable/leak-controlled" tier.

**Builder** (`build_splits_bycomplex.py`, new): read the pooled 1100 rows **with their 12 FoldX terms**
from `splits_balanced_foldxdec/*`, group by complex, emit per-fold `S1102_filtered_{train,val,test}.tsv`
in the **identical 16-col format** the trainer already consumes. Unit-test: (i) no complex appears in
>1 of {train,val,test} within a fold; (ii) union of test folds = 1100 rows; (iii) FoldX columns
preserved.

---

## 3. Arms & PLMs (focused)

- **Arms:** `base` and `foldx` (FoldX-MLP `add_scores`, 12-term). **Also run the FoldX *scalar* arm**
  on ≥2 PLMs — the §20.1 finding is that scalar ≥ MLP on small sets, and by-complex test folds are
  small; the scalar may be the safer add-on here. Skip Tier-1 aug (deprioritized).
- **PLMs (all 6, retrain is cheap):** esm2-3B, esmc600m, prostt5, saprot, ankh-large, **esm C 6B**.
  If compute-limited, the priority order is **esmc6b, ankh, saprot** (ceiling / workhorse /
  structure-aware) then the rest.
- **MINT (optional, low priority):** re-run `mint_complex` under 2a to confirm it stays worst
  out-of-distribution — one arm, uses the existing row-keyed MINT cache. Nice-to-have, not gating.

Grid = 6 PLM × {base, foldx} × 3 folds = **36 runs** (+ scalar on 2–3 PLMs, + optional MINT). Each is
a small light-attention head over cached embeddings, 300 ep / patience 30 → minutes/fold; whole grid
is **~1–3 h on one MPS/GPU driver**, no embedding regen.

---

## 4. Training

Reuse the existing harness with an isolated config: a `config_bycomplex.sh` that sets
`ES_SPLIT_DIR=scratch/splits_bycomplex_seed42`, `ES_RESULTS_DIR=scratch/results/retrain_bycomplex`,
`ES_NUM_FOLDS=3`, budget 300 ep / patience 30 (matching the balanced series so lifts are comparable).
Drive with the same `run_sweep.sh` / FoldX-MLP driver pattern already used for the balanced arms —
point them at the new split + results dirs. Save per-fold `test_predictions.tsv` (the eval input).

**⚠ Coordination with the A1 freeze.** This retrain uses the **base** `mulan-train` (no
`interface_xattn`), so it runs on the current committed code — but it must not collide on the MPS queue
with the in-flight A1 grid (#60) / C1 (#54). Queue it **after** them (or run on the GPU box). Do not
edit `mulan/`. Confirm each fold actually produced `all_results.json` (the "silent no-op" gotcha from
`TODO.md`).

---

## 5. Evaluation (reuse the rescore harness — do not re-implement)

The `rescore_perstructure/` metric core was built split-agnostic (SPEC §7). Point it at the retrain
predictions:

- Per-structure Pearson/Spearman (T=10, T=5), overall Pearson/Spearman/RMSE(+corr), AUROC (destab,
  strong), precision@k — identical definitions, so numbers line up with the leaky baseline.
- **Cluster-bootstrap CIs** (`bootstrap_ci.py`) over the test complexes, and **paired** contrasts
  (foldx − base per PLM) — this is how we test the central hypothesis.
- **Headline comparison table:** for each arm, `per-structure ρ` across **leaky (existing) → by-complex
  (2a) → homology-clustered (2c/2d)**, and the FoldX−base Δ under each. The *change in the Δ as the split
  gets stricter* is the result — the central hypothesis predicts FoldX's Δ **widens** left-to-right.

Also emit the RDE-native metrics via the reference (`../reference_ddg_splits/RDE-PPI/rde/utils/skempi.py`:
`per_complex_corr`, `overall_auroc`, single/multiple-mutation split) so the numbers are quotable
against the leaderboard lineage.

---

## 6. Decision gates & what each outcome means

- **Gate A (primary):** does FoldX−base per-structure Δ **hold or widen** under 2a vs the leaky
  baseline? Widen → physics generalizes (pre-registered win); collapse → FoldX was also complex-memo.
- **Gate B:** how far does base ρ fall (leaky 0.31–0.59 → ?). This is the leakage-controlled generalization number;
  expect a substantial drop (literature: LOCO can crater ddG predictors).
- **Gate C:** does the arm ordering (FoldX > base ≫ MINT) survive out-of-distribution?

Feed results back into `S1102_CROSSFOLD_MISPREDICT.md` (a "by-complex" column next to the pooled
numbers) and `REVIEW_gaps_and_landscape.md` §4a.

---

## 7. Phase 2 — homology-clustered split (PROMOTED: this is the leakage-controlled number, not optional)

The Dec-2025 literature is unanimous that **by-complex is not enough** (see §2 caveat): the leakage-controlled,
frontier-comparable generalization test is a **family-level** split. Do this right after 2a — it's the
number that actually matters. Two established recipes, in priority order:

- **2c. CD-HIT sequence-identity clustering (ProtBFF recipe — recommended).** Cluster the 110 S1102
  complexes' chains with **CD-HIT at ≤60% identity** (the ProtBFF protocol, doi:10.64898/2025.12.23.696257),
  GroupKFold on clusters. Directly comparable to ProtBFF's numbers (leakage-controlled frontier ~0.51 Pearson /
  ~0.48 Spearman there). CD-HIT is a small, standard install — cheaper than hand-rolling Foldseek.
- **2d. CATH-superfamily hold-out (CATH-ddG / USP-ddG recipe).** Map complexes to CATH superfamilies;
  hold out complexes sharing no superfamily with train (CATH-ddG's 813-mutation set is the reference,
  doi:10.1093/bioinformatics/btaf228). The strictest, most-cited generalization split.
- ⚠ **Coverage warning:** S1102's protease-inhibitor family (1PPF/1R0R/1CSE/3SGB/1SBN/2SIC/1TM*/1CT*…)
  collapses to **one** cluster, and antibodies to another — so at family level S1102 has few effective
  clusters. **3 clean folds may not survive**; if not, do a single **leave-one-family-out** hold-out
  (train on all but the protease family, test on it) rather than forcing K folds. Assess cluster count
  first. This is also the point where the S1102 subset starts to hurt → motivates Phase 3.

## 7b. Phase 3 — full curated SKEMPI v2 (leaderboard placement)
S1102 is a ~110-complex subset of the ~335–348-complex curated set. True leaderboard numbers (ProtBFF /
USP-ddG / CATH-ddG tables) need embeddings **+ FoldX ΔΔG for all ~7,000 muts** (big FoldX compute) — a
separate, larger effort. Until then, 2b/2c give "leakage-controlled split, S1102 subset." Note the frontier already
runs FoldX over the full set (ProtBFF/USP-ddG), so this is table-stakes for a method claim, not novel.

## 8. Sequence

1. ⬜ `build_splits_bycomplex.py` + unit test (§2) — pure data, writes only to `scratch/` (safe now).
2. ⬜ `config_bycomplex.sh` + wire the existing driver to the new split/results dirs.
3. ⬜ Retrain base + foldx, 6 PLMs × 3 folds (queue after A1/C1). Verify `all_results.json` per fold.
4. ⬜ Evaluate via the rescore harness + bootstrap; build the leaky→by-complex comparison table.
5. ⬜ **Phase 2 (not optional): CD-HIT ≤60% clustered split (2c)** — the leakage-controlled number; then (opt) the
   CATH-superfamily hold-out (2d). (opt) 2b family-stratified run; (opt) MINT arm; (opt) FoldX scalar arm.
6. ⬜ Write results into the two analysis docs; decide Phase 3 (full SKEMPI).

**Artifacts (planned, in `experiments/retrain_split/`):** `build_splits_bycomplex.py`,
`test_build_splits.py`, `config_bycomplex.sh`, `evaluate.py` (thin wrapper over the rescore metric
core), this plan.
