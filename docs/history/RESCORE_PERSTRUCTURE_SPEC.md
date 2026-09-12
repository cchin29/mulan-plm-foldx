# Per-structure and AUROC re-scoring of existing OOF predictions — specification

> **Archived — implemented.** The specification is realised by
> `experiments/rescore_perstructure/rescore.py`, which cites this document, and its outputs ship as
> `results.csv` (24 arms) and `SUMMARY.md` in the same directory. Retained as the design record:
> the metric definitions here are the contract the scorer is tested against, so this is the
> specification the shipped code implements rather than a superseded proposal.

_Written 2026-07-15. **Implementation spec.** No training, no
`mulan/` edits — pure re-analysis of predictions already on disk (safe during the A1 code freeze).
Companion to `experiments/REVIEW_gaps_and_landscape.md` §4a._

## 1. Objective

Re-score the **existing** balanced-splitter OOF predictions under the **frontier-standard metrics**
(RDE/Prompt-DDG style): **per-structure Pearson & Spearman**, overall Pearson/Spearman/RMSE/MAE, and
**AUROC** (sign-of-effect + strong-destabilizer), plus a top-k retrieval metric. Goal: see whether the
**arm ranking and the hotspot story survive the field's metrics** before committing compute to an
RDE-split retrain.

> **⚠ Scope caveat — read first.** This changes the **metric, not the split.** The predictions are
> from the **leaky per-mutation 10-fold CV** (same complex in train and test). So the per-structure
> numbers here are an **upper bound**, not a generalization figure. The leakage-controlled by-complex (RDE 3-fold)
> and CATH-clustered numbers still require retraining (separate task). State this in every output.

## 2. Inputs

**Truth + complex key** — `scratch/foldx_s1102/splits_balanced_foldxdec/fold_{f}/S1102_filtered_test.tsv`,
`f=0..9`, tab-separated, **no header**, columns:
`chain1_id  chain2_id  mutation  true_ddg  <12 FoldX terms>` (16 cols total).
- **complex/PDB key** = `chain1_id.split("_")[0]` (e.g. `1PPF_A` → `1PPF`).
- **join key** = `(pdb, mutation)`. Pooling the 10 disjoint test folds = the full **1100** OOF rows.
- Sign convention: **positive `true_ddg` = destabilizing** binding (MuLAN convention).

**Predictions** — each `.../fold_{f}/training_run/test_predictions.tsv`, tab-separated, **no header**,
columns: `chain1_id  chain2_id  mutation  pred`. Join to truth on `(pdb, mutation)` within the fold.

**Model / arm inventory** (path templates, `{f}`=fold). `RES = scratch/results/embedding_sweep_balanced`,
`INC = scratch/incoming_esmc6b_20260714`:

| arm-id | path template |
|---|---|
| `esm2_base` | `{RES}/esm2/fold_{f}/training_run/test_predictions.tsv` |
| `esmc600m_base` | `{RES}/esmc600m/fold_{f}/...` |
| `prostt5_base` | `{RES}/prostt5/fold_{f}/...` |
| `saprot_base` | `{RES}/saprot/fold_{f}/...` |
| `ankh_base` | `scratch/results/cv10_ankh_converge/fold_{f}/training_run/test_predictions.tsv` **(NOT `{RES}/ankh`, which is empty)** |
| `esmc6b_base` | `{INC}/results/base/S1102/fold_{f}/...` |
| `<plm>_aug` | `{RES}/<plm>_aug/fold_{f}/...` for plm ∈ {esm2, esmc600m, prostt5, saprot, ankh} |
| `esmc6b_aug` | `{INC}/results/aug/S1102/fold_{f}/...` |
| `<plm>_foldxmlp` | `{RES}/<plm>_foldxmlp/fold_{f}/...` for plm ∈ {esm2, esmc600m, prostt5, saprot, ankh} |
| `esmc6b_foldxmlp` | `{INC}/data/S1102/foldx/cv10_esmc6b_balanced/foldx_mlp/fold_{f}/...` |
| `mint_complex` | `scratch/results/mint_a2/mint/fold_{f}/...` |
| `mint_mono` | `scratch/results/mint_mono/mint_mono/fold_{f}/...` |

**Also compute 3 consensus arms** = per-row **mean prediction over exactly the 6 PLMs**
`{esm2, esmc600m, prostt5, saprot, ankh, esmc6b}` for that family: `consensus_base`, `consensus_aug`,
`consensus_foldxmlp`. (These reproduce the `S1102_CROSSFOLD_MISPREDICT.md` "6-PLM consensus" arm.)
- **Membership is fixed at all 6** — a consensus row is emitted **only where all 6 constituent arms
  have a prediction** for that `(pdb, mutation)`. Log and flag (in `results.csv` via `n`) any consensus
  arm whose coverage falls below 1100 so a partial mean can't silently masquerade as the full pool. Do
  **not** average over "whatever's available" — that would make the §6 `consensus_base ≈ 0.870`
  cross-check meaningless.

All arms are on the **same** balanced folds, so complex membership is identical — verify (§6).

## 3. Metrics (per arm)

Compute on the pooled OOF rows for that arm (report n; expect 1100).

**Overall (n=1100):**
- `pearson`, `spearman` (pred vs true), `rmse`, `mae` (**raw** error, `pred−true`).
- `rmse_corr`, `mae_corr` — RMSE/MAE of the **OLS-rescaled** prediction (fit `true ~ a·pred + b`,
  score `a·pred+b`). This is the **RDE-PPI / leaderboard convention** (`overall_rmse_mae` removes
  scale+offset so magnitude shrinkage isn't penalized). Report both: raw is the true error, corrected
  is the frontier-comparable one. Implement the OLS fit dependency-free (closed-form slope/intercept).

**Per-structure** (group rows by `pdb`):
- For each complex with `n_mut ≥ T`, compute within-complex Pearson and Spearman (pred vs true).
- Report the **unweighted mean** across qualifying complexes, and the **count** of qualifying complexes.
- Do this for **T=10** (primary, RDE-style) **and T=5** (secondary, wider coverage).
- Skip complexes where true or pred has zero variance (degenerate correlation) and log how many.

**Classification / retrieval (design-relevant, robust to shrinkage):**
- **Orientation (assert this):** larger `pred` ⇒ more destabilizing, and larger `true` ⇒ more
  destabilizing (MuLAN sign convention). So for every classifier below the positive class is the
  *destabilizing* label and `scores = pred` (no sign flip). Rows with `true == 0` fall in the
  **negative** class (strict `>` / `≥` cutoffs).
- `auroc_destab` — labels `y = (true > 0)`, scores = `pred`. (Can the model tell stabilizing from
  destabilizing?)
- `auroc_strong` — labels `y = (true ≥ 2.0)`, scores = `pred`. (Can it flag strong destabilizers?)
- `precision@50` and `recall@50` — rank rows by `pred` desc, take top-50; fraction that are truly
  `true ≥ 2.0` (precision), and of all `true ≥ 2.0` how many are in the top-50 (recall). Report the
  count of `true ≥ 2.0` in the pool for context.
- **Degenerate guards:** if a metric's positive class is empty (`n_pos == 0`, possible for
  `auroc_strong` on a thin per-fold slice) or the negative class is empty, report the metric as
  **NaN** with a logged warning — never crash. If the pool has `< 50` rows, `precision@50` is over the
  rows available; `recall@50` is capped by `min(50, n_pos)`. Break rank ties at the k=50 boundary
  **deterministically** (stable sort on `pred` desc, then original row index) so re-runs match.

**Notes / definitions:**
- **Spearman** = Pearson on **average-rank-transformed** values (handle ties with mean ranks).
- **AUROC** via the rank/Mann–Whitney identity: `AUROC = (R_pos − n_pos·(n_pos+1)/2) / (n_pos·n_neg)`,
  where `R_pos` = sum of average ranks of the positive-class scores. No sklearn needed.
- Expect **per-structure Spearman < overall Spearman** (per-structure removes the 3 mega-scanned
  complexes' leverage — 177/177/176 muts each, ~48% of rows) — that directional drop is itself a
  finding to report.

## 4. Outputs (write to `experiments/rescore_perstructure/`)

1. **`results.csv`** — one row per arm-id, columns: `arm, n, pearson, spearman, rmse, mae, rmse_corr,
   mae_corr, ps_pearson_T10, ps_spearman_T10, ps_ncomplex_T10, ps_pearson_T5, ps_spearman_T5,
   ps_ncomplex_T5, auroc_destab, auroc_strong, prec_at50, recall_at50`.
2. **`SUMMARY.md`** — a table sorted by **`ps_spearman_T10` desc**, plus a short written read:
   - Does the arm ranking match `S1102_CROSSFOLD_MISPREDICT.md`'s pooled-PCC ranking (FoldX > aug;
     MINT-complex bulk lift; 6B on top)? Note any re-ordering.
   - How much does overall→per-structure move each arm (the leakage/leverage proxy).
   - What AUROC_strong / precision@50 say about the *design* use-case (flagging strong destabilizers),
     independent of magnitude shrinkage.
   - Re-state the scope caveat (metric change, not split change).
3. **`rescore.py`** — the script, re-runnable, deterministic.

## 5. Implementation constraints

- **Env:** a dedicated venv lives at `experiments/rescore_perstructure/.venv` (numpy + scipy). Run with
  `experiments/rescore_perstructure/.venv/bin/python`. **Mandate:** implement Spearman + AUROC **by hand**
  (formulas above) so the metric core is dependency-free and reusable on future RDE-split predictions
  (§7). scipy is present in the venv **only** to cross-validate the hand-rolled metrics in a `--selftest`
  path (assert agreement to ~1e-9 on the pooled data); it must not be a runtime dependency of the
  metric functions themselves.
- Read TSVs as text (no torch). Pure numpy.
- **Robustness:** skip a missing arm with a logged warning (don't crash the whole run); report per-arm
  coverage `n` and flag any arm with `n ≠ 1100`.
- Deterministic; no randomness. Fast (<1 min).

## 6. Acceptance / sanity checks (must pass)

- Pooled truth loads to **1100** rows over **110** complexes; the 3 largest complexes have **177/177/176**
  mutations (matches the known S1102 skew).
- **Join key `(pdb, mutation)` is unique within every fold's truth file** — assert this (the key drops
  `chain2_id`; a duplicate would make the truth↔pred join many-to-one and silently corrupt that arm).
  Fail loudly if violated.
- Each base arm aligns to **1100/1100** rows.
- **Cross-check against the existing analysis:** `esmc6b_base` overall `pearson ≈ 0.886` and
  `consensus_base ≈ 0.870` (pooled OOF PCC in `S1102_CROSSFOLD_MISPREDICT.md`) — if these don't
  reproduce, the join is wrong.
- **Cross-check against the RDE-PPI reference metrics** (`../reference_ddg_splits/RDE-PPI/rde/utils/
  skempi.py`, see that dir's README §"metric parity"): `crosscheck_reference.py` runs the reference's
  own `per_complex_corr` / `overall_correlations` / `overall_auroc` / `overall_rmse_mae` (extracted
  verbatim) on the same pooled rows and asserts our hand-rolled `pearson`, `spearman`, `auroc_destab`,
  `ps_*_T10`, `rmse_corr`, `mae_corr` all match to <1e-9. This validates the metric math against the
  canonical SKEMPI benchmark independent of the split.
- Per-structure Spearman < overall Spearman for essentially every arm (expected leverage drop) — this is
  a **flag, not a hard gate**: log a warning if an arm violates it rather than aborting the run (an odd
  arm shouldn't sink the whole re-score).
- `n_pos`/`n_neg` for `auroc_destab` are both > 0 (S1102 has both signs). `auroc_strong` may legitimately
  have `n_pos == 0` on a per-fold slice — that yields NaN (§3 guard), not a failure.

## 7. Out of scope (explicit — do NOT do here)

- No retraining, no new splits, no `mulan/` edits (A1 grid may be running).
- This does **not** produce the RDE 3-fold by-complex or CATH-clustered generalization numbers — those
  are a separate retraining task (`REVIEW_gaps_and_landscape.md` §4a directions #1/#2). This harness is
  designed so the **same metric code can be reused** on those predictions later: keep the metric
  functions split from the path/loader so a future `--pred-root` / arm-map swap feeds RDE-split
  predictions through unchanged.

## 8. Stretch (only if quick)

- Add `spearman`/`auroc_strong` **per-fold** mean±std (not just pooled) for a variance read.
- Restrict per-structure to the **interface/charged hotspot** complexes (2O3B, 1MAH, 1BRS, 1PPF, 1JTG,
  2PCC…) to see whether any arm ranks the *hardest* complexes' mutations better — the design-relevant
  slice.
