# Residual-target training on the FoldX 12-term reference — ESM-C 6B, GPU box

Queue item 1 of `experiments/beyond_foldx/README.md`. Trains ESM-C 6B's base pathway against
`ΔΔG_exp − ŷ_FoldX` instead of `ΔΔG_exp`, on the two single-point tiers. **6 runs** (2 tiers ×
3 folds), no PLM inference, no model code changes.

Set `MULAN_ROOT` / `MULAN_VENV_BIN` for the Linux layout exactly as `RUN_BYCOMPLEX_ALL_AUG_GPU.md`
describes. Every path below is repo-relative.

## Motivation

`REDUNDANCY_RESULT.md` and `BLEND12_RESULT.md` establish that the base pathway's *output* holds
nothing usable beyond the physics channel once homology is controlled: an optimally weighted
global blend of the base prediction with the FoldX scalar gains +0.009 on clustered-SP, and with
the full 12-term decomposition +0.021 [−0.001, +0.043]. Neither is resolvable.

That bounds rules combining the *existing base prediction* with physics. It does not bound a model
trained on residual targets, which sees embeddings rather than the base scalar and can in principle
surface signal the current head discards. This tier tests exactly that, and it is the only cheap
way to.

The by-complex tier is included as a **positive control**, not for its own sake. `gain_plm` there
is +0.041 [+0.015, +0.065] and significant, so a residual arm that shows nothing on by-complex
either indicates a broken pipeline rather than a confirmed hypothesis. A clustered-only run cannot
distinguish those two outcomes.

## 1. The reference fit, and why a naive residual is wrong

**The residual target cannot be formed from the split TSVs.** Column 5 of `splits_*_foldx` is a
per-fold, train-only z-score clipped at ±4, with uncovered rows set to exactly 0
(`merge_foldx_full_skempi.py:17-18`, `CLIP = 4.0`). It is not kcal/mol.

Going back to the raw per-complex JSONs does not fix it either, because subtracting FoldX at
slope 1 removes far too much:

| quantity | value |
|---|---|
| raw FoldX Interaction Energy, SD | 1.488 kcal |
| experimental ΔΔG, SD (train, pooled 3 folds) | 1.800 kcal |
| pooled Pearson(label, FoldX channel) | 0.441 |
| **least-squares slope of label on the FoldX channel** | **0.874 kcal/SD** |

A slope-1 residual removes ≈1.49 kcal per channel SD where the regression wants 0.874 —
**over-subtracting by ≈1.7×**, which leaves an anti-correlated physics component in the target.
The resulting arm would score below plain base and would read as a clean refutation of the
hypothesis rather than as a units defect. This is the single most likely way for this experiment
to produce a confident wrong answer.

**The reference is therefore a per-fold least-squares fit on train rows only:**

    ŷ = a + Σ(j=0..11) b_j · z_j

over the twelve already-standardised terms from the 16-column `foldxdec` split. Fit per fold, on
that fold's **train rows only** — fitting on pooled or on all rows leaks the test labels into the
target.

Use the twelve terms, **not** the scalar. `BLEND12_RESULT.md` puts a linear reweighting of the
terms at 0.434 on clustered-SP against the total's 0.418, so a scalar reference would credit the
model for a +0.016 lift that reweighting already supplies.

## 2. Build the residual splits

New script, `experiments/beyond_foldx/build_residual_splits.py`. Roughly 60 lines; it derives from
the already-merged splits rather than redoing the FoldX join, so it inherits the verified 99.2%
coverage instead of re-earning it.

**Inputs** (per tier, per fold):

| tier | 4-col source split | 16-col decomposed split |
|---|---|---|
| clustered-SP | `scratch/splits_skempi_full_clustered_id60_kfold` | `scratch/foldx_skempi_full/splits_skempi_full_foldxdec` |
| by-complex-SP | `scratch/splits_skempi_full_bycomplex_seed42` | `scratch/foldx_skempi_full/bycomplex/splits_skempi_full_foldxdec` |

Basename is `skempi_sp`; files are `${basename}_{train,val,test}.tsv`; 3 folds each.

**Algorithm**, per tier per fold:

1. Read the 16-column decomposed train/val/test TSVs. Columns are
   `chain1, chain2, mutation, ddG_experimental, z_0 … z_11`.
2. Fit OLS with intercept of column 4 on columns 5–16, **train rows only**. Keep `(a, b_0…b_11)`
   as float64.
3. For every row of train, val and test, compute `ŷ` and write a 4-column TSV
   `chain1, chain2, mutation, (ddG_experimental − ŷ)` — columns 1–3 copied verbatim.
4. Write `reference.json` beside the fold's TSVs: `{"a": …, "b": [12 floats], "n_train": …,
   "train_r2": …, "label_sd": …, "resid_sd": …}`.

**Outputs:**

    scratch/splits_skempi_full_clustered_resid/fold_{0,1,2}/skempi_sp_{train,val,test}.tsv
    scratch/splits_skempi_full_clustered_resid/fold_{0,1,2}/reference.json
    scratch/splits_skempi_full_bycomplex_resid/fold_{0,1,2}/...

Uncovered rows (0.8%) carry all twelve terms as exactly 0, so `ŷ` degenerates to the intercept and
the residual is `y − a`. That is the correct neutral behaviour and needs no special case — 0 is
the standardised mean by construction.

## 3. Validation gates — all must pass before any training starts

1. **Row parity.** The residual split must carry the same rows in the same order as the 4-col
   source split, differing only in column 4. Check columns 1–3 byte-equal, per fold, per file.
   A mismatch means the decomposed and base splits are not the same partition.
2. **Exact reconstruction.** `ŷ + residual == ddG_experimental` to 1e-6 for every row of every
   fold. This catches a mis-ordered coefficient vector, which otherwise produces a plausible file.
3. **Residual variance — exact reproduction.** The reference fit is deterministic given the
   splits, so these are values to match, not a range to fall inside. Fitted on the Mac,
   2026-08-04:

   | tier | fold | n_train | train_r² | label_sd | resid_sd | ratio | intercept |
   |---|---|---|---|---|---|---|---|
   | clustered-SP | 0 | 2114 | 0.2631 | 1.323 | 1.136 | 0.858 | +0.813 |
   | | 1 | 2463 | 0.3022 | 1.944 | 1.624 | 0.835 | +1.038 |
   | | 2 | 2447 | 0.3000 | 1.987 | 1.663 | 0.837 | +1.100 |
   | by-complex-SP | 0 | 2208 | 0.2676 | 1.908 | 1.633 | 0.856 | +1.196 |
   | | 1 | 2297 | 0.2840 | 1.768 | 1.496 | 0.846 | +1.008 |
   | | 2 | 2140 | 0.2152 | 1.751 | 1.551 | 0.886 | +1.081 |

   A differing `n_train` means a different split; a differing `train_r²` at matching `n_train`
   means a different fit. **If `resid_sd ≥ label_sd` anywhere, stop** — that is the signature of
   the slope error in §1. Note `train_r²` of 0.215–0.302 puts the 12-term multiple correlation at
   0.46–0.55, above the scalar's pooled 0.441, which is the in-sample form of why §1 specifies the
   terms rather than the total.
4. **Reference strength.** The 12-term reference's own per-structure Spearman on the pooled test
   folds should land near 0.42–0.44 on clustered-SP (the FoldX total scores 0.418; a
   within-complex-normalised reweighting scored 0.434). A reference below 0.40 means the fit is
   not capturing the physics and the experiment is not testing what it claims to.
   **Record this number — §6 explains why it, and not FoldX-alone, is the comparator.**
5. **Driver log.** Every run must log `splits=scratch/splits_skempi_full_*_resid`. Anything else
   means the wrong split was resolved regardless of what the config says.

## 4. Configs

Two new files. Each sources its unaugmented parent and overrides two variables.

`experiments/full_skempi_seqonly/config_skempi_full_resid.sh`:

```sh
# Residual-target arm on the clustered-SP tier. Labels are ddG_exp - yhat, where yhat is a
# per-fold train-only OLS fit of the 12 standardised FoldX terms (see
# experiments/beyond_foldx/RUN_RESIDUAL_ESMC6B_GPU.md). Arm `base` here == residual_base:
# the head is unchanged (lightatt_default, no add_zs_scores) and only the target differs.
source experiments/full_skempi_seqonly/config_skempi_full.sh

# WITHOUT this line `ARMS=base` inherits ES_SPLIT_DIR from the parent and trains the PLAIN base
# into the residual results dir, exiting 0 - a wrong answer that reports success, not a failure.
export ES_SPLIT_DIR=scratch/splits_skempi_full_clustered_resid
export ES_RESULTS_DIR=scratch/results/full_skempi_resid
```

`experiments/full_skempi_seqonly/config_skempi_full_bycomplex_resid.sh` is the same, sourcing
`config_skempi_full_bycomplex.sh` with
`ES_SPLIT_DIR=scratch/splits_skempi_full_bycomplex_resid` and
`ES_RESULTS_DIR=scratch/results/full_skempi_bycomplex_resid`.

Nothing under `mulan/` changes. The residual arm is the driver's existing `base` arm on
`lightatt_default_config.json` with no `--add_zs_scores`. Feeding FoldX into the input *and* the
target would make the result unattributable, which is why only the base arm is run.

## 5. Training

Pull the current `run_bycomplex.sh` before starting — it gained the `all_results.json` failure
check that stops a crashed fold from logging DONE with an empty `test_pcc`.

    ES_CONFIG=experiments/full_skempi_seqonly/config_skempi_full_resid.sh \
        ARMS=base MAXPAR=1 bash experiments/retrain_split/run_bycomplex.sh esmc6b

    ES_CONFIG=experiments/full_skempi_seqonly/config_skempi_full_bycomplex_resid.sh \
        ARMS=base MAXPAR=1 bash experiments/retrain_split/run_bycomplex.sh esmc6b

Embeddings are already materialised for both tiers (`scratch/embeddings_skempi_full/esmc6b`, from
the existing plain base arms), so there is no PLM inference and no embedding GPU time. Budget
≈75–85 min/fold, ≈8–9 h for all six at `MAXPAR=1`. Early stopping runs on val loss against the
residual target, which is correct and needs no change.

## 6. Scoring — the add-back, and the comparator

**The scorer must add `ŷ` back before computing ppS.** Scoring the raw model output against
`y − ŷ` measures residual ranking skill: a different quantity, comparable to nothing in
`results_matrix_ps.csv`, and one that will produce an entirely plausible number. This is the
second way this experiment can return a confident wrong answer.

Add two tiers to `experiments/full_skempi_seqonly/score_sp_all.py`. Their `truth` is the
**original** label — i.e. the ordinary FoldX split, unchanged — and the scorer loads each fold's
`reference.json`, recomputes `ŷ` from the 16-column decomposed split, and scores `ŷ + prediction`.

Report **both** numbers:

- `ppS(ŷ + f)` — goes in the matrix, comparable to every existing arm.
- `ppS(f vs y − ŷ)` — residual ranking skill on its own. Not comparable to other arms, but it is
  the direct readout of whether the embeddings carry anything the reference misses, and it is
  worth having when the headline number comes back flat.

**The comparator is the reference's own ppS from gate 4, not FoldX-alone and not the plain base
arm.** Since the prediction is `ŷ + f(x)`, an `f` contributing nothing returns exactly `ppS(ŷ)`.
The test is therefore paired, per complex, `(ŷ + f)` against `ŷ` — same rows, same folds, same
reference. Report it as a cluster bootstrap over complexes, matching
`experiments/rescore_perstructure/foldx_alone_baseline.py`.

## 7. Reading the result

The prior from `BLEND12_RESULT.md` is +0.021 [−0.001, +0.043] on clustered-SP. Nothing suggests a
larger effect, and three considerations shape how the outcome should be read:

- **The paired arm-vs-reference contrast is better powered** than the ±0.03–0.04 half-widths
  quoted for contrasts against FoldX-alone, because both sides share the backbone, the folds and
  the complexes. How much better is not knowable until it runs — measure it, do not assume it.
- **More seeds will not help.** The CI is dominated by between-complex variance, not seed
  variance, so extra seeds shrink the noise on the mean and not the bootstrap. Only more complexes
  move that.
- **The reparameterisation is mild.** At r = 0.441 the residual target's SD is 1.616 against the
  label's 1.800 — a 10% reduction. The "make pass-through free" framing oversells how much the
  optimisation problem actually changes.

A null on clustered-SP with a positive by-complex control is the most likely outcome and is a
real result: it would say the embeddings, not merely the current head's output, hold no
complement to the physics under homology control. A null on **both** tiers is a pipeline failure
until proven otherwise — work back through the gates in §3.

## 8. Trap checklist

- [ ] Reference fitted on **train rows only**, per fold — not pooled, not on all rows.
- [ ] Twelve terms, not the scalar.
- [ ] `resid_sd < label_sd`, ratio near 0.90 (gate 3) — the slope-error signature.
- [ ] `ŷ + residual == y` to 1e-6 (gate 2).
- [ ] `ES_SPLIT_DIR` set **explicitly** in both configs.
- [ ] Driver logs `splits=…_resid` for all six runs.
- [ ] Scorer adds `ŷ` back before ppS.
- [ ] Contrast is paired against the reference's own ppS, not FoldX-alone.
- [ ] By-complex tier run and reported, not dropped as redundant — it is the control.
