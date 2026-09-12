# run4b — ProstT5 + Tier-1 augmentation (reverse + identity) — findings (2026-06-20)

Goal: does Tier-1 data augmentation (reverse-mutation antisymmetry + identity
anchors) rescue the **weaker** ProstT5 baseline (run2, PCC 0.740)? This is the
ProstT5 counterpart to run4a (Ankh). See `../../../experiments/AUGMENTATION.md`
for the augmentation rationale.

## Setup (identical to run2 except the training set)

- **Split:** the **exact same** mutation-based 70/15/15 split as run1/run2
  (seed 42 → 769 train / 157 val / 174 test). Validation and **test are left
  untouched** — evaluation stays on real forward mutations.
- **Augmentation** (`experiments/augment.py`, train split only): +769 reverse
  examples (`ΔΔG(wt→mut) = −ΔΔG(mut→wt)`, reusing cached mutant embeddings) and
  +91 identity anchors (`wt→wt`, label 0, one per complex). Train rows
  **769 → 1629**; WT FASTA entries 220 → 989.
- **PLM:** `Rostlab/ProstT5` in amino-acid mode (`<AA2fold>` prefix), per-residue,
  1024-dim, last hidden layer. Missing augmented-sequence embeddings (915) were
  generated on the fly.
- **Model:** fresh `LightAttModel` (`lightatt_default_config.json`); `LazyConv1d`
  adapts to 1024-dim. AdamW (wd 0.01), batch 32, LR 5e-4, `ReduceLROnPlateau`
  (factor 0.5, patience 5), `early_stopping_patience=10`, **50 epochs**.
- **Hardware:** Apple Silicon MPS, train ~7.5 min (run2 was ~33 min CPU).

> Note: the Linux `scratch/batch_driver2.sh` ran run4b at **25** epochs (slow
> host); this run uses **50** to match run2's protocol for a clean comparison.
> Early stopping never triggered — best eval-loss checkpoint was epoch 50, so
> val loss was still improving; a longer run might gain a little more.

## Result on held-out test set (N=174)

| Metric | Value |
|---|---|
| PCC  | **0.759** |
| RMSE | **1.547** kcal/mol |
| MAE  | 1.130 kcal/mol |
| SCC  | 0.641 |

## Comparison

| Run | PLM | Augmentation | PCC | RMSE |
|---|---|---|---|---|
| run1 | Ankh    | none              | 0.757 | 1.531 |
| run2 | ProstT5 | none              | 0.740 | 1.584 |
| **run4b** | **ProstT5** | **reverse + identity** | **0.759** | **1.547** |
| paper (10-fold CV) | Ankh-large | — | 0.868 | 1.185 |

**Verdict:** Tier-1 augmentation improves the ProstT5 baseline on every metric
(PCC +0.019, RMSE −0.037, MAE −0.030, SCC +0.004), lifting ProstT5 to roughly
the Ankh baseline level. Modest but consistent — confirms the antisymmetry +
zero-point constraints help the small-data regime.

Artifacts (gitignored): `training_run/` (`model.ckpt`, `all_results.json`,
`test_predictions.tsv`), `training_run.log`. Augmented inputs in
`../../aug_common/`.
