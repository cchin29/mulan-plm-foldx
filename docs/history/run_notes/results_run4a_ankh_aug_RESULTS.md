# run4a — Ankh + Tier-1 augmentation (reverse + identity) — findings (2026-06-20)

Goal: does Tier-1 data augmentation (reverse-mutation antisymmetry + identity
anchors) help the **strong** Ankh baseline (run1, PCC 0.757)? Companion to run4b
(ProstT5). See `../../../experiments/AUGMENTATION.md` for the rationale.

## Setup (identical to run1 except the training set)

- **Split:** same mutation-based 70/15/15 split as run1/run2 (seed 42 → 769/157/174).
  Validation and **test untouched** — evaluation on real forward mutations.
- **Augmentation** (`experiments/augment.py`, train split only, shared with run4b):
  +769 reverse + 91 identity → train rows **769 → 1629**.
- **PLM:** `ElnaggarLab/ankh-large`, per-residue, **1536-dim**, last hidden layer.
  All needed embeddings generated fresh (none cached locally).
- **Model:** fresh `LightAttModel` (`lightatt_default_config.json`); `LazyConv1d`
  adapts to 1536-dim. AdamW (wd 0.01), batch 32, LR 5e-4, `ReduceLROnPlateau`
  (factor 0.5, patience 5), `early_stopping_patience=10`, up to 50 epochs.
- **Hardware:** Apple Silicon MPS.

> Early stopping **triggered at epoch 35** (best eval-loss ~epoch 25; val loss
> rose afterward). run4b, by contrast, ran the full 50 (val loss still improving).
> Same config — the difference is each model's convergence, not protocol.

## Result on held-out test set (N=174)

| Metric | Value |
|---|---|
| PCC  | **0.767** |
| RMSE | **1.508** kcal/mol |
| MAE  | 1.133 kcal/mol |
| SCC  | 0.640 |

## Comparison — augmentation helps both PLMs

| Run | PLM | Augmentation | PCC | RMSE |
|---|---|---|---|---|
| run1 | Ankh    | none              | 0.757 | 1.531 |
| run2 | ProstT5 | none              | 0.740 | 1.584 |
| **run4a** | **Ankh** | **reverse + identity** | **0.767** | **1.508** |
| run4b | ProstT5 | reverse + identity | 0.759 | 1.547 |
| paper (10-fold CV) | Ankh-large | — | 0.868 | 1.185 |

**Verdict:** Tier-1 augmentation improves the strong Ankh baseline too
(PCC +0.010, RMSE −0.023) — **Ankh + aug (0.767) is the best single-PLM result
so far**. Augmentation is a consistent net positive for both backends; the gain
is larger for the weaker ProstT5 (+0.019) than for Ankh (+0.010), as expected.

## MPS runtimes (Apple Silicon) — for later comparison with the Linux box

Same machine, same 1629-row augmented train set, batch 32, 50-epoch cap.

| Run | PLM (dim) | Embedding gen | Train | Epochs run | ~Wall clock |
|---|---|---|---|---|---|
| run4a | Ankh (1536-d)    | ~205 s (2359 embeds) | 474.9 s | 35 (early stop) | ~11.5 min |
| run4b | ProstT5 (1024-d) | ~75 s (915 embeds)   | 452.5 s | 50 (full)       | ~9 min (est.) |

Per-epoch: Ankh ~13.6 s vs ProstT5 ~9.0 s (wider 1536-d embeddings → ~1.5× slower
training step). Reference: run2 (ProstT5, **Linux CPU**) trained in ~33 min — MPS is
roughly **4–5× faster** here. Embedding generation throughput on MPS was
~10–15 seq/s (ProstT5) and ~12 seq/s (Ankh-large).

Artifacts (gitignored): `training_run/`, `training_run.log` (timestamp-bracketed
`[START]`/`[END]`). Augmented inputs in `../../aug_common/`; Ankh embeddings in
`../../embeddings/`.
