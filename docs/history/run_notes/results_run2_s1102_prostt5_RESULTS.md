# MuLAN from-scratch training on ProstT5 embeddings — findings (2026-06-20)

Goal: train a MuLAN model **from scratch** on **ProstT5** embeddings over the
S1102 benchmark, and compare against the earlier from-scratch Ankh run
(run1, Experiment 2) and the paper. This is the run2 counterpart to
`../run1_s1102_ankh/RESULTS.md`.

ProstT5 was added as a selectable embedding backend in commit
`13e05cf`; see `../../../PLM_BACKEND.md` for how the multi-PLM support works.

## Setup (identical to run1 except the PLM)

- **Data:** `S1102_filtered.tsv` (1100 single mutations, 110 complexes), wild-type
  sequences from SKEMPI 2.0 cleaned PDBs (`wt_sequences.fasta`). Same provenance
  as run1 — see `../run1_s1102_ankh/RESULTS.md` for details.
- **Split:** the **exact same** mutation-based 70/15/15 split as run1
  (`splits/`, seed 42 → 769 train / 157 val / 174 test). Reusing the same split
  makes the PLM the only changed variable.
- **Model:** fresh, randomly-initialized `LightAttModel`
  (`models/config/lightatt_default_config.json`) — no pretrained checkpoint
  (none exists for ProstT5). `nn.LazyConv1d` adapts to ProstT5's 1024-dim input
  automatically (Ankh is 1536-dim), so no architecture/config change was needed.
- **PLM embeddings:** `Rostlab/ProstT5` in amino-acid mode (`<AA2fold>` prefix),
  per-residue, 1024-dim. Generated once over the full filtered table
  (1444 embeddings, ~19 min CPU), then reused for training.
- **Hyperparameters (matched to run1 / the paper):** AdamW (weight_decay 0.01),
  batch size 32, LR 5e-4, `ReduceLROnPlateau` (factor 0.5, patience 5),
  `early_stopping_patience=10`, up to 50 epochs (ran the full 50). CPU-only,
  training ~33 min.

## Result on held-out test set (N=174)

| Metric | Value |
|---|---|
| PCC  | **0.740** |
| RMSE | **1.584** kcal/mol |
| MAE  | 1.160 kcal/mol |
| SCC  | 0.637 |

Artifacts: `training_run/` (`model.ckpt`, `all_results.json`,
`test_predictions.tsv`, HF checkpoints), `training_run.log`, `embedding_gen.log`,
`driver.log`. ProstT5 embeddings cached at `../../embeddings_prostt5/` (reusable).

## Comparison

| Run | PLM | Embedding dim | Test PCC | Test RMSE (kcal/mol) |
|---|---|---|---|---|
| Paper (10-fold CV avg.) | Ankh-large | 1536 | 0.868 | 1.185 |
| run1 — from-scratch, single split | Ankh-large | 1536 | 0.757 | 1.531 |
| **run2 — from-scratch, single split** | **ProstT5** | **1024** | **0.740** | **1.584** |

## Interpretation

- On this **single split**, ProstT5 lands slightly below Ankh-large
  (PCC 0.740 vs 0.757; RMSE 1.584 vs 1.531) — close, within the noise a single
  70/15/15 split carries. This is **not** a robust head-to-head: both runs use
  one split and trained on 769 examples, far fewer than the paper's ~990/fold
  10-fold average, so variance is high. A 10-fold CV per PLM would be needed for
  a defensible ranking.
- Plausible reasons ProstT5 doesn't beat Ankh here for ΔΔG-from-sequence:
  ProstT5's value-add is its structure (3Di) channel, which this purely
  sequence-based pipeline does not use (we run it in AA mode); its AA-mode
  embeddings are 1024-dim vs Ankh's 1536, and the MuLAN checkpoints/paper were
  tuned around Ankh/ESM. Nothing indicates a pipeline bug — the embeddings are
  correct per-residue 1024-d tensors and flow end-to-end.

## Possible next step

Run 10-fold CV for ProstT5 (and re-run Ankh 10-fold) for a variance-controlled
comparison directly against the paper's 0.868/1.185. Each fold ~33 min training
+ embeddings already cached → ~5.5 h for 10 ProstT5 folds.
