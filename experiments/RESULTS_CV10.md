# 10-fold CV: Ankh vs ProstT5 on S1102

> Consolidated entry point for all results: [`../docs/RESULTS.md`](../docs/RESULTS.md). This
> is a deep-dive sub-doc.

Per-fold mutation-based 10-fold CV (`split_data(num_folds=10, random_state=42)`),
880/110/110 train/val/test per fold, every mutation tested exactly once.
Same training config as run1/run2: `lightatt_default_config`, 50 epochs, batch 32,
lr 5e-4, ReduceLROnPlateau, early-stop 10, best-val checkpoint tested. CPU-only.

| PLM | folds | Test PCC (mean +/- std) | Test RMSE | Test MAE | Test SCC |
|---|---|---|---|---|---|
| ankh | 10 | **0.832 +/- 0.057** | 1.310 +/- 0.118 | 0.944 +/- 0.090 | 0.731 +/- 0.060 |
| prostt5 | 10 | **0.805 +/- 0.055** | 1.426 +/- 0.123 | 1.023 +/- 0.094 | 0.674 +/- 0.051 |

## Per-fold test PCC

| fold | ankh | prostt5 |
|---|---|---|
| 0 | 0.691 | 0.674 |
| 1 | 0.892 | 0.850 |
| 2 | 0.859 | 0.814 |
| 3 | 0.812 | 0.749 |
| 4 | 0.863 | 0.827 |
| 5 | 0.894 | 0.864 |
| 6 | 0.877 | 0.863 |
| 7 | 0.827 | 0.811 |
| 8 | 0.806 | 0.802 |
| 9 | 0.802 | 0.792 |

## Findings

1. **Ankh robustly beats ProstT5 under CV** — unlike the single split, where the
   0.757-vs-0.740 gap was within noise. Paired by fold: **Ankh wins all 10/10
   folds**, mean paired Δ(Ankh−ProstT5) = **+0.0275 PCC** (std 0.0178), paired
   t = 4.65 (df=9, p ≈ 0.001); sign test 10/10 → p ≈ 0.002. So the ranking is real,
   ~+0.028 PCC, even though the per-PLM mean±std bands overlap (the std is
   cross-fold difficulty variance, which pairing cancels out).
2. **CV means are well above the single split** (Ankh 0.832 vs 0.757; ProstT5
   0.805 vs 0.740) — expected: each fold trains on 880 (vs 769) and the single
   split's 174-mutation test set was simply on the harder side.
3. **Still below the paper's 0.868** (MuLAN-Ankh) — plausibly the paper's
   pretrained-MuLAN checkpoint / config differs from this from-scratch
   `lightatt_default_config`. Same protocol family now, so the gap is a
   config/initialization question, not a measurement-protocol artifact.

Single-split baselines for reference: run1 Ankh 0.757, run2 ProstT5 0.740.
Paper's MuLAN-Ankh 10-fold CV on S1102: PCC 0.868 / RMSE 1.185.
Driver: `scratch/cv10_driver.sh`; aggregator: `scratch/cv10_aggregate.py`.
