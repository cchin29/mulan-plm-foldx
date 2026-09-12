# 10-fold CV with Tier-1 augmentation: Ankh vs ProstT5 on S1102

Same folds/config as `RESULTS_CV10.md` (`split_data(num_folds=10, random_state=42)`,
50 max ep, batch 32, lr 5e-4, early-stop 10), but each fold's TRAIN table is
augmented with reverse-mutation + identity anchors (`experiments/augment.py`);
val/test stay the original forward mutations. CPU-only.

| PLM | folds | Test PCC (mean +/- std) | Test RMSE | Test MAE | Test SCC |
|---|---|---|---|---|---|
| ankh | 10 | **0.838 +/- 0.054** | 1.286 +/- 0.125 | 0.921 +/- 0.091 | 0.738 +/- 0.057 |
| prostt5 | 10 | **0.819 +/- 0.060** | 1.386 +/- 0.142 | 0.987 +/- 0.108 | 0.681 +/- 0.055 |

## Augmented vs baseline (paired per-fold ΔPCC = aug − baseline)

| PLM | baseline PCC | aug PCC | mean ΔPCC | folds improved |
|---|---|---|---|---|
| ankh | 0.832 | 0.838 | +0.006 | 8/10 |
| prostt5 | 0.805 | 0.819 | +0.014 | 8/10 |

## Per-fold test PCC (baseline / aug)

| fold | ankh base | ankh aug | prostt5 base | prostt5 aug |
|---|---|---|---|---|
| 0 | 0.691 | 0.715 | 0.674 | 0.657 |
| 1 | 0.892 | 0.906 | 0.850 | 0.870 |
| 2 | 0.859 | 0.859 | 0.814 | 0.822 |
| 3 | 0.812 | 0.832 | 0.749 | 0.797 |
| 4 | 0.863 | 0.858 | 0.827 | 0.859 |
| 5 | 0.894 | 0.897 | 0.864 | 0.867 |
| 6 | 0.877 | 0.879 | 0.863 | 0.868 |
| 7 | 0.827 | 0.835 | 0.811 | 0.830 |
| 8 | 0.806 | 0.793 | 0.802 | 0.800 |
| 9 | 0.802 | 0.805 | 0.792 | 0.817 |

Baselines (non-aug cv10): Ankh 0.832, ProstT5 0.805. Single-split aug: run4a Ankh 0.769, run4b ProstT5 0.738. Paper MuLAN-Ankh CV: 0.868.
