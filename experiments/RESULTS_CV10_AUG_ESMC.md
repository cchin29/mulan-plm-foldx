# 10-fold CV with Tier-1 augmentation: ESM C 6B on S1102

Same folds/config as the base ESM C 6B cv10 (RESULTS.md §11) and the Ankh/ProstT5 aug
run (`split_data(num_folds=10, random_state=42)`, 50 max ep, batch 32, lr 5e-4,
early-stop 10), but each fold's TRAIN table is augmented with reverse-mutation +
identity anchors (`experiments/augment.py`); val/test stay the original forward
mutations. CPU-only training; embeddings pre-cached.

| Config | folds | Test PCC | RMSE | MAE | SCC |
|---|---|---|---|---|---|
| ESM C 6B +aug | 10 | **0.878 ± 0.043** | 1.122 ± 0.111 | 0.809 ± 0.076 | 0.794 ± 0.034 |
| ESM C 6B base | 10 | **0.859 ± 0.069** | 1.176 ± 0.164 | 0.853 ± 0.117 | 0.777 ± 0.056 |
| Ankh +aug | 10 | **0.838 ± 0.054** | 1.286 ± 0.125 | 0.921 ± 0.091 | 0.738 ± 0.057 |
| Ankh base | 10 | **0.832 ± 0.057** | 1.310 ± 0.118 | 0.944 ± 0.090 | 0.731 ± 0.060 |

**Paired Δ(ESM C +aug − ESM C base) PCC = +0.0189** (std 0.0321); +aug wins 8/10 folds.

per-fold +aug: 0.779, 0.925, 0.906, 0.848, 0.899, 0.925, 0.909, 0.876, 0.853, 0.856
per-fold base: 0.672, 0.923, 0.891, 0.851, 0.883, 0.932, 0.895, 0.837, 0.853, 0.850

_Reference base->aug: Ankh 0.832 -> 0.838 (+0.006)._
