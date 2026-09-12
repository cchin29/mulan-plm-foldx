# SKEMPI benchmarks: ESM C 6B (vs Ankh / ProstT5), 10-fold CV

ESM C 6B extended from the S1102 cv10 (RESULTS.md §11) to the standard SKEMPI
single-mutation benchmarks, same folds/config as the Ankh-vs-ProstT5 bench CV
(`lightatt_default_config`, 50 ep, batch 32, lr 5e-4, early-stop 10, best-val
checkpoint). ESM C is sequence-only (AA mode, 2560-d). CPU-only training, embeddings
generated on this box in the isolated `mulan-esmc` venv. Datasets restricted to
single-chain-per-partner complexes (MuLAN's design), nested S1131 ⊂ S4169 ⊃ S2003.

| Dataset | size | PLM | Pearson r | Spearman ρ | RMSE | MAE |
|---|---|---|---|---|---|---|
| **S1131** | 1127 muts / 110 cplx | ESM C 6B | **0.866 ± 0.036** | 0.792 ± 0.047 | 1.210 ± 0.171 | 0.872 ± 0.092 |
|  |  | Ankh | **0.822 ± 0.043** | 0.737 ± 0.050 | 1.384 ± 0.170 | 0.991 ± 0.097 |
|  |  | ProstT5 | **0.781 ± 0.050** | 0.681 ± 0.030 | 1.565 ± 0.190 | 1.114 ± 0.113 |
| **S4169** | 2497 muts / 211 cplx (single-chain subset) | ESM C 6B | **0.825 ± 0.049** | 0.680 ± 0.065 | 1.056 ± 0.106 | 0.732 ± 0.062 |
|  |  | Ankh | **0.798 ± 0.042** | 0.639 ± 0.054 | 1.133 ± 0.068 | 0.793 ± 0.040 |
|  |  | ProstT5 | **0.764 ± 0.063** | 0.607 ± 0.074 | 1.209 ± 0.093 | 0.850 ± 0.049 |
| **S2003** | 1124 muts / 174 cplx (non-Ala single-chain, derived) | ESM C 6B | **0.852 ± 0.038** | 0.752 ± 0.054 | 1.238 ± 0.148 | 0.867 ± 0.094 |
|  |  | Ankh | **0.823 ± 0.058** | 0.707 ± 0.051 | 1.347 ± 0.194 | 0.948 ± 0.123 |
|  |  | ProstT5 | **0.796 ± 0.072** | 0.660 ± 0.060 | 1.429 ± 0.168 | 1.007 ± 0.083 |

## Paired ESM C 6B − Ankh (Pearson r, per fold)

| Dataset | Δr (mean) | folds ESM C wins |
|---|---|---|
| S1131 | +0.0441 | 10/10 |
| S4169 | +0.0272 | 9/10 |
| S2003 | +0.0290 | 9/10 |

_Reference: S1102 cv10 — ESM C 6B 0.859 ± 0.069, Ankh 0.832 ± 0.057, ProstT5 0.805 ± 0.055 (Pearson)._
