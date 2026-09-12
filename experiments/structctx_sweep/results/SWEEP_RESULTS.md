# P1+P2 structural-context sweep — results

RSA is table-fixed → identical across configs; only `contact_count` varies. AUROC/Cliff's δ are **directional** (n=3 sites/protein).

## TEM-1 (P62593)

### Functional-site contact_count (percentile) by config

| config | S68 | K71 | E164 | core_med | surf_med | AUROC | δ(func-core) | ρ_vs_base |
|---|---|---|---|---|---|---|---|---|
| ca8/50 | 11 (0.67) | 11 (0.67) | 11 (0.67) | 12 | 7 | 0.61 | -0.38 | 1.000 |
| cb5/50 | 3 (0.94) | 1 (0.58) | 2 (0.83) | 2 | 0 | 0.68 | +0.17 | 0.460 |
| ca8/70 | 11 (0.67) | 11 (0.67) | 11 (0.67) | 12 | 7 | 0.61 | -0.37 | 0.998 |
| cb5/70 | 3 (0.94) | 1 (0.58) | 2 (0.84) | 2 | 0 | 0.69 | +0.17 | 0.467 |
| ca8/none | 11 (0.67) | 11 (0.67) | 11 (0.67) | 12 | 7 | 0.61 | -0.38 | 1.000 |
| cb5/none | 3 (0.94) | 1 (0.58) | 2 (0.83) | 2 | 0 | 0.68 | +0.17 | 0.460 |

**RSA / burial (config-invariant):** S68 rsa=0.052 (pctile 0.33, buried), K71 rsa=0.000 (pctile 0.18, buried), E164 rsa=0.009 (pctile 0.23, buried)

## TP53 (P04637)

### Functional-site contact_count (percentile) by config

| config | R175 | R248 | R273 | core_med | surf_med | AUROC | δ(func-core) | ρ_vs_base |
|---|---|---|---|---|---|---|---|---|
| ca8/50 | 14 (0.97) | 7 (0.58) | 12 (0.91) | 12 | 4 | 0.79 | -0.08 | 1.000 |
| cb5/50 | 3 (0.99) | 0 (0.61) | 0 (0.61) | 1 | 0 | 0.53 | -0.25 | 0.602 |
| ca8/70 | 14 (0.97) | 7 (0.60) | 12 (0.91) | 12 | 2 | 0.80 | -0.08 | 0.974 |
| cb5/70 | 3 (0.99) | 0 (0.64) | 0 (0.64) | 1 | 0 | 0.54 | -0.25 | 0.630 |
| ca8/none | 14 (0.97) | 7 (0.56) | 12 (0.91) | 12 | 5 | 0.79 | -0.08 | 0.952 |
| cb5/none | 3 (0.99) | 0 (0.51) | 0 (0.51) | 1 | 0 | 0.49 | -0.25 | 0.356 |

**RSA / burial (config-invariant):** R175 rsa=0.022 (pctile 0.08, buried), R248 rsa=0.719 (pctile 0.68, exposed), R273 rsa=0.299 (pctile 0.31, exposed)

