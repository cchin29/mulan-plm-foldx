# Embedding sweep — S1102 (paper-faithful CV)

Setup: split_method=`balanced` seed=42 epochs=300 patience=30 config=`lightatt_default_config.json`. Data: S1102_filtered (1100). Paper target: Ankh 0.868 / ESM2-3B 0.854.

| model | n | PCC (mean±std) | RMSE (mean±std) | paper PCC | Δ PCC |
|---|---|---|---|---|---|
| saprot | 10 | 0.8490±0.051 | 1.239±0.116 | — | — |
| ankh | 10 | 0.8357±0.053 | 1.288±0.121 | 0.868 | -0.032 |
| esm2 | 10 | 0.8329±0.053 | 1.308±0.114 | 0.854 | -0.021 |
| prostt5 | 10 | 0.8231±0.053 | 1.355±0.107 | — | — |
