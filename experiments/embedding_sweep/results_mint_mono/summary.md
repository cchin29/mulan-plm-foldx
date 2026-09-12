# Embedding sweep — S1102 (paper-faithful CV)

Setup: split_method=`balanced` seed=42 epochs=300 patience=30 config=`lightatt_default_config.json`. Data: S1102_filtered (1100). Paper target: Ankh 0.868 / ESM2-3B 0.854.

| model | n | PCC (mean±std) | RMSE (mean±std) | paper PCC | Δ PCC |
|---|---|---|---|---|---|
| mint_mono | 10 | 0.8114±0.065 | 1.366±0.167 | — | — |
