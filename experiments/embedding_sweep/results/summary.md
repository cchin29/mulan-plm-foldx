# Embedding sweep — S1102 (paper-faithful CV)

Setup: split_method=`random` seed=42 epochs=300 patience=30 config=`lightatt_default_config.json`. Data: S1102_filtered (1100). Paper target: Ankh 0.868 / ESM2-3B 0.854.

| model | n | PCC (mean±std) | RMSE (mean±std) | paper PCC | Δ PCC |
|---|---|---|---|---|---|
| saprot | 4/10 ⚠ | 0.8617±0.047 | 1.253±0.046 | — | — |
| ankh | 10 | 0.8217±0.048 | 1.344±0.110 | 0.868 | -0.046 |
| esmc600m | 10 | 0.8177±0.046 | 1.368±0.115 | — | — |
| prostt5 | 10 | 0.8006±0.053 | 1.438±0.121 | — | — |

> ⚠ **Incomplete/skipped models** (metrics provisional):
> - esm2: 0/10 folds found — SKIPPED. Likely a silent no-op (unknown-PLM tag) or wrong ES_RESULTS_DIR (scratch/results/embedding_sweep).
> - esmc6b: 0/10 folds found — SKIPPED. Likely a silent no-op (unknown-PLM tag) or wrong ES_RESULTS_DIR (scratch/results/embedding_sweep).
> - aido: 0/10 folds found — SKIPPED. Likely a silent no-op (unknown-PLM tag) or wrong ES_RESULTS_DIR (scratch/results/embedding_sweep).
> - ankh3_large: 0/10 folds found — SKIPPED. Likely a silent no-op (unknown-PLM tag) or wrong ES_RESULTS_DIR (scratch/results/embedding_sweep).
> - ankh3_xl: 0/10 folds found — SKIPPED. Likely a silent no-op (unknown-PLM tag) or wrong ES_RESULTS_DIR (scratch/results/embedding_sweep).
> - saprot: PARTIAL — 4/10 folds have test_pcc (missing folds [4, 5, 6, 7, 8, 9], null-pcc folds []); mean over a partial set — treat as provisional.
> - saprot13b: 0/10 folds found — SKIPPED. Likely a silent no-op (unknown-PLM tag) or wrong ES_RESULTS_DIR (scratch/results/embedding_sweep).
> - esm3: 0/10 folds found — SKIPPED. Likely a silent no-op (unknown-PLM tag) or wrong ES_RESULTS_DIR (scratch/results/embedding_sweep).
