# P3 — embedding mutation-sensitivity (TEM-1 P62593, L150A) — row-alignment gate

Each embedder must localize a point mutation to its own row (argmax = mutated position) and dominate (ratio ≫ 1). Decides row-alignment/correctness, NOT ΔΔG (see README §P3). Transformers PLMs on CPU; ESM C via the `esm` SDK in `.venv-esmc` (esmc_600m local/MPS, esmc_6b cluster).

| model | dim | argmax pos | ratio (mut/median) | gate |
|---|---|---|---|---|
| ankh | 1536 | 150 | 70.4 | ✅ (want argmax=150, ratio>5) |
| prostt5_aa | 1024 | 150 | 63.0 | ✅ (want argmax=150, ratio>5) |
| ankh3_large | 1536 | 150 | 63.2 | ✅ (want argmax=150, ratio>5) |
| ankh3_xl | 2560 | 150 | 80.0 | ✅ (want argmax=150, ratio>5) |
| saprot | 1280 | 150 | 66.3 | ✅ (want argmax=150, ratio>5) |
| saprot_1.3b | 1280 | 150 | 86.6 | ✅ (want argmax=150, ratio>5) |
| esm2_3b | 2560 | 150 | 14.0 | ✅ (want argmax=150, ratio>5) |
| esm2_650m | 1280 | 150 | 31.0 | ✅ (want argmax=150, ratio>5) |
| esmc_600m | 1152 | 150 | 27.0 | ✅ (want argmax=150, ratio>5) |
| esmc_6b | 2560 | — | — | ⏸ deferred (env) — transformers 5.12.1 does not register the `esmc` model_type; esm SDK local loader is 300M/600M only (6B is Forge/cluster). Row-alignment validated via cluster benchmarks + M5 length-guard. |

**9/9 locally-runnable models pass** the row-alignment gate (argmax=150, ratio>5) → embeddings are correctly per-residue aligned; the Ankh default (and ProstT5/ESM C swaps) are safe to use interchangeably.
