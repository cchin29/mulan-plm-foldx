# SKEMPI benchmarks: four PLMs (10-fold CV)

Mutation-level 10-fold CV (`split_data(num_folds=10, random_state=42)`), every mutation
tested once. Training config identical to the S1102 cv10 (`lightatt_default_config`, 50 ep,
batch 32, lr 5e-4, ReduceLROnPlateau, early-stop 10, best-val checkpoint). Ankh + ProstT5
trained on the Linux CPU box; **ESM2-3B + SaProt on Apple M4 Pro / MPS** (◆) — device parity
validated in RESULTS.md §16.

**Datasets** (single-mutation, restricted to single-chain-per-partner complexes — MuLAN's
one-sequence-per-partner design, same as S1102):

- **S1131** — Xiong et al.'s non-redundant interface single mutations (1127 usable / 110 complexes).
- **S4169** — mCSM-PPI2's all-single-mutations set; single-chain subset (2497 / 211). Both were
  obtained from GeoPPI's verbatim copies — see `BENCHMARK_DATASETS.md` for the provenance chain.
- **S2003** — derived: single NON-alanine SKEMPI 2.0 mutations, single-chain, deduped (1124 / 174).
  (Named for the ~2000 non-Ala singles in SKEMPI 2.0; single-chain restriction yields 1124.)
  ΔΔG = -RT·ln(Kd_mut/Kd_wt), matching GeoPPI's convention.

SaProt is run with **wild-type 3Di** (mini3di; a mutant reuses its parent WT chain's 3Di —
see RESULTS.md §15); the AA half of the SA token still changes at the mutated site.

| Dataset | size | PLM | Pearson r | Spearman ρ | RMSE | MAE |
|---|---|---|---|---|---|---|
| **S1131** | 1127 muts / 110 cplx | ProstT5 | 0.781 ± 0.050 | 0.681 ± 0.030 | 1.565 ± 0.190 | 1.114 ± 0.113 |
|  |  | ESM2-3B ◆ | 0.822 ± 0.036 | 0.730 ± 0.030 | 1.389 ± 0.140 | 1.010 ± 0.070 |
|  |  | Ankh | 0.822 ± 0.043 | 0.737 ± 0.050 | 1.384 ± 0.170 | 0.991 ± 0.097 |
|  |  | **SaProt (WT-3Di) ◆** | **0.847 ± 0.040** | **0.749 ± 0.058** | **1.301 ± 0.161** | **0.933 ± 0.101** |
| **S4169** | 2497 muts / 211 cplx (single-chain subset) | ProstT5 | 0.764 ± 0.063 | 0.607 ± 0.074 | 1.209 ± 0.093 | 0.850 ± 0.049 |
|  |  | ESM2-3B ◆ | 0.792 ± 0.048 | 0.627 ± 0.080 | 1.144 ± 0.083 | 0.801 ± 0.054 |
|  |  | Ankh | 0.798 ± 0.042 | 0.639 ± 0.054 | 1.133 ± 0.068 | 0.793 ± 0.040 |
|  |  | **SaProt (WT-3Di) ◆** | **0.815 ± 0.046** | **0.645 ± 0.047** | **1.082 ± 0.091** | **0.756 ± 0.057** |
| **S2003** | 1124 muts / 174 cplx (non-Ala single-chain, derived) | ProstT5 | 0.796 ± 0.072 | 0.660 ± 0.060 | 1.429 ± 0.168 | 1.007 ± 0.083 |
|  |  | ESM2-3B ◆ | 0.807 ± 0.068 | 0.675 ± 0.073 | 1.398 ± 0.171 | 0.980 ± 0.106 |
|  |  | Ankh | 0.823 ± 0.058 | 0.707 ± 0.051 | 1.347 ± 0.194 | 0.948 ± 0.123 |
|  |  | **SaProt (WT-3Di) ◆** | **0.832 ± 0.051** | **0.721 ± 0.051** | **1.307 ± 0.142** | **0.908 ± 0.101** |

**SaProt (WT-3Di) wins every benchmark on every metric** — leading the best sequence-only
model (Ankh) by +0.025 / +0.017 / +0.009 Pearson, with the lowest RMSE and MAE throughout,
despite being the smallest encoder (650M). Input-level AA+3Di structure fusion generalizes the
S1102 result (§15) across all three standard SKEMPI sets. **ESM2-3B sits in the Ankh tier**
(ties Ankh on S1131, just below on S4169/S2003, above ProstT5 everywhere) — the same mid-pack
placement it has on S1102 (§12). The rank order **ProstT5 < ESM2-3B ≲ Ankh < SaProt** is
identical on all three benchmarks and matches S1102.

## Paired Ankh − ProstT5 (Pearson r, per fold)

| Dataset | Δr (mean) | folds Ankh wins |
|---|---|---|
| S1131 | +0.0413 | 8/10 |
| S4169 | +0.0343 | 9/10 |
| S2003 | +0.0270 | 10/10 |

Ankh beats ProstT5 on every set (the original head-to-head); SaProt then beats Ankh on every
set. Paired SaProt−Ankh / ESM2−Ankh deltas are not tabulated here because Ankh/ProstT5 ran on
the Linux box and SaProt/ESM2 on MPS — the folds are identical (`random_state=42`) but the
per-fold arrays weren't co-located; the aggregate margins above are the cross-run comparison.

_Reference: S1102 cv10 — ProstT5 0.805, ESM2-3B 0.816, Ankh 0.832, SaProt (WT-3Di) 0.842 (Pearson)._
