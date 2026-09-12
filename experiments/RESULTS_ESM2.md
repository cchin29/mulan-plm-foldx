# ESM2-3B on S1102 — 10-fold CV (older-generation baseline)

Deep-dive for **§12** of `RESULTS.md`. ESM2-3B (`facebook/esm2_t36_3B_UR50D`) run
sequence-only through the same S1102 mutation-based 10-fold CV as Ankh / ProstT5 /
ESM C 6B, to place the *previous* ESM generation on the PLM axis at ESM C 6B's exact
embedding width.

## Headline

- **Test PCC 0.816 ± 0.054** (RMSE 1.395 ± 0.135, MAE 1.006 ± 0.093, SCC 0.691 ± 0.065).
- **Ties ProstT5, below Ankh, well below ESM C 6B** — the measured ranking is
  **ESM C 6B (0.859) > Ankh (0.832) > ESM2-3B (0.816) ≈ ProstT5 (0.805)**, two tiers,
  not the "+0.03 per step" ladder RESULTS had guessed (it had predicted ESM-2 weakest).

## Why it matters: a controlled control

ESM2-3B and ESM C 6B share the **same 2560-d final-layer width**, so the head/first-conv
geometry in MuLAN is identical — the only changed variable is the PLM's pretraining
(objective + data + scale). The **+0.043 PCC** ESM C 6B holds over ESM2-3B is therefore
a fairly clean read that the modern gain is **pretraining recipe (ESM Cambrian), not
parameter count**: 6B vs 3B at equal width, and yet ESM2-3B fails to clear even the
3.5×-smaller-encoder Ankh (1.15 B). Scale alone is not the lever; pretraining quality is.

## Per-fold and paired comparisons (same folds throughout)

| fold | ESM2-3B | ESM C 6B | Ankh | ProstT5 |
|---|---|---|---|---|
| 0 | 0.684 | 0.672 | 0.691 | 0.674 |
| 1 | 0.843 | 0.923 | 0.892 | 0.850 |
| 2 | 0.804 | 0.891 | 0.859 | 0.814 |
| 3 | 0.818 | 0.851 | 0.812 | 0.749 |
| 4 | 0.858 | 0.883 | 0.863 | 0.827 |
| 5 | 0.889 | 0.932 | 0.894 | 0.864 |
| 6 | 0.863 | 0.895 | 0.877 | 0.863 |
| 7 | 0.802 | 0.837 | 0.827 | 0.811 |
| 8 | 0.786 | 0.853 | 0.806 | 0.802 |
| 9 | 0.807 | 0.850 | 0.802 | 0.792 |
| **mean** | **0.816** | **0.859** | **0.832** | **0.805** |

| Paired Δ | mean | std | ESM2 wins | verdict |
|---|---|---|---|---|
| ESM2-3B − ESM C 6B | −0.0433 | 0.0274 | 1/10 | clearly worse |
| ESM2-3B − Ankh | −0.0166 | 0.0197 | 2/10 | worse |
| ESM2-3B − ProstT5 | +0.0109 | 0.0243 | 5/10 | **tied — within noise** |

Fold 0 is the hard fold for every PLM (all four ≈0.68), so the cross-fold std bands
are dominated by shared difficulty, which pairing cancels.

## Method & footprint

- **No new environment.** ESM2 is native to `transformers 4.44`, so it reuses the
  existing `esm` backend (`constants.py` → `facebook/esm2_t36_3B_UR50D`; `utils.py`
  generic `AutoModel` path) in the **main** venv — unlike ESM C 6B (isolated `esmc`
  venv) and AIDO (isolated bf16 venv). No code wiring was needed; the `esm` key
  pre-existed.
- **Offline embeddings, then read-only training** — the same pattern as every PLM here.
  `scratch/esm2_prewarm.sh` does one serial `MulanDataset.from_table(..., "esm")` pass
  (~37 min wall incl. the ~11 GB fp32 download + load + 1444 forward passes at 40
  threads) writing one `[L, 2560]` `.pt` per unique sequence to
  `scratch/embeddings_esm2/`; then `scratch/cv10_esm2_driver.sh` trains the 10 folds
  (`lightatt_default`, 50 ep, batch 32, lr 5e-4, early-stop 10, MAXPAR=4 × 10 threads),
  read-only from cache (`data.py:73` loads no PLM when embeddings are present).
- **Footprint:** 2.8 B params / ~12 GB peak RAM (embedding gen; ~11 GB fp32 weights +
  short-seq activations), 1.7 GB embedding cache, 19 MB trained ckpt. cv10 wall-clock
  **~8.0 h** (15:15→23:17, 2026-07-01), tracking ESM C 6B's ~8 h at the same 2560-d
  width since training reads cached tensors.
- **Artifacts** (gitignored `scratch/`): `esm2_prewarm.sh`, `cv10_esm2_driver.sh`,
  `esm2_aggregate.py`, `esm2_orchestrate.sh`, `esm2_cv10_summary.txt`,
  `embeddings_esm2/`, `results/cv10_esm2/fold_{0..9}/`.

## Next

Optionally confirm on the standard SKEMPI benchmarks (S1131/S4169/S2003) as for
Ankh-vs-ProstT5 (§10) if the ESM2-vs-ProstT5 tie is worth nailing down; otherwise the
axis now needs only AIDO-16B (§8) to place the largest scale point.
