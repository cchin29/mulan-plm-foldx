# MuLAN on S1102 — consolidated results

Overall summary of all from-scratch MuLAN experiments on the SKEMPI **S1102**
single-mutation ΔΔG benchmark. This file consolidates the per-run write-ups; the
detailed notes for each run remain in their respective directories.

Date range: 2026-06-20. Raw artifacts (data, embeddings, checkpoints, logs) live
in the gitignored per-run directories noted below.

Reference (MuLAN paper, Lombardi & Carbone, bioRxiv 2024.08.24.609515; Table 1,
mutation-based 10-fold CV):
**MuLAN-Ankh-large on S1102 → PCC = 0.868, RMSE = 1.185 kcal/mol.**

---

## Headline table — held-out test set (N=174), single seed-42 split

| Run | PLM | Embedding | Aug | Test PCC | Test RMSE | Test MAE | Test SCC |
|---|---|---|---|---|---|---|---|
| run1 | Ankh-large | last layer (1536-d) | none | 0.757 | 1.531 | — | — |
| run2 | ProstT5 (AA) | last layer (1024-d) | none | 0.740 | 1.584 | 1.160 | 0.637 |
| run3a | ProstT5 (AA) | layer 7 (1024-d) | none | 0.686 | 1.719 | 1.246 | 0.595 |
| run3b | ProstT5 (AA) | concat L7⊕L10 (2048-d) | none | 0.504 | 2.026 | 1.389 | 0.535 |
| **run4a** | **Ankh-large** | **last layer (1536-d)** | **reverse + identity** | **0.767** | **1.508** | **1.133** | **0.640** |
| run4b | ProstT5 (AA) | last layer (1024-d) | reverse + identity | 0.759 | 1.547 | 1.130 | 0.641 |
| — paper | Ankh-large | last layer | — (10-fold CV) | 0.868 | 1.185 | — | — |

**Best single-PLM result so far: run4a (Ankh + augmentation), PCC 0.767.**

Important caveat for every row above: these are **single 70/15/15 splits**
(769 train / 157 val / 174 test), not the paper's 10-fold CV average. Each run
trains on only 769 examples (vs the paper's ~990/fold), so single-split variance
is high and these numbers are **not** directly comparable to the paper's 0.868.

There is also a non-comparison data point: running the **shipped pretrained
checkpoint** (`models/pretrained/mulan_ankh.ckpt`) directly over the filtered
S1102 gives PCC 0.967 / RMSE 0.723 (N=1100). This beats the paper's own CV
number, which is the tell that the released checkpoint was trained on S1102
itself — it measures memorization, not generalization, and is only useful as a
pipeline sanity check.

---

## Shared setup (common to all from-scratch runs)

- **Data:** `S1102_filtered.tsv` — 1100 single mutations across 110 complexes.
  Wild-type sequences extracted from SKEMPI 2.0 cleaned PDBs
  (`wt_sequences.fasta`), numbered consistently with SKEMPI mutation labels.
  2 of the original 1102 mutations dropped (PDB `2I9B`, SKEMPI-v1-only, absent
  from the v2 structure bundle).
- **Split:** one mutation-based 70/15/15 split, seed 42 → **769 / 157 / 174**.
  The *same* split files are reused across runs, so the changed factor (PLM,
  layer, or augmentation) is the only variable. Validation and test are always
  real forward mutations, untouched by augmentation.
- **Model:** fresh, randomly-initialized `LightAttModel`
  (`models/config/lightatt_default_config.json`) — never a pretrained checkpoint.
  `nn.LazyConv1d` adapts to each PLM/feature width automatically (1024 / 1536 /
  2048-d), so no architecture change is needed between runs.
- **Hyperparameters (matched to the paper):** AdamW (wd 0.01), batch 32,
  LR 5e-4, `ReduceLROnPlateau` (factor 0.5, patience 5), early-stopping patience
  10, up to 50 epochs.
- **Bug fixed to make training work at all:** `scripts/train.py` called
  `logging.get_logger()` (stdlib `logging` has no such method) instead of
  `hf_logging.get_logger()`; `mulan-train` was non-functional before this fix
  (local commit `043ac4a`). ProstT5 backend support added in commit `13e05cf`
  (see `PLM_BACKEND.md`).

---

## Per-run detail

### run1 — Ankh-large, no augmentation (baseline)
*(`run1_s1102_ankh/`; full write-up: `../RESULTS.md`)*

The original from-scratch baseline. **Test PCC 0.757 / RMSE 1.531.** Established
the data provenance (SKEMPI 2.0 cleaned PDBs), the split, and the leakage finding
on the pretrained checkpoint. Training ~48 min CPU-only. Ankh embeddings cached
at `scratch/embeddings/`.

### run2 — ProstT5 (AA mode), no augmentation
*(`run2_s1102_prostt5/RESULTS.md`)*

Same setup as run1, swapping in `Rostlab/ProstT5` (amino-acid mode, `<AA2fold>`
prefix, per-residue 1024-d, last layer). **Test PCC 0.740 / RMSE 1.584.**
Marginally below Ankh — within single-split noise, not a robust ranking.
ProstT5's distinctive structure (3Di) channel is unused in this purely
sequence-based pipeline. Cheaper than Ankh (1024-d vs 1536-d → ~30% faster
training, ~⅓ less embedding disk). Embeddings cached at
`scratch/embeddings_prostt5/`. See `PLM_COMPARISON_S1102.md` for the full
runtime/memory breakdown.

### run3a — ProstT5 layer 7 (1024-d), no augmentation — NEGATIVE
*(`run3a_prostt5_L7/RESULTS.md`)*

Tested whether a single **better** ProstT5 hidden layer (the layer probe flagged
~L7 as stronger than the weak final layer) beats the last-layer baseline.
**Test PCC 0.686 — below the 0.740 baseline on every metric.** Early training is
unstable (epoch-1 loss ~196) but recovers; a single mid-layer is recoverable but
still loses to the better-scaled final layer.

### run3b — ProstT5 concat L7⊕L10 (2048-d), no augmentation — NEGATIVE
*(`run3b_prostt5_L7-10/RESULTS.md`)*

Tested whether a **richer multi-layer** concat helps. **Test PCC 0.504 — much
worse.** Training diverged early (epoch-1 loss ~988, peaking ~4000) and never
fully recovered. **Key finding:** raw mid-layer ProstT5 activations have large,
heterogeneous magnitudes that destabilize from-scratch training; stacking two
makes it worse. The layer probe's mid-layer advantage is a *normalized* linear
probe, so it does **not** transfer to raw-feature end-to-end training. To exploit
mid-layers you'd need feature normalization (LayerNorm / standardized cached
embeddings) or a learned scalar-mix — not a raw concat. (run3c, "best layer +
augmentation," was therefore not worth running and was dropped.)

### run4a — Ankh + Tier-1 augmentation — BEST
*(`run4a_ankh_aug/RESULTS.md`)*

Tier-1 augmentation = reverse mutations (`ΔΔG(wt→mut) = −ΔΔG(mut→wt)`) + identity
anchors (`wt→wt`, label 0), train split only: 769 → 1629 rows. **Test PCC 0.767 /
RMSE 1.508** — improves the strong Ankh baseline (PCC +0.010) and is the best
single-PLM result so far. Early stopping triggered at epoch 35.

### run4b — ProstT5 + Tier-1 augmentation
*(`run4b_prostt5_aug/RESULTS.md`)*

Same augmentation on the weaker ProstT5 baseline: 769 → 1629 rows. **Test PCC
0.759 / RMSE 1.547** — improves on every metric (PCC +0.019), lifting ProstT5 to
roughly the Ankh baseline level. Ran the full 50 epochs (val loss still
improving). The augmentation gain is larger for the weaker PLM (ProstT5 +0.019)
than the stronger one (Ankh +0.010), as expected.

**Augmentation verdict:** Tier-1 augmentation is a consistent net positive for
both backends — the antisymmetry + zero-point constraints help the small-data
regime.

---

## Layer-wise linear probes (diagnostic, not from-scratch training)
*(`probe_layers_RESULTS.md`; per-PLM tables in `experiments/results/probe_*.md`)*

Linear RidgeCV probe (5-fold CV), per layer, on N=1100. Features from the mutated
chain: **site delta** (`mut−wt` at the mutated residue) and **pool delta** (mean
over residues). These are *normalized* probes — they motivated the run3 layer
experiments but, as run3b showed, the mid-layer advantage does not carry over to
raw-feature end-to-end training.

| PLM | Best site layer | Best pool layer | Final layer (site / pool) |
|---|---|---|---|
| Ankh-large (49 layers) | L47, PCC 0.799 | L14, PCC 0.775 | L48: 0.796 / 0.751 |
| ProstT5 (25 layers) | L3, PCC 0.765 | L10, PCC 0.754 | L24: 0.702 / 0.731 |

Note ProstT5's final layer is comparatively weak for local ΔΔG signal (site PCC
0.702 vs 0.765 at L3), whereas Ankh's final layers stay strong — consistent with
ProstT5 underperforming Ankh in the end-to-end runs.

---

## Runtime notes

- **run1/run2** were CPU-only: Ankh ~48 min train, ProstT5 ~33 min train;
  embedding generation ~19 min (ProstT5). See `PLM_COMPARISON_S1102.md` for the
  full CPU runtime/memory table.
- **run3–run4** ran on Apple Silicon **MPS**, roughly 4–5× faster than the Linux
  CPU box: e.g. run4a (Ankh, augmented) ~11.5 min wall, run4b (ProstT5,
  augmented) ~9 min, run3a ~6 min.

## Possible next steps

- **10-fold CV per configuration** for a variance-controlled number directly
  comparable to the paper's 0.868/1.185. Embeddings are already cached, so each
  fold is training-only (~30 min CPU / ~6 min MPS) → ~5.5 h CPU per config.
- **Normalize mid-layer features** (LayerNorm / standardize cached embeddings) or
  use a **learned scalar-mix** to actually exploit the layer-probe signal that
  raw concatenation (run3b) could not.
