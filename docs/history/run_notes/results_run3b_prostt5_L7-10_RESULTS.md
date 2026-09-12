# run3b — ProstT5 layer concat 7⊕10 (2048-d), no augmentation — findings (2026-06-20)

Goal (Tier 3.1, feature augmentation): does a **richer multi-layer** ProstT5
representation beat the default last-layer embedding? Motivated by the layer
probe (`../../../experiments/LAYER_PROBE.md`), which found ProstT5's final layer
weak for local ΔΔG signal and mid-layers stronger. This run concatenates hidden
layers **7 and 10** (1024-d each → **2048-d**) and trains from scratch.

## Setup

- **Split / data:** same 769/157/174 seed-42 split as run1/run2. **No augmentation**
  (original 769-row train set) — isolates the layer choice as the only variable
  vs run2's last-layer baseline (0.740).
- **Embeddings:** `experiments/gen_layer_embeddings.py --plm prostt5 --layers 7,10`,
  per-residue concat → 2048-d (verified: `[L, 2048]`). 1444 sequences, 118 s (MPS).
- **Model / HP:** fresh `LightAttModel` (`LazyConv1d` adapts to 2048-d), 50 epochs,
  AdamW, batch 32, LR 5e-4, `ReduceLROnPlateau`, early-stopping patience 10. MPS.

## Result on held-out test set (N=174) — NEGATIVE

| Metric | run2 baseline (last layer) | **run3b (concat 7⊕10)** |
|---|---|---|
| PCC  | 0.740 | **0.504** |
| RMSE | 1.584 | **2.026** kcal/mol |
| MAE  | 1.160 | 1.389 |
| SCC  | 0.637 | 0.535 |

**Worse on every metric.** The concat embedding does *not* help — it badly hurts.

## Why: activation-scale instability (the key finding)

Training **diverged early then partially recovered**:

| epoch | 1 | 2 | 3 | 4 | … | 48 | 49 | 50 |
|---|---|---|---|---|---|---|---|---|
| train loss | 988 | 4068 | 3714 | 2141 | … | 3.6 | 4.4 | 4.4 |

(reported `train_loss` 395.9 is the mean, inflated by the early spikes; early
`eval_pcc` ≈ 0.02.) Raw **mid-layer** ProstT5 hidden states have much larger and
more heterogeneous magnitudes than the final layer (which is near-normalized), so
a 2048-d concat of two raw mid-layers makes the from-scratch optimization
unstable; it never fully recovers (final loss ~4.4 vs ~1.5 for healthy runs).

**Important caveat for the layer-probe motivation:** the probe is a *normalized
linear probe* (features standardized before fitting), so its mid-layer advantage
does **not** transfer to raw-feature end-to-end training. To actually exploit
mid-layers here you'd need feature normalization (LayerNorm / standardize the
cached embeddings) or a learned scalar-mix (Tier 3.3) — not a raw concat.

## MPS runtime

Embed (1444 seqs, 2048-d): 118 s. Train (50 ep): 438.8 s. Wall ~9.4 min.

Artifacts (gitignored): `training_run/`, `training_run.log` (timestamped);
embeddings in `../../emb_prostt5_L7-10/`.
