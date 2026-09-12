# run3a — ProstT5 layer 7 (1024-d), no augmentation — findings (2026-06-20)

Goal (Tier 3.1): does a single **better** ProstT5 hidden layer beat the default
last-layer embedding? The layer probe (`../../../experiments/LAYER_PROBE.md`)
flagged ProstT5's final layer as weak and layer ~7 as stronger for local ΔΔG.
This run trains from scratch on layer-7 embeddings (1024-d, same dim as the
last-layer baseline).

## Setup

- **Split / data:** same 769/157/174 seed-42 split as run1/run2. **No augmentation**
  — isolates the layer choice vs run2's last-layer baseline (0.740).
- **Embeddings:** `gen_layer_embeddings.py --plm prostt5 --layers 7` → 1024-d
  (verified). 1444 sequences, 118 s (MPS).
- **Model / HP:** fresh `LightAttModel`, 50 epochs, AdamW, batch 32, LR 5e-4,
  `ReduceLROnPlateau`, early-stopping patience 10. MPS.

## Result on held-out test set (N=174) — NEGATIVE

| Metric | run2 baseline (last layer) | **run3a (layer 7)** |
|---|---|---|
| PCC  | 0.740 | **0.686** |
| RMSE | 1.584 | **1.719** kcal/mol |
| MAE  | 1.160 | 1.246 |
| SCC  | 0.637 | 0.595 |

Below the last-layer baseline on every metric. (Linux box got 0.648 for the same
run; ours 0.686 — same conclusion: layer-7-alone underperforms.)

## Same instability as run3b, milder

Early training is unstable but recovers far better than the 2048-d concat (run3b):

| | epoch 1 loss | first eval_pcc | final loss | test PCC |
|---|---|---|---|---|
| run3a (L7, 1024-d)   | 196 | 0.286 | ~2.2 | 0.686 |
| run3b (L7⊕L10, 2048-d) | 988 | ~0.02 | ~4.4 | 0.504 |

Confirms the diagnosis: raw mid-layer ProstT5 activations have large scale that
destabilizes from-scratch training; a single mid-layer (1024-d) is recoverable
but still loses to the better-scaled final layer, and stacking two (2048-d) is
much worse. Neither raw mid-layer variant beats last-layer.

## Implication for run3c

run3c was planned as "best of run3a/b + Tier-1 augmentation". Since **both** layer
variants underperform the last-layer baseline (best is run3a at 0.686, vs 0.740),
there is no good layer to carry forward — augmenting a worse base is very unlikely
to beat run4b (last-layer + augmentation = 0.759). **run3c is not worth running**
as specified; a productive Tier-3 path would instead normalize the cached
mid-layer features or use a learned scalar-mix (Tier 3.3).

## MPS runtime

Embed (1444 seqs, 1024-d): 118 s. Train (50 ep, 769 rows): 235.5 s. Wall ~6 min.
(Half of run4b's train time — same 1024-d width but no augmentation, so ~769 vs
1629 rows.)

Artifacts (gitignored): `training_run/`, `training_run.log`; embeddings in
`../../emb_prostt5_L7/`.
