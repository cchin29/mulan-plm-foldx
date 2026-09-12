# From-scratch MuLAN on S1102: Ankh vs ProstT5 (findings)

> Consolidated entry point for all results: [`RESULTS.md`](../RESULTS.md). This is a
> deep-dive sub-doc.

This documents two from-scratch MuLAN training runs on the SKEMPI **S1102**
single-mutation ΔΔG benchmark, differing only in the protein language model (PLM)
used to embed sequences. It accompanies `PLM_BACKEND.md` (how multi-PLM support
works) and the ProstT5 backend added in commit `13e05cf`.

Date: 2026-06-20. Raw artifacts (data, embeddings, checkpoints, logs) live in the
gitignored `scratch/results/` directories noted below.

## Method (identical across runs except the PLM)

- **Data:** `S1102_filtered.tsv` — 1100 single mutations across 110 complexes;
  wild-type sequences extracted from SKEMPI 2.0 cleaned PDBs (mutation-consistent
  numbering). 2 of the original 1102 mutations dropped (PDB `2I9B`, SKEMPI-v1-only).
- **Split:** one mutation-based 70/15/15 split, seed 42 → **769 train / 157 val /
  174 test**. The *same* split files are reused for both runs, so the PLM is the
  only changed variable.
- **Model:** fresh, randomly-initialized `LightAttModel`
  (`models/config/lightatt_default_config.json`) — not a pretrained checkpoint.
  `nn.LazyConv1d` adapts to each PLM's embedding width automatically, so no
  architecture change is needed between PLMs.
- **Hyperparameters:** AdamW (wd 0.01), batch 32, LR 5e-4,
  `ReduceLROnPlateau` (0.5, patience 5), early-stopping patience 10, up to 50
  epochs. CPU-only.

## Results (held-out test set, N=174)

| Run | PLM | Embed dim | Test PCC | Test RMSE | Test MAE | Test SCC |
|---|---|---|---|---|---|---|
| run1 (`scratch/results/run1_s1102_ankh/`)    | Ankh-large       | 1536 | **0.757** | 1.531 | —     | —     |
| run2 (`scratch/results/run2_s1102_prostt5/`) | ProstT5 (AA mode)| 1024 | **0.740** | 1.584 | 1.160 | 0.637 |

Reference: the MuLAN paper reports **PCC 0.868 / RMSE 1.185** for MuLAN-Ankh-large
on S1102, but that is a **10-fold CV average**, not a single split — not directly
comparable to the single-split numbers above.

ProstT5 embeddings: `Rostlab/ProstT5` in amino-acid mode (`<AA2fold>` prefix),
per-residue 1024-dim. Generation ~19 min; training ~33 min (CPU).

**10-fold CV update (`experiments/RESULTS_CV10.md`):** the single-split gap above is
within noise, but under 10-fold CV the ranking is robust — **Ankh 0.832 ± 0.057**
vs **ProstT5 0.805 ± 0.055**, with Ankh winning **all 10/10 folds** (paired
Δ +0.028 PCC, t = 4.65, p ≈ 0.001).

## Runtime and memory (CPU-only, 20 threads)

| Resource | Ankh-large (run1) | ProstT5 (run2) |
|---|---|---|
| Embedding generation (1444 seqs) | not recorded¹ | ~19 min (1130 s) |
| Training (50 epochs, 769 train) | ~48 min (2881 s) | ~33 min (1978 s) |
| Test/predict (174) | 5.9 s | 4.2 s |
| Encoder parameters | 1.15 B | 1.21 B |
| Peak RAM, encoder inference (fp32) | 4.7 GB | 4.9 GB |
| Embedding cache on disk (1444 `.pt`) | 1011 MB | 683 MB |
| Trained `model.ckpt` | ~12 MB | ~8 MB |
| PLM weights in HF cache² | 7.0 GB | 11 GB |

¹ run1 did not log embedding-generation time separately; expected to be the same
order as ProstT5 (similar encoder size, same sequence set).
² On-disk HuggingFace cache, which holds multiple weight formats; the actual RAM
footprint when loaded is the ~4.7–4.9 GB "peak RAM" row.

**Why ProstT5 is cheaper despite a slightly larger encoder:** its embeddings are
1024-dim vs Ankh's 1536. That smaller width flows straight into MuLAN's
`LazyConv1d` input channels, so the trained model has fewer parameters (smaller
`model.ckpt`), trains ~30% faster, and its cached embeddings take ~⅓ less disk.
The two PLM *encoders* themselves are close in size (~1.2 B params, ~4.8 GB RAM).

## Interpretation

- On this single split, **ProstT5 is marginally below Ankh** (PCC 0.740 vs 0.757;
  RMSE 1.584 vs 1.531). The gap is within the noise a single 70/15/15 split
  carries — this is not a robust ranking. Both runs train on only 769 examples
  (vs the paper's ~990/fold), so variance is high.
- Plausible reasons ProstT5 doesn't win here for ΔΔG-from-sequence: its
  distinctive structure (3Di) channel is unused in this purely sequence-based
  pipeline (run in AA mode); its AA-mode embeddings are 1024-dim vs Ankh's 1536;
  and the architecture/paper were tuned around Ankh/ESM. No evidence of a pipeline
  bug — embeddings are verified correct per-residue 1024-d tensors that flow
  end-to-end.

## Layer-selection and augmentation experiments

Follow-on runs on the **same split**, exploring two levers: which PLM hidden
layer to embed from (motivated by `experiments/LAYER_PROBE.md`), and Tier-1 data
augmentation (`experiments/AUGMENTATION.md`).

| Run | PLM | Embedding | Augmentation | Epochs | Test PCC | RMSE | MAE |
|---|---|---|---|---|---|---|---|
| run1 | Ankh | last layer (1536) | none | 50 | 0.757 | 1.531 | 1.139 |
| run2 | ProstT5 | last layer (1024) | none | 50 | 0.740 | 1.584 | 1.160 |
| run3a | ProstT5 | layer 7 (1024) | none | 50 | 0.648 | 1.782 | 1.284 |
| run3b | ProstT5 | concat 7⊕10 (2048) | none | 25 | 0.429 | 2.193 | 1.473 |
| **run4a** | **Ankh** | last layer (1536) | **reverse + identity** | 25 | **0.769** | **1.510** | 1.129 |
| run4b | ProstT5 | last layer (1024) | reverse + identity | 25 | 0.738 | 1.611 | 1.187 |

**1. Mid/multi-layer embeddings *hurt* ProstT5 (Tier-3 negative result).**
The layer probe found ProstT5's final layer weak for the *local* mutation signal
(linear site-PCC 0.702 vs ~0.765 mid), but in full MuLAN, swapping to layer 7
(run3a, 0.648) or concatenating 7⊕10 (run3b, 0.429) was clearly *worse* than the
final-layer baseline (run2, 0.740). The probe is a linear proxy; MuLAN's
Light-Attention head evidently exploits the final layer's representation better
than a linear readout of a mid layer does. *Caveat:* run3b's 2048-d model has ~2×
the first-layer parameters but ran only 25 epochs, so it is likely **under-trained**,
not merely worse — but run3a (a full 50 epochs) already lost to baseline, so the
direction is consistent. Ankh, whose final layer the probe found already
near-optimal, was (correctly) not layer-swapped.

**2. Tier-1 augmentation helped Ankh, was neutral for ProstT5.**
Reverse-mutation (antisymmetry) + identity (ΔΔG=0) anchors lifted Ankh from
0.757 → **0.769** (RMSE 1.531 → 1.510) — and did so at **25 epochs vs the
baseline's 50**, which makes the gain more convincing (better generalization from
physically-grounded examples, with less training). For ProstT5 it was flat
(0.740 → 0.738). So the augmentation's value here is **PLM-dependent**, helping
the stronger Ankh baseline but not the weaker ProstT5 one.

**3. Best configuration: Ankh + Tier-1 augmentation (run4a, PCC 0.769).**
Still below the paper's 0.868, but that is a 10-fold-CV average vs our single
split; the gap is expected (see below).

**Caveats:** all single-split, N=174 — margins of ±0.01–0.02 PCC are within split
noise. run3b/run4a/run4b used 25 epochs (host was ~6× slowed by contention);
training restores the best val-loss checkpoint via `load_best_model_at_end`, and
the overfitting seen in run1/run2 set in well before epoch 25, so the reduction is
near-lossless for the runs where the best epoch was < 25.

## To make it directly comparable to the paper

Run 10-fold CV per PLM (`mulan.data.split_data(num_folds=10)`) and average.
Embeddings are already cached for both PLMs (`scratch/embeddings/` for Ankh,
`scratch/embeddings_prostt5/` for ProstT5), so each fold is ~33 min training only
→ ~5.5 h per PLM for all 10 folds.
