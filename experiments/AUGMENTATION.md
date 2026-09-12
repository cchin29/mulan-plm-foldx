# Augmentation strategies for MuLAN ΔΔG (S1102)

> Consolidated entry point for all results: [`../docs/RESULTS.md`](../docs/RESULTS.md). This
> is a deep-dive sub-doc.

## Why augment

From-scratch training on the S1102 split overfits: run2 (ProstT5) showed
train_loss ≈ 3.0 but a held-out test PCC of only 0.740, and run1 (Ankh) 0.757 —
both on just **769 training mutations**. Augmentation attacks this from two
directions: **data augmentation** (more / transformed training examples) and
**feature augmentation** (richer inputs). The reference target is the paper's
10-fold-CV PCC 0.868 / RMSE 1.185.

Two families, organized by value-for-effort. Status legend:
✅ implemented · 🧪 experiment in flight · 💡 idea (not yet built) · ⏸ deprioritized.

---

## Tier 1 — physically exact, cheap, high value (data augmentation)

**1.1 Reverse-mutation (antisymmetry).** ✅🧪
ΔΔG is antisymmetric: `ΔΔG(wt→mut) = −ΔΔG(mut→wt)`. For every training mutation,
add the reverse example with the label negated. Canonical in ΔΔG/stability work
(ThermoNet, DDGun) because models otherwise learn a sign bias.
*Why cheap here:* MuLAN already computes `mut − wt`, so the reverse example reuses
the **mutant** embeddings as the "wild type" and vice-versa — both already cached.
Doubles the set (769 → +769) and bakes in a correct physical constraint.
*Implementation:* `augment.py` adds the mutated chain as a FASTA entry whose label
matches MuLAN's internal mutant label, plus a row mutating it back. Validation/test
are left untouched (evaluation stays on real forward mutations).

**1.2 Identity anchors (ΔΔG = 0).** ✅🧪
Add `wt→wt` no-op "mutations" with label 0 to anchor the regression's zero point
and reinforce antisymmetry. `augment.py` adds one per complex (+91 rows on the
train split). Cheap, reuses wild-type embeddings.

> In flight as **run4a** (Ankh + 1.1/1.2) and **run4b** (ProstT5 baseline + 1.1/1.2),
> and folded into **run3c** (best ProstT5 layer + 1.1/1.2). See the matrix below.

---

## Tier 2 — the architecture already supports it (feature augmentation)

**2.1 Physics-score channel — FoldX binding ΔΔG.** ✅🧪
MuLAN's built-in `add_scores` path (`config.add_scores` + a `zs_scores` column →
concatenated before the final linear layer) fed a physics-based **FoldX 5.1 binding
ΔΔG** (RepairPDB → BuildModel repack → AnalyseComplex, mut−WT). Built for all of
S1102 (110/111 complexes, 100% coverage; total + 12 decomposed terms). Two forms:
- **Stage 1 — scalar.** The single Interaction-Energy scalar into the +1 head slot
  (`lightatt_addscores_config.json`). Ankh, paper folds (300 ep / patience 30):
  **0.8217 → 0.8310, paired ΔPCC +0.0093, 8/10 folds** — landing exactly on a
  pre-training out-of-fold linear ceiling (+0.008–0.009).
- **Stage 2 — decomposed MLP.** The 12 decomposed terms → a small MLP
  (12→16→ReLU→Dropout→1, +226 params) → the *same* head slot (`zs_mlp:true`,
  `zs_input_dim:12`, `lightatt_addscores_mlp_config.json`). Ankh, same folds:
  **0.8383, paired ΔPCC +0.0166 (~2× the scalar), 8/10 folds** (MLP-over-scalar
  +0.0073 is within fold noise). Refuted an RSM forecast that a jointly-trained MLP
  would overfit below the scalar — the regularized in-network head doesn't.

**Findings (see `../docs/RESULTS.md` §18 / §18.1 / §18.2; deep-dive `scratch/foldx_s1102/`):**
FoldX is a **floor-raiser, not a lever** — a real but small, largely-redundant signal
that helps weak folds and *hurts* the strongest (corr(baseline PCC, ΔPCC) = −0.61),
inside per-fold noise. Benefit runs along two axes: **(1)** weaker baseline → larger
gain; **(2)** sequence-only PLMs gain more than structure-aware ones (SaProt already
ingests structure → FoldX is redundant to it). Only **Ankh** is measured so far;
§18.2 predicts **ESM2-3B** (weakest seq-only) gains most and **SaProt / ESM C 6B**
least. Stage 2 is the arm to report if FoldX is used.
*Cost:* FoldX computed once, model-independent (reused across PLMs).

> **In flight (2026-07):** a **300 ep / patience 30 balanced-splitter Stage-2 sweep**
> across the balanced-plot PLM roster (Ankh, ESM2-3B, ProstT5, SaProt-650M/1.3B, ESM
> C 600M, Ankh3-large/xl) — the first multi-PLM test of the §18.2 prediction. These
> land as **Tier-2 augmentation points** on `scripts_plots/ddg_scaling_300ep_balanced`
> (distinct from the Tier-1 reverse+identity aug points). Balanced decomposed splits:
> `python experiments/foldx_s1102/merge_foldx_decomposed.py --src experiments/embedding_sweep/splits/balanced_seed42 --out balanced`.

**2.2 Embedding-space regularization.** 💡
Cheap label-preserving noise on the cached embeddings during training: Gaussian
jitter or random **residue dropout** (zero out a fraction of per-residue vectors).
Acts like augmentation views with no new compute (one addition to the collator).
Expect a mild gain (MuLAN already has dropout). Not yet built.

---

## Tier 3 — richer representations (feature augmentation)

**3.1 Multi-/mid-layer PLM embeddings.** ✅🧪
Instead of the default `last_hidden_state`, use a more informative hidden layer or
a concatenation of layers. Motivated by the **layer probe** (`LAYER_PROBE.md`):
ProstT5's final layer is *weak* for the local ΔΔG signal (site PCC 0.702 vs ~0.765
mid), while Ankh's final layer is already near-optimal — so this is worth testing
for ProstT5 but not Ankh.
*Implementation:* `gen_layer_embeddings.py` writes MuLAN-format embeddings from any
layer(s); `LazyConv1d` adapts to the resulting width for free.

> In flight as **run3a** (ProstT5 layer 7, 1024-d) and **run3b** (ProstT5 concat
> 7⊕10, 2048-d).

**3.2 Multi-PLM concatenation.** 💡
Stack per-residue embeddings from different PLMs (e.g. Ankh ⊕ ProstT5 [⊕ ESM]);
different PLMs carry complementary signal, and `LazyConv1d` absorbs any width.
Higher cost (multiple encoders). Not yet built.

**3.3 Learned scalar-mix over layers.** 💡
A learned per-layer weighted sum of all hidden states (ELMo/SeqVec-style) — the
principled "use all layers" alternative to a fixed concat; keeps dim constant.
Not yet built.

---

## Tier 4 — weak or risky here (considered, deprioritized) ⏸

- **Chain-order swap** (swap seq1/seq2 + remap mutation chain A↔B): MuLAN's
  combination ops (product, abs-difference) are already ~symmetric, so little gain.
- **Homolog / MSA substitution** (replace wt with close orthologs): the ΔΔG label
  does not truly transfer to a different complex — label drift makes it risky.
- **Test-time augmentation** (average over reverse / chain-swap views at inference):
  a small, safe variance reducer to apply *after* a model is trained, not training
  augmentation.

---

## Experiments in flight

All on the **same 769/157/174 split** as run1/run2; fresh `LightAttModel`
(`lightatt_default_config.json`), 50 epochs, AdamW, batch 32, LR 5e-4,
ReduceLROnPlateau, early-stopping patience 10. Baselines for reference:
**run1 Ankh 0.757 / 1.531**, **run2 ProstT5 0.740 / 1.584**.

| Run | PLM | Tier tested | Embedding | Augmentation | Epochs | Test PCC | RMSE |
|---|---|---|---|---|---|---|---|
| run3a | ProstT5 | 3.1 | layer 7 (1024-d) | none | 50 | 0.648 | 1.782 |
| run3b | ProstT5 | 3.1 | concat 7⊕10 (2048-d) | none | 25 | 0.429 | 2.193 |
| **run4a** | **Ankh** | 1.1+1.2 | last layer (1536-d) | reverse + identity | 25 | **0.769** | **1.510** |
| run4b | ProstT5 | 1.1+1.2 | last layer (1024-d) | reverse + identity | 25 | 0.738 | 1.611 |
| run3c | ProstT5 | 1.1+1.2 **×** 3.1 | — | — | — | ⏸ skipped | — |

Baselines: run1 Ankh 0.757 / 1.531 · run2 ProstT5 0.740 / 1.584 (both 50 ep).

What each showed:
- **run3a / run3b** — a better ProstT5 *layer* did **not** help; layer 7 (0.648)
  and concat 7⊕10 (0.429) both lost to the final-layer baseline (0.740). The
  layer-probe ranking did not transfer to MuLAN's attention head (Tier-3 negative
  for ProstT5). run3b also under-trained at 25 ep (2× params).
- **run4a** — Tier-1 augmentation **helped the strong Ankh baseline**:
  0.757 → 0.769, even at half the epochs (25 vs 50). ✅
- **run4b** — Tier-1 augmentation was **neutral for the weaker ProstT5 baseline**:
  0.740 → 0.738. So the gain is PLM-dependent.
- **run3c** — ⏸ skipped: "best layer + aug", but both ProstT5 layer variants lost
  to the baseline last layer, so the best ProstT5 config *is* the baseline — making
  run3c identical to run4b.

**Best overall:** run4a (Ankh + reverse/identity augmentation), PCC 0.769.

Tools: `augment.py`, `gen_layer_embeddings.py`. Consolidated in
`../docs/history/PLM_COMPARISON_S1102.md`. Raw artifacts in the gitignored
`scratch/results/run3*/`, `run4*/`.
