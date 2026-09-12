# ProstT5 as a MuLAN embedding — results

> Consolidated entry point for all results: [`../docs/RESULTS.md`](../docs/RESULTS.md). This
> is a deep-dive sub-doc.

What we learned from using **ProstT5** (`Rostlab/ProstT5`, amino-acid mode) as the
sequence embedder for MuLAN ΔΔG prediction on the SKEMPI **S1102** benchmark.
Companion to `LAYER_PROBE.md`, `AUGMENTATION.md`, and `../docs/history/PLM_COMPARISON_S1102.md`.

All runs: same mutation-based 70/15/15 split (769/157/174), fresh `LightAttModel`
(`lightatt_default_config.json`), AdamW / batch 32 / LR 5e-4 / ReduceLROnPlateau /
early-stop patience 10, CPU-only. Single split — margins of ±0.01–0.02 PCC are
within split noise.

## How ProstT5 is wired in

- Added as a selectable PLM (`prostt5` → `Rostlab/ProstT5`), commit `13e05cf`.
- Loaded as a `T5EncoderModel` + `T5Tokenizer(do_lower_case=False)`.
- Run in **amino-acid mode**: each sequence gets the `<AA2fold>` prefix and is
  space-separated; the prefix token (not flagged by `special_tokens_mask`) is
  sliced off so the saved embedding is exactly per-residue, **1024-dim**.
- MuLAN's `nn.LazyConv1d` adapts to 1024-d automatically — no architecture change.
- ProstT5 is *bilingual* (amino-acid ↔ 3Di structure tokens); we use AA mode only,
  since MuLAN's pipeline is purely sequence-based (no structure input).

## Results

| Run | ProstT5 embedding | Augmentation | Epochs | Test PCC | RMSE | MAE |
|---|---|---|---|---|---|---|
| run2 | last layer (24), 1024-d | none | 50 | **0.740** | 1.584 | 1.160 |
| run3a | layer 7, 1024-d | none | 50 | 0.648 | 1.782 | 1.284 |
| run3b | concat 7⊕10, 2048-d | none | 25 | 0.429 | 2.193 | 1.473 |
| run4b | last layer, 1024-d | reverse + identity | 25 | 0.738 | 1.611 | 1.187 |
| run6a | learned scalar-mix, all 25 layers, 1024-d | none | 50 | 0.685 | 1.745 | 1.276 |
| run6c | AA ⊕ real-WT-3Di concat, 2048-d | none | 44 (early-stop) | 0.663 | 1.788 | 1.307 |

For reference, **Ankh-large** (the paper's PLM) on the same split: **0.757 / 1.531**
(run1). Paper's MuLAN-Ankh on S1102: 0.868 / 1.185, but that is a 10-fold-CV
average, not a single split.

## Findings

1. **ProstT5 (AA mode) is a working but slightly weaker backend than Ankh** here:
   0.740 vs 0.757. The gap is within single-split noise, not a robust ranking.
   Plausibly ProstT5's distinctive value — its 3Di/structure channel — is unused in
   this sequence-only setup, and its AA-mode embeddings are 1024-d vs Ankh's 1536.

2. **Layer selection HURT (a negative result).** The layer probe (`LAYER_PROBE.md`)
   showed ProstT5's final layer is the *weakest* contextualized layer for the local
   mutation signal (linear site-PCC 0.702 vs ~0.765 mid-stack), because the top
   layer specializes to ProstT5's AA↔3Di translation objective. But in full MuLAN,
   embedding from layer 7 (0.648) or concatenating layers 7⊕10 (0.429) was clearly
   *worse* than the final-layer baseline (0.740). **The linear probe's ranking did
   not transfer** to MuLAN's Light-Attention head, which evidently exploits the
   final layer's representation better than a linear readout of a mid layer.
   *Caveat:* run3b's 2048-d model has ~2× first-layer params but ran 25 epochs, so
   it is likely under-trained too — but run3a (full 50 ep) already lost, so the
   direction holds.

3. **Tier-1 augmentation was neutral for ProstT5** (0.740 → 0.738), whereas the
   same reverse-mutation + identity augmentation *helped* Ankh (0.757 → 0.769).
   So the augmentation's benefit is PLM-dependent and did not rescue ProstT5.

4. **Learned scalar-mix over all 25 layers also HURT (run6a, 0.685).** The
   principled alternative to a fixed layer — an ELMo/SeqVec mix
   `γ · Σ_ℓ softmax(w)_ℓ · h_ℓ` over every hidden state, trained end-to-end
   (`mulan.modules.LayerMix`, width held at 1024) — came in *below* the run2
   last-layer baseline (0.740), between run2 and the fixed-mid-layer run3a (0.648).
   The trained weights tell the story: the softmax stayed **near-uniform** (entropy
   3.20 vs 3.22 nats; final-layer weight 0.046 = same as the earliest layers, γ 0.79),
   so the mix degenerated to ≈ a plain mean over the 25 layers — diluting the
   final-layer signal MuLAN's head reads best. The gradient to the layer weights was
   too weak to make it specialize back onto the final layer. **H-mix is not
   supported:** letting training weight the layers did not recover the mid-stack
   signal the *linear* probe found — it washed it out, the same failure direction as
   run3a/b. (All-layers cache `[L, 25, 1024]` via `gen_layer_embeddings.py
   --all-layers`; driver `scratch/run6a_driver.sh`; full 50 ep, ~6.7 h, the 25×
   per-sample I/O from the all-layers cache dominating runtime.)

5. **Structure (3Di) channel HURT (run6c, 0.663) — H-struct not supported.** The
   headline structure experiment: concatenate AA-mode (1024) with ProstT5
   structure-mode 3Di (1024) -> 2048-d, full 50 ep (early-stopped 44, so *not*
   under-trained like run3b). PCC **0.663 < run2 0.740**. Real WT 3Di came from the
   on-disk SKEMPI PDBs via `mini3di` (Foldseek-3Di-compatible), index-aligned to the
   AA fasta numbering (`gen_3di.py`), embedded in ProstT5 fold mode (`<fold2AA>`,
   `gen_struct_embeddings.py`), concatenated with the run2 AA cache
   (`concat_aa_3di.py`). The decisive limitation is the **mutant-structure
   approximation**: lacking mutant structures, each mutant reuses its WT chain's 3Di,
   so the 3Di channel is *identical* across the mutation — a constant complex-level
   offset in `mut - wt`, no mutation-specific signal — while the doubled width dilutes
   the AA signal that worked. Structure, fed this way, is worse than neutral. Genuine
   structural value would require *modeled mutant* structures (FoldX/AF). Driver
   `scratch/run6c_driver.sh`. (run6b 3Di-only was skipped — degenerate under fixed WT
   3Di: `mut_emb == wt_emb` -> zero signal by construction.)

## Runtime / footprint (CPU, vs Ankh)

ProstT5 is *cheaper* than Ankh despite a marginally larger encoder, because its
1024-d embeddings (vs Ankh's 1536) flow into a smaller first conv layer:
embedding generation ~19 min (1444 seqs), training ~33 min, cached embeddings
683 MB, `model.ckpt` ~8 MB. Encoder ~1.21 B params, ~4.9 GB peak RAM (fp32).
Full table in `../docs/history/PLM_COMPARISON_S1102.md`.

## Open next steps for ProstT5

- ~~**10-fold CV** for run2 vs Ankh run1 — the only way to turn the 0.740-vs-0.757
  single-split hint into a defensible ranking.~~ **Done (`RESULTS_CV10.md`):** Ankh
  **0.832 ± 0.057** vs ProstT5 **0.805 ± 0.055**; paired, Ankh wins **10/10 folds**
  (Δ +0.028 PCC, p ≈ 0.001). The single-split "within-noise" hint becomes a robust
  ranking — **Ankh > ProstT5** in the sequence-only setup.
- ~~**Structure mode**: feed ProstT5 actual 3Di tokens to use the channel it was
  built for — the most likely way ProstT5 would beat Ankh.~~ **Done (run6c):
  negative.** PCC 0.663 < run2 0.740 (Finding 5). Real WT 3Di concatenated with AA
  (2048-d) hurt — with the backbone held fixed across the mutation the structure
  channel carries no mutation signal and dilutes the AA signal. **Three further fusions
  also failed** (`../docs/history/PROSTT5_STRUCTURE_OPTIONS.md`): a gated concat (E1, 0.705), pooled
  3Di as head-level context outside the `mut−wt` difference (B1, 0.673), and
  interface-contact attention biasing (C3, 0.662, learnable α→≈0). Every structure knob
  trained to neutral/off; the only remaining angle is *modeled mutant* structures
  (FoldX/AF), a much larger lift.
- ~~**Learned scalar-mix** over all 25 layers instead of a fixed mid-layer/concat —
  the principled "use all layers" alternative that the probe's flat plateau hints
  at, and which avoids the failure mode of picking one wrong layer.~~ **Done
  (run6a): negative.** PCC 0.685 < run2 0.740 (see Finding 4). The learned mix
  stayed ≈uniform and degenerated to a layer mean, diluting the final-layer signal.
  H-mix is not supported; a γ-only / temperature-sharpened or final-layer-biased
  init variant is the only remaining angle.
- **All run6 enhancement bets failed** — the scalar-mix (run6a) and every structure
  fusion (run6c, E1, B1, C3; `../docs/history/PROSTT5_STRUCTURE_OPTIONS.md`) lose to AA-only run2
  0.740. ProstT5's ceiling in MuLAN's head is real; priority redirects to **AIDO**
  (`docs/history/PLAN_AIDO.md`, `../docs/RESULTS.md` §8).
