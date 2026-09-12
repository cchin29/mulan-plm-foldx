# Layer-wise probe: which PLM layers carry S1102 ΔΔG signal?

> Consolidated entry point for all results: [`../docs/RESULTS.md`](../docs/RESULTS.md). This
> is a deep-dive sub-doc.

**Question.** MuLAN embeds sequences with a PLM's `last_hidden_state` (the final
encoder layer). But the final layer of a PLM is specialized to its *pretraining*
objective — for ProstT5, amino-acid↔3Di structure translation; for Ankh, span
denoising — which can shed the local per-residue signal a ΔΔG model wants. Do
mid-depth layers carry more ΔΔG signal than the default final layer?

**Method.** `layer_probe.py`. For each of a PLM's hidden states (input embeddings
+ every encoder layer), build a per-mutation feature from the mutated chain:
- **site delta** = `mut_emb[pos] − wt_emb[pos]` at the mutated residue (local)
- **pool delta** = `mean(mut_emb) − mean(wt_emb)` over the chain (global)

Fit a linear `RidgeCV` probe (standardized, 5-fold CV) per layer and report
held-out PCC. This is a **linear proxy** for MuLAN's Light-Attention head, so
absolute PCCs are below full MuLAN — the **ranking across layers** is the point.

N = 1100 single mutations (S1102 minus `2I9B`), same data as run1/run2.
Raw tables: `results/probe_prostt5.md`, `results/probe_ankh.md`.

## ProstT5 (24 layers, 1024-d) — done

| | Best layer | PCC | Final layer 24 (default) |
|---|---|---|---|
| site delta (local) | **3** | 0.765 | **0.702** |
| pool delta (global) | **10** | 0.754 | 0.731 |

- Layer 0 (raw, no context): 0.377 — useless alone, as expected.
- Layers ~2–21: a broad plateau (~0.74–0.765); differences inside it are within
  probe noise.
- Top layers 22→24 **decay for the local signal**: 0.730 → 0.722 → **0.702**.

**Takeaway:** ProstT5's `last_hidden_state` (layer 24) is the *weakest*
contextualized layer for the local mutation signal — exactly what run2 used. A
mid layer (≈3–7 for local, ≈10 for global) carries more. The real message of the
flat plateau is **"a mid layer ≥ the final layer,"** not "layer 3 specifically."
This motivates run3a (swap to a mid layer) and run3b (concat a local + a global
layer, e.g. 7⊕10).

## Ankh-large (48 layers, 1536-d) — done

| | Best layer | PCC | Final layer 48 (default) |
|---|---|---|---|
| site delta (local) | 47 | 0.799 | **0.796** |
| pool delta (global) | 14 | 0.775 | 0.751 |

Raw table: `results/probe_ankh.md`. (Run with the command in `README.md`.)

- Layer 0 (raw): 0.377, as for ProstT5.
- site delta climbs to a high plateau (~0.77–0.80) from layer ~8 and **stays
  high all the way to the final layer** — best is 47 (0.799), but layer 48 (0.796)
  is statistically tied.
- pool delta peaks mid-depth (layer 14, 0.775) and *decays* through the deep
  layers (many 0.66–0.73 in 25–45), while site stays strong — i.e. deep Ankh
  layers keep local discriminability but lose a clean global summary.

**Takeaway — opposite of ProstT5:** Ankh's `last_hidden_state` is already
near-optimal for the local ΔΔG signal, so a layer swap offers little to gain.
This is why run1 (Ankh, final layer, 0.757) is the stronger baseline, and why we
do **not** queue an Ankh mid-layer run — only ProstT5 (run3a/b).

## ProstT5 vs Ankh — why the difference

The final layer reflects each PLM's pretraining objective. Ankh (span-denoising /
masked reconstruction) keeps its top layer general-purpose, so local per-residue
signal survives to the output. ProstT5's top layer is specialized to translate
amino acids ↔ 3Di structure tokens, which discards local AA-identity signal — so
its best ΔΔG representation sits in the middle of the stack.

**Probe validity check:** the probe's PCC ordering (Ankh ~0.80 > ProstT5 ~0.765)
matches the full-MuLAN runs (run1 Ankh 0.757 > run2 ProstT5 0.740), which supports
using it to choose layers.

## Caveats

- Linear proxy ≠ MuLAN's attention head; layer 24's *pool* score is fine (0.731),
  it is specifically the *local* readout that decays. Treat as a strong hint to
  test in full MuLAN training, not a settled conclusion.
- Single 5-fold CV on 1100 points; plateau differences are noise-level.
