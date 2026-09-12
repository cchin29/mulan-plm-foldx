# Tier-3: full-SKEMPI attention → interface head-to-head (attention→interface question)

Per-residue MuLAN LightAtt attention scored (AUROC) vs cross-chain interface labels over the **102 full-SKEMPI complexes** (204 chains), one row per PLM backbone. Unlike the S1102 view, the heads here are the **by-complex leakage-controlled** retrain heads (fold-0), evaluated on held-out complexes → an honest generalization readout. **macro** = average AUROC over sequences; 95% CI from 10k bootstrap over chains. **Δ vs Ankh** = paired bootstrap of the per-chain difference vs Ankh (CI excluding 0 ⇒ significant).

Anchors: internal Ankh (by-complex) = **0.671**; geometric contact-count floor = **0.69**; chance = 0.5. No paper anchor here — the 0.757 Fig-S1 number is S1102/INTBuilder, not this leakage-controlled full-SKEMPI setup. Absolute AUROCs run lower than S1102 (Ankh 0.788 there) precisely because these are held-out complexes.

| rank | backbone | macro AUROC [95% CI] | pooled | Δ vs Ankh [95% CI] | verdict |
|---|---|---|---|---|---|
| 1 | Ankh-large | 0.671 [0.641, 0.700] | 0.658 | — | — |
| 2 | SaProt | 0.628 [0.605, 0.653] | 0.612 | -0.042 [-0.065, -0.020] | below |
| 3 | ESM-C 600M | 0.609 [0.584, 0.634] | 0.610 | -0.062 [-0.080, -0.044] | below |
| 4 | Ankh3-xl | 0.561 [0.539, 0.582] | 0.530 | -0.110 [-0.128, -0.092] | below |
| 5 | ESM-C 6B | 0.518 [0.504, 0.532] | 0.565 | -0.152 [-0.175, -0.129] | below |
| 6 | ProstT5 | 0.516 [0.492, 0.539] | 0.533 | -0.155 [-0.180, -0.129] | below |
| 7 | ESM2-3B | 0.500 [0.478, 0.521] | 0.505 | -0.171 [-0.193, -0.148] | below |
| 8 | Ankh3-large | 0.460 [0.443, 0.476] | 0.448 | -0.211 [-0.235, -0.187] | below |
| 9 | AIDO-16B | 0.396 [0.376, 0.417] | 0.460 | -0.275 [-0.317, -0.232] | below |

**Bottom line (full-SKEMPI, leakage-controlled):** **no backbone significantly beats Ankh**. This corroborates the S1102 finding on the harder held-out set — the newer PLMs do not improve attention-based interface recovery over Ankh-large, and scaling to 16B (ESM-C 6B, AIDO) does not help.

*Caveats:* esmc6b/aido were scored on the Linux/CPU box (healthy embeddings, head-robust to ±0.006). AIDO's head is `full_skempi_bycomplex/aido_base` (its honest-ladder run) rather than `retrain_bycomplex` like the others — both by-complex leakage-controlled, but note the convention. AIDO's 0.396 sits *below chance*; the Ankh 0.671 anchor confirms the pipeline recovers signal on these masks, so it is a genuine low, but treat the sub-chance value as the model's worst-case rather than a precise point estimate. esm3/saprot13b pending full-SKEMPI WT caches.
