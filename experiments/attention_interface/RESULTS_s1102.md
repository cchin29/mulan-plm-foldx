# Tier-2: S1102 attention → interface head-to-head (attention→interface question)

Per-residue MuLAN LightAtt attention scored (AUROC) against cross-chain interface labels on the 220 S1102 wild-type chains, one row per PLM backbone (fold-0 balanced head). **macro** = average AUROC over sequences (the paper's Fig S1 statistic); 95% CI from 10k bootstrap over chains. **Δ vs Ankh** = paired bootstrap of the per-chain AUROC difference vs Ankh-large (CI excluding 0 ⇒ significant).

Anchors: paper Fig S1 Ankh-large = **0.757** (INTBuilder labels); our Ankh reproduces at **0.788** on heavy-atom-8Å labels; geometric contact-count floor = **0.69**.

| rank | backbone | macro AUROC [95% CI] | pooled | Δ vs Ankh [95% CI] | verdict |
|---|---|---|---|---|---|
| 1 | SaProt | 0.799 [0.779, 0.818] | 0.722 | +0.012 [-0.005, +0.028] | ns |
| 2 | ESM-C 600M | 0.796 [0.774, 0.816] | 0.721 | +0.008 [-0.010, +0.026] | ns |
| 3 | Ankh-large | 0.788 [0.766, 0.808] | 0.721 | — | — |
| 4 | SaProt-1.3B | 0.762 [0.745, 0.779] | 0.701 | -0.025 [-0.040, -0.010] | below |
| 5 | Ankh3-xl | 0.674 [0.657, 0.692] | 0.657 | -0.113 [-0.134, -0.092] | below |
| 6 | ESM2-3B | 0.661 [0.647, 0.676] | 0.650 | -0.126 [-0.145, -0.107] | below |
| 7 | Ankh3-large | 0.643 [0.623, 0.663] | 0.598 | -0.145 [-0.168, -0.122] | below |
| 8 | ProstT5 | 0.638 [0.624, 0.653] | 0.642 | -0.149 [-0.166, -0.132] | below |
| 9 | AIDO-16B | 0.619 [0.600, 0.638] | 0.609 | -0.168 [-0.193, -0.143] | below |
| 10 | ESM-C 6B | 0.603 [0.592, 0.614] | 0.630 | -0.185 [-0.204, -0.165] | below |
| 11 | ESM3-1.4B ⚠️ | 0.424 [0.412, 0.437] | 0.508 | -0.363 [-0.390, -0.337] | below |

⚠️ **ESM3**: provenance-verified — the esm3 head trained on these exact `scratch/embeddings_esm3` SDK embeddings, so 0.424 is **real, not a bug**. ESM3's embeddings carry massive-activation outliers (per-chain std ~325 vs ~0.04 for the other PLMs); its attention partially tracks outlier-*magnitude* positions (Spearman(att, ‖emb‖) ≈ +0.22) rather than interface residues, even though the ddG regression head still works (CV PCC ~0.82). A genuine attention↔ddG dissociation; kept out of the headline as an embedding-scale outlier, not a fair interface signal.

**Bottom line:** the newer PLMs do **not** robustly beat Ankh-large on attention-based interaction-site prediction. Only the structure-aware SaProt and ESM-C 600M match/edge it, and the margin is within noise (see Δ-vs-Ankh CIs). Pure-sequence successors (ESM2-3B, ProstT5, Ankh3) score clearly lower. **Scaling to 16B does not help:** ESM-C 6B (0.603) and AIDO-16B (0.619) both land significantly below Ankh — and ESM-C 6B is ~0.19 *below* its own 600M sibling. (esmc6b/aido from the Linux/CPU box; healthy embeddings, head-robust to ±0.006 — not scale outliers like esm3.)

*Fold robustness:* a fold-5 spot-check reproduces the ranking — top cluster SaProt 0.795 / ESM-C 600M 0.799 / Ankh 0.784 (still a 3-way tie), all successors below (SaProt-1.3B 0.748, Ankh3-xl 0.659, ProstT5 0.652, Ankh3-large 0.623, ESM2-3B 0.574). No backbone beats Ankh in either fold; the fold-0 numbers above are not an artifact.
