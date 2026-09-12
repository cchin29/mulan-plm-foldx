# MuLAN results — consolidated

_Last updated: 2026-08-03._

**Canonical results hub for this fork.** Every experimental result lives here in
summary; the detailed sub-docs (linked per section) keep the full method notes,
raw tables, and interpretation. Every result recorded for this fork lives here: a row in the
master table, a short section, and a deep-dive sub-doc where one exists.

Task: predict single-mutation binding free-energy change (ΔΔG) on the **S1102**
benchmark — 1102 single mutations from SKEMPI (a database of mutation-induced
changes in protein–protein binding) — with MuLAN, comparing protein language
models (PLMs) as the sequence embedder. All training runs are **from-scratch**
(randomly-initialized `LightAttModel`, *not* the released `mulan-ankh` checkpoint).
Runs on CPU (central processing unit), not GPU (graphics processing unit), unless
noted.

Reference target — the MuLAN paper (Lombardi & Carbone, bioRxiv 2024.08.24.609515),
Table 1, mutation-based 10-fold cross-validation (CV): **MuLAN-Ankh-large on S1102
→ PCC (Pearson correlation coefficient) 0.868 / RMSE (root-mean-square error)
1.185 kcal/mol**.

Full **paper Table 1** (PCC / RMSE kcal/mol), from the bioRxiv preprint
([2024.08.24.609515](https://doi.org/10.1101/2024.08.24.609515)). Note the
paper reports **S1400**, not the S4169 we sweep, and uses the **released
pretrained-MuLAN** config (our runs are from-scratch, so 0.868 is a ceiling, not a
like-for-like baseline):

| Split | Backbone | S1102 | S1131 | S2003 | S1400 |
|---|---|---|---|---|---|
| Mutation-based | MuLAN-Ankh-large | **0.868** / 1.185 | 0.866 / 1.211 | 0.774 / 1.068 | 0.907 / 1.172 |
| Mutation-based | MuLAN-ESM2-3B | 0.854 / 1.223 | 0.849 / 1.255 | 0.781 / 1.078 | 0.904 / 1.187 |
| Complex-based | MuLAN-Ankh-large | 0.783 / 1.508 | 0.770 / 1.599 | 0.671 / 1.366 | — |
| Complex-based | MuLAN-ESM2-3B | 0.752 / 1.625 | 0.744 / 1.675 | 0.664 / 1.417 | — |

The mutation-based S1102 values (Ankh 0.868, ESM2 0.854) are drawn as dashed-grey
target lines in `scripts_plots/ddg_variability.{png,svg}` and the ESM2/Ankh S1102
reference in `ddg_scaling`.

---

## Master results table

All single-split runs use the *same* mutation-based 70/15/15 split (seed 42 →
769 train / 157 val / 174 test), so the PLM/embedding/augmentation is the only
changed variable. Single-split margins of ±0.01–0.02 PCC are within split noise.

| Run | PLM | Embedding (dim) | Augmentation | Eval | Test PCC | RMSE | MAE | Runtime | Peak RAM | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| Paper | Ankh-large | last (1536) | — | 10-fold CV | 0.868 | 1.185 | — | — | — | reference, released config |
| Exp.1 | Ankh-large | last (1536) | — | direct inference | 0.967 | 0.723 | — | — | 4.7 GB | pretrained ckpt, **likely leakage** — not valid |
| **run1** | **Ankh-large** | last (1536) | none | single split | **0.757** | 1.531 | 1.139 | ~48 min | 4.7 GB | from-scratch baseline |
| run2 | ProstT5 (AA) | last (1024) | none | single split | 0.740 | 1.584 | 1.160 | ~33 min | 4.9 GB | from-scratch baseline |
| run3a | ProstT5 (AA) | layer 7 (1024) | none | single split | 0.648 | 1.782 | 1.284 | ~3.5 h◇ | 4.9 GB | mid-layer swap — **hurt** |
| run3b | ProstT5 (AA) | concat 7⊕10 (2048) | none | single split | 0.429 | 2.193 | 1.473 | ~3.9 h◇ | 4.9 GB | multi-layer concat — **hurt** (also under-trained, 25 ep) |
| run6a | ProstT5 (AA) | learned scalar-mix, all 25 (1024) | none | single split | 0.685 | 1.745 | 1.276 | ~6.7 h◇ | 4.9 GB | mix stayed ≈uniform (≈ layer mean) — **hurt** |
| run6c | ProstT5 (AA+3Di) | AA⊕3Di concat (2048) | none | single split | 0.663 | 1.788 | 1.307 | ~57 min | 4.9 GB | real WT 3Di structure channel (fixed across mutation) — **hurt** |
| run6-E1 | ProstT5 (AA+3Di) | AA⊕3Di + learned gate (2048) | none | single split | 0.705 | 1.679 | 1.239 | ~1.7 h◇ | 4.9 GB | gate→0.46: down-weighting structure recovers ½ the gap — still **hurts** |
| run6-B1 | ProstT5 (AA+3Di) | AA siamese + WT 3Di head context | none | single split | 0.673 | 1.707 | 1.223 | ~29 min | 4.9 GB | structure as complex context (outside mut−wt) — **hurt** |
| run6-C3 | ProstT5 (AA+iface) | AA + interface-bias attention (1025) | none | single split | 0.662 | 1.793 | 1.304 | ~22 min | 4.9 GB | learnable interface-focus α→0.02 (≈off); no benefit |
| **run4a** | **Ankh-large** | last (1536) | reverse + identity | single split | **0.769** | **1.510** | 1.129 | ~52 min | 4.7 GB | **best single-split**, 25 ep |
| run4b | ProstT5 (AA) | last (1024) | reverse + identity | single split | 0.738 | 1.611 | 1.187 | ~35 min | 4.9 GB | aug neutral for ProstT5, 25 ep |
| cv10_ankh | Ankh-large | last (1536) | none | 10-fold CV | **0.832 ± 0.057** | 1.310 | 0.944 | ~8.5 h† | 4.7 GB | robust winner |
| cv10_prostt5 | ProstT5 (AA) | last (1024) | none | 10-fold CV | 0.805 ± 0.055 | 1.426 | 1.023 | ~8.5 h† | 4.9 GB | |
| cv10_aug_ankh | Ankh-large | last (1536) | reverse + identity | 10-fold CV | **0.838 ± 0.054** | 1.286 | 0.921 | ~15.2 h‡ | 4.7 GB | aug helps under CV (+0.006, 8/10 folds) |
| cv10_aug_prostt5 | ProstT5 (AA) | last (1024) | reverse + identity | 10-fold CV | 0.819 ± 0.060 | 1.386 | 0.987 | ~15.2 h‡ | 4.9 GB | aug helps more for ProstT5 (+0.014, 8/10) |
| **cv10_esmc6b** | **ESM C 6B** | last (2560) | none | 10-fold CV | **0.859 ± 0.069** | **1.176** | **0.853** | ~8 h | ~29 GB | **best PLM — beats Ankh 9/10 folds (§11)** |
| **cv10_aug_esmc6b** | **ESM C 6B** | last (2560) | reverse + identity | 10-fold CV | **0.878 ± 0.043** | **1.122** | **0.809** | ~17.5 h | ~29 GB | **best result overall — aug +0.019 PCC, 8/10 folds (§5a)** |
| cv10_esmc600m | ESM C 600M | last (1152) | none | 10-fold CV | 0.831 ± 0.056 | 1.310 | 0.951 | ~43 min ◆ | — | ESM C's small sibling (esm SDK, 1152-d); ≈ Ankh-large / Ankh3-xl, well below ESM C 6B (0.859); MPS (§5a) |
| cv10_aug_esmc600m | ESM C 600M | last (1152) | reverse + identity | 10-fold CV | 0.828 ± 0.060 | 1.331 | 0.958 | ~1.2 h ◆ | — | **only PLM where aug does NOT help** (−0.003 paired, 3/10 folds, n.s.); contrast 6B +0.019; MPS (§5a) |
| **cv10_esm2** | **ESM2-3B** | last (2560) | none | 10-fold CV | **0.816 ± 0.054** | 1.395 | 1.006 | ~8.0 h | ~12 GB | **≈ ProstT5 (tied), below Ankh — not the weakest PLM (§12)** |
| cv10_esm2_aug | ESM2-3B | last (2560) | reverse + identity | 10-fold CV | **0.824 ± 0.061** | 1.331 | 0.943 | ~2.5 h ◆ | ~12 GB | +Tier-1 aug (+0.008, 7/10); MPS (§5a) |
| cv10_ankh3_large_nlu | Ankh3-large | last (1536) | none | 10-fold CV | 0.823 ± 0.057 | 1.340 | 0.957 | ~1 h ◆ | — | [NLU] prefix; ≈ Ankh-large−0.01, **first MPS run** (§13) |
| cv10_ankh3_xl_nlu | Ankh3-xl | last (2560) | none | 10-fold CV | 0.831 ± 0.056 | 1.315 | 0.944 | ~1.7 h ◆ | — | [NLU] prefix; ≈ Ankh-large, below ESM C (§13) |
| cv10_ankh3_large_aug_nlu | Ankh3-large | last (1536) | reverse + identity | 10-fold CV | **0.849 ± 0.042** | 1.265 | 0.885 | ~2 h ◆ | — | [NLU] +Tier-1 aug (+0.026); MPS (§13) |
| cv10_ankh3_xl_aug_nlu | Ankh3-xl | last (2560) | reverse + identity | 10-fold CV | **0.855 ± 0.043** | 1.243 | 0.885 | ~3.4 h ◆ | — | [NLU] +Tier-1 aug (+0.024), best Ankh3; MPS (§13) |
| cv10_aido | AIDO.Protein-16B | last (2304) | none | 10-fold CV | **0.828 ± 0.046** | 1.321 | 0.947 | ~7.3 h | ~32 GB (bf16) | **16B MoE ≈ Ankh (tied), below ESM C 6B — scale ≠ win (§14)** |
| cv10_aug_aido | AIDO.Protein-16B | last (2304) | reverse + identity | 10-fold CV | **0.836 ± 0.057** | 1.287 | 0.927 | ~14 h | ~32 GB (bf16) | +Tier-1 aug (+0.008 paired, 7/10 folds); still below ESM C 6B / SaProt (§14) |
| cv10_saprot | SaProt-650M | AA+3Di (1280) | none | 10-fold CV | **0.842 ± 0.050** | 1.276 | 0.926 | ~50 min ◆ | — | **WT-3Di (mini3di); +0.026 vs its #-masked seq-only ctrl (8/10); seq-only ≈ ESM2-3B; > Ankh-large (§15)** |
| cv10_saprot_aug | SaProt-650M | AA+3Di (1280) | reverse + identity | 10-fold CV | **0.852 ± 0.051** | 1.231 | 0.890 | ~1.5 h ◆ | — | WT-3Di +Tier-1 aug (+0.010, 7/10); best SaProt; structure lift stacks with aug; MPS (§15) |
| cv10_saprot13b | SaProt-1.3B (AFDB) | AA+3Di (1280) | none | 10-fold CV | 0.837 ± 0.061 | 1.289 | 0.928 | ~45 min ◆ | — | AFDB_OMG_NCBI ckpt, 66 layers (deeper, not wider — same 1280-d); ≈ SaProt-650M−0.005, no depth gain; MPS (§17) |
| cv10_saprot13b_aug | SaProt-1.3B (AFDB) | AA+3Di (1280) | reverse + identity | 10-fold CV | 0.850 ± 0.065 | 1.231 | 0.877 | ~1.6 h ◆ | — | +Tier-1 aug (+0.013, 8/10); ≈ SaProt-650M+aug−0.002; MPS (§17) |
| foldx_cv10 (base arm) | Ankh-large | last (1536) | none | 10-fold CV, 300ep | 0.822 ± 0.051 | 1.344 | 0.960 | — | — | paired baseline for FoldX add_scores; ◆ MPS (§18) |
| **foldx_cv10 (+FoldX)** | Ankh-large | last + FoldX ΔΔG scalar | none | 10-fold CV, 300ep | **0.831 ± 0.046** | 1.313 | 0.941 | — | — | FoldX binding ΔΔG in add_scores; **paired +0.009, 8/10 — at the +0.011 ceiling (§18)** |
| **foldx_cv10 (+FoldX MLP)** | Ankh-large | last + 12 FoldX terms → MLP | none | 10-fold CV, 300ep | **0.838 ± 0.047** | — | — | — | — | Stage-2 decomposed MLP head; **paired +0.017 vs base (p≈0.02), 8/10 — beats §18 ceiling; +0.007 over scalar within noise (§18.1)** |

_Key: **MAE** = mean absolute error (kcal/mol, like RMSE); **AA** = amino-acid
(ProstT5 is run in amino-acid mode); **ep** = epochs; ± is cross-fold **std**
(standard deviation)._

_**Runtime** = training/eval wall-clock for that run (single-split = training; CV =
end-to-end for the fold set); **embedding generation is cached and counted separately**
(§7). **Peak RAM** = the PLM encoder's embedding-generation peak (fp32 CPU; **bf16** for
AIDO) — the dominant footprint; cv10 **training** itself runs light on cached tensors.
Peak RAM is a per-PLM property, so rows sharing a PLM share it. **†** Ankh + ProstT5
shared one ~8h34m 20-fold cv10 run (§5); **‡** shared one ~15h13m augmented 20-fold run
(§5a). Single-split runtimes are the HF Trainer **`train_runtime`** (measured wall-clock)
from each run's `all_results.json`; **◇** flags runs that ran well above the clean
run1/run2 ProstT5/Ankh baselines (~33/48 min) because of heavier mid-/multi-layer
embedding-cache I/O and/or box load at run time (e.g. run6a is I/O-bound, §3) — elapsed
time, not intrinsic per-epoch cost. **◆** ran on **Apple M4 Pro / 48 GB / MPS** (not the
Linux CPU box), sequential folds — the first CV10 results measured on MPS (§13)._

**Headline findings:**
- **AIDO.Protein-16B — the biggest encoder does *not* win (§14).** The 16B MoE (2304-d),
  ~2.7× ESM C 6B's params and ~14× Ankh's, gives cv10 **PCC 0.828 ± 0.046** —
  **statistically tied with Ankh-large** (paired Δ −0.004, wins 2/10) and **clearly below
  ESM C 6B** (paired Δ −0.031, wins 1/10), while posting the **tightest cross-fold std** in
  the suite (0.046). Sharpest evidence yet that ΔΔG embedding quality tracks **pretraining
  objective/data, not raw parameter count**: the ladder **ESM C 6B (0.859) > {Ankh 0.832 ≈
  AIDO 0.828} > {ESM2-3B 0.816 ≈ ProstT5 0.805}** is non-monotonic in size.
  **+Tier-1 augmentation** lifts AIDO to **0.836 ± 0.057** (paired +0.008, 7/10 folds) —
  the same small, consistent aug gain seen across every PLM (§5a), leaving it still below
  ESM C 6B+aug (0.878) and SaProt+aug (0.852): scale doesn't close the gap under aug either.
- **SaProt-1.3B — going deeper (66 layers, same 1280-d) doesn't beat the 650M SaProt (§17).**
  The AFDB_OMG_NCBI 1.3B checkpoint gives cv10 **PCC 0.837 ± 0.061** base / **0.850 ± 0.065**
  +aug — both **~tied with (slightly below) SaProt-650M** (0.842 / 0.852): the depth-only
  scale-up (2× params, same width) buys nothing on S1102, reinforcing the AIDO/ESM2 finding
  that ΔΔG performance tracks **pretraining data/objective, not parameter count**.
- **ESM C 6B is the new best PLM** (§11): sequence-only, 10-fold CV **PCC 0.859 ±
  0.069**, beating Ankh-large (0.832) by a paired **+0.027 PCC** (t = 4.15,
  p ≈ 0.0025), winning **9/10 folds** and every secondary metric (RMSE 1.176,
  MAE 0.853, SCC 0.777). It's the closest any from-scratch config has come to the
  paper's 0.868 — the gain is model **scale + newer pretraining**, not structure
  (pure AA-mode).
- **ESM2-3B measured (§12) — corrects the earlier prediction.** Running ESM2-3B
  (`facebook/esm2_t36_3B_UR50D`, same 2560-d width as ESM C 6B) through the identical
  cv10 gives **PCC 0.816 ± 0.054** — **not** the weakest PLM as previously guessed.
  It ties ProstT5 (0.805; paired +0.011, 5/10 folds — within noise) and sits below
  Ankh (paired −0.017, wins 2/10) and well below ESM C 6B (−0.043, wins 1/10). So the
  measured ranking is **ESM C 6B (0.859) > Ankh (0.832) > ESM2-3B (0.816) ≈ ProstT5
  (0.805)** — a two-tier picture (ESM C 6B / Ankh above; ESM2-3B / ProstT5 a tied
  bottom pair), **not** the clean "+0.03 per step" ladder once assumed. The ESM C 6B
  → ESM2-3B gap (+0.043) shows the jump is **newer pretraining (Cambrian), not just
  parameter count** — 6B vs 3B at identical width, yet the bigger step is ESM C's
  training recipe, not size.
- **Ankh > ProstT5** for sequence-only ΔΔG, and the ranking is *robust* under
  10-fold CV (Ankh wins **10/10 folds**, paired Δ +0.028 PCC, p ≈ 0.001) even
  though the single-split gap (0.757 vs 0.740) was within noise. **Confirmed across
  the standard SKEMPI benchmarks** (§10): Ankh beats ProstT5 on S1131/S4169/S2003 by
  +0.027 to +0.041 Pearson, winning 8–10/10 folds — the ranking generalizes beyond
  S1102. On the same three benchmarks, **SaProt (WT-3Di) tops the sub-6B field** (0.847 / 0.815
  / 0.832, +0.009–0.025 over Ankh) and **ESM2-3B sits in the Ankh tier** — the S1102
  ordering holds throughout (§10, §15). **ESM C 6B, benchmarked on Linux, now sweeps all
  three sets** (S1131 0.867, S4169 0.825, S2003 0.852 — leading SaProt by +0.020/+0.010/
  +0.020); its +aug benchmarks are landing (S1131 aug complete, 0.872; §10).
- **Tier-1 augmentation** (reverse-mutation antisymmetry + ΔΔG=0 identity anchors)
  **helped Ankh** (0.757 → 0.769, at half the epochs) but was **neutral for
  ProstT5** (0.740 → 0.738) — PLM-dependent.
- **Mid/multi-layer ProstT5 embeddings hurt** in full MuLAN despite the layer
  probe favoring them — the linear probe's ranking did not transfer to MuLAN's
  attention head. A **learned scalar-mix over all 25 layers (run6a)** also hurt
  (0.685): training left the layer weights ≈uniform, so it averaged to a blurred
  embedding rather than recovering the final-layer signal run2 uses.
- **ProstT5's structure (3Di) channel did not help (run6c, 0.663 — hurt).** Feeding
  real WT 3Di alongside AA (2048-d concat) underperformed the AA-only baseline:
  with the backbone held fixed across the mutation, the 3Di channel carries no
  mutation-specific signal and the width inflation dilutes the AA signal. With both
  run6a and run6c failing to clear run2's 0.740, **ProstT5's ceiling in this head is
  real** — the effort now redirects to AIDO (§8).
- Best from-scratch config: **Ankh + Tier-1 augmentation, PCC 0.769** (single
  split) / **Ankh 0.832** (10-fold CV). Still below the paper's 0.868 — a
  config/initialization gap (paper uses a different pretrained-MuLAN setup), not a
  protocol artifact.

---

## 1. Paper reproduction

→ deep-dive: `scratch/RESULTS.md` (gitignored)

- **Exp.1 — pretrained checkpoint, direct inference.** Ran shipped
  `models/pretrained/mulan_ankh.ckpt` over all 1100 mutations: **PCC 0.967 /
  RMSE 0.723**. Above the paper's own cross-validated number, as expected for an all-data
  checkpoint evaluated on data it was trained on — the released
  checkpoint was trained on S1102 itself. **Not a valid reproduction** (measures
  memorization); useful only as a pipeline sanity check.
- **Exp.2 — from-scratch, single split.** Fresh `LightAttModel` on the 769/157/174
  split: **PCC 0.757 / RMSE 1.531** (= run1). The fair, leakage-free number;
  underperforms the paper's 0.868 because it's a single split (high variance) and
  trains on fewer examples (769 vs ~990/fold).
- **Data provenance.** `examples/S1102.tsv` ships the benchmark but no wild-type
  (WT) sequences. RCSB (Research Collaboratory for Structural Bioinformatics) FASTA
  does *not* match SKEMPI's numbering (confirmed mismatch on 1A22). Fix: extract
  per-residue sequences from SKEMPI 2.0 cleaned PDB (Protein Data Bank) files
  (`build_wt_fasta.py`), validated exact-match against `examples/sample_mut.txt`.
  110/111 complexes resolved; `2I9B` dropped (SKEMPI-v1-only) → `S1102_filtered.tsv`
  (1100 mutations).
- **Bug fixed (commit `043ac4a`):** `scripts/train.py` called stdlib
  `logging.get_logger()` (nonexistent) instead of `hf_logging.get_logger()` —
  `mulan-train` was non-functional before this.

## 2. PLM comparison — single split (run1 / run2)

→ deep-dive: `history/PLM_COMPARISON_S1102.md`, `experiments/RESULTS_PROSTT5.md`

Ankh-large (1536-d) **0.757 / 1.531** vs ProstT5 AA-mode (1024-d) **0.740 / 1.584**.
Close, within single-split noise — not a robust ranking on its own (resolved by
CV, §5). ProstT5 wired in as a selectable backend (commit `13e05cf`,
`Rostlab/ProstT5`, `<AA2fold>` prefix sliced off → exact per-residue 1024-d);
`nn.LazyConv1d` adapts to width with no architecture change. ProstT5's structure
(3Di, a 3D-structure token alphabet) channel is unused in this sequence-only pipeline.

## 3. Layer-selection experiments (run3a / run3b / run6a)

→ deep-dive: `experiments/LAYER_PROBE.md`, `experiments/AUGMENTATION.md` (Tier 3),
`experiments/RESULTS_PROSTT5.md`

**Negative result for ProstT5.** The layer probe found ProstT5's final layer
weakest for the *local* mutation signal (site PCC 0.702 vs ~0.765 mid-stack), but
in full MuLAN, swapping to layer 7 (run3a, 0.648) or concatenating 7⊕10 (run3b,
0.429) was clearly *worse* than the final-layer baseline (run2, 0.740). The linear
probe's layer ranking did **not** transfer to MuLAN's Light-Attention head. Ankh's
final layer is already near-optimal per the probe, so it was (correctly) not
layer-swapped. Embeddings generated by `experiments/gen_layer_embeddings.py`.

**run6a — learned scalar-mix, also negative (H-mix unsupported).** The principled
"don't pick one wrong layer, let training weight them all" alternative: an
ELMo/SeqVec scalar-mix `γ · Σ_ℓ softmax(w)_ℓ · h_ℓ` over **all 25 hidden states**
(`mulan.modules.LayerMix`), embedding width held at 1024, same split/optimizer as
run2. Result **PCC 0.685 — below the run2 0.740 baseline**, landing between run2 and
the fixed-mid-layer run3a. Inspecting the trained weights explains why: the softmax
stayed **near-uniform** (entropy 3.20 vs 3.22 nats for uniform; final layer weight
0.046, same as the earliest layers), so the mix collapsed to ≈ a plain mean over the
25 layers — which *dilutes* the final-layer representation MuLAN's head exploits
best. The gradient to the layer weights was too weak to make it specialize. So
letting training choose the mix did **not** recover the mid-stack signal the linear
probe saw; it washed it out — the same lesson as run3a/b, now for the "use all
layers" variant. All-layers cache (`[L, 25, 1024]`, 17 GB) via
`gen_layer_embeddings.py --all-layers`; driver `scratch/run6a_driver.sh`; full 50 ep,
~6.7 h (25× per-sample I/O from the all-layers cache dominated). Artifacts:
`scratch/results/run6a_prostt5_layermix/`.

## 4. Augmentation experiments (run4a / run4b)

→ deep-dive: `experiments/AUGMENTATION.md`

**Tier-1, physically exact, cheap:** reverse-mutation (antisymmetry:
ΔΔG(wt→mut) = −ΔΔG(mut→wt)) + identity (ΔΔG=0) anchors, via `experiments/augment.py`
(769 → 1629 train rows; val/test untouched). **Helped Ankh** (0.757 → **0.769**,
RMSE 1.531 → 1.510, at 25 ep vs baseline 50 — gain *and* less training). **Neutral
for ProstT5** (0.740 → 0.738). Tier 2 (zero-shot score channel, embedding-space
regularization) and higher Tier 3 (multi-PLM concat, learned scalar-mix) ideas are
scoped but not yet built — see the sub-doc's tiered roadmap.

## 5. 10-fold cross-validation (cv10)

→ deep-dive: `experiments/RESULTS_CV10.md`

Variance-controlled head-to-head, every mutation tested once
(`split_data(num_folds=10)`, 880/110/110 per fold, 50 max ep, same config).

| PLM | Test PCC (mean ± std) | RMSE | MAE | SCC |
|---|---|---|---|---|
| Ankh | **0.832 ± 0.057** | 1.310 ± 0.118 | 0.944 ± 0.090 | 0.731 ± 0.060 |
| ProstT5 | 0.805 ± 0.055 | 1.426 ± 0.123 | 1.023 ± 0.094 | 0.674 ± 0.051 |

_SCC = Spearman correlation coefficient (rank correlation)._

**Ankh wins all 10/10 folds**; paired Δ(Ankh−ProstT5) = +0.0275 PCC (std 0.0178),
t = 4.65 (df 9 — degrees of freedom; p ≈ 0.001); sign test 10/10 (p ≈ 0.002). The per-PLM std bands
overlap, but that's cross-fold *difficulty* variance, which pairing cancels — the
ranking is real. CV means sit well above the single split (each fold trains on 880
vs 769; the single split's test set was on the harder side). Driver
`scratch/cv10_driver.sh`, aggregator `scratch/cv10_aggregate.py`.

**Runtime:** all 20 folds (Ankh ×10 + ProstT5 ×10) ran in **~8h34m wall-clock**
end-to-end (per `scratch/cv10_driver.log`), with **4 folds running concurrently at
10 CPU threads each** on the 40-core box. Per-fold wall time under that contention
was ~97–134 min (mean ~118) for Ankh and ~68–92 min (mean ~85) for ProstT5 —
slower than the single-job §7 baselines (~48 / ~33 min) because of the 4× sharing.
Embeddings were already cached, so this is training + eval only.

### 5a. Augmented 10-fold CV (cv10_aug)

→ deep-dive: `experiments/RESULTS_CV10_AUG.md`

Same folds/config as above, but each fold's **train** table is augmented with
Tier-1 reverse-mutation + identity anchors (`experiments/augment.py`, 880 → ~1856
rows); val/test stay the original forward mutations, so the eval is directly
comparable to the non-augmented cv10.

| PLM | baseline cv10 | augmented cv10 | paired ΔPCC | folds improved |
|---|---|---|---|---|
| Ankh | 0.832 ± 0.057 | **0.838 ± 0.054** | **+0.006** | 8/10 |
| ProstT5 | 0.805 ± 0.055 | **0.819 ± 0.060** | **+0.014** | 8/10 |
| ESM C 6B | 0.859 ± 0.069 | **0.878 ± 0.043** | **+0.019** | 8/10 |
| ESM2-3B ◆ | 0.816 ± 0.054 | **0.824 ± 0.061** | **+0.008** | 7/10 |
| SaProt (WT-3Di) ◆ | 0.842 ± 0.050 | **0.852 ± 0.051** | **+0.010** | 7/10 |
| AIDO.Protein-16B | 0.828 ± 0.046 | **0.836 ± 0.057** | **+0.008** | 7/10 |
| ESM C 600M ◆ | 0.831 ± 0.056 | 0.828 ± 0.060 | −0.003 | 3/10 |

Augmentation helps **every PLM tested under CV except the smallest ESM C, the 600M**:
the six larger models gain consistently (+0.006 to +0.019, 7–8/10 folds each, small
regressions otherwise), but **ESM C 600M shows no lift** — 0.831 → 0.828, paired
Δ −0.003 ± 0.016, aug wins only **3/10** folds, t = −0.66 (df 9, not significant; one
fold at −0.044 dominates the mean). So the Tier-1 aug benefit is **not universal**: it
vanishes on the 600M even though its **6B sibling posts the suite's largest gain
(+0.019)**, hinting the effect may scale with encoder capacity/quality rather than
being automatic. **ESM C 6B + aug (0.878 ± 0.043) is the best result in the whole
suite** — the largest aug gain (+0.019) *and* the tightest cross-fold std, on the strongest
base PLM (deep-dive `experiments/RESULTS_CV10_AUG_ESMC.md`). The gain is **larger for the weaker ProstT5** (+0.014, 2.3×
Ankh's) — which *flips* the single-split read where aug was neutral for ProstT5
(run4b 0.740 → 0.738). The two **◆ MPS** rows extend the pattern across the scale range:
**ESM2-3B** (+0.008) and, notably, **SaProt (WT-3Di)** (+0.010) — so the Tier-1 aug lift
**stacks on top of SaProt's structure gain** (seq-only ctrl 0.816 → WT-3Di 0.842 →
WT-3Di + aug 0.852), the two effects being roughly additive (§15). Between the original
Ankh/ProstT5 pair, Ankh still wins every metric (RMSE 1.286, MAE 0.921), but the
head-to-head gap **narrows from +0.027 to +0.019 PCC**. Both remain below the paper's
0.868 — augmentation doesn't close the config/init gap. Driver
`scratch/cv10_aug_driver.sh`, cache pre-warm `scratch/cv10_aug_prewarm.sh`
(augmentation adds uncached sequences — identity anchors etc. — so the caches are
populated by one serial pass per PLM before the folds run read-only), aggregator
`scratch/cv10_aug_aggregate.py`.

**Runtime:** ~15h13m wall-clock for all 20 folds (4 concurrent, 10 threads each) —
~1.8× the non-augmented cv10, tracking the ~2.1× larger train set, partly offset by
early-stopping. Embeddings pre-cached, so this is training + eval only.

## 6. Layer probe (analysis, not a training run)

→ deep-dive: `experiments/LAYER_PROBE.md`; raw tables
`experiments/results/probe_{ankh,prostt5}.md`

Linear RidgeCV probe (5-fold) per hidden layer; site-delta (local, at mutated
residue) and pool-delta (global mean) features. A linear proxy for MuLAN's head —
the **ranking across layers** is the point.

| PLM | site best layer / PCC | final-layer site PCC | pool best layer / PCC |
|---|---|---|---|
| ProstT5 (24 layers) | 3 / 0.765 | **0.702** (weakest contextualized) | 10 / 0.754 |
| Ankh (48 layers) | 47 / 0.799 | 0.796 (≈ tied, near-optimal) | 14 / 0.775 |

ProstT5's top layer specializes to AA↔3Di translation and sheds local AA signal;
Ankh's span-denoising keeps its top layer general-purpose. The probe's PLM ordering
(Ankh ~0.80 > ProstT5 ~0.765) matches the full-MuLAN runs — but its *layer* ranking
did not transfer (§3).

## 7. Runtime & footprint (CPU-only)

→ deep-dive: `history/PLM_COMPARISON_S1102.md` (full table), `experiments/RESULTS_ESMC.md`
(ESM C)

**Machine (all runs):** a CPU-only Linux box, 40 cores / 62 GiB, no GPU.
(3.3 TB free). All PLM encoders run in **fp32** on CPU; training is CPU-only. The
cv10 runs share the box at **MAXPAR=4** concurrent folds × 10 threads/job.

| Resource | Ankh-large | ProstT5 | ESM2-3B | ESM C 6B | AIDO-16B |
|---|---|---|---|---|---|
| Embedding generation (1444 seqs) | ~same order¹ | ~19 min | ~37 min⁴ | **69.7 min** (~2.9 s/seq, 40 thr) | **311 min** (~13 s/seq, bf16)⁵ |
| Encoder params / peak RAM (random-access memory; fp32 = 32-bit float) | 1.15 B / 4.7 GB | 1.21 B / 4.9 GB | 2.8 B / ~12 GB | **6 B / ~29 GB** | **16 B MoE / ~32 GB (bf16)** |
| PLM weights download (fp32) | 7.0 GB² | 11 GB² | 11 GB² | **25.4 GB** (6 safetensors shards) | ~60–64 GB² |
| Embedding cache on disk | 1011 MB | 683 MB | 1.7 GB (2560-d) | **1.7 GB** (2560-d) | 1.5 GB (2304-d) |
| Trained `model.ckpt` | ~12 MB | ~8 MB | 19 MB (2560-d) | **19 MB** (2560-d → wider first conv) | 18 MB (2304-d) |
| Single-split training (50 ep, 769) | ~48 min | ~33 min | — (cv10 only) | — (cv10 only) | — (cv10 only) |
| Single-split test/predict (174) | 5.9 s | 4.2 s | — (cv10 only) | — (cv10 only) | — (cv10 only) |
| 10-fold CV wall-clock | ~8.5 h (20 folds w/ ProstT5)³ | (same run) | ~8.0 h (10 folds, 4 conc.) | **~8 h** (10 folds, 4 concurrent) | ~7.3 h (10 folds, 4 conc.) |

¹ run1 didn't log generation time separately. ProstT5 is *cheaper* despite a
marginally larger encoder: 1024-d embeddings (vs 1536) → smaller first conv layer →
fewer params, ~30% faster training, ~⅓ less embedding disk.
² On-disk HuggingFace cache (holds multiple weight formats); loaded RAM footprint is
the "peak RAM" row.
³ Ankh+ProstT5 shared the same ~8h34m 20-fold run (§5). ESM C 6B ran its 10 folds
in ~7h58m (23:49→07:47), training read-only from the cached embeddings (no PLM
load — `data.py:73`), so this is training + eval only. **ESM C peak RAM (~29 GB) is
the embedding-generation step**, which runs once in the isolated venv; cv10 training
itself is light (works on cached 2560-d tensors, like the ESM2-3B width).
⁴ ESM2-3B embedding gen ~37 min is the single `from_table` prewarm pass (2192 s
wall) — it bundles the ~11 GB fp32 download, model load, and all 1444 forward passes
at 40 threads (§12); it runs in the **main** venv (no isolated venv needed — ESM2 is
native to transformers 4.44), unlike ESM C 6B / AIDO. Peak RAM ~12 GB = 2.8 B params
fp32 (~11 GB weights) + short-seq activations; its cv10 (~8.0 h) then trains
read-only from the cache, like the other PLMs.
⁵ AIDO-16B is the heaviest generation (311 min, ~13 s/seq): a 16 B **MoE** run in
**bf16 on CPU** (Skylake bf16-upcast) in an isolated `mulan-aido` venv (§14). Peak RAM
~32 GB is the generation step; cv10 (~7.3 h) trains read-only from the 2304-d cache in
the main venv. The **+aug** cache is not yet built (aug-prewarm was OOM-killed overnight,
§14).

## 8. AIDO.Protein-16B (setup & CPU-bf16 path) — measured, see §14

→ deep-dive: `docs/history/PLAN_AIDO.md`

**Status: DONE** — base cv10 result in §14 (PCC 0.828 ± 0.046); **no GPU needed** (the
16B MoE runs in bf16 on CPU, isolated `mulan-aido` venv). Setup/loading details below.
`genbio-ai/AIDO.Protein-16B` (MoE =
Mixture-of-Experts, 16B, hidden 2304, 36 layers) downloaded to local HuggingFace
(HF) cache (2026-06-21, ~60–64 GB fp32). Confirmed it ships an `auto_map`, so a `transformers`-native
`AutoModel(..., trust_remote_code=True)` path exists (ModelGenerator not mandatory)
and `output_hidden_states` is reachable for a layer probe. **Hypothesis (H1):**
larger-scale MoE pretraining yields richer embeddings → MuLAN PCC above Ankh's
0.769. Can't run on this CPU-only box (needs ≥40 GB GPU). Recommended integration:
**Approach B (offline embeddings)** — write `[L, 2304]` `.pt` tensors on a GPU,
then `mulan-train --embeddings_dir <aido_dir>`; `LazyConv1d` adapts. **run5** numbers
land in the master table above when computed.

## 9. ProstT5 structure channel — 3Di (run6c)

→ deep-dive: `history/PLAN_PROSTT5_STRUCTURE_v2.md`, `experiments/RESULTS_PROSTT5.md`

**The headline structure experiment — negative.** H-struct's bet was that ProstT5
underperforms Ankh *because* the sequence-only pipeline never uses ProstT5's 3Di
structure channel — the one capability distinguishing it from Ankh/ESM. run6c tested
this directly: per residue, concatenate AA-mode (1024) with real-structure 3Di-mode
(1024) → **2048-d**, same split/optimizer as run2, full 50 ep (early-stopped at 44 —
*not* under-trained, unlike run3b). Result **PCC 0.663 — well below run2's 0.740**:
the structure channel **hurt**.

Real WT 3Di was obtained from the on-disk SKEMPI 2.0 PDBs: `mini3di` (pure-Python,
Foldseek-3Di compatible) encodes each WT chain's backbone into 3Di tokens
**index-aligned to the AA fasta's residue numbering** (`gen_3di.py`), then ProstT5
fold-mode (`<fold2AA>`) embeds them (`gen_struct_embeddings.py`), concatenated with the
run2 AA cache (`concat_aa_3di.py`). The **mutant-structure approximation** is the
catch: we lack mutant structures, so each mutant reuses its **WT** chain's 3Di (the AA
channel carries the mutation). That makes the 3Di channel *identical* between the WT and
mutant passes, so in MuLAN's `mut − wt` it contributes only a constant complex-level
offset — no mutation-specific signal — while the doubled width dilutes the AA signal
that worked. Same direction as run3b's concat, now confirmed with full training. The
plan's risk #1 (fixed-mutant-structure washing out) is borne out, and worse than
neutral. Testing genuine structural value would need *modeled mutant* structures
(FoldX/AF) — a much larger lift. Driver `scratch/run6c_prostt5_aa3di/`,
caches `scratch/emb_prostt5_3di` (3Di) + `scratch/emb_prostt5_aa3di` (2048-d).

**Three more structure-fusion fixes, all negative (→ `history/PROSTT5_STRUCTURE_OPTIONS.md`).**
After run6c, three more fusions were tried: **E1** (a learned scalar gate on the 3Di
block) reached 0.705 with the gate at 0.46 — down-weighting structure recovers ~half the
run6c→run2 gap (dilution is real) but structure stays net-negative; **B1** (pooled WT 3Di
fed to the head *outside* the `mut − wt` difference, AA-only siamese) reached 0.673 —
structure-as-context mildly hurts via complex-level overfitting; **C3** (interface contacts
as a learnable attention bias, focusing pooling on the binding interface) reached 0.662 with
the bias strength **α → 0.017 ≈ off** — the model *declined* the interface prior. **All four
fusions (run6c 0.663, E1 0.705, B1 0.673, C3 0.662) lose to AA-only run2 0.740, and every
learnable structure knob trained to neutral/off** — MuLAN's head consistently declines
ProstT5 structure, consistent with 3Di being a backbone descriptor that single-point
mutations barely move.

**Gate (run6 series).** Both ProstT5 enhancement bets failed to clear run2's 0.740 —
the learned scalar-mix (run6a, 0.685, §3) and the structure runs (run6c/E1/B1/C3, ≤0.705). Per
`history/PLAN_PROSTT5_STRUCTURE_v2.md`, ProstT5's ceiling in MuLAN's head is real; priority
redirects to **AIDO** (§8). (run6b 3Di-only was skipped: with fixed WT 3Di it is
algebraically degenerate — `mut_emb ≡ wt_emb` → zero signal; run6d/6e were gated on
run6c showing structure helps, which it did not. The interface-aware lever (C3) was tested
and also negative — `history/PROSTT5_STRUCTURE_OPTIONS.md`.)

## 10. Standard SKEMPI benchmarks — four PLMs (S1131 / S4169 / S2003)

→ deep-dive: `experiments/RESULTS_BENCHMARKS.md` (results), `experiments/BENCHMARK_DATASETS.md`
(dataset definitions + overlaps)

Extends the S1102 comparison to the field's standard SKEMPI single-mutation benchmarks,
mutation-level 10-fold CV, same config as the S1102 cv10 (`lightatt_default`, 50 ep,
early-stop 10). Datasets restricted to single-chain-per-partner complexes (MuLAN's
one-seq-per-partner design, as for S1102): **S1131** (1127, Xiong et al. via GeoPPI), **S4169** (2497
single-chain subset, mCSM-PPI2 via GeoPPI), **S2003** (1124, derived = non-alanine SKEMPI singles). The
three are *nested* — S4169 is the superset, S1131 ⊂ S4169, and S2003 = the non-Ala half of
S4169 (see `BENCHMARK_DATASETS.md`), so the per-dataset numbers are correlated, not
independent. **Pearson r** (± cross-fold std); Spearman ρ / RMSE / MAE in the deep-dive.

| Dataset (size) | ProstT5 | ESM2-3B ◆ | Ankh | SaProt (WT-3Di) ◆ | ESM C 6B |
|---|---|---|---|---|---|
| S1131 (1127) | 0.781 ± 0.050 | 0.822 ± 0.036 | 0.822 ± 0.043 | 0.847 ± 0.040 | **0.867 ± 0.036** |
| S4169 (2497) | 0.764 ± 0.063 | 0.792 ± 0.048 | 0.798 ± 0.042 | 0.815 ± 0.046 | **0.825 ± 0.049** |
| S2003 (1124) | 0.796 ± 0.072 | 0.807 ± 0.068 | 0.823 ± 0.058 | 0.832 ± 0.051 | **0.852 ± 0.038** |

**On S1131, ESM C 6B now leads (0.867 ± 0.036)** — beating SaProt (0.847) by **+0.020** and every
other PLM, mirroring its S1102 win (§11); size + newer pretraining edges out SaProt's structure
channel on this set. **S2003 is now complete: ESM C 6B leads again at 0.852 ± 0.038** (10/10
folds), above SaProt's 0.832 by **+0.020** — the same margin as on S1131. **S4169 is now complete
too: ESM C 6B leads at 0.825 ± 0.049** (10/10 folds), edging SaProt's 0.815 by **+0.010** — so
**ESM C 6B now tops all three benchmark sets** (0.867 / 0.825 / 0.852), a clean sweep matching its
S1102 win (§11). **Among the remaining PLMs, SaProt (WT-3Di) is the strongest** — 0.847 / 0.815 /
0.832, leading Ankh (the strongest sequence-only model here) by **+0.025 /
+0.017 / +0.009** — despite being the **smallest** encoder in the table (650M vs Ankh's 1.15B,
ESM2's 2.8B). This robustly generalizes the S1102 structure lift (§15): input-level AA+3Di fusion
beats every sequence-only PLM *except the 6B ESM C (on all three sets)*. **ESM2-3B lands in the Ankh tier** —
ties Ankh on S1131 (0.822), just below on S4169/S2003, above ProstT5 throughout — mirroring the
S1102 ordering (§12). **Ankh still beats ProstT5 on every set** (the original head-to-head: paired
Δr +0.041 / +0.034 / +0.027, Ankh wins 8–10/10 folds; Spearman agrees). Magnitudes track S1102
(S1131/S2003 land ~0.82–0.87, S4169 a touch lower as the broadest set).

**300ep update (ESM C 6B, GPU box — `results_esmc6b_gpu_20260714`).** The table above is the
**same-budget (50ep/patience-10) cross-PLM comparison** and is kept as-is so the ranking stays
apples-to-apples. Re-running **ESM C 6B alone at the 300ep/patience-30 budget** lifts all three
benchmark sets: **S1131 0.876 ± 0.030 (was 0.867), S4169 0.844 ± 0.039 (was 0.825), S2003
0.870 ± 0.034 (was 0.852)** — base, 10/10 folds each; and **S1131 +aug 0.871 ± 0.026** (10/10).
So the longer budget adds **+0.009 / +0.019 / +0.018** on top of ESM C 6B's already-leading
50ep numbers — it does not change the ranking (ESM C 6B still tops all three), it just raises the
ceiling. These are **not** dropped into the 50ep table because the other PLMs were not re-run at
300ep, so a mixed-budget row would overstate the PLM gap. (S4169 +aug is **1/10 — incomplete**,
so it is withheld; per-fold raw in `scratch/incoming_esmc6b_20260714/`.)

SaProt + ESM2 benchmarks ran on **Apple M4 Pro / MPS** (◆; device-parity-validated, §16);
Ankh + ProstT5 + ESM C 6B on the Linux box (ESM C pulled from a Linux snapshot). Pipeline: `experiments/build_benchmarks.py` (tables, 0
WT-residue mismatches), then per-PLM embeddings + CV (Ankh/ProstT5 `scratch/bench_cv_driver.sh`;
SaProt/ESM2 added via the local queue), `scratch/bench_aggregate.py`. **Caveat:** the single-chain
restriction makes S4169/S2003 MuLAN-compatible *subsets*, not the verbatim published sizes — not
directly comparable to external S4169 leaderboards, but the per-PLM contrast is exact (identical
data per PLM).

**Fairness / leakage (L4).** A split-provenance + leakage audit (`experiments/paper_fairness/`,
`audit_leakage.py`) confirms: **S1131 and S4169 are leakage-free** (zero exact/reverse-mutation
train-test contamination across all folds, incl. reverse-mutation augmentation); **S2003 carries a
disclosed ~0.18% artifact** (2 duplicate-measurement mutations in complex 3SE4, divergent ΔΔG,
negligible vs fold noise). The three sets are **nested** (S1131, S2003 ⊂ S4169), so per-set PCCs
are correlated. The paper's 0.868 is a **from-scratch-vs-pretrained ceiling** (released `mulan-ankh`
checkpoint; and the paper's largest set is S1400, not our S4169) — not a matched baseline; the
valid comparison here is the per-PLM contrast on identical splits. Full writeup:
`experiments/paper_fairness/FAIRNESS.md`.

**Tier-1 augmentation on the benchmarks (SaProt + ESM C 6B, in progress).** Both the Mac
queue (SaProt) and the Linux box (ESM C 6B, `esmc6b_aug_chain.sh`) are re-running the SKEMPI
benchmarks with reverse+identity augmentation to see whether the +0.010 S1102 aug lift (§5a)
transfers. **SaProt S1131 is complete: 0.844 ± 0.041** (10/10 folds;
[0.871, 0.795, 0.894, 0.829, 0.803, 0.886, 0.853, 0.822, 0.786, 0.901]) — **essentially neutral
vs the 0.847 base** (−0.003), unlike the small positive aug lift on S1102/CV. **ESM C 6B S1131
is now complete too: 0.872 ± 0.029** (10/10 folds) vs its 0.867 base — a slim **+0.005 but only
5/10 folds win** (magnitude, not frequency), so **also effectively neutral**. Remaining folds
are still running — **SaProt S4169 +aug** (partial), **ESM C 6B S2003 +aug** (in flight) and
**S4169 +aug** (queued; the ~2.5–3 h/fold long pole); this table updates when they land. Early
read: on the pre-augmented benchmark splits, Tier-1 aug does *not* add the signal it does on
S1102 for **either** PLM — consistent with aug being PLM- and dataset-dependent (neutral for
ProstT5 on S1102, §4).

## 11. ESM C 6B against Ankh

→ deep-dive: `experiments/RESULTS_ESMC.md`

**The first modern PLM to clear Ankh in MuLAN.** ESM Cambrian (ESM C) 6B —
EvolutionaryScale's representation-focused successor to ESM-2 — run **sequence-only**
(AA mode, 2560-d final-layer embeddings) through the same S1102 mutation-based 10-fold
CV as the Ankh/ProstT5 cv10: **PCC 0.859 ± 0.069**, vs Ankh 0.832 and ProstT5 0.805.
Paired Δ(ESM C − Ankh) = **+0.0267 PCC** (sample-std 0.0203, t = 4.15, p ≈ 0.0025,
df 9), **9/10 folds** (sign test p ≈ 0.02); ESM C also wins RMSE (1.176 vs 1.310),
MAE (0.853 vs 0.944) and Spearman (0.777 vs 0.731). It's the **closest any
from-scratch config has come to the paper's 0.868**, and the margin over Ankh is
essentially the same as Ankh's over ProstT5 — one clean step up. The gain is from
model **scale + newer pretraining**, *not* a structure channel (this is pure AA mode),
unlike the negative ProstT5 3Di experiments (§9).

**Loading was the hard part (non-obvious).** ESM C's `model_type: "esmc"` is **not in
any released transformers** (4.48/4.57/5.12 all reject it; the config's `4.57.6` is
EvolutionaryScale's own build string), and the `esm` SDK's local registry only ships
the 300M/600M (the 6B was Forge-API-gated). But the downloaded
`EvolutionaryScale/esmc-6b-2024-12` safetensors are in **native esm-package format**
(keys prefixed `esmc.`) and, prefix-stripped, map **exactly** (808/808 tensors, 0 shape
mismatches) onto a hand-built `esm.models.esmc.ESMC(2560, 40, 80, use_flash_attn=False)`
module (unused LM head loaded `strict=False`) — **no second 25 GB download** of the
biohub HF re-host. Embedding generation runs in an isolated venv (`~/.venvs/mulan-esmc`,
torch 2.12 + `esm` 3.2.3) and is fully standalone (esm + torch only; inlines mulan's
id-label logic so cache filenames match). cv10 **training** runs read-only from cache in
the **original** venv (transformers 4.44, untouched) — `data.py:73` loads a PLM only for
missing embeddings, so no ESM C code runs during training and the Ankh/ProstT5 numbers
stay reproducible. Two gotchas fixed: NaN embeddings from `meta`/`to_empty` leaving
rotary buffers uninitialized (→ real-init then overwrite params); and a torch/torchvision
clash from `pip install esm` breaking transformers' lazy import (→ removed torchvision,
kept embedding gen transformers-free). Runtime: 25.4 GB download, embedding gen 69.7 min
(~2.9 s/seq), ~29 GB peak RAM, cv10 ~8 h. Code wiring (`esmc_6b` in `constants.py`; ESM C
branch in `utils.py`) is additive. Driver `scratch/esmc6b_gen.py` + `cv10_esmc6b_driver.sh`
+ `esmc6b_orchestrate.sh`; caches `scratch/embeddings_esmc6b/`.

**Next:** confirm on the standard SKEMPI benchmarks (S1131/S4169/S2003) — the table (§10)
now has ProstT5/ESM2/Ankh/SaProt plus **ESM C 6B**, which **sweeps all three benchmark sets —
S1131 0.867, S4169 0.825, S2003 0.852** (leading SaProt by +0.020 / +0.010 / +0.020); the
esmc6b chain is complete. ESM C 6B vs AIDO-16B (§14, now measured).

## 12. ESM2-3B as an older-generation baseline

→ deep-dive: `experiments/RESULTS_ESM2.md`

**The controlled scale/pretraining control for ESM C 6B.** ESM2-3B
(`facebook/esm2_t36_3B_UR50D`) is EvolutionaryScale's *previous*-generation PLM and
shares ESM C 6B's exact **2560-d** embedding width — so running it through the
identical S1102 mutation-based 10-fold CV isolates **pretraining recipe + scale** from
embedding geometry. Result: **PCC 0.816 ± 0.054** (RMSE 1.395, MAE 1.006, SCC 0.691).

**This corrects the earlier ranking guess.** RESULTS had predicted
`ESM C 6B > Ankh > ProstT5 > ESM-2` ("+0.03 per step"). Measured, ESM2-3B is **not**
the weakest — it **ties ProstT5** and both trail Ankh:

| Contrast | Paired ΔPCC | folds ESM2 wins | read |
|---|---|---|---|
| ESM2-3B − ESM C 6B | **−0.0433** (std 0.027) | 1/10 | clearly worse |
| ESM2-3B − Ankh | −0.0166 (std 0.020) | 2/10 | worse |
| ESM2-3B − ProstT5 | +0.0109 (std 0.024) | 5/10 | **tied (within noise)** |

So the measured ordering is **ESM C 6B (0.859) > Ankh (0.832) > ESM2-3B (0.816) ≈
ProstT5 (0.805)** — two tiers, not a ladder. The most informative gap is **ESM C 6B −
ESM2-3B = +0.043 at identical 2560-d width**: the ESM Cambrian → ESM-2 jump is driven
by the **newer pretraining objective/data, not parameter count** (6B vs 3B alone would
predict a smaller step, and ESM2-3B doesn't even clear the 3.5×-smaller-encoder Ankh).
Per-fold ESM2-3B: 0.684, 0.843, 0.804, 0.818, 0.858, 0.889, 0.863, 0.802, 0.786,
0.807 (fold 0 is the hard fold for every PLM).

**Method — the cheapest run in the suite.** ESM2 is native to `transformers 4.44`, so
unlike ESM C 6B (isolated `esmc` venv) and AIDO (isolated bf16 venv), it needs **no
new environment**: the existing `esm` backend (`utils.py` generic `AutoModel` path)
generates 2560-d embeddings directly in the **main** venv. Same offline-embeddings
pattern as everything else — a serial prewarm (`scratch/esm2_prewarm.sh`, ~37 min incl.
the 11 GB download) writes one `.pt` per unique sequence, then the 10 folds train
read-only from cache (`data.py:73` loads no PLM). Footprint (§7): 2.8 B params /
~12 GB peak RAM (embedding gen), 1.7 GB cache, 19 MB ckpt; cv10 wall-clock **~8.0 h**
(10 folds, MAXPAR=4 × 10 threads, 15:15→23:17). Driver `scratch/cv10_esm2_driver.sh`
+ `esm2_orchestrate.sh`; aggregator `scratch/esm2_aggregate.py`; caches
`scratch/embeddings_esm2/`. No wiring changes (the `esm` key already existed).

**Takeaway:** ESM2-3B is a solid *older-generation* baseline that lands with ProstT5 in
the bottom tier; the modern step-ups (Ankh, then ESM C 6B) come from **pretraining
quality**. AIDO-16B (§14) now places the top of the scale axis — and confirms the
pretraining-over-scale read: the largest encoder ties Ankh, well below ESM C 6B.

---

## 13. Ankh3 (large / xl) — first CV10 runs on Apple Silicon (MPS)

First full 10-fold CV results measured on **Apple M4 Pro / 48 GB / MPS** instead of the
Linux CPU box, same S1102 protocol (`lightatt_default`, 50 ep, batch 32, lr 5e-4,
early-stop 10, best-val checkpoint). Ankh3 is a **T5 encoder**, so it runs on the main
venv (transformers 4.44) unchanged.

| PLM | dim | prefix | Test PCC | RMSE | MAE | SCC |
|---|---|---|---|---|---|---|
| Ankh3-large | 1536 | **[NLU]** | **0.823 ± 0.057** | 1.340 | 0.957 | 0.698 |
| Ankh3-large | 1536 | [S2S] | 0.818 ± 0.053 | 1.372 | 0.984 | 0.688 |
| Ankh3-xl | 2560 | **[NLU]** | **0.831 ± 0.056** | 1.315 | 0.944 | 0.727 |
| Ankh3-large **+aug** | 1536 | [NLU] | **0.849 ± 0.042** | 1.265 | 0.885 | 0.729 |
| Ankh3-xl **+aug** | 2560 | [NLU] | **0.855 ± 0.043** | 1.243 | 0.885 | 0.742 |

**Prefix.** Ankh3 is **prefix-conditioned** — its card uses `[NLU]` for embedding
extraction (`[S2S]` as an alternative). The initial `embed_sequence` fed ankh3 the raw
sequence like plain ankh-large, which is wrong (no prefix + a leading `<unk>` off-by-one);
fixed to add the prefix and strip the two leading tokens (both prefixes verified aligned).
On S1102, **[NLU] edges [S2S]** by +0.005 (large) — within noise, but [NLU] is the pick.

**Where they land.** Ankh3-large (0.823) sits **just below** the original Ankh-large
(0.832) and Ankh3-xl (0.831) ≈ Ankh-large — so on this task **Ankh3 does not beat
Ankh-large**, and neither reaches ESM C 6B (0.859). Useful negative-ish scaling datapoints:
2560-d Ankh3-xl gains only +0.008 over 1536-d Ankh3-large.

**MPS speedup.** A full 10-fold CV took **~1 h (Ankh3-large)** / **~1.7 h (Ankh3-xl)**
on MPS, vs the ~8 h these took per PLM on the 40-thread Linux CPU box — a large,
practical speedup for the from-scratch head. Embedding generation (1444 seqs) was
**~2 min (Ankh3-large)** / **~10 min (Ankh3-xl)** on MPS. See `MPS_COMPATIBILITY.md`.

**Tier-1 augmentation lifts both** (reverse-mutation + identity anchors, `[NLU]`, 10-fold):
Ankh3-large **0.823 → 0.849** (+0.026) and Ankh3-xl **0.831 → 0.855** (+0.024) — the same
~+0.025 aug lift seen for the other PLMs, and it does not change the ordering (Ankh3-xl +aug
0.855 ≈ ESM C 6B seq-only 0.859; ESM C 6B **+aug is now the suite best at 0.878**, §5a). Both
aug runs also on MPS (~2725-id union cache).

---

## 14. AIDO.Protein-16B and the effect of encoder scale

→ deep-dive: `docs/history/PLAN_AIDO.md`; setup/loading in §8. _(Pulled from the Linux
`mulan-aido` run; the ankh3 MPS runs in §13 are the parallel Mac line.)_

**The scale ceiling test — negative for "bigger is better."** AIDO.Protein-16B
(`genbio-ai/AIDO.Protein-16B`, a **16B-param Mixture-of-Experts**, hidden 2304, 36
layers) is the largest PLM in the suite — ~2.7× ESM C 6B's parameter count and ~14×
Ankh's. Run **sequence-only** (2304-d final-layer embeddings) through the identical
S1102 mutation-based 10-fold CV: **PCC 0.828 ± 0.046** (RMSE 1.321, MAE 0.947, SCC 0.738).

| Contrast | Paired ΔPCC | folds AIDO wins | read |
|---|---|---|---|
| AIDO − ESM C 6B | **−0.0308** (std 0.031) | 1/10 | clearly worse |
| AIDO − Ankh | −0.0042 (std 0.018) | 2/10 | **tied (within noise)** |
| AIDO − ESM2-3B | +0.0121 (std 0.023) | 6/10 | ≈ tied, slight edge |

So the 16B MoE lands **statistically tied with the 3.5×-smaller Ankh-large** (0.828 vs
0.832) and a clean step **below ESM C 6B** (0.859). It does earn one distinction — the
**tightest cross-fold std in the whole suite** (0.046, vs ESM C's 0.069 and Ankh's
0.057): most *stable* across folds, just not most *accurate*. Per-fold AIDO: 0.719,
0.866, 0.842, 0.842, 0.861, 0.882, 0.857, 0.823, 0.789, 0.798.

**Why this matters.** The biggest encoder does **not** top the table, so the measured
ordering — **ESM C 6B (0.859) > {Ankh 0.832 ≈ AIDO 0.828} > {ESM2-3B 0.816 ≈ ProstT5
0.805}** — is **non-monotonic in parameter count**. ΔΔG embedding quality tracks
**pretraining objective + data**, not raw size. (Caveat: AIDO is **sparse MoE**, so 16B
*total* params overstates the ~active params per forward pass, so its "16B" x-position
flatters it.)

**Method — CPU bf16, offline embeddings (§8).** No GPU needed: the MoE runs in **bf16 on
CPU** (Skylake bf16-upcast) in an isolated `mulan-aido` venv, one `[L, 2304]` `.pt` per
sequence. Base embedding generation (1444 seqs, 40 threads) took **311 min / 5.2 h**
(~13 s/seq) at **~32 GB peak RAM** — the heaviest generation in the suite; cv10 training
then ran read-only from cache (~7.3 h). Driver `scratch/cv10_aido_driver.sh`; aggregator
`scratch/aido_aggregate.py`; caches `scratch/embeddings_aido/`.

**Still pending — AIDO +aug (systemd-oomd crash + workaround).** The augmented run isn't
done: the aug-cache prewarm (A8) auto-fired at 04:54 and was killed by **systemd-oomd** at
05:00 mid model-load (checkpoint shard ~4/13). **Not a real OOM** — the 62 GB box had ~58 GB
free; oomd judges *per-cgroup* memory pressure, and the 16B bf16 load (~32–40 GB) tripped
the **VS Code terminal scope's** (`app-code-*.scope`) limit, SIGKILLing the whole process
group (no `ABORT` logged). Aug cache is still 1444 (base only; needs 2725). **Workaround —
re-run in a dedicated scope** so oomd leaves it alone:

```
systemd-run --user --scope -p ManagedOOMSwap=off -p ManagedOOMMemoryPressure=off \
  bash scratch/aido_aug_chain.sh
```

The base result above stands; the +aug point (A8 prewarm ~4–5 h → cv10 ~14 h) lands on re-run.

---

## 15. SaProt (structure-aware, AA+3Di) — the effect of WT structure on a 650M model

The first **structure-aware** PLM in the sweep. SaProt (`westlake-repl/SaProt_650M_AF2`) is
an **ESM2-650M** backbone with a **structural-alphabet (SA) vocabulary**: each residue is a
2-char token = amino acid + Foldseek-3Di state (446-token vocab). It fuses sequence and
structure at the *input*, unlike ProstT5's late 3Di concat (§9, negative). Same S1102
protocol as the rest (`lightatt_default`, 50 ep, 10-fold CV), on **Apple M4 Pro / MPS**.

**Stage 1 — WT structure, no FoldX.** The 3Di channel is the **wild-type** structure
(`experiments/gen_3di.py`, mini3di, per-residue aligned); a **mutant reuses its parent WT
chain's 3Di** (single-point mutation ⇒ backbone ~unchanged), so the only thing that changes
WT→mutant is the AA half of the SA token at the mutated site. To attribute any gain to
structure, each run has a paired **seq-only control** = the same weights fed all-`#`
(structure-masked) SA tokens.

| Run (650M, base, MPS) | Test PCC | RMSE | MAE | SCC |
|---|---|---|---|---|
| **SaProt — real WT-3Di** | **0.842 ± 0.050** | 1.276 | 0.926 | 0.729 |
| SaProt — seq-only (`#`) | 0.816 ± 0.069 | 1.362 | 0.976 | 0.699 |

**Structure helps (paired, same folds).** Δ (real − `#`) = mean **+0.026**, **8/10 folds
positive** (range −0.027…+0.098), t-like ≈ 2.26 (dof 9, ~p≈0.05). Same weights, structure
on vs off, so the lift is attributable to the 3Di channel — not the model.

**Where it lands.**
- **Seq-only SaProt (0.816) ≈ ESM2-3B (0.816).** With structure masked, a 650M SA model
  behaves like a sequence-only ESM2 — a clean sanity check (SaProt *is* an ESM2-650M).
- **Real-3Di 0.842 sits above Ankh-large (0.832)** and both Ankh3 base rows, below ESM C 6B
  (0.859) — the **strongest structure-model result** here, and it's *base* (WT-3Di only, no
  aug, no FoldX). The +0.026 structure lift is ≈ the Tier-1 aug lift (~0.025) and **does
  stack** with it: SaProt WT-3Di **+ Tier-1 aug = 0.852** (`cv10_saprot_aug`, MPS), a further
  **+0.010** over the 0.842 structure base (paired, 7/10 folds) — the two gains are roughly
  additive (§5a), making SaProt +aug the best sub-1B result in the suite.

**Method notes.** SaProt's SA vocab has only the 20 canonical AAs; non-canonical residues
(`X`, 47/171800 = 0.03 %) are mapped to the AA-mask `#` (valid `#d`/`##` tokens) — without
this, adjacent `X` residues collapse (`XdXd`→1 token) and break per-residue alignment.
Embeddings generated on CPU (keeps MPS free); CV10 trained on MPS. Loads via
`EsmForMaskedLM`/`AutoTokenizer` on the main venv (transformers 4.44), registered as
`saprot` in `PLM_ENCODERS`; gen script `experiments/gen_saprot_emb.py`.

**Stage 2 — FoldX mutant structure: a confirmed no-op (negative result).** Tested whether a
FoldX-generated mutant structure yields *mutation-specific* 3Di beyond Stage-1's reused WT 3Di.
Pipeline (FoldX 5.1, Apple-Silicon): `RepairPDB` → `BuildModel` (mutant PDB) → mini3di → mutant
3Di, compared per-residue to WT 3Di.

Probe of **39 mutations** — all 36 of 1A22's S1102 mutations **plus 1ACB Leu38→Pro and Leu38→Gly**
(the two most backbone-perturbing substitutions in the whole set): **0/39 changed the 3Di at any
position** — including →Pro/→Gly, and even at the mutated residue itself (e.g. `LI38P`: pos-38
3Di `p`→`p`).

**Why:** FoldX `BuildModel` is a fast **side-chain repacker** with only sub-Ångström backbone
movement, and Foldseek-3Di is a coarse (20-state) **backbone** descriptor — sub-Å moves never
cross a state boundary, so **FoldX-mutant 3Di ≡ WT 3Di**. Stage-2 embeddings would thus be
identical to Stage-1's (mutant AA + WT-equivalent 3Di) → identical CV10 (0.842). The 110-complex
`RepairPDB` batch was **not run** (the ~10-min probe substituted for ~hours of compute).

This is **dataset-independent** — a property of FoldX + 3Di, not of S1102 — so it holds for the
S1131/S4169/S2003 benchmarks too; the →Pro/→Gly test also covers **non-alanine** substitutions
(relevant to S2003, the non-Ala set). **Verdict: Stage-1 WT-3Di (0.842) is the ceiling for the
SaProt+FoldX structure route.** Capturing mutation-induced backbone/3Di change would require a
backbone-relaxing method (energy minimization / MD, or AlphaFold-on-mutant) — out of scope.

_Reusable gotcha:_ S1102 mutation chain letters are **partner-role** labels (A/B), not PDB
chains. FoldX needs the actual chain via `skempi_v2.csv` (e.g. 1ACB role B → chain **I**);
1A22 worked only because role A = chain A there. FoldX 5.1 needs no `rotabase.txt`.

---

## 16. Device parity — MPS vs CPU/Linux

The MPS-measured rows (Ankh3 0.823/0.831, SaProt 0.842, ankh3 +aug 0.849/0.855) are compared
against CPU/Linux-measured rows, and "Ankh3 < Ankh-large" rests on a **0.009 gap that straddles
the device boundary** (inside the ±0.01–0.02 split-noise band). To check the device isn't the
story, **Ankh-large** — the exact CPU baseline, same T5-encoder path as Ankh3 — was reproduced
**end-to-end on MPS** and paired against Linux `cv10_ankh` (identical `cv10_splits`).

**Layer 1 — embeddings (MPS vs CPU forward, same Mac).** Over all 1444 Ankh-large embeddings:
mean |Δ| **3.0e-8** (≤ fp32 epsilon), max |Δ| 1.3e-6, per-residue cosine **1.000000** (median and
min). Metal produces the same representation as CPU — every residue directionally identical.

**Layer 2 — end-to-end CV10 (paired on the same 10 folds).**

| Configuration | CV10 PCC |
|---|---|
| Linux (CPU emb + Linux train) — reference | **0.8321 ± 0.058** |
| MPS emb + MPS train | **0.8301 ± 0.060** |
| CPU emb + MPS train | 0.8266 ± 0.059 |

Paired **Δ (MPS end-to-end − Linux) = −0.0020**, inside the split-noise band; per-fold |Δ| ≤ 0.014
on **9/10** folds, the lone outlier fold 9 (−0.049) being training stochasticity (MPS/CPU op
nondeterminism diverges the trajectory even at `seed=42`), not device bias. Isolating the
**embedding device** (MPS-emb vs CPU-emb, training fixed on MPS) gives mean **+0.0035** — negligible,
consistent with the fp32-identical embeddings.

**Verdict:** MPS reproduces Linux within split noise, and the embedding device is provably
negligible. The MPS-measured rows sit **legitimately on the same axis** as the CPU/Linux rows, and
the 0.009 Ankh3-vs-Ankh gap is real, not a device artifact. Driver `scratch/ankh_parity.sh`;
`MULAN_FORCE_CPU` hook in `get_device()`. (Full engineering write-up: `MPS_COMPATIBILITY.md`.)

---

## 17. SaProt-1.3B (AFDB_OMG_NCBI) — a deeper SaProt checkpoint

A second SaProt checkpoint, `westlake-repl/SaProt_1.3B_AFDB_OMG_NCBI` (`saprot_1.3b` in
`PLM_ENCODERS`): **66 transformer layers vs the 650M's 33 (ESM2-650M arch), same 1280-dim**
AA+3Di structural-alphabet embedding (2× the depth, no width increase) and a different
pretraining corpus (AFDB+OMG+NCBI vs the 650M's AF2 set). Same S1102 protocol as §15 (WT-3Di
structure channel, `lightatt_default`, 50 ep, 10-fold CV), on **Apple M4 Pro / MPS**.

| Run (base/+aug, MPS) | Test PCC | RMSE | MAE | SCC |
|---|---|---|---|---|
| SaProt-1.3B — base | 0.837 ± 0.061 | 1.289 | 0.928 | 0.737 |
| SaProt-1.3B — **+Tier-1 aug** | **0.850 ± 0.065** | 1.231 | 0.877 | 0.742 |

**Aug lift replicates.** Paired (same 10 folds), Tier-1 augmentation (reverse-mutation +
identity anchors) gives mean **+0.013 PCC, 8/10 folds positive** — consistent with the
+0.010–0.026 aug lifts seen for every other PLM in the suite (§5a, §13, §15).

**Where it lands — depth doesn't help.** Both SaProt-1.3B rows sit **~tied with (a hair below)
the 650M SaProt** from §15: base **0.837 vs 0.842** (Δ −0.005) and +aug **0.850 vs 0.852**
(Δ −0.002) — within split-noise, i.e. **no measurable benefit from 2× the layers** at fixed
1280-d width and a different (larger, more diverse) pretraining corpus. Combined with
AIDO.Protein-16B (§14, tied with Ankh despite ~14× the params) and the ESM C 6B → ESM2-3B
read (§12, gap is pretraining recipe not size), this is the **third independent case** in the
suite where scaling a PLM upward does not move ΔΔG performance — the ceiling here looks set
by pretraining objective/data and the mutation-scale signal MuLAN extracts, not raw capacity.

**Method notes.** Same WT-3Di embedding pipeline as §15 (`experiments/gen_saprot_emb.py`,
checkpoint arg `saprot_1.3b`); embeddings generated on CPU, CV10 trained on MPS. Both base and
+aug are **S1102 only** — no S1131/S4169/S2003 sweep is planned for this checkpoint (depth
comparison was the only open question).

---

## 18. FoldX binding ΔΔG as an add_scores channel — the orthogonality ceiling

→ deep-dive: `scratch/foldx_s1102/` (driver `build_ddg.py`, merge/CV drivers, scatter
`foldx_signal_scatter.png`); memory `foldx-addscores-pipeline.md`

**A different lever than the PLM sweep:** instead of a better embedder, feed a
physics-based **FoldX 5.1 binding ΔΔG** (RepairPDB → BuildModel side-chain repack →
AnalyseComplex Interaction Energy, mut−WT) into MuLAN's existing `add_scores`/`zs_scores`
head channel (one scalar concatenated before the final linear layer). Built for **all of
S1102** (110/111 complexes, 100% coverage, 1100 rows; total + 12 decomposed terms
captured). Role→real-chain mapping via `skempi_v2.csv`, validated against each repaired
PDB. `DatasetArguments.add_zs_scores` wired through `scripts/train.py`; config
`lightatt_addscores_config.json`.

**A pre-training ceiling analysis (no CV needed) caps the whole direction.** Because
add_scores is a single scalar into a linear layer, an out-of-fold linear fit of the
FoldX feature on the baseline's *residual* is a tight ceiling on what the channel can buy:

| Signal | Value |
|---|---|
| Raw FoldX ΔΔG vs experimental label (n=1100) | **Pearson +0.407**, Spearman +0.447 |
| r(FoldX, out-of-fold Ankh **prediction**) — redundant part | +0.35 |
| r(FoldX, out-of-fold Ankh **residual**) — orthogonal part | **+0.20** |
| Best linear combo (Ankh + β·FoldX), OOF | 0.837 → **0.846** (Δ **+0.008–0.009**) |
| OOF ceiling, 11 decomposed components | +0.011 |
| OOF ceiling, all 12 terms | +0.011 |
| **RSM 2nd-order (12 + squares + 66 interactions), OLS** | in-sample 0.876 → **OOF 0.824 (WORSE than baseline)** |
| RSM 2nd-order, ridge (best α) | ≤ +0.005 (never beats the linear fit) |

Cross-checked on two independent Ankh CV runs (`cv10_ankh` and `cv10_ankh_seed43`), which
agree to ±0.001. **Findings:**
- **Mostly redundant.** Of FoldX's 0.41 standalone correlation, the bulk overlaps what
  Ankh already learned (r=0.35 with its own predictions); only **+0.20** is orthogonal.
- **Ceiling ≈ +0.011 PCC**, inside fold noise (±0.03–0.05). The FoldX signal is a
  **one-sided detector** — decent at flagging destabilizing mutations, blind to
  stabilizing ones (no negative-ΔΔG resolution; see scatter).
- **Decomposition barely helps** (+0.011 vs the +0.008 scalar). The only decomposed term
  with orthogonal signal beyond the aggregate is **Electrostatics** (+0.006); **VdW
  clashes contribute ≈0** — the "steric-clash tail" hypothesis was **wrong** (clashes are
  already summed into Interaction Energy; alone too sparse/noisy).
- **No nonlinear headroom.** A full RSM overfits hard (in-sample 0.876, OOF 0.824 — below
  baseline); ridge only claws back *toward* the linear fit, never past it. The useful
  response surface is genuinely linear — so a nonlinear **Stage-2 MLP head is forecast to
  underperform the plain scalar** and was dropped.

**Stage-1 CV10 result — confirms the ceiling exactly.** Paired baseline vs FoldX scalar
(Ankh, paper folds, `config.sh` 300 ep / patience 30):

| Arm | Test PCC | RMSE | MAE | SCC |
|---|---|---|---|---|
| baseline (no score) | 0.8217 ± 0.051 | 1.344 | 0.960 | 0.732 |
| **+ FoldX scalar** | **0.8310 ± 0.046** | 1.313 | 0.941 | 0.745 |

Paired **ΔPCC = +0.0093, FoldX wins 8/10 folds** (RMSE −0.031, MAE −0.020, SCC +0.013) —
landing **exactly on the +0.008–0.009 linear ceiling** predicted above. (This arm's
baseline 0.822 sits below the canonical `cv10_ankh` 0.832 because it uses the
embedding_sweep 300 ep/patience 30 config, not 50 ep/patience 10 — but the comparison is
*paired*, so only the Δ matters.)

**Per-fold — FoldX is a floor-raiser, not a uniform lift.** Ranked by gain:

| fold | baseline PCC | +FoldX | ΔPCC |
|---|---|---|---|
| 8 | 0.7854 | 0.8069 | **+0.0215** |
| 4 | 0.8263 | 0.8438 | +0.0175 |
| 3 | 0.8245 | 0.8418 | +0.0173 |
| 5 | 0.7845 | 0.7991 | +0.0146 |
| 2 | 0.7320 | 0.7445 | +0.0125 |
| 6 | 0.8760 | 0.8880 | +0.0120 |
| 7 | 0.7870 | 0.7942 | +0.0072 |
| 9 | 0.8431 | 0.8478 | +0.0047 |
| 1 | 0.8540 | 0.8499 | **−0.0041** |
| 0 | 0.9046 | 0.8943 | **−0.0103** |

The gains anti-correlate with baseline strength: **corr(baseline PCC, ΔPCC) = −0.61**.
FoldX helps most where Ankh is *weakest* (fold 8, baseline 0.785 → biggest gain +0.022) and
**hurts the two folds where Ankh is already strongest** — folds 0 and 1, the two highest
baselines (0.905, 0.854). It behaves as independent physics that props up the hard folds but
adds net noise to the easy ones the PLM has already nailed. Most striking, **fold-0 — the
designated hotspot fold that originally motivated the whole FoldX direction (PLMs squash
steric clashes) — is the single fold FoldX hurt** (−0.010), fully consistent with the
ceiling finding that the isolated clash terms carry ≈0 orthogonal signal. (Caveat: −0.61 on
10 folds / one seed is directional, not significant on its own — but it matches the
mechanism.)

**Verdict: a real but tiny (+0.009 PCC), largely-redundant signal — inside per-fold noise
and not a headline lever.** (A prior driver bug — `--save_model False` disables
`load_best_model_at_end`, tripping the EarlyStopping assertion — was fixed to
`--save_model True`, `save_total_limit=2` bounding disk.)

### 18.1 Stage 2 (decomposed-terms MLP head) — measured against the RSM forecast

Built for completeness to test the RSM's prediction that a jointly-trained MLP over the
12 decomposed terms would **overfit below** the +0.009 scalar. It did the **opposite**. The
head is a small MLP (12 → 16 → ReLU → Dropout(0.1) → 1, +226 params) whose scalar output
feeds the *same* +1 head slot; per-fold train-only-standardized 12-term vectors (100%
coverage). Same harness (300 ep, patience 30, seed 42), same paired folds.

| arm | PCC (10-fold) | paired Δ vs base | wins |
|---|---|---|---|
| baseline | 0.8217 ± 0.051 | — | — |
| FoldX **scalar** (§18) | 0.8310 ± 0.046 | +0.0093 | 8/10 |
| FoldX **MLP** (Stage 2) | **0.8383 ± 0.047** | **+0.0166** | 8/10 |
| MLP − scalar | — | +0.0073 | 7/10 |

- **MLP vs baseline** (+0.0166): paired *t* ≈ 2.7, *p* ≈ 0.02 — a real lift, **~2× the
  scalar's** point estimate and above the +0.011 "linear ceiling" that §18 reported.
- **MLP vs scalar** (+0.0073): paired *t* ≈ 1.45, *p* ≈ 0.18 — **within fold noise**; the
  increment over the scalar is not statistically resolvable at n=10.

**Why the RSM misled:** the RSM was a full 2nd-order OLS polynomial (12 + squares + 66
interactions) fit on the OOF residual — a high-variance estimator with no analogue to the
in-network MLP's dropout + ReLU + early-stopping + co-adaptation with the representation.
The 226-param regularized head does *not* overfit; the RSM's "OOF 0.824 < baseline" was an
artifact of the estimator, not a property of the signal.

**Fold texture matches the floor-raiser pattern:** biggest MLP gains land on the weak folds
(fold 4 +0.050, fold 7 +0.034, both baseline ~0.79→0.82), while fold 0 — the hotspot fold
that motivated the direction — is hurt *more* by the MLP (−0.019) than by the scalar
(−0.010).

**Revised verdict:** Stage 2 quietly beat §18's linear ceiling and is the better arm to
report if FoldX is used (0.838 vs 0.831), but the MLP-over-scalar edge is inside noise — so
the decomposition earns at most a modest, unproven increment. Direction **closed out** with
the scalar↔MLP choice a wash; the candid headline is still "real but small (+0.01–0.017),
floor-raising, not a lever." (Config `models/config/lightatt_addscores_mlp_config.json`;
`zs_mlp` plumbing in `mulan/{config,modules,data}.py`.)

**rng-splitter cross-check — ESM C 6B (GPU, `results_esmc6b_gpu_20260714`).** A second PLM now
has the rng/paper-splitter FoldX-MLP arm: ESM C 6B base **0.869** (S1102, 10/10 — landing right
on the paper's MuLAN-Ankh 0.868 line, vs its 0.881 under the balanced splitter), +aug **+0.002
(6/10)**, **+FoldX-MLP +0.005 (8/10)**, scalar +0.0003. So under rng the MLP again edges the
scalar (unlike the *balanced* esmc6b, where scalar won, §19), but the magnitude stays tiny —
reinforcing the §19 finding that the **strongest PLM has the least FoldX-MLP headroom** on either
splitter. Ankh's rng +0.017 vs esmc6b's rng +0.005 is the same base-strength ordering as balanced.

### 18.2 Which other PLMs would benefit (prediction — only Ankh is measured)

FoldX is model-independent physics (already computed, `results/*.json`, 100% coverage), so it
can be paired with any PLM cheaply. Whether it *helps* runs along two axes established above:

1. **Baseline strength (floor-raiser).** FoldX helped weak folds and hurt strong ones
   (corr(baseline, ΔPCC) = −0.61). Extrapolated across models: the weaker the PLM, the larger
   the FoldX gain.
2. **Sequence-only vs structure-aware.** FoldX ΔΔG *is* a structure-based calculation, so a PLM
   that already ingests structure has absorbed part of the signal → FoldX is more **redundant**
   to it. (The orthogonal residual r=0.20 was measured against sequence-only Ankh.)

Crossing both axes against the CV10 baselines:

| model | base PCC | type | predicted FoldX benefit |
|---|---|---|---|
| **ESM2-3B** | 0.816 | seq-only | **highest** — weak *and* maximally orthogonal |
| Ankh3-large | 0.823 | seq-only | high (≈ or slightly > Ankh) |
| ProstT5 (AA) | 0.805 | struct-trained | high-but-uncertain — weakest, but structure-trained → some redundancy |
| ESM C 600M / Ankh3-xl / AIDO-16B | 0.828–0.831 | seq-only | ≈ Ankh (~+0.015) |
| **Ankh** | 0.832 | seq-only | **+0.017 (measured, §18.1)** |
| SaProt-650M / 1.3B | 0.837–0.842 | AA+3Di | **low** — strong *and* explicitly structure-aware (most redundant) |
| **ESM C 6B** | 0.859 | seq-only | **lowest** — smallest residual; expect a fold-0-style regression |

**Single best candidate: ESM2-3B** — weakest sequence-only model, so FoldX physics is both
maximally orthogonal and has the most residual to fill; nearly free to test (`merge_foldx*` +
paired CV on ESM2's cached embeddings, `PLM=esm2`). It is also the one run that would actually
confirm Axis 1.

**Strategic caveat:** even a generous FoldX gain on ESM2 (0.816 → ~0.84) still lands *below*
SaProt+aug (0.852), Ankh3-xl+aug (0.855), and ESM C 6B (0.859/0.878). FoldX is a floor-raiser,
not a SOTA lever — and **augmentation is the bigger, cheaper lever** (+0.024–0.026 for Ankh3,
and it stacks). FoldX only earns its keep when locked to a specific weak/sequence-only model.

**Update — balanced-series measurements (2026-07-14).** The predictions in this table are
now partly tested; see **§19**. Headline: the **redundancy axis does not hold as predicted** —
SaProt (WT-3Di), forecast *lowest* benefit because it is structure-aware, instead gains
**+0.018**, essentially tying sequence-only ESM C 600M (+0.019). "Only Ankh is measured" is
no longer true: ESM C 600M, SaProt, and Ankh-large FoldX-MLP arms are now measured on the
balanced folds.

---

## 19. Balanced-splitter 300ep series — base CV10 + FoldX-MLP add_scores (Apple MPS)

→ artifacts: `scripts_plots/ddg_scaling_data_300ep_balanced.csv`,
`benchmarks_folds_300ep_balanced.csv`; figures `ddg_scaling_300ep_balanced.{png,svg}`,
`ddg_variability_300ep_balanced.{png,svg}`. Drivers `experiments/embedding_sweep/run_sweep.sh`
(`ES_CONFIG=config_balanced.sh`) and `scratch/foldx_s1102/cv10_foldx_mlp_balanced_driver.sh`.

The plots-facing companion to the master table. Same S1102 (1100 muts) and 300 ep /
patience 30 budget as the rng series, but with the fork-default **balanced equal-size
splitter (seed 42)** instead of the paper's `rng.integers` folds — so these numbers are a
**separate series**, not interchangeable with the master table above (kept out of it
deliberately to avoid conflating splitters). Every arm is a fresh 10-fold CV on Apple
M4 Pro / MPS unless noted.

**Base CV10 (balanced) — tracks the rng master table within fold noise:**

| PLM | params | base PCC | +Tier-1 aug |
|---|---|---|---|
| ESM C 600M | 6.0e8 | 0.834 ± 0.057 | 0.831 ± 0.062 (aug no lift) |
| SaProt (WT-3Di) | 6.5e8 | 0.849 ± 0.051 | 0.854 ± 0.050 |
| Ankh-large | 1.15e9 | 0.836 ± 0.053 | 0.838 ± 0.057 |
| **Ankh3-large** | 1536-d | **0.835 ± 0.052** | — (balanced aug not run) |
| ProstT5 | 1.21e9 | 0.823 ± 0.053 | 0.832 ± 0.049 |
| ESM2-3B | 2.8e9 | 0.833 ± 0.053 | 0.839 ± 0.054 |
| ESM C 6B | 6.0e9 | 0.881 ± 0.036 | 0.880 ± 0.039 (GPU) |

Ankh3-large (10/10, newly landed 2026-07-14) sits right on Ankh-large (0.836) — Ankh3 again
fails to beat Ankh here, matching the rng §13 finding. It is base-only in this series and is
kept out of the paired variability figure.

**Two levers head-to-head — Tier-1 aug vs FoldX-MLP add_scores (balanced, paired by fold).**
Both the augmented arm (val/test = forward mutations, so paired to base) and the Stage-2
12-term FoldX-MLP head (`merge_foldx_decomposed.py --out balanced`, ~100% coverage, built from
the *same* balanced base folds) sit on identical folds, so each arm's per-fold Δ vs its own
base is directly comparable:

| PLM | base | +aug (Tier-1) paired Δ | +FoldX-MLP paired Δ |
|---|---|---|---|
| ESM C 600M | 0.834 | −0.003 (5/10, *no lift*) | **+0.019 (9/10)** |
| SaProt (WT-3Di) | 0.849 | +0.005 (7/10) | **+0.018 (7/10)** |
| Ankh-large | 0.836 | +0.002 (6/10) | **+0.018 (9/10)** |
| ProstT5 | 0.823 | +0.009 (6/10) | **+0.014 (8/10)** |
| ESM2-3B | 0.833 | +0.007 (7/10) | *2/10 — pending* |
| SaProt-1.3B | *(base pending)* | — | 0.858 (base needed to pair) |
| **ESM C 6B** (GPU) | **0.881** | −0.001 (5/10, *no lift*) | **+0.004 (6/10)** ‡ |

**Findings (revise §18.2 and the §13 "aug is the bigger lever" framing):**
1. **FoldX-MLP beats Tier-1 aug under the balanced splitter — for every fully-measured PLM.**
   FoldX-MLP lands **+0.014–0.019** (paired); balanced aug is weak (**+0.002–0.009**), and for
   ESM C 600M it is a genuine *no-lift* (−0.003, 5/10). The FoldX-MLP gain also matches the
   rng-series MLP (+0.017, §18.1) → **splitter-robust**. ‡ **ESM C 6B is the diminishing-returns
   exception**: at 0.881 base (best PLM) *both* levers nearly flatten — aug −0.001 (5/10, no lift)
   and FoldX-MLP only **+0.004 (6/10)**, a quarter of the smaller PLMs' lift; and it is the sole
   PLM where the **Stage-1 scalar arm (+0.007) beats the 12-term MLP (+0.004)**. So the *ordering*
   (FoldX-MLP ≥ aug) still holds, but headroom for either add-on shrinks as the base PLM strengthens
   — the FoldX signal is largely already captured by a 6B encoder. (From `results_esmc6b_gpu_20260714`.)
2. **Redundancy axis refuted.** §18.2 predicted structure-aware SaProt would benefit *least*;
   instead it gains **+0.018 ≈** sequence-only ESM C 600M (+0.019). The 12-term head carries
   orthogonal signal even for an AA+3Di model.
3. **⚠ The "aug is the bigger lever" claim (§13, +0.024–0.026) is splitter-specific — UNDER
   AUDIT.** Those aug gains are rng-splitter (and partly patience-10) numbers; under the
   balanced 300/30 folds the aug lift **collapses to ~+0.005** while FoldX-MLP holds. Whether
   the cause is truly the *splitter* (vs budget) is being nailed down by a paired **rng-300/30
   aug run for ankh + prostt5** (queued) — the rng series has the identical
   300 ep / patience 30 budget, differing only in `ES_SPLIT_METHOD=random`. This claim is pending
   the queued run that completes the pair. (SaProt-1.3B pairs with its queued base run;
   Ankh-large's foldxmlp is now final at 10/10, +0.018.)

## 20. ESM C 6B on GPU — balanced 4-benchmark sweep, 300ep/p30

Run on a CUDA GPU box and transferred back (integrity-checked: per-fold JSON means reproduce
the summaries exactly).

**Base (balanced, 300ep/p30) — complete, 10/10 all four:**

| dataset | PCC mean | pstd | vs CPU 50/10 ref |
|---|---|---|---|
| S1102 | 0.8812 | 0.036 | +0.022 over cv10 0.859 (budget) |
| S1131 | 0.8763 | 0.030 | +0.009 over base 0.867 |
| S4169 | 0.8437 | 0.039 | +0.019 over base 0.825 |
| S2003 | 0.8697 | 0.034 | +0.018 over base 0.852 |

Ranking preserved; the +0.01–0.02 lift over the CPU 50/10 exploration numbers is the
300/30 budget. Ordering **S1102 > S1131 > S2003 > S4169** unchanged.

**Aug (tier-1, balanced, 300ep/p30):** S1102 **0.8800** (Δ=+0.001 vs 0.8812 base,
neutral) and **S1131 0.8712** (Δ=−0.0051 vs 0.8763, neutral) — both complete 10/10 and
consistent with the earlier read that **aug is a wash on 6B** (the residual it fills is
already captured by scale). **S4169 aug 1/10, S2003 aug 0/10 on the GPU — INCOMPLETE**,
blocked by an OOM (below); the refill was **moved to the 62 GB CPU box** (S1131 aug 10/10,
S2003 aug 7/10 in flight, S4169 aug queued at PAR=1 — see OOM note).

**GPU OOM gotcha (long benchmarks):** the light-att head runs a Conv1d over sequence
length, so per-fold VRAM scales with **complex length**. Short S1102 folds use <3 GB
(4-way packing fine), but a long **S4169** aug fold needs ~**16.5 GB** — two won't fit
on 24 GB, and even at `GPU_PAR=2` a co-scheduled S2003 fold OOMs against a resident
S4169 fold (confirmed: "GPU 0 … 1.40 GiB free, process has 16.51 GiB"). Fix: run
**S4169 at `GPU_PAR=1`** and **S2003 separately at `GPU_PAR=2`** (not in the same
invocation). `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` alone is insufficient —
it's a capacity limit, not fragmentation. **Update:** even `GPU_PAR=1` still OOMs a single
S4169-aug fold on the 24 GB card in practice (peak exceeds the ~16.5 GB steady-state
estimate), so **S4169-aug was moved off the GPU entirely to the 62 GB CPU box**. The same
~2× RAM applies there — an S4169-aug fold peaks ~40 GB RSS, so two at MAXPAR=2 (≈80 GB)
would OOM 62 GB too. Fix (`scratch/bench_cv_esmc6b_aug_driver.sh`): per-dataset parallelism
— **S4169 at PAR=1 / 40 threads** (one fold, whole box), S1131/S2003 stay at 2×20.

**FoldX phase-2 (paper split, 300/30) — complete, confirms the shrinking-residual
prediction.** On the matched paper split (apples-to-apples with the Ankh FoldX runs §18):
base 0.8690 → **+FoldX scalar 0.8693 (+0.0003)** → **+FoldX 12-term MLP 0.8744 (+0.0054)**.
Ankh's reference gains were +0.009 (scalar) / +0.017 (MLP) — so **ESM C 6B's FoldX lift is
~⅓ of Ankh's on the MLP head and ≈0 on the scalar.** This is the §18.2 forecast confirmed
empirically: FoldX's orthogonal residual shrinks as the PLM strengthens, and ESM C 6B (the
strongest sequence-only model) leaves the least for FoldX to fill.

**FoldX phase-2 (balanced split, 300/30) — now complete; scalar > MLP here, opposite of
the paper split.** On the balanced partition (`cv10_esmc6b_balanced`, 10/10 all three arms,
pull 20260714_155233): base 0.8812 → **+FoldX scalar 0.8886 (+0.0074)** → **+FoldX 12-term
MLP 0.8849 (+0.0037)**, i.e. the MLP is **−0.0036 below scalar**. This *inverts* the paper-split
ordering (there scalar ≈0, MLP +0.0054 and MLP > scalar). Same model, same FoldX physics, same
budget — only the fold partition differs, so the scalar-vs-MLP margin is within partition noise
and **the candid read is "FoldX scalar buys ~+0.005–0.007; the decomposed-MLP head is not
reliably better than the scalar."** The larger balanced lift (+0.0074 vs +0.0003) also tracks
baseline strength: the balanced base is *higher* (0.8812 vs 0.8690) yet FoldX helps it more,
consistent with the §18 floor-raiser mechanism operating per-fold rather than per-mean.
(Per-fold means reproduce the summaries exactly — integrity verified.)

### 20.1 FoldX phase-2 across the three SKEMPI benchmarks (GPU, balanced, 300/30)

The FoldX story (§18/§19) was measured only on S1102. The GPU sweep now extends the
decomposed-12-term head to **all three SKEMPI benchmarks**, 10/10 folds × 3 arms each
(`cv10_esmc6b_bench`, pull `results_esmc6b_gpu_20260715_093819.tar.gz`; per-fold means
reproduce the summaries exactly):

| dataset | base | +FoldX scalar (Δ, wins) | +FoldX 12-term MLP (Δ, wins) | MLP − scalar |
|---|---|---|---|---|
| S1131 | 0.8763 | **0.8884** (+0.0120, 9/10) | 0.8816 (+0.0052, 6/10) | −0.0068 (mlp>scalar 3/10) |
| S2003 | 0.8697 | 0.8760 (+0.0063, 7/10) | **0.8855** (+0.0158, 9/10) | +0.0095 (mlp>scalar 8/10) |
| S4169 | 0.8437 | 0.8567 (+0.0130, 9/10) | **0.8574** (+0.0137, 9/10) | +0.0006 (mlp>scalar 4/10) |

Two clean generalizations of the S1102 result:

1. **FoldX phase-2 helps on every benchmark.** The scalar arm is positive across all three
   (+0.006 to +0.013, 7–9/10 folds); the MLP arm is positive too (+0.005 to +0.016). This is
   the multi-dataset confirmation the FoldX add-scores channel lacked — the orthogonal physics
   residual survives on datasets beyond S1102, and even on ESM C 6B (the strongest sequence-only
   PLM, §18.2's "lowest-benefit" prediction) it still buys a real floor-raise.
2. **The scalar-vs-MLP winner tracks dataset size.** On the smallest set (S1131, ~1131 muts) the
   scalar wins and the MLP *underperforms* it (−0.0068, mlp>scalar only 3/10) — the same pattern
   as the S1102 balanced split (§19). On the larger S2003 (~2003) the MLP clearly wins (+0.0095,
   8/10) and on the largest S4169 (~4169) they tie (+0.0006, 4/10). Consistent with the MLP
   needing more rows to fit the 12→hidden map without the regularized-head advantage washing out.
   **Defensible narrative: use the FoldX *scalar* on small sets, the 12-term MLP on large ones;
   the scalar is the safe default (+0.006–0.013 everywhere), the MLP the upside on ≥2k-mut sets.**

**CPU↔GPU "correlation" arm (04) — NOT yet a valid device check.** The GPU paper arm
(`paper_seed42`, 300/30) was pulled complete (base 0.8690, aug 0.8711, 10/10 each), but
the CPU runs it was compared against — `results/cv10_esmc6b` / `cv10_aug_esmc6b` — use a
**different split** (`cv10_splits`; fold_0 test sets share only 73/110 IDs) **and a
different budget** (50ep/p10). So the fold-for-fold r I first computed (base r≈−0.34,
aug r≈0.93) is **confounded** and is *not* a portability metric. The correct CPU
comparand for the GPU 300/30 paper arm is the **FoldX baseline** run
(`scratch/foldx_s1102/cv10_esmc6b/baseline`, which *is* `paper_seed42` 300/30) —
**7/10 folds in** (folds 7–9 running). On the matched folds the device parity is
**strong**: per-fold Pearson **r = 0.987**, mean|Δ| = 0.0061, max|Δ| = 0.0139, and the
fold-means agree to **CPU 0.8761 vs GPU 0.8765 (Δ = −0.0004)** — differences signed both
ways (no systematic offset) and well inside the paper-split fold spread (pstd ≈ 0.033). This
provisionally **confirms CPU↔GPU numerical equivalence** — mixing/offloading results across
the two machines is safe to trust; full 10-fold confirmation lands when folds 7–9 finish.

**fold_0 finding — a systematically HARD partition, not a per-model seed bug.**
`cv10_esmc6b` fold_0 = **0.6719** looked like a broken run, but a sweep across **all 26
S1102 cv10 runs** shows **fold_0 is the worst fold in 25 of them** (the 26th has it
second-worst): every PLM / aug / scale dips at fold_0 by −0.08 to −0.21 (Ankh −0.157,
ESM2 −0.146, ProstT5 −0.145, SaProt −0.125, ESM C 6B −0.208). They all test on the
**identical `cv10_splits/fold_0` test set**, so this is a **hard split partition**, not a
seed-42 accident specific to ESM C. **Do NOT reseed-and-swap fold_0 for one model** — that
would cherry-pick ESM C's hard fold while competitors keep theirs, breaking comparability;
0.859 is legitimate under a protocol all PLMs share equally. Two residual signals: (1) ESM
C's −0.208 dip is the *largest* of the pack (~−0.14 typical), so a little extra bad-seed
luck may sit on top of the hard partition; (2) **budget mitigates it** — the same fold at
the GPU 300/30 budget scores 0.823 (Δ≈−0.06), i.e. more epochs climb partway out. A reseed
sweep (`scratch/rerun_cv10_esmc6b_fold0.sh`, seeds 43/44/45 → `cv10_esmc6b/fold_0_reseed/`)
runs as a **diagnostic only** (seed-vs-partition attribution) — it stays OUT of the
canonical mean. **Takeaway: single-seed CV at a short budget can amplify a hard fold;
report ESM C base with this shared-hard-fold_0 caveat, not a per-model correction.**

---

## 21. Leakage-controlled protocols on full SKEMPI

Everything above §20.1 is measured on S1102 or on the standard SKEMPI benchmark subsets, all of
which partition **by mutation**. A complex contributing 60 mutations therefore appears in both
train and test, and the resulting PCC is not an estimate of performance on an unseen complex.
This section and those that follow it replace that protocol with three hold-out rules, each
stricter than the last, applied to the **full SKEMPI 2.0 extraction — 5801 mutations across 337 complexes**
(4165 single-point, 1636 multi-point).

| Protocol | Held out | Folds | fold_0 train / val / test | Result tier |
|---|---|---|---|---|
| by-complex | whole PDB entries | 3 | 2208 / 568 / 1389 | `full_skempi_bycomplex` |
| clustered | MMseqs2 sequence clusters at 60% identity | 3 | 2114 / 378 / 1673 | `full_skempi` |
| CATH superfamily | every complex in a CATH superfamily | 1 | 4213 / 878 / 687 | `full_skempi_cath` |
| by-complex, combined | whole PDB entries, single **+** multi | 3 | 3275 / 592 / 1934 | `full_skempi_bycomplex_all` |
| clustered, combined | 60% clusters, single **+** multi | 3 | 2969 / 537 / 2295 | `full_skempi_clustered_all` |
| multi-point only | either rule, multi-point rows alone | 3 | 925 / 165 / 546 ◆ | `full_skempi_mp_*` |

◆ by-complex; the clustered multi-point fold_0 is 920 / 170 / 546.

The ladder is deliberately ordered: by-complex removes the identical structure, clustering removes
its close homologues as well, and the CATH hold-out removes the whole fold family. A method that
has learned complex identity rather than the effect of a substitution decays down that ladder.
The combined tiers exist because the structure-based frontier — RDE-Network, DiffAffinity,
Prompt-DDG, BA-DDG — trains one model on single and multi-point rows together with a whole-PDB
hold-out, so only the combined by-complex tier is protocol-matched to their published numbers.

**The metric is per-structure Spearman at T ≥ 10** (`ppS`): rank the mutations *within* each
complex that carries at least 10 of them, then average across complexes. Pooled Pearson over a
leakage-controlled split rewards a model that merely separates complexes by their mean ΔΔG, which
is exactly the skill the split is meant to stop crediting; per-structure rank correlation asks
the question a user actually has, which is which substitution in *this* interface is worst. At
T ≥ 10 the two SP tiers score 96 complexes, the combined tier 121, the multi-point tiers 43 and
the CATH hold-out 13. Full rationale: `experiments/METRIC_RATIONALE.md`.

Every number in §22–§25 was rescored on 2026-08-02 from the current result tree with
`experiments/full_skempi_seqonly/score_{sp_all,cath,multipoint}.py`, which re-derive the rung CSVs
that `scripts_plots/results_matrix.py --ps` consolidates.

→ deep-dive: `docs/history/PLAN_FULL_SKEMPI.md`

## 22. FoldX ΔΔG coverage and result lineage

The FoldX channel has been merged onto these splits at three different coverage levels, and a
number quoted from this campaign means nothing without knowing which one it carries. Neither
increase needed new FoldX compute — both were merge defects, not missing energies.

| Lineage | Single-point | Combined (single + multi) | Multi-point | What was wrong |
|---|---|---|---|---|
| 1 | 28% | 28% | — | The merger read one result directory when the same per-complex JSONs already sat in four others (`build_results_all.py`). |
| 2 | 87.8% | 90.8% | 98.7% | A single-key join missed the alternate chain-label convention; dual-keying fixed it. |
| 3 | **99.2%** (12393/12495 non-zero scalars) | **99.1%** (5746/5801 rows per fold) | 98.7% | Residue-numbering mismatch in the remaining keys. |

The multi-point merge (`merge_foldx_multipoint.py`) is a separate code path that the single-point
defects never touched, so **98.7% is multi-point's finished coverage, not a lag** — its 21
uncovered rows are the two quarantined groupings (`2C5D.AB.CD`, `3SE3.B.C`).

The correction is large enough to partition the results rather than merely shift them. Rescoring
the same predictions across a re-run moves every FoldX arm and no base arm, by up to **0.073
pooled Pearson** — wider than most of the between-backbone differences the tables in §23 are read
for. In the working tree the superseded per-arm directories were archived in place as
`<arm>__cov87` and `<arm>__cov_pre` rather than deleted; those archives are **not** published
here, since `results/` ships one directory per arm at its current lineage.

**Reading an arm's lineage requires its checksum, not its neighbours.** The archive suffixes look
like they encode it and do not: an arm re-run before the archiving convention existed has no
sibling at all, which is exactly what makes it easy to mistake for current. The test is
`find <arm-dir> -name all_results.json | sort | xargs cat | sha256sum` against the reference
blocks; every classification below was made that way.

| Tier | FoldX channel coverage | Arms below it |
|---|---|---|
| `full_skempi` (clustered-SP) | 99.2% | — |
| `full_skempi_bycomplex` | 99.2% | — |
| `full_skempi_cath` | 99.1% | — |
| `full_skempi_bycomplex_all` | 99.1% | — |
| `full_skempi_clustered_all` | 99.1% | — |
| `full_skempi_cath_aug` | 99.1% | — |
| `full_skempi_clustered_all_aug` | 99.1% | — |
| `full_skempi_mp_bycomplex` | 98.7% | — |
| `full_skempi_mp_clustered` | 98.7% | — |

**No arm anywhere in the campaign is still on lineage 1, and every tier is now internally
consistent.** The last stragglers were the four Ankh3 arm-pairs in the two single-point tiers,
re-run at 99.2% and merged on 2026-08-04; §23 no longer marks any row as a lineage back.

What a re-run does to a score is not uniformly positive, so a lagging row cannot simply be read as
understated. Across the 52 FoldX arms re-scored either side of the 87.8% → 99.2% step,
per-structure Spearman rose on 42 and fell on 10 — largest gain +0.116 (ESM-C 600M, clustered),
largest loss −0.037 (ESM-C 6B, CATH). More coverage means more mutations carry a real energy
instead of the fold's training mean, which usually helps and occasionally replaces a lucky
imputation with a wrong prediction. The eight Ankh3 arms re-scored on 08-03 split 5 up / 3 down and
moved at most 0.037, because they crossed 90.8% → 99.1% rather than the full step.

**The final batch is the one that most clearly repays the compute, and only in the conclusion
metric.** The eight single-point Ankh3 arms crossing 87.8% → 99.2% on 08-04 gained ppS
uniformly — 8 of 8, mean **+0.042**, largest +0.067 (Ankh3-large's clustered 12-term arm, 0.372 →
0.439). Pooled `test_pcc` over the same eight arms moved +0.005 with mixed sign, which reads as
fold noise and is how the re-run was first assessed. The disagreement is expected rather than
anomalous: ppS is a rank statistic computed within each structure and then averaged, so it responds
to the energy channel being right on the specific rows that order one complex's mutations, while a
pooled correlation over 4165 rows dilutes exactly that. A coverage fix is the kind of change that
surfaces in one and not the other — which is the argument for judging a re-run on the metric the
conclusions use, not on the cheaper one.

**The coverage step also moved the comparator, which is easy to miss.** FoldX used directly as a
predictor is not a trained arm and does not appear in the lineage table above, but it is computed
from the same channel, so it rose with it: **0.363 → 0.418** per-structure Spearman on the two
single-point tiers. The multi-point, CATH and S1102 tiers were already at ~100% coverage and did
not move. Any contrast against the physics baseline computed before 2026-08-04 is therefore against
a reference 0.055 too low on the single-point tiers — see §26.

→ deep-dive: `experiments/full_skempi_seqonly/SP_FOLDX_COVERAGE_AUDIT.md`,
`experiments/FOLDX_COVERAGE_FINDINGS.md`

## 23. Single-point results under the three protocols

Per-structure Spearman at T ≥ 10, 3-fold pooled, 4165 mutations over 96 scored complexes. Every
FoldX arm carries **99.2%** channel coverage (§22) — the two Ankh3 rows that trailed at 87.8% were
re-run and are now on the same lineage as the rest, so the tables below are internally consistent.

**By-complex hold-out.**

| backbone | base | + FoldX scalar | + FoldX 12-term | Δ scalar | Δ 12-term |
|---|---|---|---|---|---|
| Ankh-large | 0.284 | 0.435 | 0.444 | +0.151 | +0.160 |
| ProstT5 | 0.194 | 0.452 | **0.464** | +0.258 | +0.270 |
| SaProt | 0.120 | 0.411 | 0.435 | +0.291 | +0.315 |
| ESM2-3B | 0.261 | 0.459 | 0.445 | +0.198 | +0.184 |
| ESM-C 600M | 0.227 | 0.434 | **0.467** | +0.207 | +0.240 |
| ESM-C 6B | **0.310** | 0.451 | 0.457 | +0.141 | +0.147 |
| Ankh3-large | 0.203 | 0.436 | 0.449 | +0.233 | +0.246 |
| Ankh3-xl | 0.268 | 0.463 | 0.454 | +0.195 | +0.186 |
| AIDO-16B | 0.180 | 0.447 | 0.435 | +0.267 | +0.255 |

**Homology-clustered hold-out (60% identity) — the headline protocol.**

| backbone | base | + FoldX scalar | + FoldX 12-term | Δ scalar | Δ 12-term |
|---|---|---|---|---|---|
| Ankh-large | 0.142 | 0.406 | 0.427 | +0.264 | +0.285 |
| ProstT5 | 0.054 | 0.410 | 0.444 | +0.356 | +0.390 |
| SaProt | 0.082 | 0.392 | 0.427 | +0.310 | +0.345 |
| ESM2-3B | 0.119 | 0.395 | 0.432 | +0.276 | +0.313 |
| ESM-C 600M | 0.152 | 0.355 | 0.422 | +0.203 | +0.270 |
| ESM-C 6B | **0.193** | **0.417** | **0.446** | +0.224 | +0.253 |
| Ankh3-large | 0.092 | 0.388 | 0.439 | +0.296 | +0.347 |
| Ankh3-xl | 0.144 | 0.411 | 0.431 | +0.267 | +0.287 |
| AIDO-16B | 0.120 | 0.397 | 0.428 | +0.277 | +0.308 |

Four readings, in descending order of how well the data supports them:

1. **Sequence-only MuLAN barely ranks within a structure once homologues are held out.** Median
   base ppS falls from 0.227 by-complex to **0.120** clustered, and the best backbone under
   clustering (ESM-C 6B, 0.193) scores below the *weakest* FoldX arm in the same table (ESM-C
   600M's scalar, 0.355). ProstT5 reaches 0.054 — indistinguishable from no within-complex ranking
   skill at all. The S1102 CV numbers in §2–§5, which run 0.70–0.87 PCC, describe a different task.
2. **The FoldX channel's lift widens exactly where the encoder fails.** Mean Δ over base for the
   12-term arm is **+0.223 by-complex and +0.311 clustered**; every backbone gains under both,
   and no arm anywhere in either table is negative. This is the cleanest evidence the campaign has
   produced that the biophysical channel is orthogonal to what the language model encodes rather
   than redundant with it: the stricter the split, the more of the remaining signal it carries.
3. **Under clustering the 12-term decomposition beats the single scalar for all nine backbones**
   (9/9, mean +0.036), where by-complex splits it 6/9 and the CATH hold-out reverses it (2/9,
   §25). This extends §20.1's size rule rather than contradicting it — full SKEMPI is nearly 4× S1102,
   and the decomposition needs rows to fit its 12→16 map — and adds a second axis: the harder the
   generalization, the more the extra 11 terms are worth.

   **This one reads as an architectural result and is not.** The same gap is obtained with the
   backbone removed: a plain linear reweighting of the twelve terms, no PLM anywhere in it, scores
   **0.434** on this tier against the FoldX total's 0.418, and the nine 12-term arms above span
   0.422–0.446 around that line. What the 9/9 sweep measures is the decomposition carrying more
   than the total, which is a property of the FoldX output, not of the head that consumes it or
   the encoder in front of it. §26 and
   `experiments/beyond_foldx/BLEND12_RESULT.md` carry the measurement.
4. **The choice of backbone stops mattering once the channel is added.** Under clustering the
   spread across nine backbones is 0.139 at base and **0.024** with the 12-term arm — from 0.422
   (ESM-C 600M) to 0.446 (ESM-C 6B). The whole PLM comparison of §11–§17, on which most of this
   repository's compute was spent, collapses to under a thirteenth of the mean FoldX lift. The
   by-complex tier says the same thing less starkly: 0.190 at base, 0.032 with the 12-term arm.

**Every Δ column in this section is measured against MuLAN base, which is the wrong denominator
for the question "is the channel worth it".** The free alternative is FoldX used directly, at
0.418 on both tiers. Against that reference the clustered `+ FoldX scalar` column contains no
positive entry — its range is 0.355 to 0.417 — so the nine scalar arms are, on the headline
protocol, at or below the cost-free comparator while showing lifts of +0.202 to +0.356 over base.
Both statements are arithmetically correct. §26 gives the comparator table.

Pooled Pearson for the same runs is in `scripts_plots/results_matrix.csv`; it is reported for
continuity with §1–§20 and should not be used to rank models under these splits, for the reason
given in §21.

→ deep-dive: `experiments/full_skempi_seqonly/SUMMARY_esmc6b_foldx.md`

## 24. Combined single + multi, and multi-point alone

**Combined by-complex — the frontier-matched protocol.** 5801 mutations, 121 scored complexes,
3-fold; every FoldX arm at **99.1%** coverage (§22). This is the tier to quote against RDE-Network,
DiffAffinity, Prompt-DDG and BA-DDG, because it is the only one that partitions the way they do.

| backbone | base | + FoldX scalar | + FoldX 12-term | Δ scalar | Δ 12-term |
|---|---|---|---|---|---|
| Ankh-large | 0.247 | 0.445 | 0.458 | +0.198 | +0.211 |
| ProstT5 | 0.200 | 0.459 | 0.457 | +0.259 | +0.257 |
| SaProt | 0.191 | 0.455 | 0.445 | +0.264 | +0.254 |
| ESM2-3B | 0.251 | 0.449 | 0.460 | +0.198 | +0.209 |
| ESM-C 600M | 0.267 | 0.456 | 0.464 | +0.189 | +0.197 |
| ESM-C 6B | **0.295** | 0.444 | **0.479** | +0.149 | +0.184 |
| Ankh3-large | 0.158 | 0.406 | 0.442 | +0.248 | +0.284 |
| Ankh3-xl | 0.246 | 0.458 | 0.439 | +0.212 | +0.193 |
| AIDO-16B | 0.260 | 0.457 | 0.450 | +0.197 | +0.190 |

The table is complete at nine backbones. Adding the 1636 multi-point rows raises both the median
base (0.227 → 0.247) and the median 12-term arm (0.449 → 0.457) relative to the single-point tier
of §23 — more training rows per complex, and multi-point mutations are on average larger in effect
and so easier to rank.

**Combined homology-clustered (60% identity).** The same 5801 mutations and 121 scored complexes
partitioned by sequence identity instead of by complex — the strictest combined protocol here, and
the clustered analog of the table above. Every FoldX arm at **99.1%** coverage (§22).

| backbone | base | + FoldX scalar | + FoldX 12-term | Δ scalar | Δ 12-term |
|---|---|---|---|---|---|
| Ankh-large | 0.109 | **0.418** | 0.410 | +0.309 | +0.301 |
| ProstT5 | 0.061 | 0.415 | 0.422 | +0.354 | +0.361 |
| SaProt | 0.139 | 0.414 | **0.423** | +0.275 | +0.284 |
| ESM2-3B | 0.149 | 0.402 | 0.406 | +0.253 | +0.257 |
| ESM-C 600M | 0.084 | 0.387 | 0.349 | +0.303 | +0.265 |
| ESM-C 6B | **0.206** | 0.414 | 0.422 | +0.208 | +0.216 |
| Ankh3-large | 0.155 | 0.395 | 0.421 | +0.240 | +0.266 |
| Ankh3-xl | 0.160 | 0.412 | 0.411 | +0.252 | +0.251 |
| AIDO-16B | 0.157 | 0.416 | 0.404 | +0.259 | +0.247 |

This tier qualifies §23's third reading rather than extending it. The pattern there — the stricter
the split, the more the 12-term decomposition is worth — held cleanly across the single-point
rungs (CATH 2/9, by-complex 6/9, clustered 9/9). On the *combined* clustered tier it does not: the
decomposition wins **5 of 9, mean −0.001**, and the two arms are indistinguishable in aggregate
(mean lift +0.272 versus +0.273). So the axis that predicts when the extra 11 terms pay is the
split's strictness *within a mutation regime*, not strictness alone — adding the 1636 multi-point
rows changes what the decomposed head has to fit. ESM-C 600M is the one backbone the decomposition
clearly hurts here (0.387 → 0.349); no other loses by more than 0.012.

The first two readings of §23 hold undiminished. Base ppS is lower than any other combined tier —
median 0.149, ProstT5 at 0.061 — while the mean 12-term lift is **+0.272**, and the spread across
backbones falls from 0.145 at base to 0.074 with the channel.

**Augmentation on the FoldX arms: all six now measured against repaired splits, and five of the six
losses are indistinguishable from zero.** All six were first measured on the defective augmented
channel described below. Ankh-large's two were re-run on 2026-08-06 and the remaining four on
2026-08-08, so the lineage column this table used to carry is gone — every row below is repaired.
Paired cluster bootstrap over the 121 shared complexes (B = 10,000, seed 0); the augmented and
unaugmented test TSVs are byte-identical on all three folds, so every row is one test set
throughout. **The intervals are not reproducible from this repository**: they come from
`paired_aug_bootstrap.py` run against the augmented predictions, and neither the script nor those
folds ship — `results/` carries the unaugmented arms only. The Δ *point estimates* are simply
`+aug − plain` and can be checked here, since a per-structure mean is a mean over complexes; the
`plain` column is reproducible from `results/`, and the `+aug` column reconciles against
`scripts_plots/results_matrix_ps.csv`. Anything written from the intervals should say so.

| backbone | arm | plain | +aug | Δ paired | verdict |
|---|---|---|---|---|---|
| Ankh-large | + FoldX scalar | 0.418 | 0.394 | **−0.024 [−0.047, −0.002]** | excludes 0 |
| Ankh-large | + FoldX 12-term | 0.410 | 0.409 | −0.001 [−0.031, +0.033] | n.s. |
| ESM-C 6B | + FoldX scalar | 0.414 | 0.395 | −0.019 [−0.044, +0.006] | n.s. |
| ESM-C 6B | + FoldX 12-term | 0.422 | 0.416 | −0.006 [−0.032, +0.021] | n.s. |
| AIDO-16B | + FoldX scalar | 0.416 | 0.402 | −0.014 [−0.036, +0.009] | n.s. |
| AIDO-16B | + FoldX 12-term | 0.404 | 0.403 | −0.001 [−0.019, +0.017] | n.s. |

**The repair removed most of the harm on every arm it touched, and what survives is one arm of
six.** Ankh-large's 12-term loss went −0.042 [−0.080, −0.001] → −0.001 [−0.031, +0.033], a CI
centred on zero, and its scalar arm −0.065 [−0.097, −0.034] → −0.024 [−0.047, −0.002], clearing
zero only at its edge. The other four moved the same way: every one of them read as a significant
loss on the defective lineage — −0.027 to −0.062, all four CIs excluding zero — and not one of
them does now. What was measured was mostly the defect.

**The CATH tier points the other way, and it is one fold.** Its augmented arms cover three
backbones × three arms; against their unaugmented twins, **all three base arms lose and five of
the six FoldX arms gain** — the exception is AIDO-16B's scalar arm at 0.417 → 0.349. ESM-C 6B's
12-term arm goes 0.431 → 0.445 and Ankh-large's scalar 0.421 → 0.437. Read none of it as a
replication: 687 mutations over 13 complexes at a single hold-out, with no paired interval, is a
direction and not an effect.

**The base arm reverses the sign, and what that separates is narrower than it was.** Ankh-large is
the first backbone with all three arms augmented on one tier, and the only one whose FoldX arms are
repaired:

| arm | plain | +aug | Δ paired |
|---|---|---|---|
| base | 0.109 | **0.165** | **+0.056 [+0.010, +0.104]** |
| + FoldX scalar | 0.418 | 0.394 | **−0.024 [−0.047, −0.002]** |
| + FoldX 12-term | 0.410 | 0.409 | −0.001 [−0.031, +0.033] |

Two of three exclude zero rather than all three, and the opposition is now between the base arm and
*one* of the two FoldX arms. The earlier reading — that what survives is augmentation conflicting
with the FoldX channel specifically — was carried by effect sizes the repair removed: it rested on
both FoldX arms losing at −0.04 to −0.07, and one of them no longer loses at all. What the repaired
numbers support is weaker and worth stating as such: augmentation helps the base pathway and does
not transfer through the scalar channel, on one backbone, on one tier.

**The ambiguity recorded below, at the four outstanding arms, is now closed, and it closed in the
direction that prediction named.** It
turned on whether the four outstanding arms would behave like Ankh-large's after their own re-run.
They did, and further: all four lost significance, where Ankh-large's scalar arm kept a marginal
loss. So the FoldX-conflict reading is not merely weakened — five of the six arms it rested on no
longer show the effect at all, and the one that does is the weakest of the three Ankh-large arms.
Treat the surviving −0.024 as a single unreplicated result rather than as a pattern.

**The base gain does not survive a change of tier, and it does not survive a change of backbone.**
Two replications, both on base arms and therefore both unaffected by the channel defect below:

| replication | tier | backbone | plain | +aug | Δ paired |
|---|---|---|---|---|---|
| original | clustered-ALL | Ankh-large | 0.109 | 0.165 | **+0.056 [+0.010, +0.104]** |
| same backbone, new tier | CATH | Ankh-large | 0.331 | 0.262 | −0.069 [−0.173, +0.030] |
| same tier family, new backbone | bycomplex-ALL | ESM-C 6B | 0.295 | 0.294 | −0.001 [−0.033, +0.032] |

Neither replication is significant, but both **exclude an effect of +0.056**: the CATH upper bound
is +0.030 and the bycomplex-ALL upper bound is +0.032. The CATH run is the sharper of the two,
holding the backbone fixed and changing only the protocol. Between them they leave three readings
open — the gain is specific to clustered-ALL, or specific to the backbone, or a false positive —
which is what the designed test below was built to separate.

Weigh that accordingly: CATH scores 13 complexes and its CI half-width is ±0.10, wide enough that
only an effect the size of the original would have been detectable. It rules out a general
"augmentation helps the base arm" claim, not a small effect on some tiers.

**The designed test ran on 2026-08-06 and is null.** ESM-C 6B's and AIDO-16B's base arms on
**clustered-ALL** — the tier the +0.056 was measured on — paired over the same 121 complexes:

| backbone | plain | +aug | Δ paired |
|---|---|---|---|
| Ankh-large | 0.1094 | 0.1649 | **+0.0555 [+0.0101, +0.1043]** |
| ESM-C 6B | 0.2058 | 0.1738 | −0.0320 [−0.0760, +0.0100] |
| AIDO-16B | 0.1565 | 0.1104 | −0.0461 [−0.0981, +0.0057] |

Neither backbone moves, both trend negative, and **both CIs exclude +0.056** (upper bounds +0.0100
and +0.0057). "Specific to clustered-ALL" is therefore out: the tier does not confer the gain on a
backbone that lacks it.

**The Ankh-large row in that table is the original measurement, not a replication of it.** Same tag
(`ankh`, which the model table displays as *Ankh-large*), same arm, same 121 complexes, same test
TSVs as the +0.056 above — recomputed by a second implementation. That agreement is a check on the
*code*, and a useful one, but it is not independent evidence about augmentation and must not be
counted as a second confirmation. There is no separate Ankh v1 backbone in this repository's model
table, so a row labelled `ankh` cannot be a different model held constant.

What stands is narrower than the run count suggests: **+0.056 has one measurement and no
replication.** Four attempts to reproduce it have failed — same backbone/new tier (CATH, −0.069),
same tier family/new backbone (bycomplex-ALL, −0.001), and the two designed-test backbones above —
and every one of those four CIs excludes +0.056. Two readings survive: the effect is specific to
Ankh-large, or it is a false positive. Nothing yet separates them, because no second Ankh model has
been measured on this tier. **Ankh3-large base × clustered-ALL+aug is the discriminating run**, and
it would be the first test of whether the effect extends within the Ankh family at all rather than
a confirmation of something already replicated. Six backbones are still unmeasured on the `+aug`
half of this tier: Ankh3-large, Ankh3-xl, SaProt, ProstT5, ESM-C 600M and ESM2-3B.

None of this block bears on the FoldX half, which the arms above are still measuring. The three
designed-test contrasts were computed twice by two independent implementations, agreeing on all
three point estimates and all six CI bounds to four decimals.

The augmentation was checked for internal consistency and passed: reverse rows negate the label and
the FoldX scalar together (−0.499 → +0.499), all twelve decomposed terms are negated in the same
row, and the magnitudes match the unaugmented file exactly. That check verified the negation was
*applied* consistently, which is not the same as its being the right value — the exact magnitude
match is recognised below as the signature of the defect rather than as evidence against one.

Where the extra signal goes is measured directly in
`experiments/beyond_foldx/REDUNDANCY_RESULT.md`. Augmentation roughly doubles the base pathway's
**non-redundant** content — `part_base` = Spearman(base, truth | FoldX) rises 0.033 → 0.081,
paired Δ **+0.047 [+0.004, +0.093]** — while leaving the base arm no more FoldX-like than before
(`base~fx` 0.162 → 0.181, n.s.). Yet an optimally weighted leave-one-complex-out blend of the base
prediction with the FoldX scalar returns **+0.003 before and +0.002 after**: the added information
is real, is complementary to physics, and is still worth nothing once combined. The trained FoldX
arms do not merely fail to convert it — they lose ground significantly, which "no usable signal"
does not predict.

The mechanism for that loss is **not** established. The obvious candidate — that negating rows
forces an odd-symmetric response to the FoldX features, which the data would penalise — is not
supported: the FoldX→ΔΔG slope is +1.16 on stabilizing rows against +0.83 on destabilizing ones,
but the asymmetry is +0.33 [−0.15, +0.73] under a cluster bootstrap and does not clear zero.

**A second cause was found on 2026-08-05, it is not a property of the channel, and it alone
predicts the split above.** The augmentation negates the FoldX terms *after* standardisation, and
negation and standardisation do not commute: `-z(x) - z(-x) = 2·mu/sigma`, where `mu` is the raw
train mean of Interaction Energy. That mean is not near zero — most mutations are destabilising — so
every synthesized row's channel is offset by **+0.82 to +1.08 sd** depending on fold, which is 0.96×
to 1.57× the channel's own spread, on **51% of the training set**. The direction compounds: a
reverse row's negated label claims stabilising while its channel is shifted toward destabilising.
It reaches only the arms that read the channel, which is exactly the observed pattern.

The label is negated in **raw** units and is therefore exact; the channel is negated in
**standardised** units and is not. That asymmetry between the two is the whole defect.

**Repaired 2026-08-05, and the splits rebuilt.** `merge_foldx_aug.py` now writes each reverse row as
`clip((−x − mu)/sigma)`. The canonical merges do not persist their constants, so `mu` and `sigma` are
reconstructed per fold per term by regressing the committed channel on the raw FoldX terms over
unclipped rows — an exact linear relation, so this is reconstruction, not estimation, and the build
aborts unless the recovered constants reproduce every joinable committed value to within the split's
own 5-decimal rounding. On Interaction Energy: CATH `mu` +0.9255 `sigma` 1.9552; clustered-ALL
+0.9895 / 1.8128, +1.1445 / 2.0518, +0.8308 / 1.8913 — corrections of −0.879 to −1.116 sd.
Verification: 5959 reverse rows reproduce `z(−x)` computed independently from the raw FoldX JSONs to
1e-5, all 125 reverse rows whose forward row FoldX never scored stay all-zero, forward and val/test
rows remain byte-identical to the canonical split, and columns 1–4 are unchanged so the derived
4-column base splits and the augmented fasta are untouched.

The same error affected the identity (wt→wt) anchors, whose true FoldX vector is raw zero: they were
written as standardised 0, which encodes "an average mutation" and collides with the 0 the canonical
merge writes for rows FoldX never scored. They now carry `−mu/sigma`.

The pre-repair splits are kept as `*_aug__negz`, and the superseded results beside their
replacements as `<arm>__negz`, so every before/after pair above is recoverable. **Six of the 28
affected runs were re-run first** — Ankh-large's `foldx` and `foldx_scalar` on clustered-ALL, three
folds each, 2026-08-06 — and both arms moved toward zero harm by **+0.041 ppS**, the same amount to
three decimals. **The remaining 22 were re-run on 2026-08-08, so all 28 are now on the repaired
lineage** and no augmented FoldX result in this document predates the repair. The base arm reads
only the 4-column split and is unaffected, so its
**+0.056 [+0.010, +0.104]** stands as a measurement — though see above for four replications that
exclude an effect that size.

That the repair moved both arms in the same direction, and by roughly the offset's size, was
consistent with the defect having caused the loss without proving it: n = 2 arms on one backbone,
and the 12-term arm's before-value was itself the weaker of the two. The four outstanding arms were
the test, and unlike most re-runs this one had a prediction attached to it in advance.

**The test ran, and the prediction held.** All four moved toward zero, and all four crossed from a
CI excluding zero to one containing it (§24). n is now 6 arms on three backbones rather than 2 on
one, which is what turns "consistent with the defect having caused the loss" into the reading this
document carries.

Separately, FoldX is **not** exactly antisymmetric: over 1744 reversals computed directly, reverting
recovers 0.742 [0.633, 0.866] of the forward effect, and the error grows sixfold with |ΔΔG|. That is
a real second effect but a smaller one, and it lands in the pre-registered rule's inconclusive
branch. Both are written up in `experiments/beyond_foldx/FOLDX_ANTISYMMETRY_RESULT.md`.

**Multi-point rows alone.** 1636 mutations, 43 scored complexes, 3-fold. FoldX arms at **98.7%**
coverage — the multi-point merge is a separate code path that neither single-point join defect
touched, so this is finished coverage rather than a lag (§22).

| backbone | by-complex: base → 12-term | clustered: base → 12-term |
|---|---|---|
| Ankh-large | 0.340 → 0.391 | 0.091 → 0.355 |
| ProstT5 | 0.277 → **0.444** | 0.067 → 0.342 |
| SaProt | 0.176 → 0.269 | −0.008 → 0.194 |
| ESM-C 6B | **0.374** → 0.423 | 0.137 → 0.320 |
| Ankh3-large | 0.272 → 0.415 | 0.201 → 0.288 |
| Ankh3-xl | 0.339 → 0.432 | 0.144 → **0.355** |
| AIDO-16B | 0.263 → 0.323 | 0.053 → 0.172 |

The clustered multi-point tier is the hardest rung anywhere in this document: SaProt's base arm is
*negative*, and every backbone's base sits below 0.21. It is also where the FoldX lift is largest
relative to the base it starts from. Paired bootstrap over shared complexes puts the by-complex
FoldX contrast at P(Δ > 0) ≥ 0.98 on both arms for ProstT5, SaProt and Ankh3-large, but at 0.885
for the Ankh-large 12-term arm and 0.662 for the ESM-C 6B scalar arm — the gains of the backbones
that were already strong on this tier are not separable from noise at 43 scored complexes.

→ deep-dive: `experiments/full_skempi_seqonly/SUMMARY_mp_bycomplex.md`,
`experiments/full_skempi_seqonly/SUMMARY_mp_clustered.md`

## 25. CATH superfamily hold-out

A single hold-out rather than a k-fold: 687 mutations, 50 complexes, of which **13** carry the
10 mutations needed to be scored. It reproduces USP-ddG's literal 813-mutation CATH test set from
the static `cath_fold` column that project ships — 687 of those rows survive this repository's
SKEMPI extraction — so it is the one rung directly comparable to a published structure-based
split.

Every FoldX arm carries **99.1%** coverage (§22).

| backbone | base | + FoldX scalar | + FoldX 12-term | Δ scalar | Δ 12-term |
|---|---|---|---|---|---|
| Ankh-large | 0.331 | 0.421 | 0.387 | +0.090 | +0.056 |
| ProstT5 | 0.070 | 0.415 | 0.398 | +0.345 | +0.328 |
| SaProt | 0.229 | 0.398 | 0.397 | +0.169 | +0.168 |
| ESM2-3B | 0.243 | 0.375 | 0.352 | +0.132 | +0.109 |
| ESM-C 600M | 0.117 | 0.351 | 0.370 | +0.234 | +0.253 |
| ESM-C 6B | **0.360** | **0.438** | **0.431** | +0.078 | +0.071 |
| Ankh3-large | 0.148 | 0.355 | 0.392 | +0.207 | +0.244 |
| Ankh3-xl | 0.266 | 0.414 | 0.414 | +0.148 | +0.148 |
| AIDO-16B | 0.248 | 0.417 | 0.377 | +0.169 | +0.129 |

**Thirteen complexes is too few to rank backbones on**, and the single fold gives no spread to
compare a difference against. What the tier does support is the two coarse findings that survive
at this sample size: the FoldX lift is positive for all nine backbones on both arms, and the
scalar arm beats the 12-term arm for 6 of 9 — the reversal predicted by §20.1's size rule, since
687 mutations is the smallest set the decomposed head has been fitted on. Ankh3-xl is a 9th case
only nominally: its two arms are separated by 0.0004, which at 13 complexes is a tie.

FoldX alone scores **0.383** on this tier, so of the eighteen arms above, one clears the free
comparator by a margin that excludes zero — ESM-C 6B's scalar, +0.055 [+0.001, +0.118] (§26).

**One of eighteen is what chance predicts, and no correction is applied anywhere in this
document.** At a 95% interval, 18 arms give 0.9 expected false positives under a global null, and
the surviving lower bound is +0.0014. Nothing here selects a hypothesis in advance, so this arm is
the maximum of eighteen and its interval is not a confidence interval for the arm that won. Two
tiers behave differently and are worth the contrast: `bycomplex-SP` clears zero on 10 of 18 and
`bycomplex-ALL` on 9 of 18, an order of magnitude above chance — but those are the easier
partitions, and they are not the tier the frontier comparison is on. Read the CATH result as
consistent with a real effect and not as evidence of one.
The 12-term reweighting is *negative* here with the backbone removed (−0.038 at 13 complexes),
which is the same size-rule reversal seen in the trained arms and reached without training.

Splitting the hold-out by mutation order sharpens that: on the 421 single-point rows the 12-term
arm wins 8 of 9 (best: ESM-C 6B 0.495), while on the 266 multi-point rows it wins 4 of 9. Two
backbones lose outright on the multi-point subset — ESM2-3B (0.604 base → 0.525) and Ankh3-xl
(0.587 → 0.573) — the only place in §21–§25 where adding the channel hurts a backbone, and both
on 5 scored complexes.

→ deep-dive: `experiments/full_skempi_seqonly/SUMMARY_cath.md`,
`docs/history/runbooks/CATH_HANDOFF.md`

### 25.1 Coverage of the tiers, and what is still missing

Every tier in §23–§25 is now complete at nine backbones × three arms. The last gaps closed in one
day: ESM2-3B's combined by-complex FoldX pair filled §24's only empty row, and five base arms
(Ankh, ESM2-3B, ESM-C 600M, ProstT5, SaProt) completed the combined clustered tier, which is
reported above for the first time.

Two gaps remain, and neither is a queueing job.

**The augmented tiers have three base arms, all on `clustered_all_aug`.** Ankh-large came first and
is the only backbone there with all three arms, which is what §24's ambiguity turned on — with only
FoldX arms, augmentation failing at this scale and augmentation conflicting with the FoldX channel
fit the evidence equally well. ESM-C 6B and AIDO-16B landed 2026-08-06 and rule out one of the
readings: the base gain is not a property of the tier, because neither backbone moves on it (§24).
`full_skempi_cath_aug` now carries base arms for Ankh-large, ESM-C 6B and AIDO-16B as well — all
three negative, none significant at 13 complexes.

What is still missing is a **second Ankh backbone on `clustered_all_aug`**, with Ankh3-large the
candidate. Every measurement of the +0.056 so far is the same arm of the same backbone (tag `ankh`
= Ankh-large; there is no separate Ankh v1 in the model table), so nothing yet distinguishes a
backbone-specific effect from a false positive. Six backbones remain unmeasured on the `+aug` half
of this tier: Ankh3-large, Ankh3-xl, SaProt, ProstT5, ESM-C 600M and ESM2-3B.

Building them is smaller than it was: a configuration change, not a data build. The augmented four-column split does not
need a builder: the 4-column base split is the 5-column FoldX split with its last column removed,
which holds byte-for-byte across train, val and test on every fold of both augmented tiers. What
the aug configs are missing is an `ES_SPLIT_DIR` override — they set the FoldX split dirs and the
WT fasta and inherit `ES_SPLIT_DIR` from the unaugmented parent, so `ARMS=base` under one of them
trains the plain base into the augmented results directory and exits 0. A wrong answer that reports
success, not a failure. The Ankh-large run confirms the override works: its driver line resolves to
`splits=…/splits_skempi_full_clustered_all_aug`, and the FoldX arms to `…/splits_cath_foldx_aug`
and `…/splits_cath_foldxdec_aug`.

**Augmentation on the frontier-matched protocol is measured for one backbone and settles
nothing.** The by-complex-ALL augmented arms exist for ESM-C 6B only, recorded in
`experiments/full_skempi_seqonly/results_bycomplex_all_aug.csv`: at
T ≥ 10 over 121 complexes the 12-term arm moves 0.4794 → 0.4506 and the scalar arm 0.4444 →
0.4707, and the base arm's pair is the −0.001 row above. Two arms on one backbone moving in
opposite directions do not say whether the clustered result generalizes;
`docs/history/runbooks/RUN_BYCOMPLEX_ALL_AUG_GPU.md` is the recipe those runs followed.

Beyond these, augmentation covers only two of the nine backbones on any tier. Extending it to the
seven sequence-only backbones would be 56 FoldX runs plus 28 base — worth costing only after the
base arm establishes whether the effect is real.

## 26. FoldX used directly as the comparator

§23–§25 report every arm as a lift over MuLAN base. The published SKEMPI frontier — RDE-Network,
Prompt-DDG, DiffAffinity, CATH-ddG, USP-ddG — instead reports a FoldX baseline column, which is
also the free alternative to training anything. Because the FoldX ΔΔG already sits beside the
experimental label in every `splits_*_foldx` TSV, the comparison costs no compute.

Per-structure Spearman at T ≥ 10, same complex keys, same pooling. Contrasts are paired on the
shared complex set with a cluster bootstrap over complexes (B = 10,000).

| tier | FoldX alone | best base | best scalar arm | best 12-term arm |
|---|---|---|---|---|
| by-complex-SP | 0.418 | 0.310 | 0.463 | 0.467 |
| clustered-SP | 0.418 | 0.193 | 0.417 | 0.446 |
| MP-by-complex | 0.393 | 0.374 | 0.416 | 0.444 |
| MP-clustered | 0.393 | 0.201 | 0.344 | 0.355 |
| CATH-all | 0.383 | 0.360 | 0.438 | 0.431 |
| CATH-single | 0.398 | 0.377 | 0.456 | 0.495 |

Across 232 paired contrasts over ten backbones — **the nine tiers excluding `bycomplex-ALL` and
`clustered-ALL`**, which is worth stating because those two are the combined single+multi rungs
this section identifies as the protocol every frontier comparator was measured under. Including
them gives 286 contrasts and 28 / 83 / 175, so the win rate moves 8.2% → 9.8% and the loss rate
26% → 29%; the scope changes little here, but it has to be named to be quotable:

| | count |
|---|---|
| significantly **beat** FoldX alone | **19** |
| significantly **lose** to FoldX alone | 61 |
| indistinguishable | 152 |

**No `base` arm beats FoldX alone anywhere — 0 wins in 78 contrasts.** Without the channel, no
frozen-PLM MuLAN configuration in this repository reaches an unsupervised physics score under
leakage control. All 19 wins carry the FoldX channel.

**The clustered tiers are where the method comes closest to failing.** Three wins in 76 contrasts,
and all three are 12-term arms: ESM-C 6B +0.031 [+0.001, +0.061], ProstT5 +0.030 [+0.005, +0.053],
Ankh3-large +0.028 [+0.004, +0.052]. Not one scalar arm clears the baseline on either clustered
tier, and on MP-clustered several are significantly worse — ESM-C 6B's scalar at −0.105
[−0.181, −0.027]. The 12-term wins are the same effect §23's third reading describes, and a linear
reweighting of the terms with no backbone reaches 0.434 on clustered-SP unaided.

**ESM-C 6B is the backbone that clears the bar, and by less than it did.**

Counts are over that backbone's own contrasts, so the totals differ where a backbone has fewer
tiers on disk. The last column is its **largest significant** FoldX-arm contrast, or its largest
of any kind where none is significant.

| backbone | beats | loses | n.s. | largest significant FoldX-arm contrast |
|---|---|---|---|---|
| **ESM-C 6B** | **6** | 4 | 17 | S1102 by-complex 12-term +0.109 [+0.016, +0.206] |
| ProstT5 | 4 | 7 | 16 | S1102 by-complex scalar +0.107 [+0.037, +0.182] |
| Ankh3-xl | 3 | 6 | 18 | CATH-single 12-term +0.076 [+0.001, +0.150] |
| ESM2-3B | 2 | 3 | 16 | by-complex-SP scalar +0.046 [+0.015, +0.078] |
| SaProt-1.3B | 1 | 1 | 5 | S1102 by-complex scalar +0.141 [+0.047, +0.238] |
| ESM-C 600M | 1 | 6 | 14 | by-complex-SP 12-term +0.053 [+0.020, +0.087] |
| AIDO-16B | 1 | 8 | 12 | by-complex-SP scalar +0.034 [+0.005, +0.062] |
| Ankh3-large | 1 | 10 | 16 | clustered-SP 12-term +0.028 [+0.004, +0.052] |
| Ankh-large | 0 | 5 | 22 | CATH-single 12-term +0.084 [−0.019, +0.203] *(n.s.)* |
| SaProt-650M | 0 | 11 | 16 | S1102 by-complex scalar +0.110 [−0.002, +0.225] *(n.s.)* |

ESM-C 6B's CATH-single 12-term arm, **+0.097 [+0.017, +0.194]**, is the result the CATH tier of
§25 rests on; it is not this backbone's largest, but it is the one on a tier the frontier reports.

Two consequences for how §23–§25 should be quoted.

**The frontier comparison needs this row in the same table.** The CATH figure to place beside
USP-ddG 0.493 and CATH-ddG 0.494 is ESM-C 6B's 0.438, against a FoldX-alone row of 0.383. Without
that row a reader cannot tell that most of the 0.438 was free.

**A lift over base is not evidence the channel earns its keep.** "The FoldX channel triples
ProstT5's clustered ppS from 0.054 to 0.444" and "no ProstT5 arm on that tier is separable from
reporting the FoldX number directly" describe the same two runs.

These figures supersede a 2026-07-27 computation made at 87.8% channel coverage, where FoldX alone
scored 0.363 on the single-point tiers (§22). The win count did not fall when the baseline rose —
the FoldX arms gained about as much as the comparator did — but its composition changed: the
clustered tiers went from 1 win in 56 to 3 in 76, ESM-C 6B's CATH-single result fell from +0.127 to
+0.097, and Ankh-large lost its only win.

### Distance between this FoldX and the frontier's

Four tiers publish a FoldX number the one above can be set against. These rows are the **combined
single+multi** rungs (n = 5,801), which is the protocol every frontier comparator was measured
under — so they do not restate the single-point figures in the table above, where FoldX alone is
0.418. The gap is not uniform, and that is what identifies its cause.

| tier | metric | here | published | gap | n | source |
|---|---|--:|--:|--:|--:|---|
| CATH-superfamily | AUROC | 0.760 | 0.754 | **+0.006** | 687 | USP-ddG Table 1 |
| by-complex | per-structure Sp | 0.430 | 0.369 | +0.061 | 5801 | Prompt-DDG Table 1 |
| by-complex | AUROC | 0.717 | 0.658 | +0.059 | 5801 | BA-DDG Table 1 |
| clustered id60 | pooled Sp | 0.512 | 0.294 | **+0.218** | 5801 | ProtBFF Table 2 |

The CATH row rules out a stronger pipeline: 687 mutations at 100% channel coverage on both sides,
agreeing to 0.006. The two by-complex rows are one effect measured twice — a consistent +0.06 from
two independent tables on two metrics over the same 5,801 rows, which is the coverage advantage
§22 records, sized by the 87.8% → 99.2% step that alone moved FoldX alone 0.363 → 0.418.

The clustered row is 3.6× that and carries something further. Pooling over the combined rung
rewards the multi-point rows (0.572 alone, against 0.437 for single-point) and the between-complex
ordering (0.596) — the first two components a lower-coverage FoldX loses, since coverage fails by
whole structure and multi-point mutations are the hardest to map. Degrading this data that way
reproduces the published figure: multi-point unusable plus ~40% of complexes unusable gives 0.287.

### Lift over the FoldX each method was built on

ProtBFF is the same construction as the arms in §23–§25 — biophysical features scaled into a frozen
PLM — so its headline inherits its FoldX baseline, and the comparable quantity is the lift.

| tier | method | its FoldX | its result | lift |
|---|---|--:|--:|--:|
| clustered id60 (fold-avg Sp) | ProtBFF | 0.294 | 0.477 | **+0.183** |
| | MuLAN + FoldX | 0.521 | 0.547 | **+0.025** |
| by-complex (per-structure Sp) | RDE-Network | 0.369 | 0.401 | +0.032 |
| | Prompt-DDG | 0.369 | 0.426 | +0.057 |
| | **MuLAN + FoldX** | 0.430 | 0.479 | **+0.049** |
| | CATH-ddG | 0.369 | 0.460 | +0.091 |
| | BA-DDG | 0.369 | 0.513 | +0.144 |

**CATH is the primary tier, and on AUROC the baseline mismatch does not reach it.** §25's
comparison is on the one rung where this FoldX and the frontier's are verified to agree — 0.006
apart at 100% coverage on both sides — and that agreement is on AUROC. On per-structure Spearman
they do not agree: the frontier's own FoldX row is 0.430 against 0.383 here. Neither number is a
check on the other, since the two are computed on different rows (687 here against their 813) with
no pairing between them; note that the AUROC row that agrees sits on the same row-set mismatch, so
the mismatch is not what separates the two metrics. Both FoldX rows belong in the leaderboard, and
with the nine published `cath`/`ps_spearman_T10` rows of `frontier.tsv` beside the two measured
here it reads:

| # | method | CATH ps-Sp |
|--:|---|--:|
| 1 | CATH-ddG | 0.494 |
| 2 | USP-ddG | 0.493 |
| 3 | flex-ddG | 0.454 |
| 4 | **MuLAN + FoldX** (ESM-C 6B, scalar) | **0.438** |
| 5 | FoldX | 0.430 |
| 6 | BA-DDG | 0.402 |
| 7 | **FoldX alone** | **0.383** |
| 8 | Prompt-DDG | 0.303 |
| 9 | RDE-Network | 0.288 |
| 10 | DiffAffinity | 0.249 |
| 11 | PPIformer | 0.216 |

Fourth of eleven, on a lift of +0.055 [+0.001, +0.118] over the physics it is handed — and FoldX
alone outranks four of the seven published *learned* models unaided (flex-ddG and the FoldX row are
physics, not models), which is a statement about the benchmark as much as about any method. The
rank rests on which FoldX row is the comparator. Against the one measured here the lift is +0.055,
paired over the same 13 complexes with a CI excluding zero. Against the published row it is +0.008
— but that is a difference of two unpaired point estimates on different row sets with no interval,
and the +0.055 CI is ±0.06 wide, so it establishes no sign in either direction. Read it as: this
store's FoldX and the published FoldX are not separable on this tier, which is enough to stop the
+0.055 from being quoted as a margin over the field's physics baseline.
[`OPEN_QUESTIONS.md`](../OPEN_QUESTIONS.md) § "Repair count" attributes that difference to the
single `RepairPDB` pass this store ships at. Measured on the CATH-**single** subset only — 11
complexes and 421 rows, not the 13/687 of the row above — five repairs move FoldX alone from 0.398
to 0.459, paired Δ +0.060 [+0.020, +0.100]. That clears the published CATH-all row of 0.430, but
the pilot never touched multi-point and the CATH test set is 39% multi-point, so it is an
indication of where the gap comes from rather than a like-for-like recovery of it.

**by-complex and clustered are reference tiers**, for setting this work against papers that report
no CATH number. Of the two, by-complex is sound: baselines within 0.061, and the +0.049 lift sits
mid-field — above RDE-Network, level with Prompt-DDG, below CATH-ddG and BA-DDG.

**The clustered headline is not quotable at all.** 0.547 against 0.477 is a 0.070 win resting on a
0.218 difference in the baseline both methods start from, and the lift column runs the other way.

This is inference from one side of the comparison: the frontier's FoldX predictions are not
published, so the mechanism is consistent with the evidence rather than demonstrated. The first table
above regenerates with `experiments/rescore_perstructure/foldx_gap_analysis.py`; the second needs
the clustered FoldX splits, which do not ship, and the script says so and stops.

→ deep-dive: `experiments/rescore_perstructure/FOLDX_ALONE_BASELINE_RESULT.md` (§v3),
`experiments/beyond_foldx/REDUNDANCY_RESULT.md`, `experiments/beyond_foldx/BLEND12_RESULT.md`

---

## Sub-doc index

| Doc | Scope |
|---|---|
| `history/PLM_COMPARISON_S1102.md` | Ankh vs ProstT5 narrative + full runtime table |
| `experiments/RESULTS_ESMC.md` | ESM C 6B: loading, results (new best PLM), method |
| `experiments/RESULTS_ESM2.md` | ESM2-3B: 10-fold CV (ties ProstT5), controlled-width read, method |
| `experiments/RESULTS_PROSTT5.md` | ProstT5 backend: wiring, results, next steps |
| `history/PLAN_PROSTT5_STRUCTURE_v2.md` | run6 series plan (scalar-mix, 3Di structure) + results |
| `history/PROSTT5_STRUCTURE_OPTIONS.md` | post-run6c structure-fusion options + E1/B1 results + verdict |
| `experiments/BENCHMARK_DATASETS.md` | S1131/S4169/S2003 definitions, overlaps & differences |
| `experiments/RESULTS_BENCHMARKS.md` | Four-PLM (ProstT5/ESM2/Ankh/SaProt) 10-fold CV (Pearson/Spearman/RMSE/MAE) on S1131/S4169/S2003 |
| `experiments/RESULTS_CV10.md` | 10-fold CV per-fold tables + significance |
| `experiments/RESULTS_CV10_AUG.md` | Augmented 10-fold CV: per-fold tables + paired Δ vs baseline |
| `experiments/LAYER_PROBE.md` | Layer-probe method & interpretation |
| `experiments/results/probe_{ankh,prostt5}.md` | Raw per-layer probe tables |
| `experiments/AUGMENTATION.md` | Tiered augmentation roadmap + run3/run4 results |
| `experiments/HYPERPARAMETERS.md` | Full hyperparameters for every run |
| `history/PLAN_AIDO.md` (archived) | AIDO.Protein integration plan & hypotheses |
| `docs/history/PLAN_FULL_SKEMPI.md` | Full-SKEMPI campaign design: the protocol ladder and what each rung tests |
| `experiments/METRIC_RATIONALE.md` | Why per-structure Spearman at T ≥ 10 is the conclusion metric |
| `experiments/full_skempi_seqonly/SP_FOLDX_COVERAGE_AUDIT.md` | The 28% / 87.8% / 99.2% FoldX merge lineage and the join defect behind it |
| `experiments/FOLDX_COVERAGE_FINDINGS.md` | Which mutations the FoldX channel does not cover, and why |
| `experiments/full_skempi_seqonly/SUMMARY_cath.md` | CATH-superfamily hold-out, split by mutation order |
| `experiments/full_skempi_seqonly/SUMMARY_mp_bycomplex.md` | Multi-point by-complex tier + paired bootstrap |
| `experiments/full_skempi_seqonly/SUMMARY_mp_clustered.md` | Multi-point clustered tier + paired bootstrap |
| `experiments/rescore_perstructure/FOLDX_ALONE_BASELINE_RESULT.md` | FoldX used directly as the comparator: every arm contrasted against it, with cluster-bootstrap CIs |
| `experiments/beyond_foldx/REDUNDANCY_RESULT.md` | How much of the base arm's signal the FoldX channel already carries |
| `experiments/beyond_foldx/BLEND12_RESULT.md` | The 12 FoldX terms reweighted linearly with no backbone — what the decomposition is worth on its own |
| `scripts_plots/results_matrix.py` | Status + data matrix over every tier; `--ps` builds the per-structure ladder |
| `experiments/README.md` | Scripts and how to reproduce |
| `PLM_BACKEND.md` | How multi-PLM support works (design) |
| `scratch/RESULTS.md` (gitignored) | Paper-reproduction round, data provenance |
