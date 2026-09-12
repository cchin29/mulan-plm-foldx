# FoldX & structure in MuLAN — the complete evaluation, and the augmentation plan

**2026-08-05.** The full account behind the FoldX figures, and the augmentation plan.

> **The lineage every number here carries.** Figures are computed at **99.2 %** single-point FoldX
> coverage. That matters because the FoldX-alone baseline on the single-point tiers is **0.418** at
> this coverage and 0.363 at the 87.8 % it replaced, so a number quoted from an earlier copy of
> this document is not comparable. The multi-point and S1102 tiers are the same under both, because
> their coverage never moved, and every `base` arm is the same everywhere, because it never reads
> the channel.
>
> The headline consequence: ESM-C 6B's CATH-all scalar arm reads **0.438**, not the 0.468 the
> 87.8 % lineage gave, and it beats FoldX alone by **+0.055 [+0.001, +0.118]** rather than +0.085.
>
> **The comparator this document is built around** is the FoldX-alone baseline (§6). Without it,
> the FoldX channel's apparent "+0.21 to +0.31 lift" reads as the model's, when it is largely the
> model learning to read a number that already scores ~0.4 by itself.

Two halves:

- **Part I — what was evaluated.** The structure thread end to end: ProstT5 + hidden layers →
  embedding/SaProt → FoldX mutant-3Di → the FoldX channel (scalar & 12-term MLP) → **the FoldX-alone
  baseline** → the measured ceiling and flexible-backbone extensions.
- **Part II — where it goes next.** FoldX ΔΔG as *data augmentation*, a leakage-minimising design,
  and the evaluation on ΔΔG + interface prediction.

_Conventions, as used throughout this repository: headline ΔΔG metric = **per-structure Spearman ρ (T≥10)**;
"clustered" always means **CD-HIT ≤60 % sequence identity** (the retrain split), never the iDist
interface-cluster leakage diagnostic. Every number is traceable to a repo file; the map is in the
Appendix._

---

# Part I — The structure evaluation, end to end

## 0. The question and the four routes

MuLAN predicts binding ΔΔG from **frozen PLM embeddings of sequence alone**: embed WT and mutant,
take the per-residue difference `mut − wt`, pool through a light-attention head, regress ΔΔG. The
head has exactly one auxiliary input slot — `add_scores` / `zs_scores` — where one precomputed
per-mutation number can be concatenated before the final linear layer.

Binding is a **structural, cross-chain** phenomenon, so the standing question was: *does adding
structure help?* Four routes were tried. They are not four unrelated experiments — each was the
diagnosis of the previous failure.

```
ProstT5 has a 3Di structure channel, but MuLAN never uses it
        │
        ├─→ [1] feed WT 3Di in by late concat ────────────── HURTS (−0.077)
        │        └─ diagnosis: 3Di is fixed across the mutation, so it
        │           cancels in mut − wt, and the extra width dilutes AA
        │
        ├─→ [1b] hidden-layer probe: is it just the wrong layer? ─ NO
        │
        ├─→ [2] let the embedder own the fusion (SaProt AA+3Di) ─ HELPS +0.026
        │
        ├─→ [3] make the structure mutation-specific:
        │        FoldX BuildModel → mutant PDB → mutant 3Di ── NO-OP (guaranteed)
        │        └─ diagnosis: BuildModel repacks side chains only;
        │           3Di reads backbone only ⇒ mutant 3Di ≡ WT 3Di
        │
        └─→ [4] stop using FoldX as a structure generator; use its native
                 output — a physics binding ΔΔG — as a score channel ──── REAL
                 └─ scalar (Stage 1) and 12-term MLP (Stage 2)
                    …but see §6: measured against FoldX itself, not against
                    MuLAN-without-it
```

---

## 1. Thread 1 — ProstT5's 3Di channel, and the hidden-layer probe

### 1.1 The hypothesis

ProstT5 underperforms Ankh-large on S1102 (CV10 pooled 0.805 vs 0.832) *despite* being the one PLM
with an explicit structure modality. Hypothesis **H-struct**: the gap exists because MuLAN's
sequence-only pipeline never touches ProstT5's 3Di (fold-mode) channel. Feed the structure in and
the gap closes.

### 1.2 The four fusions

WT 3Di from the SKEMPI 2.0 PDBs via mini3di, index-aligned to the AA FASTA, embedded through
ProstT5 fold-mode. Same split / optimizer / budget as the AA-only baseline.

| Run | Structure input | How it enters | Test PCC |
|---|---|---|---|
| run2 — AA-only baseline | none | — | **0.740** |
| run6c | WT 3Di | per-residue concat AA(1024) ⊕ 3Di(1024) → 2048-d | 0.663 |
| run6-E1 | WT 3Di | + learned scalar gate on the 3Di block | 0.705 (gate → 0.46) |
| run6-B1 | WT 3Di | 3Di as head context, *outside* the `mut − wt` difference | 0.673 |
| run6-C3 | interface contacts | contacts as an attention bias | 0.662 (α → 0.017 ≈ off) |

Two things for the slide:

- **Every fusion lost to sequence-only.** The headline concat cost −0.077 PCC.
- **Every learnable structure knob trained itself off.** The E1 gate collapsed to 0.46, the C3
  attention bias to ≈0.017. The head was *given the option* to use structure and declined.

### 1.3 Diagnosis — the 3Di cancellation

No experimentally determined mutant structures exist, so every mutant reuses its **wild-type**
3Di. Held constant across the mutation, the 3Di block contributes only a constant complex-level
offset to `mut − wt` — **zero mutation-specific signal** — while doubling the input width dilutes
the AA signal that was working. (run6b, 3Di-only, was skipped as algebraically degenerate: fixed
WT 3Di ⇒ `mut_emb ≡ wt_emb` ⇒ no signal.)

This diagnosis forks into threads 2 and 3: either the problem is *how* structure is fused
(→ SaProt), or the problem is that structure isn't *mutation-specific* (→ FoldX mutant structures).

### 1.4 The hidden-layer probe

One alternative had to be excluded first: MuLAN reads `last_hidden_state`, but a PLM's final layer
is specialised to its *pretraining objective*, which may shed the local per-residue signal ΔΔG
needs. ProstT5's top layer in particular is trained to translate AA ↔ 3Di.

Method (`layer_probe.py`): for every hidden state, build **site delta** (`mut_emb[pos] − wt_emb[pos]`,
local) and **pool delta** (`mean(mut) − mean(wt)`, global); fit a RidgeCV probe (5-fold CV,
standardised); report held-out PCC. N = 1100. A *linear proxy* for the attention head — the
**ranking across layers** is the result, not the absolute values.

| PLM | | Best layer | PCC | Default final layer |
|---|---|---|---|---|
| **ProstT5** (24 layers, 1024-d) | site delta (local) | **3** | 0.765 | layer 24 → **0.702** |
| | pool delta (global) | 10 | 0.754 | layer 24 → 0.731 |
| **Ankh-large** (48 layers, 1536-d) | site delta (local) | 47 | 0.799 | layer 48 → **0.796** |
| | pool delta (global) | 14 | 0.775 | layer 48 → 0.751 |

- **ProstT5's final layer is its *weakest* contextualised layer for the local signal** (0.702 vs a
  0.74–0.765 plateau over layers ~2–21) — consistent with an AA↔3Di translation objective
  discarding local AA identity.
- **Ankh's final layer is already near-optimal** (0.796 vs best 0.799, statistically tied) —
  span-denoising keeps the top layer general-purpose.
- Probe validity: its ordering (Ankh 0.80 > ProstT5 0.765) matches full MuLAN (run1 Ankh 0.757 >
  run2 ProstT5 0.740).

**Acting on the probe — also negative.** In full MuLAN, run3a (ProstT5 layer 7) scored **0.648** and
run3b (concat 7⊕10) **0.429**, both far below the final-layer baseline 0.740. The layer ranking is
real in a linear probe and does **not** survive contact with the attention head.

**Net of thread 1:** ProstT5's structure modality is not recoverable inside MuLAN — not by fusing
3Di, not by gating it, not by picking a better layer. And explicitly: **no Ankh mid-layer run was
queued**, because the probe says there is nothing there to win.

---

## 2. Thread 2 — the embedding sweep and SaProt

The first fork: maybe structure is fine and late concatenation is the problem. SaProt tests this
directly — every input token is a **fused amino-acid + 3Di symbol**, and the model was *pretrained*
on that fused vocabulary, so structure enters at the input rather than bolted on downstream.

| SaProt-650M (S1102, 10-fold) | Test PCC | RMSE | MAE | SCC |
|---|---|---|---|---|
| **real WT 3Di** | **0.842 ± 0.050** | 1.276 | 0.926 | 0.729 |
| 3Di masked with `#` (sequence-only control) | 0.816 ± 0.069 | 1.362 | 0.976 | 0.699 |

**+0.026 PCC, 8/10 folds, paired.** The same WT 3Di that *hurt* ProstT5 by −0.077 *helps* SaProt by
+0.026. The slide line: **it is the fusion, not the data, that decides.**

Two corollaries that matter later: SaProt's WT-3Di run (0.842) is the **ceiling for the entire
SaProt+FoldX structure route** (§3), and the lift stacks with Tier-1 augmentation (→ 0.852), so
structure and data augmentation are not competing for the same headroom.

---

## 3. Thread 3 — FoldX as a mutant-structure generator

The other fork: get **real, mutation-specific** structure. Pipeline: `RepairPDB` → `BuildModel`
(mutant PDB) → mini3di → compare per-residue 3Di vs WT.

**Result: 0 of 39 mutations changed the 3Di at any position.** The probe was deliberately
adversarial — all 36 of 1A22's S1102 mutations **plus 1ACB Leu38→Pro and Leu38→Gly**, the two most
backbone-perturbing substitutions available. No change even for →Pro/→Gly, and **not even at the
mutated residue itself** (`LI38P`: position-38 3Di `p` → `p`). Atom-level confirmation on 1KNE:
**0 of 232 backbone (N/Cα/C/O) atoms moved.**

**Mechanism — why this is not a sampling artefact.** FoldX `BuildModel` is a fast side-chain
repacker with sub-Ångström backbone motion. Foldseek-3Di is a coarse **20-state backbone**
descriptor. Sub-Å moves never cross a 3Di state boundary ⇒ **FoldX-mutant 3Di ≡ WT 3Di**,
*structurally guaranteed*, not statistically likely. It is a property of FoldX + 3Di, so it holds
for S1131 / S4169 / S2003 too, and no gentler mutation or larger benchmark rescues it.

**Consequence:** a full SaProt Stage-2 run (110-complex RepairPDB batch) would produce
byte-identical embeddings to Stage 1 and land on the same 0.842. It was **not run** — a ~10-minute
probe replaced hours of compute. Genuine mutation-induced backbone change requires a
**backbone-relaxing** method (energy minimisation / MD / flex-ddG / AlphaFold-on-mutant) — §7.

One more line worth saying: **FoldX was never needed for this anyway** — foldseek/mini3di on the WT
PDB gives the identical 3Di directly.

---

## 4. Thread 4 — FoldX binding ΔΔG as a score channel (**the "FoldX channel"**)

This is the definition used throughout this document.

### 4.1 How the FoldX ΔΔG is computed

Per complex, serial, resumable, cached as one JSON per complex (`scratch/foldx_s1102/build_ddg.py`
and its full-SKEMPI descendants):

1. **`RepairPDB`** on the crystal structure — one-time per structure (~3 min), reused forever.
2. **`BuildModel`** — build every mutant of that complex (side-chain repack → mutant PDB).
3. **`AnalyseComplex`** on the two SKEMPI interface chain groups (`#Pdb = PDB_<g1>_<g2>`), for both
   WT and mutant.
4. **ΔΔG_bind = InteractionEnergy(mutant) − InteractionEnergy(WT)**, and the same difference term
   by term.

**Chain mapping is validated, not assumed.** The dataset mutation carries a *role* chain (A/B =
seq1/seq2); SKEMPI's `Mutation(s)_cleaned` carries the *real* PDB chain. Mutations are matched on
`(wt_aa, pos, mut_aa)`, duplicates disambiguated role→group, and **every mapping is checked against
the actual WT residue in the repaired PDB**. Unresolvable cases are recorded `unresolved`, never
guessed (§5).

### 4.2 The two arms — precise definitions

Both arms feed **the same single `add_scores` head slot**; the regression head width is unchanged
between them. They differ only in what produces the number in that slot.

**Arm A — "FoldX scalar" (Stage 1).** The single **Interaction Energy** difference (mut − WT), per
mutation, standardised per fold, written to the `zs_scores` column and concatenated before the final
linear layer. Config `lightatt_addscores_config.json`, `add_scores: true`. **Added parameters: 1.**

**Arm B — "FoldX 12-term MLP" (Stage 2).** FoldX's decomposed terms — the mut−WT difference of each
— form a 12-vector per mutation, passed through a small in-network MLP
(**12 → 16 → ReLU → Dropout → 1**, **+226 parameters**) whose scalar output feeds *the same* slot.
Config `lightatt_addscores_mlp_config.json`, `add_scores: true`, `zs_mlp: true`, `zs_input_dim: 12`,
`zs_mlp_hidden: 16`. The MLP trains **jointly** with the head, not separately.

**The 12 terms** — the exact answer to "which FoldX parameters are included" — each as
mutant − WT of the `AnalyseComplex` interaction-energy decomposition:

| # | Term | # | Term |
|---|---|---|---|
| 1 | Interaction Energy *(= the scalar in Arm A)* | 7 | Solvation Hydrophobic |
| 2 | Backbone Hbond | 8 | Van der Waals clashes |
| 3 | Sidechain Hbond | 9 | entropy sidechain |
| 4 | Van der Waals | 10 | entropy mainchain |
| 5 | Electrostatics | 11 | torsional clash |
| 6 | Solvation Polar | 12 | backbone clash |

Term 1 is the aggregate Arm A uses, so **Arm B strictly contains Arm A's information** — the MLP
arm's question is whether the decomposition carries anything beyond the total. *(Note for the
methods slide: FoldX's Interaction Energy is a weighted sum of the components, so the 12-vector is
near-collinear by construction. Benign for an MLP, but state it rather than let a reviewer find it.)*

**Key property: the FoldX channel is model-independent.** The ΔΔG is physics computed from
structure alone — computed **once**, reused across every PLM backbone and every split, no re-run
when the embedder changes. This is what makes Part II affordable.

### 4.3 The ceiling analysis, done *before* any training

Out-of-fold, no training required (n = 1100, Ankh-large):

| Signal | Value |
|---|---|
| Raw FoldX ΔΔG vs experimental label | Pearson **+0.407**, Spearman +0.447 |
| r(FoldX, OOF Ankh **prediction**) — the *redundant* part | +0.35 |
| r(FoldX, OOF Ankh **residual**) — the *orthogonal* part | **+0.20** |
| Best linear combination (Ankh + β·FoldX), OOF | 0.837 → **0.846** (Δ +0.008–0.009) |
| OOF ceiling using all 12 terms | **+0.011** |
| RSM 2nd-order OLS (12 + squares + 66 interactions) | in-sample 0.876 → **OOF 0.824** (worse than baseline) |

Before a single training run: FoldX correlates 0.41 with truth, **but ~80 % of that is already
inside Ankh's predictions**; only ~+0.20 is orthogonal, capping a linear scalar near +0.01 PCC.

### 4.4 What it delivered — measured

**S1102, CV10, paired, Ankh-large, 300 ep / patience 30 (pooled Pearson, the paper's metric):**

| Arm | Test PCC | RMSE | MAE | SCC | Paired Δ | Folds won |
|---|---|---|---|---|---|---|
| baseline (no score) | 0.8217 ± 0.051 | 1.344 | 0.960 | 0.732 | — | — |
| **+ FoldX scalar** | 0.8310 ± 0.046 | 1.313 | 0.941 | 0.745 | **+0.0093** | 8/10 |
| **+ FoldX 12-term MLP** | 0.8383 ± 0.047 | — | — | — | **+0.0166** (p ≈ 0.02) | 8/10 |

The scalar lands **exactly on its predicted +0.008–0.009 linear ceiling** — a satisfying validation
of the pre-CV analysis. The MLP **clears** it (+0.0166 > +0.011), refuting the forecast that a
jointly-trained MLP would overfit below the scalar.

**Replication across PLMs (balanced splitter, 300 ep / p30), FoldX-MLP arm:**

| PLM (base PCC) | + FoldX-MLP, paired |
|---|---|
| ESM-C 600M (0.834) | **+0.019 (9/10)** → 0.853 |
| SaProt WT-3Di (0.849) | **+0.018 (7/10)** → 0.867 |
| Ankh-large (0.836) | **+0.018 (9/10)** → 0.854 |
| ProstT5 (0.823) | **+0.014 (8/10)** → 0.837 |
| ESM-C 6B (0.881) | +0.004 (6/10) → 0.885; scalar +0.007 |
| AIDO-16B (0.837, balanced 10CV) | scalar +0.011 → 0.849; MLP **+0.020** → 0.857 |

Two axes had been predicted; **one confirmed, one refuted:**

- ✅ **Weaker baseline → bigger lift.** ESM-C 600M (weakest) gains most (+0.019); ESM-C 6B
  (strongest) gains least (+0.004). FoldX is a **floor-raiser**.
- ❌ **"Structure-aware PLMs won't benefit" is wrong.** SaProt (+0.018) gains as much as
  sequence-only ESM-C 600M (+0.019). The 12-term head carries orthogonal signal even for an AA+3Di
  model.

**The floor-raiser pattern, quantified.** Per-fold ΔPCC anti-correlates with baseline strength:
**corr(baseline PCC, ΔPCC) = −0.61**. Biggest gain on the weakest fold (fold 8, +0.0215); the only
two losses are the two strongest folds (fold 0 −0.0103, fold 1 −0.0041).

**The irony worth a slide.** Fold 0 is *enriched in interface steric-clash hotspots* — precisely the
regime FoldX physics was meant to rescue — and is the single fold FoldX hurt most. The decomposition
explains it: only **Electrostatics** carries orthogonal signal beyond the aggregate (+0.006);
**Van der Waals clashes ≈ 0**. The "steric-clash tail" hypothesis was **wrong**.

**Supporting diagnostic — 18 interface hotspots:**

| | Pearson r vs truth | closer-to-truth |
|---|---|---|
| PLM ensemble | 0.28 | 8 |
| **FoldX** | **0.55** | 8 (2 ties) |

FoldX roughly **doubles** hotspot rank correlation and nails the gain-of-bulk steric subset
(r = 0.99 on those 4) — but misses cavity, electrostatic and enzyme-side cases, and its absolute
magnitudes are unreliable (1PPF LB18W: FoldX 11.9 vs true 7.4; 2FTL IB18A: 1.05 vs true 5.0).
**Real signal, wrong magnitudes** — the single most important observation for Part II.

**Scalar vs MLP.** The winner tracks dataset size: on the smallest benchmark (S1131) and on ESM-C 6B
the **scalar** wins; the MLP clearly wins on the larger S2003 and on AIDO; they tie on S4169.
Defensible: **scalar is the safe default (+0.006–0.013 everywhere); the 12-term MLP is the upside on
≥2k-mutation sets** — and, per §6, on the strong backbone it is the arm that clears the physics
baseline.

---

## 5. FoldX coverage on SKEMPI

Two bugs were masking already-computed values, and fixing them moved the science.

| Milestone | CATH-all (SP+MP) | single-point | multi-point |
|---|---|---|---|
| Original per-run FoldX dirs | ~48 % | ~28 % | — |
| **Fix 1 — unified `results_all`** (union across 5 campaigns) | ~69.5 % | 58.0 % | 98.7 % |
| **Fix 2 — dual-key `load_foldx`** | 90.8 % | 87.7 % | 98.7 % |
| **Fix 3 — the residual computed** | **99.1 %** | **99.2 %** | 98.7 % |

**Fix 3 closed the gap Fix 2 left, and it is the lineage every number below is computed on.** The
residual analysed in the paragraphs that follow was not, in the end, an artefact of the split
builder: most of it was computable and was computed. Multi-point coverage never moved, which is why
the multi-point tiers in §6 are identical before and after.

- **Fix 1 (plumbing, zero compute).** FoldX had run in five campaigns (full-SKEMPI +
  S1102/S1131/S2003/S4169), each writing its own results dir keyed to a different mutation subset;
  the mergers read only one. Symlinking each complex to its most-complete JSON across all five
  lifted single-point coverage **28 → 58 % at zero compute**.
- **Fix 2 (merger bug).** `load_foldx` force-remapped every mutation's chain to the split's
  `g1→A / g2→B` convention. Standard complexes need this; the **S4169-sourced complexes**
  (3BT1 / 1PPF / 1R0R / 3SGB …) already carry the split's native keys, so the remap returned `None`
  and **silently dropped all of them**. Dual-keying lifted single-point coverage **58 → 88 %**.

**The residual ~12 % is not a compute gap.** 510 mutations / 65 complexes, fully classified: 0
complexes missing a JSON, 0 remap failures, 502 mutations the split builder emits that `build_ddg`'s
SKEMPI list doesn't contain, 8 `unresolved`. They concentrate in large multi-chain **antibody /
TCR–pMHC** complexes (1DAN 42, 4NKQ 26, 3C60 24, 4P23/4P5T 24 …) where each has a **twin mutation
with identical `(wt, pos, mut)` on a sibling chain in the same interface group** (1DVF `YA32A` vs
`YB32A` — heavy vs light, different residues, shared numbering). Filling them means guessing which
copy the experiment mutated, with a real chance of writing a **wrong** ΔΔG. **Decision at the time: stop at
87.7 % single / 90.8 % combined**; uncovered rows fall back to the per-fold standardised mean (0) —
signal dropout, not leakage. That decision was later reversed for everything except the genuinely
ambiguous twin-mutation rows, taking coverage to 99.2 % single / 99.1 % combined (Fix 3 above); the
residual that remains is this twin-chain ambiguity, and it is still not filled by guessing.

**The fix moved results, not just plumbing.** CATH split, mean `test_pcc`, 90.8 % vs the preserved
28 %-coverage runs:

| model | base | foldx (pre → post) | Δ |
|---|---|---|---|
| ESM-C 6B | 0.5200 | 0.5163 → **0.6389** | **+0.1226** |
| Ankh3-large | 0.2673 | 0.4636 → **0.5265** | +0.0629 |
| Ankh3-xl | 0.6162 | 0.4988 → **0.5402** | +0.0414 |
| AIDO-16B | 0.4870 | 0.5103 → **0.5554** | +0.0450 |

> **Framing note.** In the published SKEMPI frontier (RDE-Network, Prompt-DDG, DiffAffinity,
> CATH-ddG, USP-ddG), **FoldX is an unsupervised _baseline predictor_, not an input feature.**
> MuLAN + a FoldX `add_scores` channel is the less common setup, so "coverage of a FoldX feature" is
> our own concern — no published FoldX-coverage percentage exists to compare 88 % against. What *is*
> verified: those methods evaluate on essentially all of SKEMPI 2.0, dropping only missing-ΔΔG rows
> and individual un-processable structures (CATH-ddG's concrete case: PDB **1KBH**, 92 mutations).
>
> **That framing note is also the clue we should have followed sooner.** If the frontier uses FoldX
> as a *baseline*, then so must we — §6.

---

## 6. The FoldX-alone baseline ⚠️

### 6.1 The missing comparator

Every MuLAN+FoldX number above is compared against **MuLAN base**. None was compared against
**FoldX by itself**, scored directly as a predictor. That is the comparator the frontier reports,
and it is the *free* alternative to training anything. The FoldX ΔΔG was already sitting in column 5
of every `splits_*_foldx` TSV next to the experimental label in column 4 — the comparison cost
nothing.

**Method** (`experiments/rescore_perstructure/foldx_alone_baseline.py`) is deliberately identical to
`rescore.py`: complex key `chain1_id.split("_")[0]`; per-structure Spearman over complexes with
n ≥ 10, skipping zero-variance complexes; test folds **pooled** before grouping; **95 % CI = cluster
bootstrap over complexes** (B = 10,000, seed 0); MuLAN-vs-FoldX contrasts **paired** on the shared
complex set. **Validation: every `base` value reproduces `results_matrix_ps.csv` to 4 decimals.**

### 6.2 Ankh-large

Recomputed at 99.2 % coverage. Every value here is the mean per-structure Spearman at T ≥ 10, folds
pooled before grouping by complex; the CATH single/multiple breakouts live in `RESULTS.md` §25.

| Split tier | cplx | **FoldX alone** | base | + scalar | + 12-term MLP |
|---|---|---|---|---|---|
| CATH-all | 13 | 0.383 | 0.331 | 0.421 | 0.387 |
| by-complex-SP | 96 | 0.418 | 0.284 | 0.435 | 0.444 |
| clustered-SP | 96 | 0.418 | 0.142 | 0.406 | 0.428 |
| MP-by-complex | 43 | 0.393 | 0.340 | 0.397 | 0.391 |
| MP-clustered | 43 | 0.393 | 0.091 | 0.331 | 0.355 |
| S1102 by-complex | 24 | 0.443 | 0.289 | 0.495 | 0.482 |
| S1102 clustered | 24 | 0.443 | 0.127 | 0.426 | 0.432 |

**FoldX alone is stable at ρ ≈ 0.38–0.44 on every tier — including the ones where MuLAN
collapses.** It does not care whether the test complex is out-of-cluster or out-of-superfamily,
because it is physics computed from the *test* structure. That stability is what makes it a
demanding baseline under leakage control, and raising it from 0.363 to 0.418 on the single-point
tiers made it more demanding still.

**Ankh-large now has 0 significant wins in 27 contrasts.** Its largest is CATH-single's 12-term arm
at +0.084, with a CI spanning zero on 11 complexes. At 87.8 % coverage it had one.

### 6.3 All ten backbones — 232 contrasts

| | count |
|---|---|
| significantly **beat** FoldX alone | **19** |
| significantly **lose** to FoldX alone | 61 |
| indistinguishable | 152 |

The win count barely moved when the baseline rose — the FoldX arms gained about as much as the
comparator did — but its *composition* changed, and that is the finding. Counts are over each
backbone's own contrasts, so totals differ where a backbone has fewer tiers on disk. The last
column is its largest **significant** FoldX-arm contrast, or its largest of any kind where none is
significant.

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

Two backbones lost their only win (Ankh-large, SaProt-650M) and four gained one. ESM-C 6B's
CATH-single result fell from +0.127 to **+0.097 [+0.017, +0.194]**.

**ESM-C 6B, every tier:**

| tier | FoldX alone | base | + scalar | + 12-term MLP | moved by Fix 3? |
|---|---|---|---|---|---|
| CATH-all | 0.383 | 0.360 | **0.438** | 0.431 | yes — scalar 0.468 → 0.438 |
| by-complex-SP | 0.418 | 0.310 | 0.451 | 0.457 | yes |
| clustered-SP | 0.418 | 0.193 | 0.417 | 0.446 | yes |
| MP-by-complex | 0.393 | 0.374 | 0.391 | 0.423 | no |
| MP-clustered | 0.393 | 0.137 | 0.270 | 0.320 | no |
| S1102 by-complex | 0.443 | 0.452 | 0.546 | 0.551 | no |
| S1102 clustered | 0.443 | 0.370 | 0.460 | 0.384 | no |

The `base` column is identical at both coverage levels on every tier, because the base arm never
reads the channel. That is the control: the movement in the other two columns is the coverage
change and nothing else.

### 6.4 Five conclusions

**1. No `base` arm beats FoldX alone anywhere — 0 wins in 78 contrasts.** Without the channel, no
frozen-PLM MuLAN configuration reaches an unsupervised physics score under leakage control. This
conclusion is *strengthened* by the higher baseline, not weakened.

**2. The "+0.30 lift" framing must be retired.** *"The FoldX channel triples Ankh-large's clustered
ρ from 0.14 to 0.43"* is arithmetically true and materially misleading: the channel hands the model
a number that already scores 0.42 on its own. The correct statement is that on the clustered tiers
MuLAN+FoldX performs **the same as reporting FoldX directly**.

**3. The frontier comparison survives — for ESM-C 6B, with the baseline shown, and by less.** The
CATH figure is now **0.438**, and it still significantly beats FoldX alone, at
**+0.055 [+0.001, +0.118]** rather than v2's +0.085 — a CI whose lower bound is now essentially at
zero. Placing it beside USP-ddG 0.493 / CATH-ddG 0.494 is defensible **only with the FoldX-alone
row (0.383) in the same table**, and the top-level README additionally carries the *published* FoldX
row for that tier (0.430), against which the margin is within rounding. See `OPEN_QUESTIONS.md` on
the single-`RepairPDB` default, which is the likely reason our own baseline sits below theirs.

**4. Backbone choice decides whether the method clears the bar.** Ankh-large does not — it now has
zero significant wins. ESM-C 6B does. Its largest CATH result remains the **12-term MLP** on
CATH-single (+0.097), against the scalar's non-significant margin there.

**5. The clustered tiers are where the method comes closest to failing — 3 wins in 76 contrasts.**
All three are 12-term arms: ESM-C 6B +0.031 [+0.001, +0.061], ProstT5 +0.030 [+0.005, +0.053],
Ankh3-large +0.028 [+0.004, +0.052]. Not one scalar arm clears the baseline on either clustered
tier, and on MP-clustered several are significantly worse. At 87.8 % coverage this was 1 win in 56;
the tiers are still where the method is weakest, and they remain **what Part II should target**.

---

## 7. Cross-thread synthesis, orthogonality, and the ceiling

| # | Route | How structure enters | Result vs sequence-only |
|---|---|---|---|
| 1 | ProstT5 + WT 3Di | late concat of a separate 3Di block | **hurts** (0.663 < 0.740) |
| 1b | ProstT5 mid-layer swap | different hidden layer | **hurts** (0.648 / 0.429) |
| 2 | SaProt WT-3Di | **input-level** AA+3Di token, jointly pretrained | **helps +0.026** → 0.842 |
| 3 | FoldX mutant 3Di | remodel each mutant → new 3Di | **no-op** (≡ WT 3Di, guaranteed) |
| 4 | FoldX ΔΔG channel | physics scalar / 12-term into the head | **+0.009–0.017 (leaky, vs base)**; **0 to +0.13 vs FoldX alone, backbone-dependent** |

**Answer: yes — but only WT structure, only when the embedder owns the fusion, and (for physics)
only on a strong backbone, mostly on the by-complex and CATH tiers.**

### 7.1 The orthogonality measurement

CV10 runs save per-mutation OOF predictions on **identical folds**, so residuals can be correlated
directly.

| | ESM-C 6B | Ankh | ESM2-3B | SaProt | ProstT5 | FoldX |
|---|---|---|---|---|---|---|
| **ESM-C 6B** (0.869) | — | 0.84 | 0.80 | 0.84 | 0.78 | **0.46** |
| **Ankh** (0.837) | 0.84 | — | 0.86 | 0.83 | 0.86 | **0.46** |
| **ESM2-3B** (0.815) | 0.80 | 0.86 | — | 0.85 | 0.87 | **0.46** |
| **SaProt** (0.847) | 0.84 | 0.83 | 0.85 | — | 0.86 | **0.49** |
| **ProstT5** (0.811) | 0.78 | 0.86 | 0.87 | 0.86 | — | **0.55** |
| **FoldX** (0.442) | 0.46 | 0.46 | 0.46 | 0.49 | 0.55 | — |

- **PLM ↔ PLM residuals: 0.78–0.92, mean ≈ 0.84** — a shared blind spot, not decorrelated noise.
- **FoldX is the one orthogonal channel: 0.46–0.55.**
- **Ensembling confirms it.** Best single ESM-C 6B **0.869** → +SaProt 0.871 (noise) → +Ankh 0.866
  (*worse*) → 5-model average 0.860 (*worse*). Correlated errors don't cancel.

**Takeaway:** orthogonality here is **cross-modality (physics vs PLM), not cross-PLM.** Moving the
ceiling requires pairing a strong PLM with a physics/structure channel, not with more PLMs. §6 adds
the sharper version: even that pairing only reliably beats the physics channel *alone* on the
strongest backbone.

### 7.2 The ceiling, and what would break it

**Everything traces to one fact: FoldX `BuildModel` moves side chains, not backbone.** For the 3Di
route that rigidity is fatal and the failure is structural. For the ΔΔG-feature route the same
physics is *useful but shallow*: real cross-chain energetics (r = 0.99 on gain-of-bulk clashes,
doubled hotspot rank correlation), but a strong PLM has independently learned most of it, so only
~20 % is orthogonal.

**Three directions that could break it, in increasing cost:**

**(a) Flexible-backbone physics — Rosetta flex-ddG.** Runbook written
(`scripts_plots/FLEX_DDG_RUNBOOK.md`): 18 fold-0 hotspot targets with resfiles, chain-groups and
validated numbering; ~35-backrub-model ensemble per mutation, PackRotamers+Min on WT and mutant,
bound − separated, then the mandatory GAM/REF2015 recalibration. **Cost: ~15–25 core-hours per
mutation, ~300–450 core-hours for the 18** — 1–1.5 days on 14 cores. **Status: deferred**
(PyRosetta is license-gated). **Success criterion is mechanism-specific, not accuracy:** does
relaxing the backbone tame the clash *overshoot* (1PPF LB18W 11.9 → ~7.4) and lift the cavity
*undershoot* (2FTL IB18A 1.05 → ~5.0)? Charge-changing cases stay hard — that's the FEP regime.
**§6 raises this direction's value:** if MuLAN mostly re-reports FoldX, then **improving FoldX** is
the lever, and better labels would be strictly better.

**(b) A complex-aware channel.** Current structural tooling is monomer-only, but FoldX's genuine
signal is cross-chain. Encoding the mutant side chain *in the partner's pocket* rather than as a
single-chain 3Di is where this physics could become a lever rather than a floor-raiser.

**(c) More data — Part II.** The ceiling analysis assumes a fixed training set. FoldX ΔΔG is not
only a *feature*; it is a cheap, model-independent **label generator**.

---

# Part II — The FoldX augmentation plan

The objective this part addresses: generate FoldX ΔΔG predictions for additional mutations and
incorporate them into a leakage-minimising training set, then evaluate whether MuLAN on Ankh-large,
with and without the FoldX channel, improves on both ΔΔG prediction and interface prediction under
that data extension.

> **Status as of 2026-08-06 — read before the design below.** Part II is a plan, and one branch of
> it has since been executed with a result that the plan did not anticipate, then revised twice.
>
> **Source C (Tier-1: reverse-mutation and identity rows, §10.4) is built and measured.** On
> `fullSK clustered-ALL` it lost on six of six FoldX arms and *gained* on the one base arm then
> available — Ankh-large, **+0.056 [+0.010, +0.104]**. Three significant deltas in two directions
> inside one backbone appeared to rule out "augmentation fails at this scale" and to point at a
> conflict with the FoldX channel. **Neither half of that reading survived.**
>
> **The FoldX half was measured on a defective channel.** On 2026-08-05 the augmented channel was
> found to negate the FoldX terms *after* standardisation rather than before, putting a constant
> offset of about one channel width on 51 % of the training rows while the label was negated
> correctly in raw units — so label and channel pointed in opposite directions on half of train. The
> builder is repaired and the splits rebuilt. Ankh-large's two arms have been re-run against the
> repaired splits: the 12-term arm's loss disappears entirely (−0.042 → −0.001, CI centred on zero)
> and the scalar arm's falls to a third (−0.065 → −0.024). Most of the measured harm was the defect.
> Four arms are outstanding and are still marked uninterpretable.
>
> **The base half has one measurement and four failed replications.** The +0.056 is unaffected by
> the channel defect — the base arm reads only the 4-column split — but CATH (same backbone, new
> tier, −0.069), bycomplex-ALL (new backbone, −0.001) and the two designed-test backbones on
> clustered-ALL itself (ESM-C 6B −0.032, AIDO-16B −0.046) all fail to reproduce it, and all four CIs
> exclude +0.056. What remains open is whether the effect is specific to Ankh-large or is a false
> positive; no second Ankh model has been measured on this tier.
> `RESULTS.md` §24 and `experiments/beyond_foldx/FOLDX_ANTISYMMETRY_RESULT.md` §1 carry the
> derivation and the verification.
>
> A second finding bears on §10.4's premise: FoldX's own antisymmetry is **approximate**. Over 1744
> reversals computed directly, reverting a mutation recovers 0.742 [0.633, 0.866] of the forward
> effect, and the error grows with |ΔΔG|. Tier-1 assumes exact antisymmetry, and no arithmetic in
> the builder can repair that.
>
> Sources A and B (§10.2, §10.3) remain unbuilt.

## 8. The reframe

So far FoldX has been used **once per real SKEMPI mutation, as an input feature**. The proposal is
to use it **on mutations SKEMPI never measured, as a label**.

| | FoldX as **feature** (done) | FoldX as **label** (proposed) |
|---|---|---|
| Applied to | the 7,085 SKEMPI mutations | arbitrary mutations on any PPI structure |
| Supplies | 1 or 12 extra numbers per row | **new training rows** |
| Bounded by | experimental data availability | compute only |
| Failure mode | redundancy with the PLM | label noise (FoldX ≠ truth) |
| Available at test time | **yes** — the channel is an input | **no** — it only shapes training |

**§6 changes the motivation, and makes it sharper rather than weaker.** The channel route is
*exhausted*: the model already extracts essentially all of FoldX's information, and on 8 of 9 tiers
adding the channel just reproduces FoldX. So augmentation is only interesting if it does something
the channel structurally **cannot**:

> **The reframed goal.** Not "add more FoldX signal" — the channel already delivers that. Instead:
> **distil the physics into the sequence pathway**, so the model carries it *without* needing a
> FoldX computation at inference, and — the harder target — **fix the clustered tiers**, where the
> learned model currently degrades the signal it is handed (§6.4 conclusion 5).

That reframing has a concrete, falsifiable form (§12, criterion A): a model trained **with**
pseudo-labels and evaluated **without** the channel that reaches FoldX-alone accuracy would be a
genuinely new object — a FoldX-free predictor at FoldX-level skill — and is a claim the channel
cannot make by construction.

**The standing risk, stated up front:** FoldX's magnitudes are unreliable (§4.4 — overshoots
clashes, undershoots cavities, misses electrostatics). Training on FoldX labels risks teaching
MuLAN *FoldX's biases*. The design below puts real SKEMPI labels in charge of every reported number.

---

## 9. Measured cost model

Measured from the repo, with the v1 errors corrected:

| Quantity | Value | Confidence |
|---|---|---|
| FoldX BuildModel per mutation per core | ~7.7 s (median over 54 complexes with ≥20 mutants) | **measured** |
| FoldX **end-to-end** per mutation (incl. `AnalyseComplex` on 2 chain groups) | **~12–15 s — not yet measured** | ⚠️ v1 quoted 7.7 s as end-to-end; that was BuildModel only |
| FoldX RepairPDB | ~3 min/structure, **one-time, already paid for 211 structures** | measured |
| Repaired structures on disk | **211** (`work`, `work_S1131`, `work_S2003`, `work_S4169`) | measured |
| SKEMPI PDBs on disk | 690 | measured |
| Interface positions per complex (both partners, heavy-atom ≤5 Å) | **39.5** (12 sampled complexes, range 23–49) | **measured** |
| Ankh-large embedding | ~13 s/mutant chain (6,422 files / 21.9 h, Mac) | ⚠️ **upper bound** — wall-clock span includes idle |
| Ankh-large embedding storage | ~1.3 MB/mutant chain (8.5 GB / 6,422) | measured |
| GPU embedding throughput | **unmeasured** — v1's "5–10× on the 3090" was invented | ⚠️ measure before scheduling |
| Free disk on Mac / GPU box | **unchecked** | ⚠️ check before committing 33–65 GB |

**The binding constraint is embeddings, not FoldX.** FoldX costs ~12 s and ~1 KB per mutation; the
PLM embedding costs ~13 s and **1.3 MB**.

| Augmentation size | FoldX (14 cores, at 12 s/mut) | Ankh-large embeddings | Storage fp32 / fp16 |
|---|---|---|---|
| 10 k mutations | **~2.5 h** | ~36 h Mac; not measured on GPU | 13 GB / 6.5 GB |
| **50 k mutations** | **~12 h** | ~180 h Mac; not measured on GPU | 65 GB / **33 GB** |
| 158 k (full saturation, 211 complexes × 750) | ~38 h | prohibitive on Mac | 205 GB / 103 GB |

**Recommended target: ~50 k FoldX-labelled mutations** (a 7× expansion over SKEMPI's 7,085), with
augmentation embeddings stored in **fp16** — they are only used for pre-training.

---

## 10. Design

### 10.1 The leakage control — the structural point

> FoldX ΔΔG is **split-independent physics**. Compute it **once over all complexes**; control
> leakage entirely at the *inclusion* step. For each fold, the augmented training set contains
> **only rows whose complex is in that fold's training set** — and on the clustered / CATH tiers, no
> complex in a test *cluster* or test *superfamily*. Validation and test sets remain **untouched
> real SKEMPI mutations**.

One FoldX campaign serves every split tier; leakage control is a filter, not a recompute.

**Train/test sizes per configuration** (fold 0) — this table also answers the dataset-size
question:

| Split | train muts | train complexes | test muts | test complexes |
|---|---|---|---|---|
| full-SKEMPI CATH-superfamily | 4,213 | 257 | 687 | 50 |
| full-SKEMPI by-complex (single+multi) | 3,275 | 192 | 1,934 | 112 |
| full-SKEMPI clustered CD-HIT ≤60 % (SP) | 2,114 | 115 | 1,673 | 112 |
| full-SKEMPI by-complex (MP) | 925 | 83 | 546 | 48 |
| full-SKEMPI clustered CD-HIT ≤60 % (MP) | 920 | 71 | 546 | 39 |
| S1102 clustered CD-HIT ≤60 % | 428 | 50 | 593 | 37 |

### 10.2 Source B (primary) — PPI structures outside SKEMPI **[promoted from v1]**

**Why this is now primary.** The measured failure is **between-complex** (base ρ falls 0.39 → 0.28
→ 0.14 as splits tighten). Source A densifies mutations *inside complexes the model already sees* —
the wrong axis. Only Source B adds new structural families, so it is the only source that targets
the CATH-superfamily and clustered tiers directly, which §6.4 identifies as where the method fails.

Select PDB heterodimers with **no CATH superfamily and no ≤60 % sequence-identity match to any
SKEMPI complex**, repair them, saturate their interfaces. **Leakage is zero by construction** — these
complexes are in no test set.

Selection criteria to fix before running: 2 chains (MuLAN's single-chain-per-partner rule), both
50–500 residues, resolution ≤ 2.5 Å, buried surface ≥ 500 Å², no CATH/CD-HIT overlap with SKEMPI,
family-diverse. Cost: + ~3 min RepairPDB per structure (100 structures ≈ 5 core-hours — negligible)
plus interface detection and **both** WT and mutant embeddings (WT chains are not free here).

### 10.3 Source A (secondary) — interface saturation on SKEMPI complexes

Cheapest per row, because RepairPDB is already paid for 211 structures and WT chain embeddings
already exist — the marginal cost is one mutant chain embedding. **Site selection already exists:**
`experiments/gen_interface.py` marks every residue whose heavy atoms are within a cutoff (default
**5.0 Å**) of any partner-chain heavy atom, index-aligned to the same FASTA numbering MuLAN and the
FoldX mapping use.

At the measured 39.5 interface positions/complex, full saturation (×19) ≈ **750 mutations per
complex**; 211 complexes ⇒ ceiling **~158 k**. On the clustered-SP tier, whose training set is only
**2,114 real mutations across 115 complexes**, full saturation on those same complexes yields
**~86 k rows** (or ~37 k with an 8-substitution chemistry panel) — a **17–40× expansion of exactly
the tier where the base PLM scores 0.14**.

**Its role in the design is as the control arm**, not the treatment: run it at equal mutation budget
against Source B. *More mutations on old complexes* vs *the same budget on new complexes* is the
scientific question, and the contrast is a better result than either arm alone.

### 10.4 Source C (implemented) — Tier-1 physical augmentation

`experiments/augment.py` already produces **reverse mutations** (ΔΔG antisymmetry: reverse example,
label negated, reusing the cached mutant embedding as the "wild type") and **identity anchors**
(wt→wt, ΔΔG = 0, one per complex). Measured: +0.012 for Ankh on the single split (0.757 → 0.769),
+0.002 to +0.009 on the balanced splitter — smaller than the FoldX-MLP channel's +0.014–0.019. Free
and orthogonal, so carry it as a **factor**, not a competing arm. Note it currently **skips
multi-point mutations** for reversal — worth extending, since the MP tiers are the weakest.

---

## 11. Training protocol

`mulan/data.py` has **no per-example sample-weight support**. Recommended: **two-stage
pre-train → fine-tune.**

1. **Stage A (pre-train):** train the head on the *pseudo-labelled* augmented rows only (that fold's
   training complexes), FoldX ΔΔG as target.
2. **Stage B (fine-tune):** re-initialise the optimizer, load Stage-A weights, train on the *real*
   SKEMPI training rows to convergence, early-stopping on the real validation set.
3. Report **only** Stage-B test numbers, on untouched real mutations.

### 11.1 ⚠️ The Stage-A channel switch

**This is a correctness requirement, not a preference.** In the "+channel +augmentation" cell, if
Stage A runs with the channel on, the **training target and the `add_scores` input are the same
number**. The head drives the loss to ~0 by learning the identity map from the channel slot and
learns nothing from the embeddings — and **the loss curve would look excellent**, so nothing in the
logs would flag it. The 12-term MLP arm has the same problem in milder form (term 1 *is* the target).

So: **Stage A trains the pure sequence pathway to imitate FoldX (channel off); Stage B re-enables
the channel and fine-tunes on real labels.** This is also exactly what makes criterion A in §12
measurable.

### 11.2 The `--init-from` code change

v1 said `scripts/train.py` "has no checkpoint-loading path." Reading it: `mulan.load_pretrained`
**is** in the training path (it initialises the model when the argument names a registered model),
and `save_model_ckpt` already writes `{state_dict, config}`. `--init-from <ckpt>` is a **~5-line
change reusing existing machinery.**

### 11.3 Required controls

- **Shuffled-label control** — same augmented rows, FoldX labels permuted **across complexes** (not
  within: within-complex shuffling preserves too much). If the gain survives, what we measured was
  extra *sequence* exposure, not FoldX information. This is what makes the result publishable.
- **Rank-vs-kcal target** — train Stage A on within-complex *ranks* rather than kcal/mol. Ranks
  discard FoldX's documented weakness (unreliable magnitudes) and match the evaluation metric.

---

## 12. Experiment matrix and success criteria

### 12.1 Backbone choice

Ankh-large is the fork's reference backbone and should be reported — but §6.3 shows Ankh-large does
not clear the FoldX baseline (0 wins / 5 losses), while **ESM-C 6B does** (6 / 4). Running augmentation
only on Ankh-large risks measuring a lift on a configuration that is below the free alternative in
the first place. **Report both; lead with ESM-C 6B; state plainly why.**

The un-augmented half of the matrix already exists for both backbones on all three tiers.

**Task 1 — ΔΔG (per-structure Spearman ρ, T≥10), per backbone:**

| | no augmentation | + FoldX augmentation |
|---|---|---|
| base | ✅ done | **new** |
| + FoldX scalar | ✅ done | **new** |
| + FoldX 12-term MLP | ✅ done | **new** |
| **base, trained with pseudo-labels, channel off at test** | — | **new — the interesting cell** |

× 3 tiers (CATH-superfamily · by-complex · clustered CD-HIT ≤60 %) × 2 backbones.

**Task 2 — interface prediction.** The pipeline exists in `experiments/attention_interface/` and
reproduces the paper's Interactome AUROC to three decimals (0.642). Re-score the augmented heads
against the same labels.

### 12.2 ⚠️ The Task-2 hypothesis, corrected

v1 predicted that interface-saturation augmentation would sharpen the attention map "because it
densely supervises the interface region that currently gets attention only incidentally."
**Measured: 99.1 % (233/235) of SKEMPI single-point mutations already sit on a cross-chain 5 Å
interface residue** (most complexes 100 %; 3SGB 177/177). The training signal is *already*
~100 % interface-localised — **the stated mechanism is void.**

**What augmentation actually changes is *within-interface* coverage.** SKEMPI is wildly non-uniform:
a handful of saturated protease/inhibitor complexes (3BT1 240 muts, 1R0R 191, 3SGB 191, 1PPF 190)
alongside a long tail of complexes with **1–13 mutations each**. Saturation moves a 1-mutation
complex to ~750.

**And the sign is genuinely ambiguous.** SKEMPI's positions are *hotspot-biased* — experimentalists
mutate residues that matter — and that bias may be exactly what makes the attention map peak
sharply. Uniform interface coverage could **flatten** it. The corrected hypothesis:

> Augmentation redistributes training mass from hotspot-biased sampling to uniform interface
> coverage. Whether that **sharpens or blurs** the attention map is the experiment. A blur is a real,
> reportable result.

**Run first, before any training:** interface-AUROC of the existing heads restricted to hotspot vs
non-hotspot interface residues. That measurement decides which direction to expect.

**Note the definitional mismatch to state, not paper over:** site selection uses cross-chain
**heavy-atom ≤5 Å** (`gen_interface.py`); the Task-2 AUROC labels are the Zenodo-18175031
**Cα ≤8 Å** Interactome set (heavy-atom 8 Å proxy on S1102). A 5 Å heavy-atom shell is a strict
subset of an 8 Å Cα shell, so augmentation is biased toward the interface *core*. Run a sensitivity
arm at heavy-atom 8 Å.

### 12.3 Success criteria

| Criterion | Reading |
|---|---|
| **A. Channel-off distillation.** Pseudo-label-trained base, evaluated with no FoldX channel, reaches FoldX-alone ρ | **The headline result.** A FoldX-free model at FoldX-level accuracy. Not a claim the channel can make. |
| **B. Clustered-tier repair.** Any arm significantly beats FoldX alone on clustered-SP or MP-clustered | Attacks the one place the method currently fails (3 wins in 76 contrasts today). Highest scientific value. |
| **C. CATH-tier confirmation.** ESM-C 6B + MLP + augmentation extends the +0.127 CATH-single win, with the CI clear of 0 across more folds | Consolidates the strongest existing positive. |
| **D. Interface AUROC moves ≥ 0.02 in *either* direction, with the hotspot/rim decomposition explaining it.** | Interpretability result; direction is the finding, not the sign. |
| **E. Nothing moves anywhere** | Also informative and worth reporting: FoldX's orthogonal signal is fully captured by the 1–12 numbers of the channel, and more of it as labels adds nothing. Bounds the whole FoldX direction. |
| **F. Gain survives on the leaky split but vanishes under leakage control** | Memorisation aid, not generalisation — report and stop. |

**Abandonment rule (absent in v1):** if after the 10 k pilot no arm reaches FoldX-alone on any tier,
and criterion A shows no distillation, **stop and write up §6 as the result.** "A physics baseline
that a frozen-PLM model cannot beat under homology control" is a publishable finding on its own.

### 12.4 ⚠️ Power and detectable effect size

From the cluster bootstrap, paired CI half-widths against FoldX alone:

| tier | complexes | ≈ half-width |
|---|---|---|
| by-complex-SP / clustered-SP | 96 | **±0.04** |
| MP tiers | 42 | ±0.09 |
| S1102 tiers | 24 | ±0.11 |
| CATH-all / single | 13 / 11 | **±0.08 to ±0.13** |
| CATH-multiple | 5 | ±0.16 |

v1's "+0.03" target is **inside the noise on every tier except the two 96-complex ones.** Two
consequences: (i) **pilot on clustered-SP / by-complex-SP**, the only tiers that can resolve a small
effect; (ii) **the CATH tier needs more folds before it can test anything** — today it is one fold,
13 complexes, with 1JTG contributing 194 of 687 mutations.

---

## 13. Risks

| Risk | Mitigation | Cost if it bites |
|---|---|---|
| FoldX label bias (clash overshoot, cavity undershoot, electrostatics missed) is learned | Pseudo-labels only pre-train; real labels own the fine-tune; train Stage A on within-complex ranks | Gain fails to transfer; ~1 week |
| **Stage A degeneracy** (target == channel input) | **Channel off in Stage A** (§11.1) | Silent total invalidation — highest-severity item |
| The gain is "more sequences seen", not FoldX | Shuffled-label control, permuted **across** complexes | Result reframed, not lost |
| Effect below the detection floor | Pilot on the 96-complex tiers only (§12.4) | Wasted GPU on an unresolvable target |
| Interface definition mismatch (5 Å heavy-atom selection vs 8 Å Cα labels) | State both; sensitivity arm at 8 Å | AUROC effect understated on rim residues |
| Embedding storage / GPU time | fp16; prefer the shorter chain; start at 10 k | Smaller augmentation set |
| FoldX failure rate on *chosen* mutations (→Pro/→Gly, buried small→large) is unknown | 500-mutation pilot measuring failure rate + ΔΔG distribution; clip or rank-transform | Fat destabilising tail dominates the pre-training loss |
| Multi-point tiers stay weak (augmentation is single-point) | Extend `augment.py` reversal to multi-point; sample double mutants from the saturation set | MP tiers unchanged |
| `results_all` symlink farm breaks when a 6th campaign is added | Verify coverage before and after the augmentation campaign | This class of bug already cost 28→88 % coverage once |

---

## 14. Sequenced plan

**Week 0 — measure, don't estimate (1 day).** FoldX end-to-end per mutation with `time`;
Ankh-large / ESM-C 6B embedding throughput on the 3090 (200-sequence run); `df -h` both machines;
`gen_interface.py` mean interface fraction over all 211 complexes; hotspot-vs-rim interface AUROC on
the existing heads (§12.2). **Also: re-run `foldx_alone_baseline.py` once the by-complex/clustered
rebuild lands** — FoldX-alone won't move, but the contrasts will.

**Week 1 — 10 k pilot, on the resolvable tier.** Interface enumeration over the 211 repaired
complexes → 10 k mutations from **clustered-SP fold-0 training complexes only** → FoldX → embeddings
→ `--init-from` → Stage A (channel off) / Stage B on ESM-C 6B and Ankh-large, base and +MLP →
**plus the shuffled-label control** and the **channel-off evaluation** (criterion A).
*Gate: does clustered-SP move off 0.19 (ESM-C 6B) / 0.14 (Ankh-large) toward FoldX-alone 0.418?*

**Week 2 — the Source A vs Source B contrast at equal budget.** ~100 leakage-free non-SKEMPI
heterodimers, repaired and saturated, vs the same mutation budget on training complexes. This is the
scientific question (§10.2), not a week-3 extra.

**Week 3 — scale the winner to ~50 k, complete the matrix, re-score interface AUROC.**

**Parallel, not blocking:** flex-ddG (§7.2a) pending the PyRosetta licence. §6 raises its priority —
if MuLAN largely re-reports FoldX, then improving FoldX is the lever.

---

## 15. Open questions at the time of writing

1. **Source B scope** — how many non-SKEMPI complexes, and by what family-diversity criterion?
2. **Panel vs full saturation** — 19 substitutions at ~12 positions/complex, or a chemistry-stratified
   8-substitution panel across all ~40? (Leaning: panel, for site coverage.)
3. **kcal/mol vs within-complex ranks** as the Stage-A target. (Leaning: ranks.)
4. **How many CATH folds** to build before the CATH tier can test anything (§12.4).
5. **Whether to report Ankh-large as primary** given §6.3, or lead with ESM-C 6B and report
   Ankh-large, the paper's backbone, as the secondary comparison.

---

## Appendix — file map

| Thing | Path |
|---|---|
| **FoldX-alone baseline scorer** (CIs, paired contrasts, multi-model) | `experiments/rescore_perstructure/foldx_alone_baseline.py` |
| FoldX-alone results (all 10 backbones), current | `…/foldx_alone_baseline_allplm_cov99.csv` |
| FoldX-alone results, pre-fix lineage — not current | `…/foldx_alone_baseline.csv`, `…_allplm.csv` |
| FoldX-alone write-up | `…/FOLDX_ALONE_BASELINE_RESULT.md` |
| FoldX-alone figure — pre-fix lineage, banner-marked | `…/foldx_alone_baseline.html`, `…/foldx_alone_baseline.png` |
| FoldX ΔΔG builder (+12 terms), resumable | `scratch/foldx_s1102/build_ddg.py` |
| Full-SKEMPI / CATH / MP mergers (dual-key `load_foldx`) | `experiments/full_skempi_seqonly/merge_foldx_*.py` |
| Unified results across the 5 FoldX campaigns | `experiments/full_skempi_seqonly/build_results_all.py` |
| Per-complex FoldX JSON | `scratch/foldx_s1102/results{,_S1131,_S2003,_S4169}/` |
| Repaired PDBs + mutant models (211 structures) | `scratch/foldx_s1102/work*/` |
| Channel configs | `models/config/lightatt_addscores_config.json`, `…_mlp_config.json` |
| Interface-residue selector (heavy-atom cutoff, FASTA-aligned) | `experiments/gen_interface.py` |
| Tier-1 augmentation (reverse + identity) | `experiments/augment.py` |
| Layer probe | `experiments/layer_probe.py`, `experiments/LAYER_PROBE.md` |
| Orthogonality / residual matrix | `scratch/orthogonality/residual_scatter_matrix.py` |
| Coverage analysis | `experiments/FOLDX_COVERAGE_FINDINGS.md`, `COVERAGE_BASELINE.txt` |
| Structure-thread sources for Part I | `history/FOLDX_SUMMARY.md`, `history/FINDINGS_FOLDX_MUTANT_3DI.md`, `history/PROSTT5_STRUCTURE_OPTIONS.md` |
| Per-structure Spearman matrix | `scripts_plots/results_matrix_ps.csv` |
| Interface-prediction pipeline (Task 2) | `experiments/attention_interface/` |
| flex-ddG runbook (deferred) | `scripts_plots/FLEX_DDG_RUNBOOK.md` |
| Splits | `scratch/splits_skempi_full_{cath_kfold,bycomplex_all_seed42,clustered_id60_kfold,mp_*}/` |
