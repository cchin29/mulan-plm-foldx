# Structure in MuLAN — Summary

_Updated 2026-07-15 — folded in the GPU ESM C 6B FoldX results (RESULTS §20/§20.1): 6B FoldX is no
longer "untested", and the scalar-vs-MLP recommendation is now dataset-size dependent._

**Scope:** does structural information help MuLAN predict protein–protein binding **ΔΔG** on the
**S1102** benchmark? Three routes were tried, all covered here:

- **(A) ProstT5's 3Di structure channel** — Appendix A.
- **(B) SaProt's fused amino-acid + 3Di structure-aware embedding** — the working baseline that
  experiment #1 measures against.
- **(C) FoldX**, used two ways — as a mutant-3Di *generator* (**experiment #1**) and as a
  binding-ΔΔG *feature channel* (**experiment #2**); these form the main body.

*Figures are pulled from `RESULTS.md` §9 / §15 / §17 / §18, `FINDINGS_FOLDX_MUTANT_3DI.md`,
`PROSTT5_STRUCTURE_OPTIONS.md`, and `scratch/foldx_s1102/`. Terminology and tooling are defined in
Appendices B–C.*

**Bottom line (structure overall):** structure helps only when a structure-pretrained embedder
fuses it at the *input* (SaProt, **+0.026 PCC**); bolted-on 3Di (ProstT5) and FoldX-modeled
mutant 3Di do not. FoldX's own two roles:

- **#1 (mutant 3Di): dead end.** FoldX cannot produce mutation-differentiated 3Di — a
  structurally guaranteed no-op, so it adds nothing to SaProt beyond the WT structure you
  already have for free.
- **#2 (ΔΔG feature): real, and stronger than first thought.** The decomposed-12-term MLP adds a
  **splitter-robust +0.014–0.019 PCC** across four sub-6B PLMs and *beats* augmentation on the
  balanced splitter (§19) — the best-supported sub-6B add-on, though still below ESM C 6B (scale
  wins). The plain single-scalar version is smaller (+0.009).

> **Reading this for "does structure help?"** — the one-line answer and the cross-thread
> comparison are in the **Does structure help? — cross-thread synthesis** section; **Appendix A**
> covers the ProstT5 3Di route. Term definitions are in **Appendix B**, installed tooling in
> **Appendix C**.

---

## The investigation thread (chronology)

All of this is one continuous line of inquiry — "**does structure help MuLAN's ΔΔG?**" — that
FoldX enters twice. (Unfamiliar terms — PLM, 3Di, `add_scores`, CV10, `mut − wt` — are defined in
**Appendix B**.)

1. **The puzzle.** ProstT5 (which *can* encode structure) underperforms sequence-only Ankh on
   S1102. Hypothesis: MuLAN never actually uses ProstT5's 3Di channel — so feed structure in.
2. **ProstT5 3Di channel** (Appendix A, `run6c` + three fix attempts) → **all negative.**
   Diagnosis: with no mutant structures available, each mutant reuses the **WT** 3Di, which then
   cancels in the `mut − wt` difference (no mutation-specific signal), and the extra width dilutes
   the working AA signal.
3. That diagnosis forks two ways:
   - **Would a better-*fused* WT structure help at all?** → **SaProt** (which fuses AA+3Di at the
     input) → **yes, +0.026 PCC.** This is the baseline experiment #1 measures against.
   - **Could we get *real*, mutation-specific 3Di?** → **FoldX experiment #1**: model each mutant
     structure with FoldX, recompute its 3Di → **confirmed no-op** (FoldX doesn't move the
     backbone, so the mutant 3Di equals the WT 3Di).
4. **Repurpose the tool.** FoldX was now installed and proven useless for *3Di*, but its native
   output is a physics **binding ΔΔG**. → **FoldX experiment #2**: feed that ΔΔG as an `add_scores`
   number → a small, orthogonal lift.

So #1 and #2 both involve FoldX for different reasons: #1 tries FoldX as a *structure generator*
(fails), #2 uses FoldX as a *physics score* (marginal). Appendix A is what motivated #1.

---

## #1 — FoldX mutant structure → mutation-aware 3Di (SaProt) — **confirmed no-op**

**Goal.** SaProt's input token is `amino-acid + Foldseek-3Di`. The 3Di half is a *structure*
descriptor; the hope was that modeling each mutant with FoldX would give a mutation-specific
3Di (a real structural signal per variant) rather than reusing the wild-type 3Di.

**Pipeline.** `RepairPDB` → `BuildModel` (mutant PDB) → mini3di → per-residue compare vs WT 3Di.

### Key points

- **0 of 39 mutations changed the 3Di at any position** — probe = all 36 of 1A22's S1102
  mutations **+ 1ACB Leu38→Pro and Leu38→Gly** (the two most backbone-perturbing substitutions
  in the whole set). No change even for →Pro/→Gly, and **not even at the mutated residue** (e.g.
  `LI38P`: position-38 3Di `p`→`p`).
- **Atom-level (1KNE):** 0 of 232 backbone (N/Cα/C/O) atoms moved between the WT-remodel and the
  mutant. `BuildModel` repacks **side chains only**; the backbone is fixed.
- **Mechanism.** FoldX `BuildModel` is a fast side-chain repacker with sub-Ångström backbone
  motion; Foldseek-3Di is a coarse **20-state backbone** descriptor. Sub-Å moves never cross a
  3Di state boundary ⇒ **FoldX-mutant 3Di ≡ WT 3Di**, structurally guaranteed.
- **Dataset-independent.** A property of FoldX + 3Di, not of S1102 — holds for S1131 / S4169 /
  S2003 too, and the →Pro/→Gly test covers non-alanine substitutions (relevant to S2003).
- **You don't need FoldX for this anyway:** foldseek/mini3di on the WT PDB gives the identical
  3Di directly.

### The SaProt WT-3Di baseline this ceilings against (§15, 650M, MPS)

This experiment ran in two stages: **Stage 1** = SaProt with the WT 3Di (below), **Stage 2** =
SaProt with a *FoldX-modeled mutant* 3Di (the no-op probed above). Stage 1 works and sets the
ceiling; the point of #1 is that Stage 2 can add nothing to it.

| SaProt-650M run (Stage 1, base) | Test PCC | RMSE | MAE | SCC |
|---|---|---|---|---|
| **Real WT-3Di** | **0.842 ± 0.050** | 1.276 | 0.926 | 0.729 |
| Seq-only (3Di masked with `#`) | 0.816 ± 0.069 | 1.362 | 0.976 | 0.699 |

Structure (WT-3Di) lift = **+0.026 PCC, 8/10 folds** (paired). Because a FoldX-modeled mutant's
3Di is identical to the WT 3Di, a full **Stage 2** run would produce byte-identical embeddings to
Stage 1 and land on the *same* 0.842.

**Verdict.** SaProt **Stage 1** (WT-3Di, 0.842) is the ceiling for the whole SaProt+FoldX
structure route. The 110-complex `RepairPDB` batch (Stage 2) was **not run** — a ~10-minute probe
made it pointless, replacing hours of compute. Capturing genuine mutation-induced backbone/3Di
change would need a **backbone-relaxing** method (energy minimization / MD, or
AlphaFold-on-mutant), not FoldX.

---

## #2 — FoldX binding ΔΔG as an `add_scores` channel (§18) — **real but small**

**Approach.** Compute a physics-based FoldX 5.1 binding ΔΔG (`RepairPDB` → `BuildModel`
side-chain repack → `AnalyseComplex` Interaction Energy, mut−WT) and feed it as one number into
MuLAN's `add_scores` head slot (Appendix B). Built for all of S1102 (110/111 complexes, 100%
coverage, 11,000 rows). FoldX also reports its total ΔΔG as **12 decomposed physical terms** (van
der Waals, electrostatics, H-bonds, solvation, clashes, entropies, …), all captured. Two arms
were tested: the single **scalar** total, and a small **MLP** over the 12 terms. PLM = Ankh-large.

### Key points

- **Signal is mostly redundant.** FoldX correlates **+0.41** with the experimental label
  standalone, but most of that overlaps what Ankh already learned (**r=0.35** with its own
  predictions); only **+0.20** is orthogonal.
- **Orthogonality ceiling ≈ +0.011 PCC** — inside per-fold noise (±0.03–0.05).
- **Measured lift:** **+0.009 PCC** (scalar, 8/10 folds) → **+0.017 PCC** (decomposed MLP,
  p≈0.02). The scalar lands *exactly* on its predicted linear ceiling.
- **A floor-raiser, not a uniform lift.** Gain anti-correlates with baseline strength
  (**corr = −0.61**): helps weak folds, **hurts the two strongest** (folds 0 and 1).
- **One-sided detector.** Good at flagging destabilizing mutations, blind to stabilizing ones.
- **Decomposition barely helps.** Only **Electrostatics** carries orthogonal signal beyond the
  aggregate (+0.006); **VdW clashes ≈ 0** — the "steric-clash tail" hypothesis was **wrong**.

### Pre-CV ceiling analysis (out-of-fold, no training needed)

| Signal | Value |
|---|---|
| Raw FoldX ΔΔG vs experimental label (n=1100) | Pearson **+0.407**, Spearman +0.447 |
| r(FoldX, OOF Ankh **prediction**) — redundant part | +0.35 |
| r(FoldX, OOF Ankh **residual**) — orthogonal part | **+0.20** |
| Best linear combo (Ankh + β·FoldX), OOF | 0.837 → **0.846** (Δ **+0.008–0.009**) |
| OOF ceiling — 11 decomposed terms / all 12 terms | **+0.011** |
| RSM 2nd-order OLS (12 + squares + 66 interactions) | in-sample 0.876 → **OOF 0.824 (worse than baseline)** |
| RSM 2nd-order, ridge (best α) | ≤ +0.005 (never beats the linear fit) |

### CV10 result (paired, Ankh, 300 ep / patience 30)

| Arm | Test PCC | RMSE | MAE | SCC | Paired Δ | Wins |
|---|---|---|---|---|---|---|
| baseline (no score) | 0.8217 ± 0.051 | 1.344 | 0.960 | 0.732 | — | — |
| **+ FoldX scalar** | **0.8310 ± 0.046** | 1.313 | 0.941 | 0.745 | **+0.0093** | 8/10 |
| **+ FoldX MLP** (12 terms → 16 → ReLU → 1) | **0.8383 ± 0.047** | — | — | — | **+0.0166** (p≈0.02) | 8/10 |

- **MLP vs baseline** (+0.0166): paired *t* ≈ 2.7, **p ≈ 0.02** — a real lift, above the +0.011
  "linear ceiling."
- **MLP vs scalar** (+0.0073): paired *t* ≈ 1.45, **p ≈ 0.18** — within noise on the paper split,
  but the MLP has since **replicated at +0.014–0.019 across four PLMs on the balanced splitter**
  (§19; see "Which PLMs benefit" below), so the decomposed MLP is the arm to report.

### Per-fold texture — the floor-raiser pattern

| fold | baseline PCC | + FoldX scalar | ΔPCC |
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

Biggest gain on the weakest fold (8); the only losses are the two strongest baselines (0, 1).
Fold 0 happens to be enriched in interface steric-clash "hotspots" — the hard cases (a bulky
mutation jamming the partner's pocket) that FoldX physics was *specifically* meant to rescue,
since sequence-only PLMs tend to under-predict them. Yet fold 0 is the single fold FoldX **hurt**
(−0.010), consistent with the finding that FoldX's isolated clash terms carry ≈0 orthogonal
signal.

### Supporting diagnostic — hotspot rank correlation (18 interface hotspots)

| | Pearson r | closer-to-truth |
|---|---|---|
| PLM ensemble | 0.28 | 8 |
| **FoldX** | **0.55** | 8 (2 ties) |

FoldX roughly **doubles** rank correlation and nails the gain-of-bulk steric-clash subset
(r=0.99 on those 4) — but misses cavity / electrostatic / enzyme-side cases and its absolute
magnitudes are unreliable. The missing signal is real and lives in the *complex*; a
rigid-backbone method captures only one regime of it.

### Which PLMs benefit — prediction (§18.2) vs. measurement (§19, balanced splitter)

§18.2 originally predicted the benefit along two axes: **weaker PLM → bigger floor-raise**, and
**sequence-only → more orthogonal** (a structure-aware PLM has already absorbed the signal). The
balanced-splitter 300 ep series (§19) has since measured the FoldX-MLP arm on four PLMs and
**refuted the second axis while confirming the first.** (Balanced is a separate, higher-PCC
series than the paper splits used elsewhere in this doc — compare lifts, not absolute values.)

| PLM (balanced base) | §18.2 prediction | +FoldX-MLP measured (§19), paired |
|---|---|---|
| ESM C 600M (0.834) | (not predicted) | **+0.019 (9/10)** → 0.853 |
| SaProt WT-3Di (0.849) | low — structure-aware, "most redundant" | **+0.018 (7/10) → 0.867** — *prediction wrong* |
| Ankh-large (0.836) | ~+0.015 | **+0.018 (9/10)** → 0.854 |
| ProstT5 (0.823) | high-but-uncertain | **+0.014 (8/10)** → 0.837 |
| ESM2-3B (0.833) | highest | pending (task #37) |
| ESM C 6B (0.881) | lowest | **+0.004 (6/10, balanced MLP)** → 0.885; scalar **+0.007** — prediction confirmed (§20) |

- **"Lowest" confirmed on the 6B (§20).** The GPU sweep measured it: balanced FoldX-MLP +0.0037,
  paper +0.0054 — the smallest lift of the suite, exactly as the §18.2 strength axis predicts. On the
  6B the **scalar** arm (+0.0074 balanced) actually edges the MLP (see §20 / the verdict caveat below).
- **Weak-vs-strong axis holds:** the weakest PLM (ESM C 600M) gets the biggest lift (+0.019).
- **Seq-vs-structure axis fails:** structure-aware SaProt gains +0.018 ≈ seq-only ESM C 600M's
  +0.019 — the 12-term head carries orthogonal signal *even for an AA+3Di model*.
- **Splitter-robust:** these balanced +0.014–0.019 lifts match the rng-split MLP +0.017 (§18.1).

**Verdict (updated).** FoldX-MLP is a **real, splitter-robust ~+0.017 lever, now measured across
four sub-6B PLMs** — not a one-PLM curiosity, and the *stronger* of the two add-ons on the
balanced splitter, where it beats Tier-1 augmentation on every PLM tested (**+0.014–0.019 vs aug's
+0.002–0.009**). Two things still keep it off the headline: (1) **scale wins** — the best FoldX
arm, SaProt+FoldX **0.867**, matches the *published* MuLAN-Ankh 0.868 but stays below **ESM C 6B's
0.881** base; FoldX on the 6B is **now measured (§20)** and, as predicted, gives the suite's smallest
lift (+0.004 MLP / +0.007 scalar, balanced) — the strongest sequence-only model leaves the least for
FoldX to fill. (2) Whether FoldX genuinely *out-levers* augmentation, or aug merely underperforms on
the balanced splitter, is **under audit**: the earlier "aug +0.024–0.026" figure is an rng-splitter /
patience-10 number, and a paired rng-300/30 aug run (task #43/#50) is queued to separate splitter from
budget — treat the aug-vs-FoldX reversal as provisional until it lands.

**Scalar vs. 12-term MLP — refined by the GPU multi-benchmark sweep (§20.1).** The "decomposed MLP
is the arm to report" call holds *on the sub-6B S1102 balanced series*, but the three-SKEMPI GPU
extension shows the winner **tracks dataset size**: on the smallest set (S1131) and on ESM C 6B the
**scalar** wins (MLP underperforms it); the MLP clearly wins only on the larger S2003, and they tie on
S4169. The defensible recommendation is therefore **scalar as the safe default (+0.006–0.013 across
every benchmark), the 12-term MLP as the upside on ≥2k-mut sets** — not the MLP unconditionally.

---

## Discussion — why FoldX behaves this way here

Both results trace to one fact: **FoldX `BuildModel` moves side chains, not backbone.**

For **#1** (using FoldX to generate a mutation-specific 3Di for SaProt), that rigidity is fatal.
Foldseek-3Di reads only the backbone, so a fixed backbone
means a fixed 3Di — the mutant token differs from wild-type only in its amino-acid half, which
is information SaProt already has from the sequence. FoldX contributes literally nothing new to
the structural channel, and the failure is structural, not statistical, so no larger benchmark
or gentler mutation would rescue it. The true structural ceiling for SaProt on this task is
the WT-3Di run (0.842), reachable without FoldX at all.

For **#2** (feeding FoldX's binding ΔΔG into MuLAN's head as an `add_scores` feature), the same
physics is *useful but shallow*. FoldX's binding ΔΔG does encode real
complex-level energetics — most visibly on gain-of-bulk steric clashes at tight
protease–inhibitor interfaces, exactly the "cross-chain steric information" a monomer PLM
embedding can't see (hence r=0.99 on that subset and the doubled hotspot rank correlation). But
a strong PLM has independently learned most of that regime, so only ~20% of FoldX's correlation
is orthogonal — which caps the *single scalar* near +0.01 PCC. It props up folds where the PLM is
weak and adds noise to folds it has already nailed — including, ironically, the very hotspot fold
that motivated the experiment. The **decomposed 12-term MLP does better**: it clears the linear
scalar ceiling and, in the balanced-splitter series (§19), replicates at **+0.014–0.019 across
four PLMs** — including structure-aware SaProt, refuting the idea that only weak sequence-only
models have room for it. So "useful but shallow" holds only against *scale* (it stays below
ESM C 6B); as a sub-6B add-on the decomposition earns real credit.

**Where FoldX would actually pay off** is neither of these: it is the **interface/complex**
direction. FoldX's genuine signal is cross-chain, and the current structural tooling is
monomer-only, so a complex-aware channel — encoding the mutant side chain in the *partner*
pocket rather than a single-chain 3Di — is the natural place this physics becomes a lever rather
than a floor-raiser. That is scoped as Phase 2 future work (flexible-backbone physics such as
Rosetta flex-ddG or FEP for faithful magnitudes).

---

## Does structure help? — cross-thread synthesis

Four ways structure was injected into MuLAN ΔΔG, four outcomes:

| # | Method | How structure enters | Result vs seq-only |
|---|---|---|---|
| 1 | ProstT5 + WT 3Di (§9, run6c) | late concat of a separate 3Di block | **hurts** (0.663 < 0.740) |
| 2 | SaProt WT-3Di (§15) | **input-level** AA+3Di token, jointly pretrained | **helps +0.026** (→ 0.842) |
| 3 | FoldX mutant 3Di (#1) | remodel each mutant → new 3Di | **no-op** (≡ WT 3Di) |
| 4 | FoldX ΔΔG feature (#2) | physics scalar into the head | **small +0.009–0.017** |

**The answer: yes — but only WT structure, only when fused right, and only modestly.**

- **It's the fusion, not the data, that decides.** ProstT5 and SaProt were fed the *same* WT 3Di
  and split in opposite directions. SaProt co-embeds AA+3Di at the input as a model pretrained to
  do so, so structure sharpens the representation (+0.026, the best sub-1B result, and it stacks
  with augmentation → 0.852). ProstT5's late concat just doubles the width and dilutes the AA
  signal. Structure helps when the *embedder* owns the fusion, not when it's bolted on downstream.
- **The mutation-specific structural lever is out of reach.** Every route that tried to make
  structure *change with the mutation* — ProstT5's fixed-3Di concat, FoldX's remodeled mutant —
  collapsed to a constant WT term, because single-point mutations barely move the backbone and
  rigid-backbone tools (FoldX BuildModel) can't move it either. Real per-variant structure would
  need backbone-relaxing modeling (MD / AlphaFold-on-mutant).
- **Physics adds a real, orthogonal increment.** FoldX ΔΔG isn't 3Di — it's an energy — and it
  carries genuine cross-chain signal. The decomposed-12-term MLP delivers a **splitter-robust
  +0.014–0.019 PCC across four sub-6B PLMs** (§19), *beating augmentation* on the balanced
  splitter. Notably it helps structure-aware SaProt as much as a sequence-only model, refuting the
  earlier "structure-aware = redundant" prediction. It's still a floor-raiser, not a scale-beater
  (best arm 0.867 < ESM C 6B 0.881), but it is the best-supported sub-6B add-on — not the
  "caps near +0.01" footnote §18 first framed it as.
- **The unrealized win is complex/interface structure.** The one regime where structure is both
  real and *not* already in the PLM is cross-chain interface energetics (FoldX doubled hotspot
  rank correlation; monomer contacts predict interface membership at chance). That — a
  complex-aware channel, not a monomer 3Di — is the open Phase 2 lever.

**Practical read:** if you want the best sub-6B model today, use a structure-pretrained embedder
(SaProt) with WT 3Di and add the FoldX-MLP score — SaProt+FoldX (0.867) is the top sub-6B arm and
matches the published MuLAN-Ankh; FoldX-MLP is now the strongest single add-on measured (beating
augmentation on the balanced splitter, §19, pending the #43 audit). If you can run it, ESM C 6B
(0.881) still beats every sub-6B option outright. Don't spend effort synthesizing mutant 3Di with
rigid-backbone tools. The "orthogonality is cross-modality" claim above is measured directly in the
next section.

---

## Orthogonality check — do the models make independent errors?

The redundancy argument that runs through this doc (FoldX is ~80% redundant; PLM ensembling
can't break the ceiling) is directly testable: the CV10 runs save per-mutation out-of-fold
predictions on **identical folds**, so we can correlate models' **residuals** (true − predicted).
If two models were orthogonal, they would err on *different* mutations and their residuals would
be uncorrelated. They are not.

*(The residual scatterplot matrix across 5 PLM backbones + FoldX was written to
`scratch/orthogonality/residual_scatter_matrix.png`, which is not published — `scratch/` is
working state, not an artifact. The correlation matrix below carries the same information in the
form the argument actually uses.)*

**Residual correlation matrix (1100 OOF points; lower = more independent errors):**

| | ESM-C 6B | Ankh | ESM2-3B | SaProt | ProstT5 | FoldX |
|---|---|---|---|---|---|---|
| **ESM-C 6B** (0.869) | — | 0.84 | 0.80 | 0.84 | 0.78 | **0.46** |
| **Ankh** (0.837) | 0.84 | — | 0.86 | 0.83 | 0.86 | **0.46** |
| **ESM2-3B** (0.815) | 0.80 | 0.86 | — | 0.85 | 0.87 | **0.46** |
| **SaProt** (0.847) | 0.84 | 0.83 | 0.85 | — | 0.86 | **0.49** |
| **ProstT5** (0.811) | 0.78 | 0.86 | 0.87 | 0.86 | — | **0.55** |
| **FoldX** (0.442) | 0.46 | 0.46 | 0.46 | 0.49 | 0.55 | — |

*(parenthetical = each model's pooled OOF PCC.)*

**Findings:**

- **PLM↔PLM residuals are highly correlated: 0.78–0.92, mean ≈ 0.84** (the full 9-model version
  reaches 0.92 for same-family pairs, e.g. SaProt ↔ SaProt-1.3B). The models miss the *same*
  mutations by nearly the same amounts — a shared blind spot, not decorrelated noise. There is
  essentially **no orthogonality to harvest inside the PLM family.**
- **FoldX is the one orthogonal channel: residual corr 0.46–0.55**, visibly a rounder cloud than
  the PLMs' tight diagonal bands. Tighter still is the §18 "added-value" metric —
  **corr(raw FoldX ΔΔG, PLM residual) = +0.19 to +0.23** — reproducing the reported +0.20. FoldX
  correlates slightly more with ProstT5's errors (0.55), the one structure-trained PLM.
- **Ensembling confirms it.** Equal-weight prediction averaging (pooled PCC): best single
  ESM-C 6B **0.869** → ESM-C + SaProt **0.871** (+0.002, noise) → ESM-C + Ankh **0.866**
  (*worse*) → 5-model average **0.860** (worse). Correlated errors don't cancel.

**Takeaway:** orthogonality in the current results is **cross-modality (physics vs. PLM), not
cross-PLM.** This is the quantitative backbone of two claims above — that FoldX buys a real but
small lift (#2), and that a PLM-only ensemble cannot (synthesis). Any ensemble that aims to move
the ceiling must pair a strong PLM with an orthogonal structure/physics channel, not with more
PLMs.

*Reproducible:* `scratch/orthogonality/residual_scatter_matrix.py` (run from the `mulan/` repo
root) regenerates the figure, the matrix, and the added-value check; the `MODELS` dict and
`FOLDX_SPLIT_DIR` at the top are the only knobs.

---

## Appendix A — ProstT5 3Di structure channel (§9) — the precursor to FoldX #1

The FoldX mutant-3Di experiment (#1) did not start the structure-channel question — it was the
attempt to *rescue* it. The original bet lived in ProstT5.

**Hypothesis (H-struct).** ProstT5 underperforms Ankh (0.805 vs 0.832 CV10) *because* MuLAN's
sequence-only pipeline never uses ProstT5's defining feature — its 3Di structure channel. So
feed the structure in and close the gap.

**run6c — the headline test, negative.** Per residue, concatenate ProstT5 AA-mode (1024-d) with
real WT 3Di-mode (1024-d) → 2048-d; same split/optimizer as the AA-only baseline (run2), full
50 ep (early-stopped at 44 — *not* under-trained). WT 3Di from the SKEMPI 2.0 PDBs via mini3di,
index-aligned to the AA fasta, embedded through ProstT5 fold-mode.

| Run (single split) | Structure input | PCC |
|---|---|---|
| run2 — AA-only baseline | none | **0.740** |
| run6c — AA ⊕ WT-3Di concat (2048-d) | WT 3Di | 0.663 |
| run6-E1 — + learned scalar gate on 3Di | WT 3Di | 0.705 (gate → 0.46) |
| run6-B1 — WT 3Di as head context, outside `mut−wt` | WT 3Di | 0.673 |
| run6-C3 — interface contacts as attention bias | contacts | 0.662 (α → 0.017 ≈ off) |

**Key points:**

- **Every structure fusion lost to AA-only (0.740).** run6c *hurt* by −0.077; the three repair
  variants all stayed below baseline.
- **Every learnable structure knob trained itself to neutral/off** — the E1 gate down to 0.46
  (recovering ~half the dilution gap but still net-negative), the C3 interface bias to ≈0. The
  head consistently *declined* ProstT5 structure.
- **Root cause = the same trap as FoldX #1.** No mutant structures exist, so each mutant reuses
  its **WT** 3Di. Held fixed across the mutation, the 3Di channel contributes only a constant
  complex-level offset in MuLAN's `mut − wt` — **zero mutation-specific signal** — while doubling
  the width *dilutes* the AA signal that worked. (run6b, 3Di-only, was skipped as algebraically
  degenerate: fixed WT 3Di ⇒ `mut_emb ≡ wt_emb` ⇒ zero signal.)
- **CV10 (final):** ProstT5 AA-only 0.805 ± 0.055; + Tier-1 aug 0.819 ± 0.060 (aug +0.014, 8/10).
  ProstT5's ceiling in MuLAN's head is real; priority was redirected to other PLMs / AIDO.

**This is exactly why FoldX #1 was attempted** — run6c's own writeup says "testing genuine
structural value would need *modeled mutant* structures (FoldX/AF)." FoldX #1 then showed that a
rigid-backbone modeler can't produce them either, closing the loop: the WT-3Di stand-in isn't a
data gap FoldX can fill.

---

## Appendix B — Background: what MuLAN does and the terms used here

**MuLAN** is the model under study. It predicts **ΔΔG** — the change in a protein–protein
complex's binding free energy caused by a single-residue point mutation (positive = the mutation
destabilizes binding). It takes the wild-type (WT) complex sequence plus one mutation, embeds
both the WT and the mutant sequence with a **frozen protein language model (PLM)**, forms a
per-residue **`mut − wt`** difference, pools it, and regresses ΔΔG through a small "light
attention" head. The head has one optional input slot — **`add_scores` / `zs_scores`** — where a
single precomputed per-mutation number (e.g. a physics score) can be concatenated before the
final linear layer. This slot is the entire mechanism behind FoldX experiment #2.

**Key terms:**

- **S1102** — the benchmark: 1102 single mutations across ~110 complexes from **SKEMPI 2.0**
  (a curated database of experimental ΔΔG measurements). Related SKEMPI subsets referenced:
  S1131 / S4169 / S2003.
- **PCC / SCC / RMSE / MAE** — Pearson corr., Spearman corr., root-mean-square error, mean
  absolute error between predicted and experimental ΔΔG. Higher PCC/SCC and lower RMSE/MAE are
  better. **PCC is the headline metric**, reported as mean ± std across folds.
- **CV10 / fold / paired / single-split** — CV10 is 10-fold cross-validation (10 train/test
  splits). "**Paired**" means two arms (e.g. with vs without FoldX) use identical folds and seed,
  so only their per-fold difference (**Δ**) matters; "**8/10**" = the change helped in 8 of the 10
  folds. The early "**runN**" experiments instead used one fixed **single split**.
- **OOF** — out-of-fold (held-out) predictions/analysis, i.e. estimated without training on the
  point being scored. **RSM** — response-surface model, a 2nd-order polynomial regression used
  here only as a ceiling probe.
- **The PLMs** (the frozen embedder MuLAN sits on): **Ankh-large** (1536-d, sequence-only, the
  strong workhorse baseline); **ProstT5** (1024-d, "bilingual" — has an amino-acid *AA mode* and a
  structure *3Di/fold mode*); **SaProt-650M / 1.3B** (1280-d, structure-aware — every input token
  is a fused *amino-acid + 3Di* symbol, pretrained jointly); **ESM2-3B** and **ESM C 6B**
  (sequence-only, other size points).
- **3Di / Foldseek / mini3di** — **3Di** is a 20-letter structural alphabet (from Foldseek) that
  encodes each residue's **backbone** geometry as one symbol; **mini3di** is a pure-Python encoder
  that produces 3Di from backbone coordinates. Because it reads only the backbone, 3Di is
  **invariant to any change that doesn't move the backbone**. "**WT 3Di**" = 3Di computed from the
  wild-type structure and reused for every variant of that chain.
- **Augmentation ("Tier-1 aug")** — a training-data trick (add reverse mutations with the ΔΔG
  sign flipped, plus zero-ΔΔG identity anchors); gives a general +0.01–0.026 PCC lift across PLMs,
  and is the cheaper competing lever referenced throughout.
- **"Stage 1 / Stage 2"** is used in two unrelated experiments — disambiguated where it appears:
  (a) the **SaProt structure route** — *Stage 1* = SaProt with WT 3Di, *Stage 2* = SaProt with
  FoldX-modeled mutant 3Di (experiment #1); (b) the **`add_scores` route** — a *scalar* arm
  and a *decomposed-MLP* arm (experiment #2).

---

## Appendix C — Tooling & artifacts (installed, reusable regardless of outcome)

- **FoldX 5.1** — `~/tools/foldx` (→ `~/.local/bin/foldx`); no `rotabase.txt` dependency.
- **foldseek AVX2** — `~/tools/foldseek/bin/foldseek` (→ `~/.local/bin/foldseek`); the WT-3Di
  encoder (via mini3di) used across both experiments.
- **FoldX ΔΔG dataset** — the full binding-ΔΔG set over S1102: **110/111 complexes, 100%
  coverage**, `scratch/foldx_s1102/results/*.json` (total ΔΔG **+ 12 decomposed energy terms** per
  mutation). Role→real-chain mapping via `skempi_v2.csv`, validated against each repaired PDB.
- **Reuse.** The dataset is model-independent physics, so pairing it with any other PLM needs only
  the merge + paired-CV drivers (`scratch/foldx_s1102/`), **no FoldX re-run** — the "which PLM"
  table above nominates **ESM2-3B** as the best untested candidate.
- **Numbering gotcha (any batch pipeline must handle).** S1102 mutation chain letters are
  **partner-role** labels (A/B), not PDB chains — map via `skempi_v2.csv` (e.g. 1ACB role B →
  chain **I**). And use the SKEMPI `Mutation(s)_cleaned` column, which matches the actual PDB ATOM
  numbering, not `Mutation(s)_PDB`.

---

*Sources: `mulan/RESULTS.md` §9 (ProstT5 3Di), §15 (SaProt WT-3Di), §17-Stage 2 (FoldX
mutant-3Di), §18 / §18.1 / §18.2 (FoldX add_scores, paper splits), **§19 (balanced-splitter
300ep — FoldX-MLP across 4 PLMs; `scripts_plots/ddg_scaling_300ep_balanced.png`,
`ddg_scaling_data_300ep_balanced.csv`, `benchmarks_folds_300ep_balanced.csv`)**;
`mulan/PROSTT5_STRUCTURE_OPTIONS.md`,
`mulan/PLAN_PROSTT5_STRUCTURE_v2.md`; `mulan/FINDINGS_FOLDX_MUTANT_3DI.md`;
the phase-1 hotspot diagnostic; `mulan/scratch/foldx_s1102/`
(drivers, `cv10/summary*.json`, `results/*.json`); `mulan/scratch/orthogonality/`
(residual analysis — script, figure, per-model OOF predictions from `scratch/results/cv10_*`).*
