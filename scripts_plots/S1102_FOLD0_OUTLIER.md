# S1102 fold-0 low outlier in the variability plot

Analysis of the recurring low-outlier fold seen in the per-fold PCC variability
plot (`scripts_plots/benchmarks_folds.csv`). Every S1102 run — all 7 models ×
{base, aug} — shows **fold 0** as the lowest of the ten folds, by a wide margin
(~0.07–0.15 PCC below the pack).

## It is the same fold, and the same mutations, every time

The 10-fold CV split is **deterministic**: `split_data(num_folds=10,
random_state=42)` (`mulan/data.py:222`) builds a fixed `fold_index` array and
shuffles it under a fixed seed. So **fold 0's test set is the identical 110
mutations in every run and every model**. The dip reproduces because the data
does, not because of run-to-run noise. (Ankh fold-0 = 0.691 and ProstT5 fold-0 =
0.674 in `experiments/RESULTS_CV10.md` match the CSV to the digit.)

## What is NOT special about fold 0

- **Label distribution is typical**: mean ΔΔG 1.18 (global 1.13), std 2.12 —
  actually one of the *narrowest* spreads of the ten folds.
- **Composition is typical**: 36 complexes, top-3 complexes = 49% of rows, 30%
  alanine mutations — indistinguishable from folds 1–9.

The low PCC is not a skewed label range or an odd complex mix.

## What IS different: a shared cluster of extreme interface hotspots

Fold 0 catches an enrichment of extreme-magnitude interface hotspots — the P1
protease-inhibitor tail (OMTKY3 systems: 1PPF, 3SGB, 1R0R, 1CSE) plus other
high-impact interfaces (1KTZ TGF-β·receptor, 1MAH AChE·fasciculin, 1A4Y) — and
**all PLMs regress these toward the mean in the same way** (see the FoldX section
below for the per-mechanism breakdown). Worst-predicted points (mean |error|
across Ankh, ESM-C 6B, SaProt, Ankh3-XL):

| complex | mutation | true ΔΔG | all models predict |
|---|---|---|---|
| 1PPF | L→W @18 | **7.44** | ~1.9 |
| 1PPF | L→Y @18 | **6.54** | ~1.3 |
| 1PPF | Y→E @20 | **6.33** | ~3.4 |
| 1MAH | Y→N @69 | **5.22** | ~0.5 |
| 2FTL | I→A @18 | **4.97** | ~2.2 |
| 1KTZ | S→L @28 | **4.48** | ~0.2 |

These are extreme interface-hotspot substitutions (5–7.4 kcal/mol) — several at
protease-inhibitor P1/P2′ sites — that PLM+regressor heads cap at ~2–4. Because
they are high-leverage and fold 0's overall spread is narrow, they crater the
correlation.

**Clincher** — dropping just the 10 worst-predicted points lifts every model's
fold-0 PCC from ~0.69 to ~0.80, back into the normal fold band:

| model | fold-0 PCC | minus 10 worst |
|---|---|---|
| Ankh | 0.691 | 0.801 |
| ESM-C 6B | 0.672 | 0.778 |
| SaProt | 0.730 | 0.810 |
| Ankh3-XL | 0.689 | 0.800 |

## Why *these specific* mutations, and not just large ones

The failure is **selective**, not simple mean-reversion. The single most extreme
label in fold 0 — 3BTM M13K at **−7.61** — is predicted almost perfectly
(−7.84), and 1ACB L38S (+4.95) is fine too. So the models *can* emit extreme
values. The blown points all share a three-part signature:

| mutation | complex | pos | true | pred | shrink |
|---|---|---|---|---|---|
| 3BTM M→K @13 | (control) | 13 | −7.61 | −7.84 | 3% ✓ |
| 1ACB L→S @38 | (control) | 38 | +4.95 | +4.66 | 6% ✓ |
| 1PPF L→W @18 | elastase·OMTKY3 | **18 (P1)** | +7.44 | +1.90 | **75% ✗** |
| 1PPF L→Y @18 | elastase·OMTKY3 | **18 (P1)** | +6.54 | +1.34 | **80% ✗** |
| 1MAH Y→N @69 | AChE·fasciculin | 69 | +5.22 | +0.46 | **91% ✗** |
| 2FTL I→A @18 | protease·inhib | 18 | +4.97 | +2.18 | **56% ✗** |

1. **Reactive-site P1 (or flanking P2′) of a tight protease–inhibitor
   interface.** 1PPF is human leukocyte elastase · turkey ovomucoid third domain
   (OMTKY3), whose reactive-site P1 residue is **Leu18** — the dominant
   specificity determinant. The top misses (L18W, L18Y, Y20E/Y20Q) are the P1
   and P2′ of that one loop.
2. **The substitution forces bulk/chemistry that clashes with the *partner*
   enzyme's pocket.** Elastase's S1 pocket is small/hydrophobic; forcing a bulky
   aromatic (Leu→Trp, Leu→Tyr) into it costs 6–7 kcal/mol. The energy is set by
   steric complementarity with the *other* chain.
3. **That information is not in the inhibitor's own sequence** — which is all the
   PLM embedding encodes. The model learns "P1 matters" generically and hedges
   to ~2 kcal/mol because it cannot compute how a Trp ring collides with
   elastase's specific S1 geometry.

Ensemble regression across fold 0 is `yhat ≈ 0.33 + 0.51·y` (global compression),
but the *selective* blow-ups are pocket-steric P1 substitutions whose magnitude
depends on the binding partner — a **cross-chain steric information ceiling**,
not a general inability to predict big numbers.

## Structure-based check: does FoldX have the information the PLMs lack?

If the ceiling really is *cross-chain steric*, a structure-based method that sees
the bound complex should recover the signal. Tested with **FoldX 5.1** (CPU,
`../foldx5_MacSilicon_0/foldx5.1_20261231`) on the SKEMPI WT complex PDBs
(`scratch/skempi2/PDBs/`). Pipeline per complex: `RepairPDB` (~45–60 s) →
`BuildModel` (mutant + WT) → `AnalyseComplex` on the two interacting chains, so
ΔΔG_bind = IntEnergy(mutant) − IntEnergy(WT) — the **binding** ΔΔG, not folding
stability. PDB chain IDs vary per complex, so each mutation was mapped to the PDB
chain carrying its WT residue at the given position (e.g. dataset chain B →
1PPF/3SGB/1R0R chain I, 1KTZ chain B, 1S0W chain C, 1FCC chain C…). Driver +
outputs in `scratch/foldx_fold0/` (`run_all.py`, `results_extended.json`).

Extended to **18 fold-0 hotspots** (the top |error| points across all models):

| mutation | type | true ΔΔG | PLM ens. | **FoldX** | closer |
|---|---|---|---|---|---|
| 1PPF LB18W | P1 gain-of-bulk clash | 7.44 | 1.90 | **11.89** | FoldX |
| 1PPF LB18Y | P1 gain-of-bulk clash | 6.54 | 1.34 | 12.59 | PLM |
| 1KTZ SB28L | gain-of-bulk clash | 4.48 | 0.17 | **8.03** | FoldX |
| 1S0W AB142F | gain-of-bulk (buried) | −2.10 | 0.11 | **−1.16** | FoldX |
| 1PPF YB20E | P2′ charge swap | 6.33 | 3.51 | 2.47 | PLM |
| 1PPF YB20Q | P2′ polar swap | 4.63 | 3.51 | 1.92 | PLM |
| 2FTL IB18A | P1 loss-of-contact | 4.97 | 2.18 | 1.05 | PLM |
| 2O3B QB74A | loss-of-contact | 3.23 | 0.61 | **1.06** | FoldX |
| 1XD3 LB8A | loss-of-contact | 2.74 | 0.77 | **3.01** | FoldX |
| 1XD3 RB74L | charge→hydrophobic | 2.43 | 0.50 | 5.84 | PLM |
| 1MAH YA69N | enzyme loss-of-aromatic | 5.22 | 0.46 | 0.19 | tie |
| 1MAH YA121Q | enzyme polar swap | 3.00 | 1.60 | −0.57 | PLM |
| 3SGB KB7A | charge loss (Ala) | −2.54 | −0.06 | 0.79 | PLM |
| 1A4Y DA435A | charge loss (Ala) | 3.48 | 1.11 | **1.76** | FoldX |
| 1FCC WB43A | aromatic hotspot (Ala) | 3.77 | 1.85 | 1.73 | tie |
| 1R0R AB10P | Pro backbone | 3.30 | 1.30 | 0.29 | PLM |
| 1CSE LB38S | P1 loss-of-bulk (overpred) | 1.17 | 3.64 | **2.76** | FoldX |
| 1R0R LB13N | polar swap (overpred) | 1.48 | 3.91 | **2.06** | FoldX |

Aggregate over the 18 hotspots:

| | Pearson r | Spearman | MAE |
|---|---|---|---|
| PLM ensemble | 0.28 | 0.28 | 2.80 |
| **FoldX** | **0.55** | 0.34 | 2.90 |

Head-to-head "closer to truth": **FoldX 8, PLM 8, 2 ties** — a dead heat on point
accuracy. So the candid reading is **mechanism-dependent, not a blanket win**:

- **Gain-of-bulk / steric-clash cases: FoldX decisively recovers the signal.**
  Restricting to the 4 clash mutations (LB18W, LB18Y, SB28L, buried AB142F),
  FoldX r = **0.99** vs PLM 0.79 — and it flags them as the strongest
  destabilizers, exactly where the PLMs squash to ~1.5. This is the marquee
  confirmation of the cross-chain steric ceiling: FoldX sees the side chain jam
  into the partner pocket; the sequence model (fed only the monomeric inhibitor
  sequence + WT 3Di) cannot. Magnitudes **overshoot** (12 vs 7): rigid backbone
  can't relax the clash — the predicted artifact.
- **Loss-of-contact / cavity / electrostatic / enzyme-side cases: FoldX misses
  too.** 2FTL IB18A (cavity, 1.05 vs 4.97), 1MAH YA69N (0.19 vs 5.22), 1R0R AB10P
  (Pro, 0.29 vs 3.30) — FoldX undershoots as badly as or worse than the PLM.
  Rigid-backbone empirical ΔΔG cannot collapse a new cavity or capture longer-
  range electrostatics, and enzyme-side aromatic losses aren't a simple pocket
  clash.
- **Bonus:** on the two *over*-prediction hotspots (PLM too high: 1CSE LB38S,
  1R0R LB13N), FoldX pulls the estimate back toward truth — structure helps in
  both directions there.

**Conclusion:** the structural information *is* present in the complex, and FoldX
exploits it to roughly **double the rank correlation** on fold-0 hotspots
(r 0.28 → 0.55) and to correctly re-rank the steric-clash misses the PLMs discard
— so the ceiling is real information, not label noise. But it is **not a
drop-in oracle**: absolute magnitude is unreliable (MAE unchanged, overshoots
clashes, undershoots cavities/electrostatics), and its wins are concentrated on
gain-of-bulk mutations. Faithful magnitudes on these extremes would need
flexible-backbone (Rosetta flex-ddG) or FEP.

## For faithful magnitudes: flex-ddG vs FEP

FoldX showed the information is in the structure; the remaining magnitude errors
(overshot clashes, undershot cavities/electrostatics) come from missing *physics*
— backbone motion and explicit solvent. Two methods add it, on a cost/accuracy
ladder above the rigid-backbone empirical calc:

| method | backbone | solvent | energy | cost / mut | SKEMPI r |
|---|---|---|---|---|---|
| FoldX (ran here) | ~rigid, side-chain repack | implicit | empirical fit | seconds | ~0.5 |
| Rosetta flex-ddG | **flexible (backrub ensemble)** | implicit | physics scorefn | min–~1 h CPU | ~0.6–0.7 |
| FEP / alchemical | **full MD** | **explicit water** | rigorous free energy | h–days GPU | ~0.8, ≤1 kcal/mol |

**Rosetta flex-ddG** (Barlow et al. 2018, Kortemme lab): adds backbone
flexibility via *backrub* moves (local geometry-preserving Cα-axis hinges),
generating an ensemble (~35 models), repacking+minimizing WT and mutant in each,
and averaging ΔΔG. Because the backbone can now *move to relieve* a clash, it
should tame the rigid-backbone **overshoot** (e.g. 1PPF LB18W 11.9→ toward 7.4)
and model **cavity-creating** losses (2FTL IB18A) better than FoldX. Still
implicit solvent, so charge swaps (YB20E, RB74L) stay only partially captured.
Pragmatic next step: CPU-minutes/mutation, feasible for the whole S1102 set as a
feature or baseline.

**FEP / alchemical free energy**: the rigorous gold standard. "Morphs" WT→mutant
along a coupling parameter λ with **explicit-solvent MD** at each window; binding
ΔΔG via a thermodynamic cycle (ΔG_mut in complex − ΔG_mut in free protein),
estimators BAR/MBAR or TI. Full MD + real water samples backbone reorganization,
repacking, *and* desolvation with rigorous electrostatics — so it's the one
method that could also nail the **electrostatic hotspots** both rigid and
flexible structure methods miss. Caveats: hours–days/mutation on GPU,
convergence-limited, and charge-changing mutations (e.g. Tyr→Glu) need finite-
size corrections even here. Only practical for a handful of validation points,
not benchmark scale.

**For this analysis:** flex-ddG is the realistic follow-up to check whether
relaxing the backbone recovers the steric-clash magnitudes; FEP is a
gold-standard spot-check for the few electrostatic outliers, not a routine
feature.

## Takeaways

1. The fold-0 dip is a **genuine data artifact** of the fixed `seed=42` split —
   the same 6–10 extreme interface mutations (led by 1PPF LB18W/LB18Y/YB20E)
   underpredicted identically by every model — not model or run instability.
2. It is a reproducible per-fold difficulty offset that **pairing cancels** (as
   `RESULTS_CV10.md` notes for Ankh-vs-ProstT5). So the cross-fold std shown in
   the variability plot partly reflects this fixed fold-0 penalty rather than
   true model instability; prefer paired/fold-matched comparisons when ranking
   models.
3. The misses are **partly a cross-chain steric information ceiling**, confirmed
   by FoldX over 18 hotspots: seeing the partner pocket roughly doubles rank
   correlation (r 0.28 → 0.55) and nails the gain-of-bulk steric-clash subset
   (r 0.99), but FoldX still misses cavity-creating / electrostatic / enzyme-side
   hotspots and its absolute magnitude is unreliable (MAE unchanged, 8–8 head-to-
   head). So fold 0 is not one failure mode — it's an enrichment of *several* hard
   biophysical regimes, only one of which (steric clash) a rigid-backbone
   structure method cleanly fixes.
4. **Direction for the models:** a complex-aware structural channel that encodes
   the mutant side chain in the partner pocket (not just monomeric WT 3Di) should
   lift the steric-clash points; the cavity/electrostatic points likely need
   flexible-backbone physics (flex-ddG / FEP) and may remain a genuine ceiling
   for any single-structure predictor.

## Does this explain the paper (0.868) vs local Ankh (0.832) CV10 gap?

> **⚠ Later refinement — see `S1102_CROSSFOLD_MISPREDICT.md`.** The pooled
> cross-fold analysis there finds **fold 0 is *not* uniquely special**: every fold carries
> its own interface hotspots. Read the “concentrated in fold 0” framing below as specific
> to this earlier pass — the invariant both analyses share is that the **hotspot mutations
> themselves (not the fold partition) are the lever.**

Short answer: **yes — the same hotspot cluster, not the fold structure.** Tested by
joining each fold's `test_predictions.tsv` (pred) with the split's true ΔΔG and
computing Pearson two ways, then peeling off the worst points (whole-set, pooled).

**Aggregation is not the lever.** Pooling all 1100 predictions into one Pearson instead
of averaging per-fold r barely moves Ankh — so the gap is *not* an artifact of how folds
are combined:

| model | mean-of-fold r | pooled r (all 1100) | Δ |
|---|---|---|---|
| Ankh | 0.8321 | 0.8373 | +0.005 |
| ProstT5 | 0.8046 | 0.8105 | +0.006 |
| ESM C 6B | 0.8588 | 0.8687 | +0.010 |

**~18 hotspots account for essentially the entire gap.** Dropping the highest-|error|
points from the *pooled* Ankh set (out of 1100) walks it straight to the paper number:

| pooled Ankh | r |
|---|---|
| all 1100 | 0.8373  (paper gap +0.031) |
| minus top-5 | 0.8496 |
| minus top-10 | 0.8577 |
| **minus top-18** | **0.8668**  ≈ paper 0.868 |
| minus top-30 | 0.8770 |

Those top-|error| points are the very hotspots enumerated above (1PPF LB18W/LB18Y/YB20E,
1KTZ, 1MAH, 2FTL…) — the cross-chain steric ceiling, concentrated in fold 0.

**Interpretation.** The mismatch is dominated by ~18 extreme interface-hotspot mutations
that are an **information ceiling for any sequence model**, not a reproduction error or a
training-budget shortfall — stronger PLMs hit the same wall (ESM C 6B's pooled 0.869 only
*matches* the paper because it fits the non-hotspot bulk slightly better while capping the
same hotspots). So local Ankh **0.832 is a legitimate leakage-free number**; 0.868 is
reachable only if those points are diluted/absent in the paper's exact S1102 build or its
protocol fit them (which the FoldX analysis says a pure sequence model cannot).

**Caveat.** This presumes the paper faced the same S1102 composition and split family. If
it used the *same* `seed=42` split it would share the same hard fold 0 — in which case the
residual gap points at data-version/protocol differences rather than fold 0. Either way the
pooled-vs-per-fold test rules out fold **aggregation** as the cause; the lever is the
hotspot mutations themselves.
