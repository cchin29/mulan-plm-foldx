# FoldX antisymmetry under mutation reversal — probe

> **Executed 2026-08-05 on the Mac.** 1744 single-point mutations over 112 complexes, zero failed
> runs, ~35 min across 8 parallel workers. Results and the decision-rule outcome:
> `experiments/beyond_foldx/FOLDX_ANTISYMMETRY_RESULT.md`. Scope was widened from the augmented
> rows to every mutation with a retained mutant structure — antisymmetry is a property of FoldX,
> not of which rows were augmented — and the augmented rows are kept as a stratum. The run also
> surfaced a second, larger and unrelated defect in the augmented channel's standardisation, which
> is §1 of that document.

Closes the open mechanism in `RESULTS.md` §25.1. FoldX runs only; no PLM inference, no training,
no changes under `mulan/`.

## 1. What is being tested

Tier-1 augmentation builds each reverse row by **negating** the label and all twelve FoldX terms.
For the label that is exact — ΔΔG(A→B) = −ΔΔG(B→A) is thermodynamics. For FoldX it is an
**assumption**, and a different kind of quantity: FoldX ΔΔG is computed from a *modelled* mutant
structure, so the value it would return for the reverse mutation is not guaranteed to be the
negative of the forward one.

If that assumption is false, every augmented FoldX row carries a fabricated value, and the arms
that read those columns would degrade while the base arm — which has no FoldX column — would not.
That is exactly the sign split §25.1 reports, under a different mechanism than the one recorded
there:

| reading | what the negative FoldX deltas mean |
|---|---|
| currently recorded | augmentation conflicts with the FoldX channel |
| **not yet excluded** | the augmented FoldX column is mislabelled, so the arms are trained on values FoldX would not produce |

These call for opposite responses. The first says the channel and the augmentation do not combine;
the second says the augmentation is repairable by computing the reverse rows instead of negating
them. Nothing in the existing checks separates them — the "well-formed" verification in §25.1
confirms the negation was *applied* consistently, not that negation is the *correct value*.

**This cannot be answered from data already on disk.** There are zero forward/reverse mutation
pairs among the FoldX-annotated rows (checked across all three folds of the clustered-ALL tier), so
there is no measured FoldX value to compare a negated one against.

## 2. What is already on disk, and what that saves

`RepairPDB` is the expensive step and it does not need repeating. Each complex's work directory
retains both the repaired wild type and every mutant structure BuildModel produced:

```
scratch/foldx_skempi_full/work/<PDB>/
  <PDB>_Repair.pdb          repaired wild type
  <PDB>_Repair_<i>.pdb      mutant structure for the i-th mutation
  individual_list.txt       the mutation list, one per line, in that same order
```

The mapping is positional and exact: line *i* of `individual_list.txt` corresponds to
`<PDB>_Repair_<i>.pdb`, and to the *i*-th key of `results/<PDB>.json`'s `muts` object. Verify on
one complex before trusting it across the sample — `1AHW` line 1 is `DC167A;`, so
`1AHW_Repair_1.pdb` should carry alanine at chain C position 167.

Four result trees hold these: `results/`, `results_all/`, `results_multipoint/`,
`results_remainder/`. Sample only from complexes whose rows appear in the **augmented** train
splits, since those are the rows whose FoldX values were fabricated.

## 3. Protocol

Per sampled forward mutation `<wt><chain><pos><mt>` on complex `<PDB>`, index *i*:

1. Copy `<PDB>_Repair_<i>.pdb` to a fresh working directory as the input structure.
2. `BuildModel` the **reverse** mutation on it — `<mt><chain><pos><wt>`, i.e. back to the wild-type
   residue at the same position.
3. `AnalyseComplex` on the resulting pair, with the **same interacting-group definition** the
   forward run used. Take the same twelve terms, and `Interaction Energy` as the scalar.
4. Record ΔΔG_reverse beside the stored ΔΔG_forward.

**RepairPDB is deliberately not re-run on the mutant.** That makes the probe conservative rather
than exact: less structural relaxation means the reverse calculation stays closer to the forward
geometry, which biases the result *toward* antisymmetry. So a null result is weak evidence and a
positive one is strong — see §5.

## 4. Sample

Target **~200 mutations across ≥20 complexes**, drawn only from rows present in an augmented train
split. Stratify on two axes, because both plausibly drive any asymmetry:

- **|ΔΔG_forward|** — small, medium, large thirds. Hysteresis should grow with the magnitude of the
  structural change.
- **mutation type** — to/from glycine and proline, to/from charged, and size-changing substitutions
  separately from conservative ones. Reversing a large-to-small substitution is not the same
  operation as reversing a small-to-large one, and averaging over both would hide it.

Single-point only. Multi-point reversal compounds the question and can follow if the single-point
answer is positive.

## 5. Readouts and the decision rule

Perfect antisymmetry gives correlation +1 and a mean sum of exactly zero.

| statistic | under antisymmetry |
|---|---|
| `corr(ΔΔG_fwd, −ΔΔG_rev)` — Pearson and Spearman | +1 |
| `mean(ΔΔG_fwd + ΔΔG_rev)` | 0 |
| `sd(ΔΔG_fwd + ΔΔG_rev)` | 0 |

Report all three with a cluster bootstrap over complexes (B = 10,000, seed 0), matching every other
contrast in this repository. The third matters most: a high correlation with a large spread still
means individual augmented rows carry wrong values, and it is individual rows the model trains on.

**Decision rule, fixed in advance.**

- **Correlation ≥ 0.95 and the mean sum's CI covers zero** → antisymmetry holds well enough that the
  negation is not what degraded the FoldX arms. §25.1's recorded reading stands, and the mechanism
  stays open for another explanation. Weak evidence, per §3's conservative bias.
- **Correlation ≤ 0.8, or `sd(sum)` comparable to the spread of ΔΔG itself** → the augmented FoldX
  column is substantially fabricated. §25.1's conclusion must be narrowed from "augmentation
  conflicts with the FoldX channel" to "the antisymmetric construction of the FoldX column is
  invalid", and the augmentation becomes repairable rather than abandoned.
- **In between** → report the numbers, do not pick a reading, and treat the ESM-C 6B base arm
  (`RUN_AUG_BASE_ESMC6B_GPU.md`) as the decisive experiment instead.

## 6. Cost

Two FoldX calls per mutation, no RepairPDB. On the order of a few seconds to a minute each, so ~200
mutations is well under an hour of compute — the same order as one fold of one training arm, and
far cheaper than the augmented reruns whose interpretation it decides.

Exclude any complex on the `RepairPDB`-intractable list before building the worklist rather than
killing runs afterwards.

## 7. Output

Write `experiments/beyond_foldx/foldx_antisymmetry.csv` with one row per sampled mutation:

```
pdb, mut_fwd, mut_rev, ddg_fwd, ddg_rev, sum, chain, pos, wt, mt, abs_bin, class
```

and a short result doc beside `REDUNDANCY_RESULT.md` carrying the three statistics, their CIs, and
whichever branch of §5 the numbers select. Update `RESULTS.md` §25.1 with the outcome either way —
including a null, which is the branch most likely to go unrecorded.
