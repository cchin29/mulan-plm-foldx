# Limitations and open questions

What a reader should know before relying on the numbers, the shipped FoldX store or the figures in
this repository. Each item is either a measured caveat or a question the evidence here does not
settle.

---

## FoldX determinism

The widely-repeated claim that FoldX's side-chain optimisation is stochastic does not hold for the
version used here, and the distinction changes how the store should be regenerated.

**FoldX 5.1 `BuildModel` is deterministic.** It exposes one RNG control, `--timeSeedRotabase`,
default `false`; there is no `--seed`. A wall-clock-seeding flag that is off by default only makes
sense if the default seed is fixed. The vendor's description of `numberOfRuns` covers variation
*within* an invocation — the rotamer set and the order of rotamer moves — not between them. Every
campaign here ran `numberOfRuns=1`.

**A mutation's ΔΔG is a function of (repaired structure, every entry before it in the mutation
list).** `BuildModel` walks `individual_list.txt` sequentially in one process, and each entry's
optimisation inherits the state left by the preceding entries. It also re-optimises a fresh
wild-type reference per entry, so both terms of `IE(mut) − IE(wt)` shift together.

Measured over 5873 pairwise comparisons of the same mutation of the same complex:

| | pairs | differ | median \|Δ\| | max \|Δ\| |
|---|---|---|---|---|
| identical structure **and** identical list prefix | 1722 | **0** | 0 | **0** |
| different list prefix | 4151 | 3621 | 0.067 | 11.03 |

Zero counterexamples. Ruled out along the way: across the 182 complexes appearing in more than one
campaign, the repaired structure is shared rather than recomputed — each was repaired once and
copied — and the `AnalyseComplex` chain groups are identical everywhere. The only variable was
which mutations each campaign requested, and a campaign covering a different SKEMPI subset places
a given mutation at a different line number.

**Magnitude**, on `Interaction Energy` over the same population with the excluded `1KBH` dropped —
1527 shared mutations, 5870 comparisons:

| | |
|---|---|
| pooled RMSD between campaigns | **0.524 kcal/mol** |
| median \|Δ\| | 0.008 over all pairs; 0.067 over the pairs that differ |
| 95% limits of agreement | ±1.03 |
| intraclass correlation | **0.972** |
| 99th percentile \|Δ\| | 2.09 |
| maximum | 11.04 |

Per-term ICC runs from 0.991 (solvation hydrophobic) and 0.984 (Van der Waals) down to **0.923
(entropy mainchain)**, 0.935 (sidechain H-bond) and 0.941 (torsional clash). The least stable terms
are ones the scalar arm never sees, which is why the **12-term arm is roughly twice as exposed**:
about 5% of covered rows move more than 0.5 standardized units under a campaign swap, against about
1.3%.

**The effect is concentrated, not diffuse.** 115 of the 181 complexes holding a shared mutation
agree on every one of them and 12 disagree on every one; three complexes — `1PPF`, `1R0R`, `3SGB`,
the OMTKY3 saturation sets, about 14% of split rows — account for **68%** of all absolute
disagreement, and the top ten for 87%. Two campaigns disagree on the sign of `Interaction Energy`
for 175 of 5870 comparisons (3.0%), but almost all are noise around zero: flips where both sides
exceed 0.5 kcal/mol number **9 of 5870**, and at ±2.0 kcal/mol there are none.

**Nothing published is within an order of magnitude of the 11 kcal/mol tail.** Reported FoldX
variance is 0.46–0.61 kcal/mol (calibration SD, and structural sensitivity across different PDBs of
the same protein), with a ±3.5 kcal/mol *prediction interval* as the widest figure of any kind.
That gap is independent support for reading the spread as a pipeline effect rather than FoldX
noise.

**Consequences for anyone regenerating the store.** Re-running a campaign with its original
mutation list reproduces it exactly, so the published store is reproducible. But subset computation
is not interchangeable with full computation: the union of a complex's mutations must be computed
once and sliced per dataset afterwards, which is what `worklist_single_point` does.
`worklist_from_table`, which takes a per-dataset subset, is retained only to reproduce the
historical campaigns. `--numberOfRuns > 1` is not the remedy — there is nothing random to average.

**Single- and multi-point energies for one complex are not poolable.** They come from separate
`BuildModel` invocations with separate mutation lists. The canonical store splits them for this
reason, and the split is what makes them comparable to it.

**Cross-platform agreement is unestablished.** The multi-point campaign ran on Linux and the
single-point ones on Apple Silicon, and there is no vendor statement on bitwise agreement across
platforms. It cannot affect the current store, which has zero shared mutation keys between the two,
but one platform per dataset is the safe convention.

## Repair count

The store ships at a **single** `RepairPDB` pass. That is the FoldX default and it is not neutral.

*Structural convergence* (13 CATH T≥10 complexes, unsuperposed heavy-atom RMSD between rounds):

| round | median | max |
|---|---|---|
| 1 → 2 | **0.2496 Å** | 0.3828 |
| 2 → 3 | 0.0797 | 0.1916 |
| 3 → 4 | 0.0216 | 0.1768 |
| 4 → 5 | 0.0132 | 0.0229 |

Convergence arrives by round 4 — faster than the literature's 7–10 rounds — but the first repair
leaves a quarter-Ångström on the table.

*Energy effect* (5× vs 1× repair, identical union mutation lists, so repair count is the only
variable): **97.6% of 334 mutations change** on `Interaction Energy`, median 0.070, p90 0.62, max
8.51 kcal/mol.

*Metric effect* — FoldX-alone per-structure Spearman, **CATH single-point only**, T≥10, 11
complexes, paired cluster bootstrap over complexes. This pilot did not touch multi-point, and the
CATH test set is 39% multi-point, so this is the CATH-*single* number:

| | ρ | 95% CI |
|---|---|---|
| 1× repair | 0.3984 | [0.2759, 0.5193] |
| **5× repair** | **0.4585** | [0.3455, 0.5695] |
| paired Δ | **+0.0601** | **[+0.0204, +0.1000]** — excludes zero |
| published FoldX row | 0.4303 | |

The 1× baseline reproduces this repository's existing CATH-single FoldX-alone figure exactly, which
validates the scoring path. Five repairs move it past the published FoldX row, which is the most
likely reason our FoldX baseline sat below theirs — not a coverage or metric difference.

**The energies behind both arms ship**, at
[`data/foldx_repair_ablation/`](data/foldx_repair_ablation/) — 13 complexes at each repair count,
all twelve terms. Its README gives the three denominators these results use (13, 13 and 11
complexes, which is easy to misread as one set) and the author-form chain remap a join needs. The
repaired *structures* are not redistributable and are not there, so the RMSD rows cannot be
recomputed from this repository while the ΔΔG rows can.

*Controls.* The list-composition effect on this set is exactly zero (293 shared mutations, 0
differ), because the campaign that produced these complexes already used the union list, so the
measured delta is purely repair count. The accepted mutation list is itself derived from the
repaired structure — a confound that would leave no trace in the logs — and it is clean: 13/13
complexes have byte-identical mutation lists between the 1× and 5× arms, and the full sweep checked
**344/344 complexes with 0 residue-set and 0 accepted-list differences** across both arms.

*No contradiction with the published null result on repair count*, which measured **folding** ΔΔG
and self-consistency bias. This measures **binding** ΔΔG correlation with experiment.

**What follows.** A stronger FoldX baseline makes the comparator this work must beat *harder*. The
existing analysis already finds that the base arm never beats FoldX alone; raising the baseline
sharpens that rather than softening it. The honest reading is that the single-repair default
understates the FoldX channel, and this should be treated as a known sensitivity of the shipped
store rather than a settled protocol.

**Open:** a four-round repair sweep is complete over 322 single-point complexes (4334 mutations on the sweep's union list; 4238 ship here)
and 152 multi-point complexes (1765 variants), but the analysis is not written — nothing yet scores
round against Spearman, so *whether the energy converges earlier than the structure* is unanswered.
The sweep is four rounds, so it cannot measure the 5× endpoint the pilot above used.

## Metric and split decomposition

`docs/SPLITS_AND_METRICS.md` states that on the same trained model the metric change costs ~0.88
pooled Pearson → ~0.49 per-structure Spearman, and the split change ~0.49 → ~0.37. Both figures are
inherited from `experiments/METRIC_RATIONALE.md` (ESM-C 6B) rather than independently re-derived.
They are the leading claim of that section and have not been confirmed against a final re-score.

## Test coverage

The three suites (164 tests) cover `plm`, `foldx` and `metrics`/`splits` — the machinery this fork
adds. **The model itself is close to untested.** `modules.py`, `config.py` and `interface_xattn.py`
have no tests at all, and neither do the five CLI entry points. `constants.py`, `data.py`,
`utils.py` and `train_utils.py` are each reached at one or two points, incidentally, from suites
aimed at something else.

The uncovered code is upstream's and unmodified, so tests there would guard against future edits
here rather than against anything currently known to be wrong. A green suite should not be read as
covering the model.

## Headline figures

`README.md` quotes one: 0.438 per-structure Spearman on the CATH hold-out, fourth of eleven. It is
quoted only alongside the two rows that qualify it — FoldX as measured here at 0.383, and FoldX as
published on the same tier at 0.430 — because the margin over the published physics baseline is
within rounding, and the single-`RepairPDB` default is the likely reason our own baseline sits
lower (see *Repair count* above). `docs/SPLITS_AND_METRICS.md` quotes none.

Figures that circulated earlier in this project's life came from superseded FoldX-coverage
lineages; they are not comparable to current arms and are not quoted as results.
`docs/FOLDX_STRUCTURE_ARC_AND_AUGMENTATION_PLAN.md` was written against the 87.8%-coverage lineage
and was recomputed against the current one on 2026-08-05; its §5 records both, and its header states
what moved. Where a number there disagrees with `RESULTS.md`, `RESULTS.md` is correct.

For orientation, the frontier reference points on the CATH protocol are CATH-ddG 0.494 and USP-ddG
0.493.

## Comparator figures

One protocol mismatch remains visible in the figure sources:

- **Cross-metric reference line.** ProtBFF reports the clustered split **pooled only** (0.477
  Spearman / 0.514 Pearson), so the clustered panel has no per-structure comparator. The
  generalization ladder's reference line sources 0.477 and its annotation names the metric, rather
  than showing a bare "~0.48" on a per-structure axis.

Two mismatches that this section carried until 2026-09-12 are settled. `plot_ppS_generalization.py`
now draws all three of its panels on the combined single+multi rungs, as `plot_ppS_scaling.py` already
did, so the five by-complex comparators (each one model trained over single **and** multi-point
mutations) sit against bars trained the same way. And the two physics protocols USP-ddG re-evaluates
on its CATH hold-out, flex ddG (0.454) and FoldX (0.430), are drawn on both CATH panels beside the
seven learned methods; the values were checked against Table 1 of the paper (Per-PPI Spearman,
all-mutation rows: 0.4542 and 0.4303) before the figures were regenerated. Both fall inside the
seven's range, so the shaded band is unchanged; what is new is that they bracket MuLAN's best CATH
bar. The single-point rows stay in `results_matrix_ps.csv`.

Comparator values are transcribed third-party results and live in `data/benchmarks/frontier.tsv`,
one row per `(method, metric, protocol)` with the source paper named per row. Numbers measured in
this repository are never mixed into that file.
