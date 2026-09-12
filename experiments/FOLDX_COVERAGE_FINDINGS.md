# FoldX coverage on full-SKEMPI — gap analysis & findings

_Last updated: 2026-07-24_

> **⛔ Superseded 2026-07-30. Every coverage figure in this document is the pre-fix state, and
> the "Merger fix" it recommends was afterwards RETRACTED.**
>
> Measured now: single-point **99.2%** (12393/12495 non-zero scalars), combined SP+MP **99.1%**
> (5746/5801 per fold). Do not quote the 87.7% / 87.8% / 90.8% figures below, or the 28% before
> them, as current.
>
> The retraction is the important part, because a coverage percentage cannot express it. The
> **dual-key `load_foldx`** that this document presents as the second fix — and whose 90.8% it
> adopts as the stopping point — was wrong, not merely superseded: its second key landed on other
> real rows and handed them a *different mutation's* ΔΔG. The corrected merger is single-key.
> Arms trained under the dual-key store carry a `__cov87` archive directory.
>
> The "residual ~12% that cannot be closed cheaply" is therefore also void: it was a
> residue-numbering join bug, and the current uncovered tail is 102 of 12495 rows. See
> [`full_skempi_seqonly/SP_FOLDX_COVERAGE_AUDIT.md`](full_skempi_seqonly/SP_FOLDX_COVERAGE_AUDIT.md)
> for the diagnosis that produced the fix.
>
> Kept unedited below because it is the record of what was believed, and of a decision
> ("stop at 87.7%") that the later audit overturned.
>
> **The per-complex table below is the 502 class, not all 510 uncovered rows** (audited
> 2026-08-06). Its column is headed "uncovered", which is what makes 1DAN read 42 where a
> recount of uncovered rows gives 44 — the difference is the two rows that `map_mutations`
> marked `unresolved`, discussed in their own right immediately after the table. 1VFB (22 vs 23)
> and 1DVF (18 vs 19) differ for the same reason. All ten rows and the 510/65 and 502/8 totals
> reproduce exactly under the retracted dual-key join, so the table is an accurate record of the
> superseded computation and is left unedited. 1DAN is the largest gap because tissue factor is
> deposited as two chains that both sit in `g2 = 'UT'`, so the chain-only remap collapses two
> rows there where every other complex loses one.

FoldX binding-ΔΔG feeds MuLAN's `add_scores` channel (Stage-1 scalar Interaction
Energy / Stage-2 12-term decomposed MLP). This note records how much of full-SKEMPI
FoldX actually covers, the two bugs that were masking coverage, the residual gap that
cannot be closed cheaply, and how the published frontier handles the same problem.

## TL;DR

| Milestone | CATH-all (SP+MP) | single-pt | multi-pt |
|---|---|---|---|
| Original per-run FoldX dirs | ~48% | ~28% | — |
| **Plumbing fix** — unified `results_all` (union of all runs) | ~69.5% | 58.0% | 98.7% |
| **Merger fix** — dual-key `load_foldx` | **90.8%** | **87.7%** | 98.7% |

Single-point coverage across the three splits after both fixes: CATH-all **87.7%**,
clustered-SP **87.8%**, by-complex-SP **87.8%**. The residual ~12% single-point gap
(510 muts / 65 complexes) is a hard multi-chain-ambiguity tail, not a compute gap.

## Two fixes, in order

### 1. Plumbing: unified `results_all` (28→58% single)
FoldX had been run in five separate campaigns (full-SKEMPI + the S1102/S1131/S2003/S4169
benchmark subsets), each writing its own `results*/` dir keyed to a *different* mutation
subset. The mergers read only one dir, so most already-computed values were invisible.
`experiments/full_skempi_seqonly/build_results_all.py` symlinks each complex to its
most-complete JSON across all five dirs (audit: the largest JSON is always the union — 0
complexes need a real dict-merge). This lifted single-point coverage 28→58% at **zero
compute**. The `#138` delta run (BuildModel-only, RepairPDB already on disk) then filled
107 more complexes incl. the big protease scans (3BT1 240/240, 1PPF 190/190, 1R0R
191/191, 3SGB 191/191).

### 2. Merger bug: chain-remap dropped every benchmark complex (58→88% single)
Symptom: after `#138` filled 3BT1 with all 240 muts, `merge_foldx_cath.py` still showed
**0/240** for 3BT1 — the delta compute wasn't reaching the splits.

Root cause in `merge_foldx_full_skempi.py::load_foldx`: it force-remapped every mutation's
chain to the split's `g1→A / g2→B` convention. **Standard complexes need this** (their
FoldX JSONs are keyed by original PDB chain, e.g. json `HA123T` → split `A123T`). But the
**S4169 benchmark-sourced complexes** (3BT1/1PPF/1R0R/3SGB/…) already carry the split's
*native* keys — json `AB18S` **==** split col2 `AB18S` — and their chain letter isn't in
`g1/g2`, so `remap_mut` returned `None` and **silently dropped all of them**.

Fix: dual-key — index each mutation under **both** its remapped key *and* its raw original
key (additive, no collisions; the remapped form stays authoritative for standard
complexes). Committed `6319caf`. Impact:

```
CATH-all   single 58.0% -> 87.7%   (combined 69.5% -> 90.8%)
clustered  single 58.2% -> 87.8%
bycomplex  single 58.2% -> 87.8%
```

`merge_foldx_cath.py` (the mixed SP+MP merger) imports `spm.load_foldx`, so it inherited
the fix for free. The multi-point loader is separate and was already at 98.7%.

## The residual single-point gap: 510 muts / 65 complexes (~12% of CATH single)

Fully classified — it is **one clean category**:

- **0** complexes missing a JSON entirely.
- **0** remap failures remaining (dual-key closed those).
- **502** muts: the split needs a mutation that is **not in `build_ddg`'s SKEMPI
  mut-list** — the fork's split builder emits interface mutations that
  `build_ddg.load_skempi()` drops. The delta driver computes `sk[pdb]["muts"]`, so it
  *structurally cannot* produce these.
- **8** muts: present in `build_ddg`'s list but dropped as **`unresolved` in
  `map_mutations`** (not `validation_failed`).

### Why the gap is a multi-chain-ambiguity tail, not a compute gap
It concentrates in large multi-chain **antibody / TCR–pMHC** complexes:

| complex | uncovered | type |
|---|---|---|
| 1DAN | 42 | FVIIa / tissue factor (4-chain) |
| 4NKQ | 26 | antibody |
| 3C60 | 24 | TCR–pMHC |
| 4P23 / 4P5T | 24 / 24 | antibody |
| 1MHP / 1VFB / 1DVF | 22 / 22 / 18 | antibodies |
| 1AO7 / 1OGA | 19 / 19 | TCR–pMHC |

The mechanism (verified on all 8 of the "fillable" muts): each has a **twin mutation with
identical `(wt, pos, mut)` on a sibling chain in the same interface group**, e.g.:

```
1DVF  YA32A  vs  YB32A   (A = antibody heavy, B = light — DIFFERENT residues, shared numbering)
1DQJ  YA50A  vs  YB50A
1DAN  DU39A  vs  DT39A   (both in g2 = 'UT')
3HFM  YH50A  vs  YL50A
```

`map_mutations` refuses to guess which copy the SKEMPI record intended, so it marks them
`unresolved`. Filling them would mean choosing a chain-copy (or mutating both) — a modeling
judgment with a real chance of writing a **wrong** ΔΔG (mutating the light chain when the
experiment mutated the heavy). Not safely automatable, for 0.06% of rows.

**Decision: stop at 87.7% single / 90.8% combined.** Uncovered rows fall back to the
per-fold standardized mean (0) — a mild signal dropout, not leakage. Closing the 502 would
require a *split-driven* FoldX pass (feed FoldX the split's exact muts, reverse-mapped to
original chains), which risks importing the fork's own chain-numbering ambiguities into the
worst-behaved complexes.

## How the published frontier handles coverage

Framing correction: in the SKEMPI frontier (RDE-Network, Prompt-DDG, DiffAffinity,
CATH-ddG, USP-ddG), **FoldX is used as an unsupervised _baseline predictor_, not as an
input feature/augmentation**. MuLAN+FoldX (FoldX as an `add_scores` channel) is the less
common setup, so "coverage of a FoldX feature" is largely our own concern.

The accurate picture (line-verified against the local PDFs): these methods use **nearly all
of SKEMPI 2.0** and prune only a **narrow** set of entries — missing-ΔΔG rows and a handful
of individually un-processable structures. They don't chase FoldX to 100% because FoldX is
just a baseline column, not a feature; the un-processable structures they *do* drop are the
same class that caps our merge.

**CATH-ddG** _(verified from local `CATH-ddG_fulltext.xml`)_. From 352 PPIs / 7,085 muts it
removes just two things → **343 PPIs / 6,706 muts**:

| filter | removed |
|---|---|
| missing experimental ΔΔG | 287 muts |
| **PDB 1KBH — "problematic structure"** (attributed to Luo et al. / RDE-Network) | **92 muts** |
| → benchmark | **343 PPIs / 6,706 muts** |

The 1KBH row is the concrete, sourced structural-coverage exclusion: an entire complex
thrown out because the structure can't be handled — directly analogous to our
un-scoreable multi-chain complexes. CATH-ddG otherwise keeps the hard cases and manages
distribution shift through its CATH-superfamily split rather than dataset pruning.

**RDE-Network / Prompt-DDG / DiffAffinity** (the per-structure-Spearman eval protocol)
— _verified against the two papers_.
Contrary to a widely-repeated secondary summary, these do **not** apply a heavy up-front
filter to a ~4,541-mut / 201-complex subset. RDE-Network's own text: _"We split the dataset
into 3 folds by structure… [which] ensures that **every data point in SKEMPI2 is tested
once**"_ — i.e. essentially the full ~7,085-mut set under by-structure 3-fold CV. Prompt-DDG
states it splits the 7,085 muts by structure and _"follow[s] (Luo et al., 2023)"_.

Their only exclusion is a **metric-reporting** one, not a coverage cut: for the
per-structure correlation they _"group mutations by structure, **discard groups with less
than 10 mutation data points**"_ — which is exactly the **T≥10 per-structure Spearman**
threshold this project uses. The FoldX baseline is run with _"Structures … first relaxed by
the RepairPDB command"_; **no FoldX coverage/failure percentage is reported**.

> ⚠️ Correction: an earlier draft cited a "4,541 muts / 201 complexes, 8 complexes with 162
> unresolved-residue muts" breakdown attributed to this pipeline. Line-checking RDE-Network
> locally shows **that breakdown is not in the paper** (a web-summary fabrication). The only
> verified frontier structural exclusion is CATH-ddG's 1KBH drop above.

So the frontier's answer to a structure it can't handle is to **drop that structure**
(CATH-ddG's 1KBH), not to reach full FoldX coverage — but the drops are narrow, and the
evaluation is otherwise on ~all of SKEMPI2. Our 87.7%/90.8% keeps even the hard multi-chain
complexes and zero-fills only the individual un-scoreable rows — so our eval set is at least
as complete as the frontier's, on the same by-structure / T≥10 protocol.

Honest caveat: none of the surveyed papers report a headline "FoldX covers X% of SKEMPI"
number. FoldX is used as a baseline predictor whose ΔΔG is computed per mutation
(RepairPDB → BuildModel), and coverage/failure statistics are simply not a reported
quantity — they report FoldX *correlation*, not FoldX *coverage*. The defensible claims are
therefore: (a) frontier methods evaluate on essentially all of SKEMPI2, dropping only
missing-ΔΔG rows and individual un-processable structures (verified: CATH-ddG → 6,706 muts;
RDE-Network → "every data point tested once"); (b) no published FoldX-coverage percentage
exists to compare our 88% against.

## Sources
- [SKEMPI 2.0 (Jankauskaitė et al. 2019)](https://academic.oup.com/bioinformatics/article/35/3/462/5055583) — 7,085 mutations / 348 complexes.
- [RDE-Network (Luo et al., ICLR 2023 / bioRxiv 2023.02.28.530137)](https://www.biorxiv.org/content/10.1101/2023.02.28.530137v1) — full SKEMPI2, 3-fold by-structure CV, per-structure T≥10; FoldX baseline via RepairPDB.
- [Prompt-DDG (Wu et al., arXiv 2405.10348)](https://arxiv.org/html/2405.10348) — FoldX as baseline; splits 7,085 muts by structure, follows Luo et al.
- [Energy-Based Models for Mutational Effects (arXiv 2508.10629)](https://arxiv.org/html/2508.10629v1) — frontier baselines incl. FoldX on SKEMPI.
- [CATH-ddG (Bioinformatics 2025)](https://academic.oup.com/bioinformatics/article/41/Supplement_1/i362/8199350) — CATH-superfamily out-of-distribution split.
- [Predicting mutational effects from folding energy (arXiv 2507.05502)](https://arxiv.org/pdf/2507.05502) — folding-energy predictor on SKEMPI.
- [FoldX force field revisited (Bioinformatics 2025, btaf064)](https://academic.oup.com/bioinformatics/article/41/2/btaf064/8003679) — updated FoldX; RepairPDB behaviour.
