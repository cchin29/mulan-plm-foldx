# Shipped data

Small, hard-to-regenerate artifacts. Between these and [`../results/`](../results/README.md),
every reported *model* metric can be recomputed without a GPU, a FoldX licence, or a large
download. Two exceptions. The CATH-superfamily split is built locally rather than shipped — a
licence question rather than a size one; see the `splits/` row below. And the FoldX-alone
comparator (0.383 on CATH) is scored from FoldX-annotated split files that do not ship, so it is
the one reported number this directory cannot reproduce —
[`../docs/REPRODUCE.md`](../docs/REPRODUCE.md) explains why.

| | size | what |
|---|---|---|
| `splits/` | 3.4 MB | The leakage-controlled partitions themselves — by-complex, sequence-clustered (MMseqs2), and the multi-point variants — all of them our own partitioning. The most reusable thing here: a new method can be evaluated on exactly our hold-outs, and our numbers are checkable. **The CATH-superfamily split is the exception and is not shipped**: it joins our rows onto the static `cath_fold` column published with USP-ddG, a partition originating with CATH-ddG, whose terms cannot be established. `splits/splits_skempi_full_cath_kfold/` carries a build recipe and a checksum manifest instead; its README has the build and verify steps. Credit and citations are in [`../NOTICE`](../NOTICE). The `S1102_filtered_*` variants carry an upstream row set. |
| `skempi_full/` | 5801 rows, 257 KB | Our SKEMPI 2.0 curation — `single_point.tsv` (4165) and `multi_point.tsv` (1636) — as `chain1  chain2  mutation  ddG`. Every row also appears, partitioned, in the shipped splits; they are kept unpartitioned here because `build_split_cath.py` joins against them. |
| `foldx/results_sp/` | 322 complexes, 4238 mutations | Per-complex single-point FoldX ΔΔG, twelve terms each. Needs a FoldX licence — free for academic and non-profit use, paid for commercial — and substantial CPU to regenerate. |
| `foldx/results_mp/` | 152 complexes, 1765 variants | The multi-point equivalent. |
| `foldx/CONSOLIDATION_REPORT.txt` | | What each source campaign contributed, and where they disagree. |
| `foldx_repair_ablation/` | 26 files, 320 KB | The same 13 CATH T≥10 complexes scored at one and at five `RepairPDB` rounds — the energies behind `OPEN_QUESTIONS.md` § "Repair count", and behind the README's explanation for why our FoldX baseline sits below the published one. Outputs only; the repaired structures are not redistributable. Its README gives the three denominators and the chain remap a join needs. |
| `splits/clusters_id60/` | 3 tables, 25 KB | The mmseqs `easy-cluster` output the clustered splits are built from — chain-cluster membership at ≤60% identity, one table per subset. Ours: `s1102.tsv` from `experiments/retrain_split/cluster_split.py`, the two full-SKEMPI tables from `experiments/full_skempi_seqonly/build_split_{clustered,multipoint}.py`, which drive the same clustering over their own row sets. Kept because it is what turns a complex list into a *family* list, and without it the collapse from complexes to families cannot be recomputed from this repository. |
| `benchmarks/` | 40 rows | Frontier comparator numbers transcribed from the source papers, one row per `(method, metric, protocol)`. Read via `scripts_plots/benchmarks.py`; see [`benchmarks/README.md`](benchmarks/README.md) — `protocol` is load-bearing. |

## Known defects in the vendored FoldX store

`foldx/` is a byte-copy of [skempi-foldx](https://github.com/cchin29/skempi-foldx)'s store at its
**v0.1.0** tag. Two documented defects travel with it. Neither is corrected here — that project is
canonical, and patching the copy in place would make the two disagree on values while both claimed
to be v0.1.0. **Both are fixed in skempi-foldx 0.2.0**, which is a rebuilt store rather than a
patch: adopting it means re-running everything scored against these energies, not swapping files.

**Two SKEMPI rows are malformed upstream.** In `skempi_v2.csv`, exactly 2 of 7085 rows repeat a
substitution in `Mutation(s)_cleaned`, and both are `2C5D`:

| shipped | almost certainly meant |
|---|---|
| `RA32E,KA34E,RB32E,`**`KA34E`** | `…,RB32E,`**`KB34E`** |
| `EC30R,EC33R,ED30R,`**`EC33R`** | `…,ED30R,`**`ED33R`** |

Both are in `foldx/results_mp/2C5D.json`. They are 3-substitution mutants labelled as
4-substitution: the energies are correct for what FoldX was asked and wrong for what the row
meant. Deliberately not corrected — substituting the intended string would produce a record
matching no SKEMPI row. The pinned release carries no registry for them, so they are identified
by the literal `Mutation(s)_cleaned` strings in the table above.

**Three codes are scored against a collapsed interface.** SKEMPI defines `2C5D`, `3SE3` and
`3SE4` under two chain pairings each; the parser keys on the PDB code, so one pairing wins and
mutations belonging to the other are scored against an interface SKEMPI does not pair them with.
31 records affected. At the pinned v0.1.0 they are registered as
`skempi_foldx.COLLAPSED_INTERFACES` and queryable with
`skempi_foldx.FoldxLookup.is_interface_suspect(pdb, mutation)`. Both names must be imported from
`skempi_foldx` directly: `mulan.foldx` re-exports 34 names from it and neither of these is among them.
Both were removed in 0.2.0, which keys records
by interface definition and so has nothing left to flag. Of the 26 records that v0.1.0 returned as
zero on all twelve terms under these codes, 15 carry real signal when recomputed against their own
pairing, up to 6.85 kcal/mol.

The two overlap on the same two `2C5D` variants by coincidence, not cause.

**Consequence for anything computed from this store:** filter both before use, or state that a result
did not. `FoldxLookup` warns about the collapsed interfaces at load; the two malformed rows carry
no such guard and have to be excluded by hand.

## Splits

Headerless TSV, `<complex>/fold_<k>/<basename>_{train,val,test}.tsv`, four columns:
sequence-A label, sequence-B label, comma-separated mutations, ΔΔG. See
`../docs/SPLITS_AND_METRICS.md` for what each protocol controls and how to compare it.

> **The mutation strings here are not SKEMPI mutation strings. Do not join them to a structure.**
> The chain letter is a *role* — `A` for the first interface group, `B` for the second — and the
> number is a **1-based index into that group's concatenated sequence**, not a PDB author residue
> number. For `1VFB` this store holds `DB100A` where the split says `DA161A`: different chain
> letter, different number, same mutation. A join keyed on chain and residue number will not
> fail — it will silently return the wrong residue for every complex whose interface group has
> more than one chain.
>
> Treat these strings as opaque keys unless you have the author-form mapping.
> `experiments/full_skempi_seqonly/build_skempi_full.py:14-22` documents the construction, and the
> `<CODE>.mapping` files that invert it are **not redistributed**. The FoldX store's `cleaned`
> field carries SKEMPI's own author-form string and is the only author-form key that ships;
> `data/foldx_repair_ablation/README.md` shows the remap in use.

**The ΔΔG column is a replicate mean, and some replicates disagree.** SKEMPI's `Affinity_wt` is a
per-row reference state rather than a per-complex constant, so averaging across rows can average
across baselines: 609 `(#Pdb, mutation)` keys cover 1501 rows, **560 of them disagreeing by up to
5.37 kcal/mol**, and 362 of 585 averaged groups span more than one `Reference`.
`build_skempi_full.py:180-189` records this; the per-group sidecar that would let you identify
them is written to `scratch/` and does not ship. ΔΔG is
`RT·ln(Kd_mut) − RT·ln(Kd_wt)` with **RT fixed at 25 °C regardless of the measured temperature**
— the convention the frontier comparators use — and positive means destabilizing.

**`(PDB code, mutation)` is not a unique key.** Six single-point and two multi-point pairs collide,
all under `2C5D`, `3SE3` and `3SE4`, which SKEMPI defines under two chain pairings each. Key on the
full interface label (`code.g1.g2`) instead.

## FoldX store

One JSON per complex: `{"muts": {"<mutation>": {<12 terms>, "cleaned": ..., "_source": ...}}}` in
`results_sp/`. `results_mp/` has the same shape but keys the payload **`"variants"`** rather than
`"muts"`, so code that hardcodes `"muts"` reads zero multi-point records rather than failing.
`_source` records which campaign produced the value. FoldX 5.1 `BuildModel` is **deterministic**,
but a mutation's ΔΔG depends on every entry preceding it in `individual_list.txt` — so two
campaigns that asked for different subsets of a complex's mutations get different numbers for the
same mutation. Provenance is therefore part of being able to explain a number; see
[`../docs/FOLDX.md`](../docs/FOLDX.md). Rebuild or audit with:

```python
from mulan.foldx import consolidate, audit
print(audit([dir1, dir2, ...]))
```

The twelve terms and their order are defined once, in `skempi_foldx.terms`, and re-exported
through `mulan.foldx` for callers here. The order is
load-bearing: it is the column order of a decomposed split file and therefore the input order of
the model's score MLP.
