# SKEMPI single-mutation benchmarks — S1131 / S4169 / S2003

Documentation of the three benchmark datasets used for the Ankh-vs-ProstT5 10-fold CV
(`RESULTS_BENCHMARKS.md`), how each was built, and — importantly — **how they overlap**.
All are single-point-mutation protein–protein binding ΔΔG sets derived from SKEMPI 2.0,
restricted to **single-chain-per-partner** complexes (MuLAN takes one sequence per partner,
the same restriction `build_wt_fasta.py` applied for S1102).

Built by `experiments/build_benchmarks.py`; tables in `scratch/benchmarks/<ds>/<ds>.tsv`,
shared WT sequences in `scratch/benchmarks/wt_sequences.fasta`.

---

## The three datasets

| Dataset | Mutations | Complexes | Ala / non-Ala | ΔΔG min/mean/max | Source |
|---|---|---|---|---|---|
| **S1131** | 1,127 | 110 | 389 / 738 | −12.35 / +1.24 / +12.35 | Xiong et al., via GeoPPI (exact) |
| **S4169** | 2,497 | 211 | 1,375 / 1,122 | −12.35 / −0.98 / +12.35 | mCSM-PPI2, via GeoPPI; single-chain subset |
| **S2003** | 1,124 | 174 | 0 / 1,124 | −12.22 / −1.06 / +12.22 | derived (non-Ala SKEMPI singles) |

- **S1131** — the non-redundant *interface* single-mutation set of **Xiong et al.**
  (BindProfX, 2017), downloaded verbatim from `Liuxg16/GeoPPI`
  (`data/benchmarkDatasets/S1131.csv`). 4 entries dropped (no on-disk structure, e.g. `2I9B`);
  100 % single-chain already, so essentially the full set.
- **S4169** — the "all single mutations" set of **mCSM-PPI2** (Rodrigues et al., 2019): 4,169
  variants across 319 complexes, obtained here from GeoPPI's verbatim copy. After the
  single-chain-per-partner restriction it is **2,497 / 211** (the other 1,672 are multi-chain
  partners — antibodies etc. — which MuLAN's one-sequence-per-partner design cannot take).
- **S2003** — **derived locally** (no canonical published file exists). Defined as the single
  **non-alanine** mutations of SKEMPI 2.0 (the "~2,000 non-Ala singles" the database is known
  for), single-chain, with duplicate measurements averaged. Single-chain restriction yields
  **1,124** (1,122 unique complex+mutation keys). The name "S2003" is kept for the ~2,000-row non-alanine set it approximates.

**ΔΔG convention** (all three): `ΔΔG = −RT·ln(Kd_mut/Kd_wt)`, matching the benchmark files
(calibrated: 1ACB L38D computed −6.77 vs the file's −6.867). S1131/S4169 carry the values as
published in those files, which trace to SKEMPI;
S2003's are computed from SKEMPI 2.0 `Kd_wt`/`Kd_mut` at the recorded temperature. Sign is
immaterial to Pearson/Spearman (both PLMs train on identical labels per dataset).

---

## Overlaps and differences (the key structure)

The three are **nested**, not independent — **S4169 is the superset**:

```
                S4169  (2,497 muts / 211 complexes)
        ┌─────────────────────────────────────────────┐
        │  1,375 alanine            1,122 non-alanine  │
        │  ┌───────────────┐        ┌────────────────┐ │
        │  │               │        │   = S2003      │ │   S2003 ≡ the non-Ala
        │  │   S1131 (1,127, ⊂ S4169)                │ │   mutations of S4169
        │  │   389 Ala  +  738 non-Ala ──────────────┼─┤   (1,122 = all of S2003)
        │  └───────────────┘        └────────────────┘ │
        └─────────────────────────────────────────────┘
```

- **S1131 ⊂ S4169** — all 1,127 S1131 mutations are in S4169 (S1131 is Xiong et al.'s
  non-redundant subset; S4169-only adds 1,370 mutations).
- **S2003 ⊂ S4169** — all 1,122 S2003 mutations are in S4169. In fact **S2003 = the
  non-alanine mutations of S4169** (S4169 has exactly 1,122 non-Ala single-chain mutations,
  identical to S2003). This is a clean cross-validation that the local derivation matches
  the S4169 file on the non-Ala entries.
- **S1131 ∩ S2003 = 738** — exactly S1131's 738 non-alanine mutations (S1131's other 389 are
  alanine, hence absent from non-Ala S2003; S2003 adds 384 non-Ala beyond S1131).
- **All three share 738 mutations** (= S1131's non-Ala set).

### Pairwise mutation overlap

| Pair | shared | A-only | B-only | relationship |
|---|---|---|---|---|
| S1131 vs S4169 | 1,127 | 0 | 1,370 | **S1131 ⊂ S4169** |
| S2003 vs S4169 | 1,122 | 0 | 1,375 | **S2003 ⊂ S4169** (= its non-Ala half) |
| S1131 vs S2003 | 738 | 389 (Ala) | 384 | overlap = S1131's non-Ala |

### Complex overlap

| Pair | shared complexes | union |
|---|---|---|
| S1131 ∩ S4169 | 110 | 211 |
| S2003 ∩ S4169 | 174 | 211 |
| S1131 ∩ S2003 | 102 | 182 |
| **all three** | **102** | **211** |

S4169 spans all 211 complexes; S1131's 110 and S2003's 174 are subsets. The embedding union
across all three is therefore just S4169's id set: **3,183 unique sequences** (1,445 already
in the S1102 cache, ~1,738 newly generated per PLM).

### What distinguishes them, despite the nesting

Although nested, each stresses the model differently:
- **S1131** — non-redundant *interface* mutations; smallest, cleaned for redundancy. Mean
  ΔΔG **+1.24** (skews destabilizing) — the most "alanine-scanning-like" of the three.
- **S4169** — the broadest set (most complexes, both Ala + non-Ala); the standard headline
  benchmark. Mean ΔΔG **−0.98**.
- **S2003** — non-alanine only, so it isolates *substitution* effects (not just side-chain
  deletion to Ala), generally considered harder. Mean ΔΔG **−1.06**.

Because S1131 and S2003 are subsets of S4169, their CV numbers are **not statistically
independent** of S4169's — treat them as three lenses on one SKEMPI pool, not three
independent benchmarks. This matters when reading the per-dataset Ankh-vs-ProstT5 gaps.

---

## Relation to S1102 (this repo's existing benchmark)

S1102 (the repo's original set, 1,100 muts) is almost entirely contained in S4169:
S4169 ∩ S1102 = 1,100 (all of S1102), S1131 ∩ S1102 = 1,019, S2003 ∩ S1102 = 767. So these
benchmarks extend the *same* SKEMPI 2.0 single-chain pool that S1102 was drawn from, to more
complexes (110 → 211) and the full single-chain single-mutation set.

---

## Caveats

1. **Single-chain restriction** drops 40 % of the literature S4169 (multi-chain antibody
   complexes), so these are MuLAN-compatible **subsets**, not the verbatim published sizes.
   The Ankh-vs-ProstT5 comparison remains fair (identical data per PLM) but numbers are not
   directly comparable to published S4169 leaderboards.
2. **S2003 is a local derivation**, not a canonical published benchmark — it equals the
   non-alanine half of S4169. Labelled "S2003" only for traceability.
3. **Nesting** (above) means the three results are correlated, not independent.

## Provenance / sources

- **S4169's selection**: mCSM-PPI2 — Rodrigues, Myung, Pires & Ascher, *Nucleic Acids
  Research* 47(W1):W338–W344, 2019, doi:10.1093/nar/gkz383. CC BY 4.0.
- **S1131's selection**: BindProfX — Xiong, Zhang, Zheng & Zhang, *J. Mol. Biol.*
  429(3):426–434, 2017, doi:10.1016/j.jmb.2016.11.022.
- **Both obtained from GeoPPI's verbatim copies** — Liu et al., *PLOS Comput. Biol.*
  17(8):e1009284, 2021; `github.com/Liuxg16/GeoPPI` (MIT),
  `data/benchmarkDatasets/S1131.csv`, `S4169.csv`. GeoPPI renumbered residues and renamed
  columns; the row sets and ΔΔG values are unchanged from the originating files, which is why
  the selections are credited above rather than to it.
- SKEMPI 2.0: Jankauskaitė et al., *Bioinformatics* 2019; `scratch/skempi_v2.csv` +
  cleaned PDBs in `scratch/skempi2/PDBs`.
- Build + validation: `experiments/build_benchmarks.py` (0 WT-residue mismatches across all
  three after structure-based validation).
