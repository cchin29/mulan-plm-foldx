# Single-point FoldX coverage audit — the origin of the 87.8% figure

_Audited 2026-07-28 on the Linux box (`consolidate-results`). Companion to
`MULTIPOINT_FOLDX_AUDIT.md`, which covered the multi-point arm only — the single-point arm had
never had the equivalent audit. Written as a spec for the next round; the fix it specifies landed
on 2026-07-30, as the banner below records._

> **The fix landed on 2026-07-30. Every coverage figure below is the pre-fix state.**
> Measured after: SP **99.2%** (12393/12495 non-zero scalars), combined SP+MP **99.1%**
> (5746/5801 per fold) — close to, and slightly better than, the 99.6% / 99.4% this document
> projected. The projection was sound; the diagnosis was right.
>
> Two things to carry away rather than the numbers. The 87.8% / 90.8% pair is **not** what
> current arms are trained on, and arms that were carry a `__cov87` archive directory. And the
> dual-key merge that produced 90.8% was later **retracted outright** — not merely superseded:
> its second key landed on other real rows and handed them a different mutation's ΔΔG, so it was
> wrong in a way a coverage percentage cannot express. The corrected merger is single-key.
>
> Kept as written because it is the diagnosis that produced the fix, and because a document that
> predicted a result is worth more unedited than corrected.

---

## 1. Coverage as it stands today

Measured by re-running the production join logic (`merge_foldx_full_skempi.load_foldx`,
`merge_foldx_multipoint.foldx_for`) over the label tables and every built split.

| tier | table | rows covered | complexes (full / partial / zero) |
|---|---|---|---|
| **SP only** | `scratch/skempi_full/single_point.tsv` (4165) | 3655 — **87.8%** | 315: 250 / 40 / 25 |
| **MP only** | `scratch/skempi_full/multi_point.tsv` (1636) | 1615 — **98.7%** | 146: 144 / 0 / 2 |
| **ALL (SP+MP)** | `scratch/skempi_full/all_point.tsv` (5801) | 5270 — **90.8%** | 337: 270 / 49 / 18 |

ALL is exactly the union with no interaction (SP half 87.8% + MP half 98.7%). Every built split
reproduces its tier's figure, confirming the 87.8% / 90.8% numbers:

| split dir | folds | rows | coverage |
|---|---|---|---|
| `splits_skempi_full_clustered_id60_kfold` (SP) | 3 | 12495 | 87.8% |
| `splits_skempi_full_bycomplex_seed42` (SP) | 3 | 12495 | 87.8% |
| `splits_skempi_full_mp_clustered_id60_kfold` (MP) | 3 | 4908 | 98.7% |
| `splits_skempi_full_mp_bycomplex_seed42` (MP) | 3 | 4908 | 98.7% |
| `splits_skempi_full_cath_kfold` (ALL) | 1 | 5778 | 90.8% |
| `splits_skempi_full_bycomplex_all_seed42` (ALL) | 3 | 17403 | 90.8% |
| `splits_skempi_full_clustered_all_id60_kfold` (ALL) | 3 | 17403 | 90.8% |

Per-split, the aggregate hides real fold variance — clustered fold_1 is the weak one:

| tier | train | val | test | per-fold test |
|---|---|---|---|---|
| SP clustered | 86.0% | 97.0% | 87.8% | 88.2 / **78.2** / 96.8 |
| MP clustered | 99.9% | 92.7% | 98.7% | 99.6 / 96.5 / 100.0 |
| ALL bycomplex | 90.8% | 91.0% | 90.8% | 91.1 / 92.8 / 88.6 |
| ALL clustered | 90.1% | 94.5% | 90.8% | 91.4 / **83.7** / 97.3 |
| ALL cath | 88.3% | 95.7% | **99.9%** | 99.9 |

CATH's held-out test set is effectively fully covered; the clustered ladder is the one whose test
folds actually lose FoldX signal.

---

## 2. Root cause — residue numbers are never remapped

**The MP arm has no compute gap at all**: its 21 uncovered rows are exactly the two quarantined
wrong-interface groupings (`2C5D.AB.CD`, `3SE3.B.C`) from `multipoint_foldx_exclude.tsv`. MP joins
through the `multi_point_foldx_key.tsv` sidecar, which carries the raw SKEMPI variant string, so it
is immune to the bug below.

**The SP arm's 510 uncovered rows are a join-convention failure, not missing FoldX.** Every one of
the 315 SP complexes has a FoldX JSON; FoldX genuinely failed on only 16 mutations in total.

The two conventions:

* `build_skempi_full.py:152` numbers a mutation as `pos = off[mchain] + resseq2idx[resseq]` —
  position within the **concatenated interface group**, where `off` comes from `group_seq()` and
  `resseq2idx` from the `<CODE>.mapping` file. Chains of a group are concatenated in group-string
  order (`g1="HL"` → H then L), so a mutation in the 2nd chain is shifted by the 1st chain's length.
* The FoldX JSONs are keyed by **original PDB chain + per-chain sequence index**.
* `merge_foldx_full_skempi.py:71 remap_mut()` translates only the **chain letter** (`g1`→A,
  `g2`→B) and passes the residue number through untouched.

So the join succeeds only when the offset happens to be 0 *and* the author resseq happens to equal
the sequence index — i.e. single-chain interface groups with clean numbering. Every mutation in the
2nd-or-later chain of a multi-chain group silently misses and is written as standardized 0.

Inverting the builder's numbering exactly (reusing `build_skempi_full.load_mapping` /
`group_seq`) resolves the 510:

```
uncovered SP rows                          510
  recovered via (chain, per-chain seq-index) key   316
  key identical under both conventions             178   <- single-chain groups, offset 0
  neither -> genuine FoldX compute gap              16
```

Those 16 are exactly the mutations FoldX itself reports as `unresolved` across all 323 SP JSONs —
an independent confirmation that nothing else is actually absent:

| complex | unresolved mutations |
|---|---|
| 1DAN | DT39A, DU39A, TT16A, TU16A |
| 1DQJ | YA50A, YB50A |
| 1DVF | YA32A, YB32A |
| 1VFB | YA32A, YB32A |
| 3HFM | YH50A, YH50F, YH50L, YL50A, YL50F, YL50L |

### Why this matters beyond the headline number

The loss is **not random** — it lands on multi-chain interface groups, i.e. the antibody/TCR
complexes. The FoldX arm is currently zeroed precisely on the hardest and most interesting part of
the benchmark, which biases the measured FoldX lift **downward** on the SP and ALL tiers.

**Projected after the fix, with zero new FoldX compute:**

| tier | now | after fix |
|---|---|---|
| SP only | 3655/4165 (87.8%) | **4149/4165 (99.6%)** |
| MP only | 1615/1636 (98.7%) | unchanged (already clean) |
| ALL (SP+MP) | 5270/5801 (90.8%) | **5764/5801 (99.4%)** |

---

## 3. Affected structures (65 complex-groupings, 494 recoverable rows)

<details><summary>full table — grouping / rows / covered now / recoverable / genuine gap</summary>

| grouping | rows | covered now | recoverable | genuine gap |
|---|---|---|---|---|
| `1DAN.HL.UT` | 89 | 45 | 40 | 4 |
| `4NKQ.C.AB` | 34 | 8 | 26 | 0 |
| `4P5T.CD.AB` | 35 | 11 | 24 | 0 |
| `4P23.CD.AB` | 35 | 11 | 24 | 0 |
| `3C60.CD.AB` | 35 | 11 | 24 | 0 |
| `1MHP.HL.A` | 50 | 28 | 22 | 0 |
| `1VFB.AB.C` | 48 | 25 | 21 | 2 |
| `1OGA.ABC.DE` | 48 | 29 | 19 | 0 |
| `1AO7.ABC.DE` | 86 | 67 | 19 | 0 |
| `1DVF.AB.CD` | 25 | 6 | 17 | 2 |
| `3QIB.ABP.CD` | 25 | 8 | 17 | 0 |
| `1MI5.ABC.DE` | 39 | 22 | 17 | 0 |
| `2AK4.ABC.DE` | 29 | 16 | 13 | 0 |
| `1JRH.LH.I` | 43 | 32 | 11 | 0 |
| `4NM8.ABCDEF.HL` | 9 | 0 | 9 | 0 |
| `5E9D.AB.CDE` | 8 | 0 | 8 | 0 |
| `4OZG.ABJ.GH` | 12 | 4 | 8 | 0 |
| `3NGB.HL.G` | 41 | 33 | 8 | 0 |
| `3BN9.B.CD` | 35 | 27 | 8 | 0 |
| `1YY9.CD.A` | 16 | 8 | 8 | 0 |
| `1MLC.AB.E` | 11 | 3 | 8 | 0 |
| `3HFM.HL.Y` | 71 | 58 | 7 | 6 |
| `4B0M.A.BM` | 9 | 2 | 7 | 0 |
| `3SE8.HL.G` | 28 | 21 | 7 | 0 |
| `3N85.A.LH` | 9 | 2 | 7 | 0 |
| `2BDN.HL.A` | 12 | 5 | 7 | 0 |
| `4PWX.AB.CD` | 25 | 19 | 6 | 0 |
| `4JFF.ABC.DE` | 6 | 0 | 6 | 0 |
| `3SE9.HL.G` | 25 | 19 | 6 | 0 |
| `3QDJ.ABC.DE` | 15 | 9 | 6 | 0 |
| `1BD2.ABC.DE` | 9 | 3 | 6 | 0 |
| `3QDG.ABC.DE` | 14 | 9 | 5 | 0 |
| `1N8Z.AB.C` | 12 | 7 | 5 | 0 |
| `1DQJ.AB.C` | 21 | 15 | 4 | 2 |
| `3D3V.ABC.DE` | 4 | 0 | 4 | 0 |
| `2NYY.DC.A` | 23 | 19 | 4 | 0 |
| `1XGU.AB.C` | 4 | 0 | 4 | 0 |
| `1XGT.AB.C` | 4 | 0 | 4 | 0 |
| `1XGR.AB.C` | 4 | 0 | 4 | 0 |
| `1XGQ.AB.C` | 4 | 0 | 4 | 0 |
| `1XGP.AB.C` | 4 | 0 | 4 | 0 |
| `1NMB.N.LH` | 8 | 4 | 4 | 0 |
| `1NCA.N.LH` | 4 | 0 | 4 | 0 |
| `4I77.HL.Z` | 12 | 9 | 3 | 0 |
| `5C6T.HL.A` | 18 | 16 | 2 | 0 |
| `4MNQ.ABC.DE` | 2 | 0 | 2 | 0 |
| `4FTV.ABC.DE` | 6 | 4 | 2 | 0 |
| `3G6D.LH.A` | 2 | 0 | 2 | 0 |
| `4L3E.ABC.DE` | 7 | 6 | 1 | 0 |
| `4JFE.ABC.DE` | 1 | 0 | 1 | 0 |
| `4JFD.ABC.DE` | 1 | 0 | 1 | 0 |
| `4HFK.A.BD` | 8 | 7 | 1 | 0 |
| `3QFJ.ABC.DE` | 1 | 0 | 1 | 0 |
| `3PWP.ABC.DE` | 1 | 0 | 1 | 0 |
| `3L5X.A.HL` | 1 | 0 | 1 | 0 |
| `2VLR.ABC.DE` | 1 | 0 | 1 | 0 |
| `2OI9.AQ.BC` | 1 | 0 | 1 | 0 |
| `2E7L.EQ.AD` | 1 | 0 | 1 | 0 |
| `2BNR.ABC.DE` | 26 | 25 | 1 | 0 |
| `2BNQ.ABC.DE` | 1 | 0 | 1 | 0 |
| `2B2X.HL.A` | 3 | 2 | 1 | 0 |
| `1QSE.ABC.DE` | 1 | 0 | 1 | 0 |
| `1KIQ.AB.C` | 1 | 0 | 1 | 0 |
| `1KIP.AB.C` | 1 | 0 | 1 | 0 |
| `1CZ8.HL.VW` | 1 | 0 | 1 | 0 |

**TOTAL: 494 recoverable, 16 genuine gap.**

</details>

Note the 21 groupings currently at **0/N coverage** (`4NM8`, `5E9D`, `4JFF`, `3D3V`, the five
`1XG*` lysozyme variants, `1NCA`, `4MNQ`, `3G6D`, `4JFE`, `4JFD`, `3QFJ`, `3PWP`, `3L5X`, `2VLR`,
`2OI9`, `2E7L`, `2BNQ`, `1QSE`, `1KIQ`, `1KIP`, `1CZ8`) — every one of them is fully recoverable.
They are not missing FoldX data, they simply never joined.

---

## 4. Second, independent bug — 3SE4 gets the wrong interface

`merge_foldx_full_skempi.foldx_for()` keys the SP lookup on the **bare PDB code**
(`row[0].split(".")[0]`), while splits are per `code.g1.g2` grouping. Applying the MP audit's §2
grouping-trust criterion to the SP set:

* SKEMPI lists **3 codes** under more than one `#Pdb` grouping; only **3SE4** has more than one
  grouping present in `single_point.tsv` (`3SE4.B.A` and `3SE4.B.C`).
* `build_ddg` collapses each pdb to its **first-seen** grouping, which for 3SE4 is `B_C`.
* Result: **18 rows of `3SE4.B.A` carry `3SE4.B.C`'s interface ΔΔG** — all 18 currently "covered".

This is the same defect quarantined for `2C5D`/`3SE3` on the MP side, but the exclude list is only
consulted on the MP path — `merge_foldx_cath.py:41` dispatches single-point rows straight to
`sp_fx.get()`, bypassing it. 0.43% of the SP set, so low impact, but it is a **wrong** value rather
than a missing one, which is worse per row.

---

## 5. Proposed fix (next round)

1. **Remap residue numbers, not just chain letters.** In `merge_foldx_full_skempi.load_foldx` /
   `remap_mut`, build the inverse of the builder's numbering — reuse
   `build_skempi_full.load_mapping()` + `group_seq()` to get
   `pos -> (orig_chain, per-chain seq-index)` per `(code, group)` and index FoldX values under the
   split's own `<wt><A|B><pos><mt>` key. Keep the existing raw/remapped dual-key entries so the
   benchmark-sourced complexes (3BT1/1PPF/1R0R/3SGB) still resolve. Expected: SP 87.8% → 99.6%.
2. **Extend the grouping-trust guard to single-point.** Add `3SE4.B.A` to a shared exclude list
   (or generalize `multipoint_foldx_exclude.tsv`) and consult it on the SP path in both
   `merge_foldx_full_skempi.py` and `merge_foldx_cath.py`, so those 18 rows go to standardized 0
   instead of a wrong-interface value. Better: promote the SP lookup key from bare PDB code to
   `code.g1.g2` so the collision cannot recur.
3. **Rebuild + re-run.** Regenerate `splits_skempi_full_foldx{,dec}`, `splits_cath_foldx{,dec}`,
   and the `bycomplex{,_all}` / `clustered_all` variants, then re-run the FoldX arms. Everything
   downstream of `merge_foldx_aug.py` inherits the fix for free (it reads the canonical
   standardized `foldxdec` split verbatim), but the `_aug` splits must be rebuilt from the new
   canonical ones.
4. **Re-checksum.** The "low-coverage FoldX result checksums" block (kept with the run notes, not shipped) fingerprints
   the pre-dual-key-fix results; this fix creates a **third** lineage. Add a new checksum block
   (or a coverage marker inside the split dir) so 28% / 87.8% / 99.6% results can be told apart —
   coverage is still a property of the split file, not of `all_results.json`.

### Scope of the re-run

Affects every **SP** and **ALL** FoldX arm: `full_skempi` (clustered-SP), `full_skempi_bycomplex`,
`full_skempi_cath`, `full_skempi_bycomplex_all`, `full_skempi_clustered_all` (+ their `_aug`
variants). **Does not affect**: the MP tiers (`full_skempi_mp_*`, already 98.7% and immune), and
the S1102 / benchmark FoldX lineage (`foldx_s1102/`, S1131/S2003/S4169), which uses `merge_foldx.py`
at 100% coverage — those numbers stand.

### Validity of results already in hand

All existing FoldX comparisons remain **internally valid** — every arm within a tier reads the same
split, so `base` vs `foldx` vs `foldx_scalar` and `_aug` vs reference are apples-to-apples. The
expectation is that the fix **increases** the measured FoldX lift on the SP and ALL tiers, since the
zeroed rows are concentrated in the antibody/TCR complexes. The clustered-all headline
(esmc6b 0.409→0.565, aido 0.247→0.561) is therefore a floor, not a ceiling.

---

## 6. Minor / non-bugs

* The CATH tier has **5778 rows, not 5801** — `1E50.A.B` (17 rows) and `1E96.A.B` (6 rows) are
  dropped upstream of FoldX (no CATH superfamily annotation). Not a coverage issue.
* `results_all/` holds 323 SP JSONs for 315 SP complexes; `results_multipoint/` 152 for 146 MP
  complexes. The surplus is unused benchmark complexes, harmless.
* `1KBH` remains intentionally excluded from curation (RDE BLOCK, IDP complex) — its absence from
  the full-SKEMPI FoldX set is by design, not a gap.

---

## 7. Reproducing this audit

The audit scripts were one-shot and are not committed. To
reproduce, the essential checks are:

* **coverage per tier** — import `merge_foldx_full_skempi.load_foldx` (SP, over
  `scratch/foldx_skempi_full/results_all`) and `merge_foldx_multipoint.foldx_for` (MP, over
  `results_multipoint` + `multi_point_foldx_key.tsv` + `multipoint_foldx_exclude.tsv`), dispatch
  per row on whether col2 contains a comma, and count against `single_point.tsv` /
  `multi_point.tsv` / `all_point.tsv`.
* **gap classification** — for each uncovered SP row, invert `pos = off[chain] + resseq2idx[resseq]`
  via `build_skempi_full.load_mapping()` / `group_seq()` in both the author-resseq and per-chain
  seq-index conventions, and check the resulting key against the complex's JSON `muts`. The
  residual should equal the JSONs' `meta.unresolved` count (16).
* **grouping trust** — parse `#Pdb` from `scratch/skempi_v2.csv`, take the first-seen grouping per
  code, and flag split rows whose own `code.g1.g2` differs.
