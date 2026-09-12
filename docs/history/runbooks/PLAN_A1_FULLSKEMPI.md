# A1 interface cross-attention under the leakage-controlled protocols

Queue item 3 of `README.md`. `docs/history/PLAN_A1_INTERFACE_XATTN.md` remains the plan of record
for the module, the mask builder and the controls; it is written for the S1102 balanced folds and
judged on worst-40 tail MAE. This document covers what changes when the same module is aimed at
the full-SKEMPI clustered tier and judged on the beyond-FoldX diagnosis instead.

## Why A1 is the queued item

`REDUNDANCY_RESULT.md` locates the failure in **extraction, not combination**. The base pathway
and the physics channel are not redundant (within-complex Spearman 0.15–0.27), so there is room in
principle; but an optimally weighted blend of the base output with the full 12-term decomposition
gains +0.021 [−0.001, +0.043] on clustered-SP, and nothing for the other eight backbones. No
combination rule recovers anything, because the pathway's output has nothing to recover.

A1 is the only queued change that alters what the pathway extracts. `modules.py:233-246` pools
each chain independently and combines them *after* pooling; no residue is ever paired with its
cross-chain partner. Ranking mutations within one complex — what ppS measures — is the regime
where that design carries least signal.

## Status inventory

| asset | state |
|---|---|
| `mulan/interface_xattn.py` + wiring (`config.py`, `modules.py`, `data.py`, `train_utils.py`, `scripts/train.py`) | **done**, branch `a1-interface-xattn`, gated init-0, exact no-op verified |
| S1102 interface masks, heavy-8 Å | **done**, 110/110 in `scratch/interface_masks/` |
| A1 config JSON + driver shell | **not built** (PLAN_A1 §8 step 4) |
| dense / shuffled mask variants for the controls | **not built** |
| full-SKEMPI interface masks | **not built** — 0 of 315, and blocked, see §2 |
| Ankh-large embeddings, full SKEMPI | cached, 11563 files, **D = 1536** |
| ESM-C 6B embeddings, full SKEMPI | **0 files on the Mac** — GPU box only |

## 1. The success metric changes, and it gets more sensitive

PLAN_A1 judges on worst-40 tail MAE and named hotspots. Under the leakage-controlled protocols the
right readout is **`gain_plm` from `redundancy.py` / `blend12.py`**, not ppS against the physics
baseline.

The reason is sensitivity. Asking "does the A1 arm beat FoldX-alone" folds the question through a
0.418 baseline the pathway is nowhere near. Asking "does the A1 base arm carry more non-redundant
information than the plain base arm" is measured directly off base-arm predictions, needs no FoldX
arm trained at all, and is the exact quantity the diagnosis says is broken:

| quantity | plain base, ESM-C 6B, clustered-SP | what A1 must do |
|---|---|---|
| `part_base` — Spearman(base, truth \| FoldX) | 0.130 | rise |
| `gain_plm` — base added on 12 terms | +0.021 [−0.001, +0.043] | rise, and clear zero |
| ppS of the base arm itself | 0.193 | may or may not move |

`part_base` can rise while base ppS is flat — the pathway can trade redundant signal for
complementary signal at constant total. That is a success and the base-ppS reading would miss it.

**Mechanically this is free.** `redundancy.py` and `blend12.py` take a results directory; pointing
them at the A1 results dir reproduces every column. No new analysis code.

## 2. Two blockers on the full-SKEMPI port, both quantified

### 2a. The alignment proof does not carry over

PLAN_A1 §2 closes the mask↔sequence alignment risk empirically, and the proof rests on one fact:
*"there are **0 multi-chain role groups** in the S1102 set … so a role is always one real chain and
resnums never collide."*

That does not hold here. Of the 315 complex-ids in the clustered-SP tier, **108 carry a
multi-chain group** (`1AHW.AB.C_AB`, `2C5D.AB.CD`, …). `build_skempi_full.py` concatenates each
interface group into one pseudo-chain and renumbers residues contiguously from 1, so PDB ATOM
resnum is **not** FASTA index + 1 for any row in those 108.

This is the same coordinate-system defect that held the FoldX single-point join at 87.8% until
2026-07-29. **Do not re-derive the mapping.** Reuse the authoritative helpers:

- `build_skempi_full.py:load_mapping(code)` and `group_seq(chains, group)` — the numbering source.
- `merge_foldx_full_skempi.py:remap_mut_resnum(mut, groups_for_pdb, code)` — inverts it, and its
  docstring records the measured finding that the FoldX residue number is a per-chain 1-based
  *sequence index*, not the author resseq, plus what happens when a second convention is emitted
  as an extra candidate key (15 wrong joins in SP fold_0 alone).

The mask builder needs the forward direction — `(pdb chain, author resseq) → pseudo-chain index` —
which is `off[chain] + per-chain seq index`. Port it from those functions rather than alongside
them.

Mask keys also change convention: S1102 masks are `1A22_A__1A22_B.pt`; full-SKEMPI chain ids are
dotted, so the loader will look for `1A22.A.B_A__1A22.A.B_B.pt`.

### 2b. About a third of the structures are not on disk

314 distinct PDB codes are needed for clustered-SP. Across all four FoldX work directories:

| store | PDBs | covers |
|---|---|---|
| `scratch/foldx_s1102/work` | 111 | 109 / 314 |
| `scratch/foldx_skempi_full/work` | 113 | 108 / 314 |
| **union** | 223 | **217 / 314** |

**97 missing.** They are ordinary RCSB downloads, and — this is the part that makes it cheap —
**masks need geometry only, not energies.** No FoldX pass, no RepairPDB, so none of the
FoldX-intractable structures matter here and none of the repair runtime is incurred. Budget the
download plus a heavy-atom `NeighborSearch` pass, not a FoldX campaign.

`build_masks.py` aborts a complex at <90% ATOM-vs-FASTA agreement; with 108 multi-chain groups
that check is now doing real work rather than confirming a known-good mapping, and its abort list
is the acceptance gate for §2a.

## 3. Staging — screen on S1102 before paying for the port

§2 is a few days of careful work whose value is zero if the module does not engage at all. S1102
has everything already: masks 110/110, alignment closed, embeddings cached, controls specified,
folds shared with base/aug/FoldX/MINT.

**Stage 1 — mechanism screen, S1102 balanced folds, Ankh-large.** Build the config JSON and driver
(PLAN_A1 §8 step 4), run A1a plus the two capacity controls. Three outcomes, and only one of them
justifies §2:

| outcome | reading |
|---|---|
| gate stays ~0 | the head cannot use interface pairing under MSE. A1 is not the lever; record it and stop. |
| shuffled ≈ real | A1 is added capacity, not interface structure. Stop. |
| gate opens **and** real > shuffled | the mechanism engages. Proceed to §2. |

This is a **necessary-condition screen, not a test of the hypothesis.** S1102 balanced folds are
not homology-controlled, so a positive result there says the module works, not that it fixes what
beyond_foldx measured. Passing stage 1 and then doing nothing on clustered-SP is a live and
unremarkable outcome — the whole point of the diagnosis is that in-distribution performance and
out-of-cluster performance come apart.

**Stage 2 — clustered-SP and by-complex-SP, 3 folds each.** Base arm only; the FoldX arms are not
needed because the metric in §1 is computed from base predictions. Report `part_base` / `gain_plm`
against the plain base arm as a paired per-complex contrast.

## 4. Capacity is the main hazard, and the default should be the bottleneck

PLAN_A1 §3c prices A1 at `xattn_dim=1536` as **~9.4 M new parameters on a ~3.0 M head** — roughly
4× the trainable model. Train rows per fold on clustered-SP are 2114 / 2463 / 2447 (measured while
fitting the residual reference). That is more data than S1102's ~990 but nowhere near enough to
justify 9.4 M parameters.

Take PLAN_A1's own recommendation as the **default rather than the cheap alternative**: bottleneck
D→d→D at d = 64, ≈0.2 M params and ≈0.12 GMAC, about 5–7% on top of the existing head instead of
~340%. Treat `xattn_dim = 1536` as the high-capacity end of a sweep, to be run only if the
bottleneck shows life.

The **shuffled-mask control is mandatory, not optional**, at either width. A gain that survives
shuffling is capacity, and at this parameter-to-row ratio that is the expected failure mode.

## 5. Backbone and scheduling

ESM-C 6B is the backbone the result would matter for — it is the only one with non-redundant
signal that survives homology control (`part_base` 0.130 vs 0.026–0.099 for the rest). Its
embeddings are **not on the Mac**, and the GPU box is committed to the residual runs.

So stage 1 runs on the **Mac, Ankh-large, D = 1536**, which is fully cached. Note what that costs
in interpretation: Ankh-large has **zero** wins over FoldX-alone across 27 contrasts, so it is a
good subject for "does the mechanism engage" and a poor one for "does this clear the baseline."
PLAN_A1's errata already records that no executed A1 result carries the v1 reference model — stage
1 would close that too.

Stage 2 on ESM-C 6B queues behind the residual experiment on the GPU box.

## 6. Gates

1. ⬜ Config JSON + driver, writing to `scratch/results/a1_xattn/` (PLAN_A1 §8 step 4).
2. ⬜ `--dense` / `--shuffle` mask variants from `build_masks.py`.
3. ⬜ Stage 1 on Ankh-large, bottlenecked d = 64: A1a, dense, shuffled. **Gate: gate opens and
   real > shuffled**, else stop and record.
4. ⬜ Port `build_masks.py` to full SKEMPI via `load_mapping` / `group_seq` / `remap_mut_resnum`.
   **Gate: `build_masks.py`'s <90% abort list is empty across all 315 pairs**, which is the
   acceptance test for the 108 multi-chain groups.
5. ⬜ Fetch the 97 missing PDBs (geometry only, no FoldX).
6. ⬜ Stage 2: clustered-SP + by-complex-SP base arms. **Gate: `gain_plm` rises against the plain
   base arm, paired per complex.**
7. ⬜ If positive: ESM-C 6B on the GPU box, then reconsider A1b.
