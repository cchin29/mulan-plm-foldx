# Directions beyond the FoldX channel and augmentation

_Opened 2026-08-04. Scope: what to run after the FoldX-channel and tier-1 augmentation arms._

## The finding this responds to

`experiments/rescore_perstructure/FOLDX_ALONE_BASELINE_RESULT.md` establishes that **no `base`
arm beats FoldX-alone on any tier — 0 wins in 78 paired contrasts across ten backbones**, and
that all 19 wins over the physics baseline carry the FoldX channel, concentrated on the CATH and
by-complex tiers. The clustered tiers produce three wins in 76 contrasts.

(Figures on this page are the 99.x%-coverage lineage, matching that document's v3 section. The
0 in 79 / 18 / one in 56 triple stated here before 2026-08-07 was the pre-fix computation.)

Three mechanisms are consistent with that, and each is separately checkable.

**Homology lookup, not energetics.** The by-complex and clustered tiers are the same 4165 rows
partitioned differently. FoldX-alone is invariant to the partition (0.418 both ways) because it
is computed from the test structure. The base arms are not:

| backbone | base bycomplex-SP | base clustered-SP | drop |
|---|---|---|---|
| ESM-C 6B | 0.310 | 0.193 | −0.116 |
| Ankh-large | 0.284 | 0.142 | −0.143 |
| Ankh3-xl | 0.268 | 0.144 | −0.124 |
| ESM2-3B | 0.261 | 0.119 | −0.142 |
| ESM-C 600M | 0.227 | 0.152 | −0.075 |
| Ankh3-large | 0.203 | 0.092 | −0.111 |
| ProstT5 | 0.194 | 0.054 | −0.140 |
| AIDO-16B | 0.180 | 0.120 | −0.060 |
| SaProt | 0.120 | 0.082 | −0.038 |

Since ppS is a within-complex rank statistic, complex-level offsets cannot be the thing lost —
what is lost is position-specific mutational effect transferred from homologous interfaces.

**Architectural dilution.** `mulan/modules.py:229-246` pools each chain independently
(LightAttention), combines the two chains *after* pooling as `enc0 * enc1 ⊕ |enc0 − enc1|`, and
takes mut−wt as a difference of those pooled products. A one-residue substitution enters as a
1/L perturbation, and no residue is ever paired with its cross-chain partner. Ranking ten or
more mutations *within one complex* — the exact quantity ppS measures — is the regime where that
design carries least signal. A1 interface cross-attention (`mulan/interface_xattn.py`) was built
for this and defaults to `interface_xattn: bool = False`; no run in `results_matrix_ps.csv` used it.

**Scale is not the discriminating variable.** ESM-C 6B holds the highest out-of-cluster base
(0.193 SP / 0.206 ALL) and 5 significant wins over FoldX-alone, best CATH-single MLP
+0.127 [+0.048, +0.216]. AIDO-16B is larger and has 0 wins; ESM2-3B has 0. Whatever separates
ESM-C is in the training recipe, not the parameter count.

Tier-1 augmentation does not address any of the three: it is negative on all four clustered-ALL
arms (ESM-C 6B scalar −0.038, MLP −0.062; AIDO-16B −0.031, −0.027) and 1-of-4 positive on
CATH-all, where 13 complexes cannot resolve the effect.

## Queue

Ordered by information per GPU-hour.

### 2. Redundancy between the base pathway and the physics channel — **done 2026-08-04**

No training. Per-complex correlation between base predictions and the FoldX scalar, partial
correlation of each against truth given the other, and a global leave-one-complex-out linear
blend of the two. Script: `redundancy.py`. Result: **`REDUNDANCY_RESULT.md`**.

Outcome: the channels are **not** redundant (base~fx 0.15–0.27), non-redundant PLM signal exists
but halves under homology control (ESM-C 6B part_base 0.243 by-complex → 0.130 clustered), and an
optimally weighted blend of the two returns **+0.009 at best on clustered-SP**, zero for the other
nine backbones. On by-complex the trained scalar arm is already within 0.002 of the linear
optimum. The `fx_mlp` advantage on clustered-SP exceeds any base+scalar blend, so it comes from
the twelve decomposed terms, not the backbone.

This reorders the rest of the queue: the failure is in what the pathway extracts, not in how the
two channels are combined.

**Follow-up, done same day — `BLEND12_RESULT.md`.** Same machinery with the twelve decomposed
FoldX terms in place of the scalar. A linear blend of the terms alone reaches 0.434 on
clustered-SP with no backbone involved, and the nine observed `fx_mlp` arms span 0.422–0.446
around it. Every observed `fx_scalar` arm is at or below FoldX alone (0.355–0.417 against 0.418).
The decomposition's contribution is +0.016 [−0.007, +0.038] and disappears under rank encoding,
so it lives in term magnitudes. With the full decomposition in the model, the backbone still adds
nothing resolvable on either clustered tier.

### 0. Regenerate the FoldX-alone baseline — **done 2026-08-04**

`FOLDX_ALONE_BASELINE_RESULT.md` now carries a v3 section at 99.2% coverage; v1/v2 are retained
below it as the 87.8% lineage record. FoldX-alone on the SP tiers moved **0.363 → 0.418**; MP,
CATH and S1102 did not move. The win count went 18 → **19**, not down — the FoldX arms gained
about as much as the baseline did. Composition changed: the clustered tiers went from 1 win in 56
to **3 in 76, all `fx_mlp`**; ESM-C 6B's CATH-single flagship fell +0.127 → **+0.097**, still
significant; Ankh-large dropped to **zero** wins. No `base` arm beats FoldX alone, still 0, now
in 78 contrasts.

### 3. A1 interface cross-attention, enabled

Promoted by **2**. The only queued change that alters extraction rather than combination, already
written, gated at init-0 so it is an exact no-op until trained. Untested against the matrix.

**Planned: `docs/history/runbooks/PLAN_A1_FULLSKEMPI.md`**, which extends `docs/history/PLAN_A1_INTERFACE_XATTN.md` (still
the plan of record for the module, masks and controls) to the leakage-controlled regime. Three
things change or block.

The **metric** becomes `gain_plm` / `part_base` from `redundancy.py`, not tail MAE and not ppS
against physics. It is computed off base-arm predictions alone, needs no FoldX arm trained, and is
the quantity the diagnosis says is broken — `part_base` can rise while base ppS is flat if the
pathway trades redundant signal for complementary signal.

The **full-SKEMPI port is blocked twice.** PLAN_A1's alignment proof rests on there being zero
multi-chain role groups in S1102; the clustered-SP tier has **108 of 315**, so PDB resnum is not
FASTA index + 1 for any of them — the same coordinate-system defect that held the FoldX join at
87.8%. And only **217 of 314** required PDBs are on disk. The 97 downloads are cheap (masks need
geometry, not energies, so no FoldX pass), the renumbering is not, and it should be ported from
`load_mapping` / `group_seq` / `remap_mut_resnum` rather than re-derived.

So the plan **screens on S1102 first** — everything there exists — as a necessary-condition check
that the gate opens and real beats shuffled, before paying for the port. A pass is not evidence
the hypothesis holds: S1102 balanced folds are not homology-controlled.

### 1. Residual learning — targets of ΔΔG_exp − ΔΔG_FoldX

Retained with a narrower target. A relabel, no new embeddings. The bound in **2** covers rules
that combine the *existing base prediction* with FoldX; a residual-trained model sees embeddings
instead, so it can in principle surface signal the base head discards. But **2** removes the
by-complex motivation (the head is already at the linear optimum there) and shows the base
pathway's output is empty on clustered tiers. The experiment now tests whether the embeddings
hold clustered-tier signal the current head loses.

Target the **12-term** residual, not the scalar residual: the scalar arms are pinned at
FoldX-alone on clustered-SP, and `blend_12` sits +0.016 above it, so a scalar residual would
credit the model for a lift a linear reweighting already supplies.

**Specified for execution: `docs/history/runbooks/RUN_RESIDUAL_ESMC6B_GPU.md`.** ESM-C 6B, both single-point tiers,
6 runs, ~8–9 h at `MAXPAR=1`, no PLM inference and no changes under `mulan/`. Three things in it
are load-bearing. The residual cannot be formed from the split TSVs — column 5 is a clipped
per-fold z-score, and subtracting raw FoldX at slope 1 over-subtracts by ≈1.7× (the least-squares
slope is 0.874 kcal/SD against a raw SD of 1.488), which would return a confident wrong answer
rather than a failure. The scorer must add `ŷ` back before ppS or it measures a quantity
comparable to nothing in the matrix. And the comparator is the reference's own ppS, not
FoldX-alone and not the plain base arm, because a null `f` returns exactly `ppS(ŷ)`.

The by-complex tier is in scope as a positive control (`gain_plm` +0.041 [+0.015, +0.065] there),
so that a flat clustered result can be told apart from a broken pipeline.

### 4. Depth over breadth on ESM-C 6B

Ten backbones × seven tiers is well sampled and the answer is uniform outside ESM-C. The binding
constraint is power, not coverage: paired CI half-widths are ≈ ±0.04 on the 96-complex tiers and
≈ ±0.07 to ±0.16 elsewhere, so the CATH tier as folded cannot resolve its own target effect.

Unblocked by **0**. The figure this item exists to power up is now **+0.097 [+0.017, +0.194]**,
not +0.127. `BLEND12_RESULT.md` widens the case: `gain_12` +0.016 [−0.007, +0.038] and `gain_plm`
+0.021 [−0.001, +0.043] both straddle zero at 95 complexes, so power is the binding constraint on
the whole analysis line, not only on CATH.

## Standing caveat on framing

Every method in the frontier comparison — RDE-Network, DiffAffinity, CATH-ddG, USP-ddG — reads
the structure. MuLAN is sequence-only over a frozen encoder, and its physics channel is a scalar
or 12-term summary of the structure FoldX read. Items 1–3 test whether that is sufficient. If
all three return flat, the supported conclusion is that the frozen-sequence pathway is a
pass-through, and the next step is supplying the structure rather than a digest of it.
