# Options for using ProstT5 structure in MuLAN ΔΔG — after run6c

Companion to `PLAN_PROSTT5_STRUCTURE_v2.md` (the run6 series) and `RESULTS.md` (§3/§9).
Written after **run6c (AA⊕3Di concat) failed (PCC 0.663 < run2 0.740)** — a catalogue of
the remaining ways to feed ProstT5 structure into MuLAN, why most inherit run6c's flaw,
and the two we are running next (E1, B1).

---

## The reframing insight (why run6c really failed)

It is **not** simply "we reused the WT structure for the mutant." The deeper reason:

> **3Di tokens describe *backbone* geometry, and single-point mutations are almost always
> backbone-preserving.** So the 3Di string near a mutated residue barely changes whether
> you reuse the WT backbone, FoldX-repack it, or even re-fold the mutant.

MuLAN predicts from the **siamese difference** `mut_emb − wt_emb`. If the 3Di channel is
~constant under the mutation, it contributes ≈0 to that difference and only **dilutes**
the AA signal that does work (run2). This is structural, not a tuning issue — and it means
any approach that pushes 3Di *through the `mut − wt` difference* is fighting physics.

Two consequences:
1. Making the structure channel "mutation-aware" (predicted/modeled mutant 3Di) has a
   low ceiling for single-point mutations — the backbone barely moves.
2. The promising direction is to use structure as **complex-level context that bypasses
   the difference**, or to let the model **down-weight** it so it cannot dilute.

---

## Option catalogue

| # | Idea | Mutation-aware? | Effort | Expected payoff |
|---|---|---|---|---|
| **A1** | **Predicted 3Di** — ProstT5 AA→3Di *generation* for wt and mut seqs separately, embed each in fold mode | partial (local) | med (needs the seq2seq model, not just the encoder) | low–med |
| **A2** | **Modeled mutant structure** — FoldX repair / ESMFold the mutant, extract real mut 3Di | marginal (backbone barely moves) | high (FoldX license / GPU) | low — same backbone-invariance trap |
| **B1** | **Structure as complex-level context** to the regression head, *outside* the siamese difference (extends MuLAN's `add_scores` side channel) | n/a — context, not a delta | **low** (plumbing exists) | **med — most principled** |
| **B2/B3** | **Conditioning fusion** — AA stream cross-attends to / is FiLM-modulated by the WT structure embedding | indirect | med–high | med, overfit risk on 769 |
| **E1** | **Gated concat** — a learned scalar gate on the 3Di block so the model can down-weight it; diagnostic for *dilution vs no-signal* | no | **low** (reuses run6c cache) | diagnostic |
| **D1** | **Auxiliary structure objective** — aux head predicts 3Di from the AA embedding during training; inference stays 1024-d | training-time only | med | med — no dilution |
| **C3** | **Interface-aware features** — use structure to mark/weight interface residues (binding ΔΔG lives at the interface), not the 3Di token | yes (which residues matter) | high | **highest ceiling**, least "ProstT5-native" |

### Sequencing rationale
1. **E1 first (cheap, diagnostic).** A learned gate on the existing 2048-d cache tells us
   whether run6c lost to **dilution** (gate suppresses 3Di → recovers ~0.740) or to
   **genuinely no signal** (gate stays moderate but PCC ≤ 0.740). One training run, no new
   embeddings.
2. **B1 (most principled).** Feed the pooled WT structure as **head-level context**,
   bypassing the `mut − wt` cancellation entirely — the one design that respects the
   backbone-invariance insight (structure as a prior on interface rigidity/environment,
   not a per-mutation delta). Reuses MuLAN's `add_scores`-style head plumbing.
3. **A1** only to settle whether a mutation-sensitive structure view helps — expect a small
   effect (predicted 3Di changes little for a point mutant except dramatic substitutions).
4. **C3** is the highest-ceiling bet but drifts from "structure from ProstT5" toward "use
   the PDB to weight residues" — likely the most effective for *binding* ΔΔG.

Candid read: given backbone-invariance, **E1 and B1 are the high-information, low-cost next
steps**; A2/AF are expensive and likely hit the same wall. The cleanest eventual framing may
be that **3Di is the wrong structural primitive for single-point ΔΔG** — interface
contact/burial features (C3) are a better match.

---

## Results

### E1 — gated 3Di concat
_Diagnostic: does a learned scalar gate on the 3Di block recover run2 (→ dilution) or not
(→ no usable signal)?_

**Result: PCC 0.705 (50 ep); learned gate sigmoid 0.459 (raw −0.16).** Two-sided read:
- **Dilution was part of run6c's failure** — letting the model down-weight structure
  recovered about half the run6c→run2 gap (0.663 → 0.705).
- **But structure is genuinely net-negative** — the gate did *not* collapse to 0 (a single
  scalar is a weak, flat-gradient knob on 769 examples), and the residual ~0.46-weight
  structure still costs ~0.035 PCC vs pure AA (0.705 < run2 0.740). No path back to the
  baseline, let alone past it. Config `lightatt_structgate_config.json`, driver
  `scratch/run6_E1_driver.sh`, artifacts `scratch/results/run6_E1_structgate/`.

---

### C3 — interface-biased attention
_Use the WT structure to focus MuLAN's pooling on the binding interface, where ΔΔG lives._
Per-residue interface contacts (heavy-atom <5 Å with the partner chain, `gen_interface.py`,
mean 17% of residues) are appended as a trailing channel ([L,1025], `concat_aa_interface.py`);
MuLAN's Light-Attention (`config.interface_bias`) splits it off and adds it to the attention
logits with a learnable strength `α` (init 0 ⇒ starts as run2 AA-only). Unlike the other
fusions this does **not** get cancelled by `mut − wt` — it modulates *where* the model pools,
and the difference there is mutation-specific.

**Result: negative — PCC 0.662; learned α 0.017 (≈ off).** α started at 0 and barely moved,
so the model found **no benefit** in focusing attention on the interface beyond its default
Light-Attention — it left the bias effectively off. With the mechanism then ≈ run2, the
0.662 is **largely initialization variance** (the extra α parameter shifts the RNG-dependent
weight init; train_loss 3.72 vs run2 3.03, early-stop 35 → a worse local optimum), *not*
evidence that interface focus actively hurts. Candid claim: interface biasing gave no usable
signal — the model turned it off. Caveat: this is one formulation (5 Å contacts, additive
attention bias, α-init-0); interface-restricted cropping or distance-decay weights are
untested, but the consistent "knob trains to off" pattern makes them low-prior. Config
`lightatt_interface_config.json`, `gen_interface.py` / `concat_aa_interface.py`, driver
`scratch/run6_C3_driver.sh`, artifacts `scratch/results/run6_C3_interface/`.

## Verdict

| run6c (concat, 2048) | E1 (gated concat) | B1 (head context) | C3 (interface attn) | run2 (AA-only) |
|---|---|---|---|---|
| 0.663 | 0.705 | 0.673 | 0.662 | **0.740** |

**Every way of feeding ProstT5 structure underperformed the AA-only baseline**, and — the
stronger signal — **every learnable structure knob trained to neutral/off**: the layer-mix
weights went ≈uniform (run6a), the concat gate to 0.46 (E1), the interface-bias α to ≈0 (C3),
and the head-context projection overfit (B1). MuLAN's head **consistently declines ProstT5
structure** for single-point ΔΔG. This is coherent with the backbone-invariance insight: 3Di
is a backbone descriptor, single mutations are backbone-preserving, and even interface
*location* (C3) adds nothing the attention doesn't already capture.

**Conclusion: structure is not the lever for single-point binding ΔΔG in this head.** The
run6 gate is fully exhausted → redirect to **AIDO** (`docs/history/PLAN_AIDO.md`).

**If structure is revisited later** (lower prior): mutation-aware 3Di via *modeled mutant*
structures (A1/A2) for datasets with backbone-perturbing or multi-mutations; or interface
*cropping* rather than soft biasing (C3 variant). Neither is worth CPU time now.

### B1 — structure as head-level complex context
_Pooled WT 3Di embedding per complex (2048-d), projected to 64-d and concatenated to the
regression head, outside the siamese `mut − wt` difference; the four siamese inputs stay
AA-only (= run2)._

**Result: negative — PCC 0.673 < run2 0.740** (RMSE 1.707; early-stopped epoch 29).
Because the siamese inputs are AA-only, the *only* delta from the 0.740 baseline is the
added structure-context vector — and it **mildly hurt**. The 64-d structural features let
the small model (769 examples) overfit complex-level structure that doesn't transfer
(early-stops 21 epochs before run2). So structure-as-complex-context does not calibrate
ΔΔG here. Config `lightatt_structctx_config.json`, context `experiments/gen_struct_context.py`,
driver `scratch/run6_B1_driver.sh`, artifacts `scratch/results/run6_B1_structctx/`.
