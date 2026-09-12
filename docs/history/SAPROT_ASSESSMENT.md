# SaProt — assessment for MuLAN ΔΔG

> **Archived — the verdict below was overturned by measurement.** SaProt was subsequently run
> across the sweep and leads several benchmarks; see [`../RESULTS.md`](../RESULTS.md) §15 and §17,
> and `experiments/RESULTS_BENCHMARKS.md`. Retained because the reasoning that produced the wrong
> call — WT-only structure input looked like ProstT5's negative 3Di channel — is the useful part.

_2026-07-02._ Evaluated SaProt as a candidate PLM for the S1102 ΔΔG benchmark.
**Verdict: deprioritized — not added to the run plan (`TODO.md`).** It requires 3D
structure input, and the one structural signal we can supply (WT backbone) is exactly
the channel that was **empirically negative** for ProstT5's 3Di (RESULTS.md §9). Notes
below so the decision is recoverable.

## What SaProt is

SaProt (Su et al., *"Protein Language Modeling with Structure-aware Vocabulary"*, ICLR
2024; `westlake-repl`) is an ESM2-architecture masked-LM trained over a **structure-aware
vocabulary**: each token is an **amino acid fused with a Foldseek 3Di structural token**
(the "SA" alphabet ≈ 20 AA × 20 3Di → 400 combined tokens + specials, **vocab 446**). So
the model input is one AA+3Di token *per residue* — structure is not an optional side
channel (as in ProstT5), it is baked into every input token.

## HF availability — yes, ungated

| Model | dim (hidden) | layers | weights | note |
|---|---|---|---|---|
| `westlake-repl/SaProt_650M_AF2` | **1280** | 33 | 2.6 GB (5.2 GB repo) | flagship, ~117k dl, AF2-structure pretraining |
| `westlake-repl/SaProt_650M_PDB` | 1280 | 33 | ~2.6 GB | PDB-structure variant |
| `westlake-repl/SaProt_35M_AF2` | 480 | 12 | 0.14 GB | small |
| `westlake-repl/SaProt_35M_AF2_seqOnly` | 480 | 12 | small | sequence-only variant |

All `model_type: esm`, `vocab_size: 446`, `max_position_embeddings: 1026`. Not gated —
downloadable like ESM2. (So availability is **not** the blocker.)

## Why it's a large integration lift here (the blocker)

1. **Structure tokens are mandatory.** Every residue must be presented as an AA+3Di
   "SA" token. We *can* produce 3Di (the ProstT5 structure thread already built this:
   `mini3di` over the SKEMPI 2.0 PDBs, `gen_3di.py`, `scratch/3di`), so WT chains are
   coverable — but this is a different tokenization/embedding path than the plain-sequence
   PLMs (Ankh / ProstT5-AA / ESM2 / ESM C), not a drop-in `--plm_model_name`.
2. **The mutant-structure problem — the real killer.** We have no *mutant* structures, so
   every mutant would reuse its **WT** 3Di (the AA half of the SA token carries the point
   mutation, the 3Di half stays fixed). In MuLAN's `mut − wt` difference the structure
   half is then **constant across the mutation** → contributes only a complex-level offset,
   no mutation-specific signal — and SaProt's representation is *structure-entangled at the
   token level*, so unlike ProstT5 you can't even cleanly isolate the AA-varying part.
   This is precisely the failure mode that made **ProstT5's 3Di negative** (run6c 0.663,
   E1/B1/C3 all ≤0.705, all below AA-only run2 0.740 — RESULTS.md §9): single-point
   mutations barely move the backbone, so a fixed-WT structure channel dilutes more than
   it helps. SaProt would very likely inherit this, with less room to down-weight it.
3. **Seq-only mode defeats the purpose.** SaProt can run "sequence-only" by setting every
   structure token to `#` (unknown) — there's even a `SaProt_35M_AF2_seqOnly` — but then
   it's just a 650M ESM2-class sequence model and would be expected to land near ESM2-650M,
   well below the ESM C 6B / Ankh tier. Not worth a bespoke tokenization path for a point
   that adds little to the scale curve.

## Cost (if ever revisited)

Compute is cheap — 650M / 1280-d (ESM2-650M class): embedding gen would be fast (~ESM2
speed), cv10 ~5–6 h (narrow 1280-d), ~5 GB peak RAM. The cost is **integration + the
structure-signal ceiling**, not FLOPs.

## Recommendation

**Skip for now.** Revisit only if **modeled mutant structures** (FoldX / AlphaFold per
mutant) become available — the same precondition noted in RESULTS.md §9 for testing
genuine structural value. At that point SaProt (structure fused end-to-end) would be a
more natural vehicle than bolting 3Di onto ProstT5, and worth a real run. Until then it's
a large lift chasing a signal we've already shown is near-zero under the WT-structure
approximation.

→ related: RESULTS.md §9 (ProstT5 3Di negative results), `PLAN_PROSTT5_STRUCTURE_v2.md`,
`PROSTT5_STRUCTURE_OPTIONS.md`.
