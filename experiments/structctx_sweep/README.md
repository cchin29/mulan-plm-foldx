# structctx P1+P2 sweep — contact-knob decisions

Self-contained analysis dir (parallel to `../embedding_sweep/`), kept out of the `mulan/`
package. Evaluates the two highest-value `decisions.yaml` contact knobs from **2026-07-09 plan §4**
against the functional-site set, to decide whether the locked defaults should change.

- **P1 — D2 pLDDT contact mask:** `{50 (default), 70, none}`
- **P2 — D1 contact primary:** `{Cα ≤ 8 Å (default), Cβ ≤ 5 Å}`
- **P4 — D3 RSA MaxASA table:** `{Tien-theoretical (default), Tien-empirical, Sander&Rost}`
  (a separate guardrail script, `run_p4_rsa_table.py`, piggybacked on the TEM-1↔1BTL crystal cross-check)

3 × 2 grid on **TEM-1 (P62593)** and **TP53 (P04637)**. RSA is DSSP-derived and depends only
on the (fixed) MaxASA table, so it is **identical across all six configs** — the sweep is
entirely about `contact_count`, the packing signal. Embedding disabled (`model: none`); the
new L2 DSSP disk cache means mkdssp runs once per protein and is reused across configs.

## Run

```bash
.venv-structctx/bin/python experiments/structctx_sweep/run_sweep.py   # needs network + mkdssp
```

Writes `results/SWEEP_RESULTS.md` (the tables), `results/sweep_raw.json`, and per-config JSON.

## Metrics (per 2026-07-09 plan §4 "Eval criteria")

1. **Descriptive primary** — each functional site's RSA/contact + percentile rank (the candid
   readout at n=3).
2. **Physical-sanity gate** — median `contact_count` of the buried core (RSA<0.20) must stay
   **~10–20**, surface (RSA>0.25) **~4–8**. A config that pushes the core below ~10 fails.
3. **Discrimination (directional, n=3)** — AUROC(contact→functional-vs-bulk) and Cliff's δ
   (functional vs buried-core).
4. **Rescale-vs-reorder** — Spearman ρ of per-residue `contact_count` vs baseline (Cα/50).

**Decision rule:** change a default only if an alternative improves separation *consistently on
both proteins*; a one-protein gain is noise.

## Verdict — keep both defaults (Cα-8 Å primary, pLDDT<50 mask)

**P1 (pLDDT mask): immaterial here → keep 50.** 50 vs 70 vs none barely move `contact_count`
(ρ ≈ 0.95–1.0 vs baseline; core/surface medians essentially unchanged). Both proteins are
well-folded with high pLDDT almost everywhere, so there are few low-confidence partners to mask.
The stricter `70` cutoff would only bite on proteins with genuine disordered/low-pLDDT regions —
untested here, so no reason to move off the safer `50` default.

**P2 (contact primary): Cβ-5 Å is disqualified → keep Cα-8 Å.** Two independent reasons:
- **Fails the physical-sanity gate.** Cβ-5 Å collapses the buried-core median to **1–2** contacts
  (vs the required ~10–20); surface → 0. A 5 Å Cβ–Cβ shell captures only very-close packing, so
  it is not a usable burial/packing count. Cα-8 Å sits cleanly in-band (core median **12** on
  both proteins; surface 4–7).
- **No consistent discrimination gain.** Cβ-5 Å changes AUROC in *opposite* directions on the two
  proteins (TEM-1 0.61→0.68 but TP53 0.79→0.49, ≈ chance), and it's a genuine **reorder** not a
  rescale (ρ ≈ 0.36–0.60). By the decision rule, an inconsistent + sanity-failing change is not
  adopted.

The default `ca8/50` reproduces the functional-site report exactly (TEM-1 S68 = 11 contacts @ 0.67 pctile;
TP53 R175 = 14 @ 0.97), so the baseline column is a correctness check as well as the winner.

**P4 (RSA table): all three pass, keep Tien-theoretical.** On the TEM-1↔1BTL cross-check,
SS3 agreement is table-invariant (0.9962) and RSA Pearson r barely moves (0.9841 / 0.9842 /
0.9841 for theoretical / empirical / Sander&Rost) — a rescale, not a reorder — with only MAE
shifting (0.027 / 0.028 / 0.032). All clear the >0.85 gate; the default has the lowest MAE.
Confirms D3 is a reporting recalibration; no re-ranking, keep the default. (`results/P4_RSA_TABLE.md`.)

## P3 — embedding default (document-only; decided 2026-07-10)

P3 asked whether the D5 embedding default should change. **Decision: keep Ankh-large, document the
deviation.** The row-alignment / mutation-sensitivity gate was **executed**
(`run_p3_mutation_sensitivity.py`, TEM-1 L150A, transformers PLMs on CPU): **9/9 locally-runnable
models pass** — argmax at the mutated row and ratio ≫ 5 (ankh 70, prostt5 63, ankh3_large 63,
ankh3_xl 80, saprot 66, saprot_1.3b 87, esm2_3b 14, esm2_650m 31, **esmc_600m 27**;
`results/P3_MUTATION_SENSITIVITY.md`). This reproduces PHASE2_REPORT §4 and adds esm2_650m
(previously "wired, not run"). No ΔΔG re-benchmark (that is the S1102 benchmark, `RESULTS.md` §10–14). **ESM C is now
runnable locally via the `esm` SDK:** `esmc_600m` loads on **MPS in bf16 (~1.4 s/protein)** through
`ESMC.from_pretrained` in `.venv-esmc` — a transformers-independent path (`embedding.py` branches on
the new `sdk` spec flag; forward is `encode` → `logits(return_embeddings=True)`, strip BOS/EOS, cast
bf16→f32). **`esmc_6b` remains deferred (env):** it needs the transformers `esmc` model_type, which
5.12.1 dropped, and the SDK's local loader only exposes 300M/600M (6B is Forge/cluster) — so 6B is
cluster-only, its row-alignment standing on the cluster benchmark runs + the length guard (see
`SETUP_NOTES.md` §5). Rationale for keeping Ankh:

- **ProstT5** was the original candidate; the package defaults to **Ankh-large** because Ankh is the
  stronger ΔΔG representation on S1102 (**0.832 vs 0.805**; RESULTS §10–14) at a similar size.
- The strongest PLM is **ESM C 6B** (0.859), but it's heavy and `mps_ok=False` (needs the separate
  `.venv-esmc` on the Linux box / cluster), so it's a documented *swap*, not the default.
- The embedding is **auxiliary** — RSA + contacts carry the structural signal, and the agent tool
  omits the raw vector by default. So the default is chosen on **fit for the tool and cost**, not ΔΔG
  accuracy (the S1102 benchmark, not re-run here).
- All candidates pass the local **mutation-sensitivity** check (mutated-row change ÷ median ≫ 1,
  argmax at the mutated residue): PHASE2_REPORT §4 records Ankh ~70×, ProstT5 ~63×, ESM C-family
  and others verified end-to-end. SaProt is **redundant** (the tool already returns explicit
  structure) → deprioritized.

So: **default Ankh; ProstT5 = spec-faithful/lightest swap; ESM C 6B = strongest/heaviest swap** —
all row-aligned and interchangeable via `embedding.model` (the `LightAttModel` takes the PLM width
as config, no head surgery). Nothing to change in `decisions.yaml`.

## Caveat

AUROC/Cliff's δ here are **directional only** (3 sites/protein); the descriptive percentile table
is the candid primary. n reaches meaningful levels only with a larger labeled set (SKEMPI, below).

## SKEMPI fuller sweep — **done** (see `../skempi_interface/`)

> **Update 2026-07-10:** built as framing A (complex) + B (monomer contrast) in
> `../skempi_interface/`. Result: cross-chain **Cα-8 Å AUROC 0.690** > **Cβ-5 Å 0.600**
> (n=4510, non-overlapping CIs) — Cα-8 Å is the better interface cutoff too, agreeing with the
> monomer P2 verdict. Monomer intra-chain contacts predict interface at AUROC ≈ 0.49 (chance),
> quantifying the monomer-only blindness to interfaces. The framing analysis that led there is
> kept below for the record.

### Original framing analysis (why it was scoped this way)

2026-07-09 plan §4 proposed a SKEMPI-scale expansion to make P1/P2 *quantitative*. On inspection the
data is ready — `scratch/skempi_v2.csv` carries per-residue interface labels in
`iMutation_Location(s)` (Levy classes: `COR`/`SUP`/`RIM`/`INT` = interface, `SUR` = surface;
~5,000 single-mutation rows) and `scratch/skempi2/PDBs/` has 690 local crystal complexes — but
building it now was **deliberately deferred**. Recording why so it can be picked up cleanly:

**It reduces to a P2-interface study, and P1 can't ride along.** The pLDDT mask (P1) is an
AlphaFold concept; SKEMPI complexes are *crystals* with no pLDDT, so SKEMPI cannot test P1. P1 is
already settled on the AF set above (immaterial for well-folded proteins). Only the P2 cutoff
question — "Cα-8 Å or Cβ-5 Å *for interface residues*?" — benefits from SKEMPI.

**The monomer/complex framing fork (the crux).** An interface residue only exists in the
*complex*, but the shipped tool is *monomer-only* (AF single-chain). Two framings, different
meanings:
- **Complex framing** (the only one that literally answers P2-interface): compute *cross-chain*
  contacts on the SKEMPI crystal complexes at both cutoffs, use the `COR/SUP/RIM/INT` vs `SUR`
  labels as ground truth, report AUROC + bootstrap CIs, interface-class-resolved. But this tests
  a complex-contact computation the **v1 tool does not ship** — it informs the *future
  complex-context extension* (a documented out-of-scope item), not the v1 monomer default.
- **Monomer framing**: run the shipped tool on AF monomers for SKEMPI residues. Expected weak and
  potentially misleading — an interface residue's binding face is *surface-exposed* in the
  monomer, so monomer contact_count barely reflects interface burial (this IS the monomer-only
  limitation, restated).

**Why defer.** (1) The v1 monomer P1/P2 defaults are already settled by the committed small-set
sweep (keep Cα-8 Å / pLDDT<50 / Tien-theoretical). (2) The complex framing needs new cross-chain
contact code (`contacts.py` is single-chain-scoped) and answers a *roadmap* question, not a v1
default. (3) It is off the critical path for the work this repository reports.

**To resume (complex framing):** parse each `scratch/skempi2/PDBs/<PDB>.pdb`, build a KD-tree over
*all* chains, for each mutated residue count cross-chain neighbors at Cα≤8 Å and Cβ≤5 Å, join to
the SKEMPI location label, and score AUROC(cross-chain-contacts → interface) per cutoff with
bootstrap CIs — **reporting interface vs surface separately** (burial runs opposite for
catalytic-buried vs interface-exposed, so pooling cancels the signal). The L2 persistence layer
already removes the repeat-DSSP/embedding cost if AF structures are brought in.
