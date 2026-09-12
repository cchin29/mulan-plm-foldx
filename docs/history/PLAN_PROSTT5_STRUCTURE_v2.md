# ProstT5 structure & multi-layer enhancements for MuLAN ΔΔG — plan

Forward-looking plan for getting **more** out of ProstT5 as a MuLAN embedder than
the plain amino-acid-mode last-layer baseline, by exploiting the two things we have
*not* yet used: (1) ProstT5's **3Di / structure channel** — the thing it was
actually built for — and (2) a **principled use of its hidden layers** instead of a
single fixed layer. Companion to `PLAN_AIDO_v2.md` (the orthogonal "bigger PLM" bet),
`RESULTS_PROSTT5.md`, `LAYER_PROBE.md`, `AUGMENTATION.md`, and
`../RUNS_CONSOLIDATED_v2.md`.

Status: **run6a + run6c done (both negative) → gate tripped, redirect to AIDO.**
Most of this is CPU-feasible (ProstT5 is the same ~1.2 B encoder already cached),
except where noted.

> **Update (2026-06-25): both ProstT5 enhancement bets failed to clear run2 0.740.**
> - **run6a (H-mix): PCC 0.685.** The learned scalar-mix over all 25 layers left the
>   softmax ≈uniform (entropy 3.20/3.22 nats; final-layer weight 0.046), degenerating
>   to a layer mean that dilutes the final-layer signal — same direction as run3a/b.
> - **run6c (H-struct, AA⊕3Di concat): PCC 0.663.** Real WT 3Di (mini3di → ProstT5
>   fold mode) concatenated with AA (2048-d) hurt. With the backbone held fixed across
>   the mutation (no mutant structures), the 3Di channel is identical for wt and mut →
>   no mutation signal in `mut − wt`, and the width inflation dilutes the AA signal.
> - **run6b skipped** as degenerate (3Di-only + fixed WT 3Di ⇒ `mut_emb ≡ wt_emb` ⇒
>   zero signal); **run6d/6e** were gated on run6c showing structure helps — it did not.
>
> **Gate tripped:** ProstT5's ceiling in MuLAN's head is real; priority redirects to
> **AIDO** (`docs/history/PLAN_AIDO.md`). The only untried structure lever is *modeled mutant*
> structures (FoldX/AF) — a much larger lift. Details: `RESULTS_PROSTT5.md` Findings
> 4–5, `../RESULTS.md` §3/§9. Implementation landed: `mulan.modules.LayerMix` + config
> (run6a); `gen_3di.py`, `gen_struct_embeddings.py`, `concat_aa_3di.py` (run6c);
> drivers `scratch/run6{a,c}_driver.sh`.

---

## Where we are

Consolidated baselines (`../RUNS_CONSOLIDATED_v2.md`):

| | PLM | Embedding | Test PCC |
|---|---|---|---|
| run2 | ProstT5 (AA) | last layer 24 | 0.740 (single split) |
| cv10_prostt5 | ProstT5 (AA) | last layer 24 | 0.805 ± 0.057 (10-fold) |
| — | Ankh (best) | last layer + aug | 0.769 single / 0.832 CV |

Two **negative** results constrain this plan and must not be repeated blindly:

- **Fixed mid-layer / hard concat hurt** (run3a layer 7 → 0.648; run3b concat 7⊕10
  → 0.429, both vs run2 0.740). The layer probe is a *linear* proxy; its ranking
  did not transfer to MuLAN's attention head. **Lesson:** don't pick a single
  "best" layer by linear probe and swap it in.
- **Tier-1 augmentation was neutral for ProstT5** (0.740 → 0.738). So we cannot
  lean on augmentation to rescue ProstT5 the way it helped Ankh.

The probe did show ProstT5's final layer 24 is its *weakest* contextualized layer
for the local site signal (site-PCC 0.702 vs 0.765 at layer 3, broad ~0.74–0.765
plateau through the middle; pool best at layer 10, 0.754). The signal is spread
across the stack — which points at a *learned mix*, not a hand-picked layer.

---

## The core hypothesis

**H-struct (primary).** ProstT5 underperforms Ankh in MuLAN *because the sequence-only
pipeline never uses ProstT5's structure (3Di) channel* — the one capability that
distinguishes it from Ankh/ESM. If we feed ProstT5 real structural context for the
S1102 complexes (whose WT structures we already have on disk), structure-aware
embeddings should close or reverse the gap to Ankh. This is the single most likely
way ProstT5 *beats* Ankh, flagged as an open step in `RESULTS_PROSTT5.md`.

**H-mix (secondary).** A *learned* scalar-mix over all 25 hidden states recovers the
mid-stack signal the probe found, without the failure mode of run3a/b (committing to
one wrong layer). Keeps embedding width at 1024, so it is a cheap, low-risk test.

---

## Why structure mode is actually feasible here (the key realization)

The pipeline is nominally "sequence-only," but the **WT structures already exist on
disk**: `scratch/skempi2/` holds SKEMPI 2.0's cleaned PDBs (`SKEMPI2_PDBs.tgz`),
the same files `build_wt_fasta.py` reads to produce `wt_sequences.fasta`. So real
per-residue **3Di tokens are obtainable for every WT chain** via Foldseek's
structure→3Di conversion — no new structure prediction needed for wild types.

This converts H-struct from "out of scope, needs structures" (its framing in
`RESULTS_PROSTT5.md`) into a concrete, runnable experiment.

**The mutant-structure problem (and the cheap, defensible answer).** ΔΔG needs both
WT and mutant embeddings. We have WT structures but not mutant ones. For
**single-point** mutations the backbone is almost unchanged, so the first cut is:
use the **WT backbone 3Di for both** the WT and the mutant pass, and let the **AA
channel carry the mutation identity**. This injects structural context while keeping
the mutation signal where MuLAN already reads it (`mut_emb − wt_emb`). Modeling the
mutant structure (FoldX repair / AF) is a later refinement, not a blocker.

---

## Experiment ladder (run6 series)

Ordered cheap→expensive; each gated on the previous. All on the **same** 769/157/174
split as run1–4 unless noted, fresh `LightAttModel`, identical optimizer/schedule —
so the embedding is the only changed variable (the discipline that made run1↔run2
clean). Baselines to beat: ProstT5 run2 **0.740**; stretch target Ankh **0.757 /
0.769**.

| Run | Idea | Embedding | Width | Cost | Pre-registered bar |
|---|---|---|---|---|---|
| ~~run6a~~ **done** | **Learned scalar-mix** (H-mix) | softmax-weighted Σ of all 25 layers | 1024 | low (CPU) | > run2 0.740 → **FAILED: 0.685**, mix stayed ≈uniform |
| ~~run6b~~ **skipped** | **3Di-only structure embedding** (H-struct) | ProstT5 over real WT 3Di tokens | 1024 | low–med | degenerate: fixed WT 3Di ⇒ `mut≡wt` ⇒ zero signal |
| ~~run6c~~ **done** | **AA ⊕ 3Di concat** (H-struct, main) | AA-mode 1024 ⊕ 3Di-mode 1024 | 2048 | med | > 0.757 → **FAILED: 0.663**, structure channel hurt |
| ~~run6d~~ **off** | **AA + 3Di cross-fusion** | small learned fusion before LazyConv1d | 1024–2048 | med | gated on 6c showing signal — it did not |
| run6e | **Multi-PLM: Ankh ⊕ ProstT5(AA)** | 1536 ⊕ 1024 | 2560 | med | orthogonal complementarity test; > Ankh 0.757 (not run) |

Notes per rung:

- **run6a — learned scalar-mix. DONE (negative): PCC 0.685 < run2 0.740.**
  ELMo/SeqVec-style learned per-layer weights `γ · Σ_ℓ softmax(w)_ℓ · h_ℓ`, dim held
  at 1024 (`AUGMENTATION.md` Tier 3.3). Implemented as wiring **(ii)** — a learnable
  `LayerMix` module inside MuLAN consuming all 25 cached hidden states (`[L, 25, 1024]`
  per id; no collator change needed since the layer axis is residue-second so
  `pad_sequence` still pads along L). Outcome: the trained softmax stayed **≈uniform**
  (entropy 3.20/3.22 nats; final-layer weight 0.046, argmax layer 5, γ 0.79), so it
  degenerated to a layer mean and *diluted* the final-layer signal — same failure
  direction as run3a/b. The gradient to the layer weights was too weak to specialize.
  H-mix not supported. Remaining angle if revisited: bias the init toward the final
  layer, or sharpen with a temperature, but the structure runs are higher priority.

- **run6b — 3Di-only. SKIPPED (degenerate).** The idea was to embed each WT chain's
  real 3Di string in ProstT5 fold mode (`<fold2AA>`) and train MuLAN on that alone.
  But the mutant-handling trick (reuse WT 3Di) makes the mutant and WT 3Di *identical*,
  and with **no AA channel** to carry the mutation, `mut_emb ≡ wt_emb` exactly →
  `mut − wt = 0` for every example → the model can only predict a constant. So the
  literal real-3Di run6b is algebraically zero-signal, not an empirical question; it
  was skipped. (A predicted-mutant-3Di variant could differ, but that is the scoped
  fallback, weaker than real 3Di.) The plumbing it would have shaken out was validated
  inside run6c instead.

- **run6c — AA ⊕ 3Di concat. DONE (negative): PCC 0.663 < run2 0.740.** Per residue,
  concatenate AA-mode (1024) and 3Di-mode (1024) → 2048-d; `LazyConv1d` absorbs the
  width. Got the full schedule (early-stopped at 44 ep), so the run3b under-training
  confound is ruled out — this is a clean negative. Real WT 3Di came from the on-disk
  SKEMPI PDBs via **`mini3di`** (pure-Python, Foldseek-3Di-compatible; chosen over the
  Foldseek binary for exact per-residue index alignment to the AA fasta and no external
  install), embedded in ProstT5 fold mode. The decisive limitation is the
  mutant-structure approximation: with WT 3Di reused for the mutant pass, the structure
  channel is identical across the mutation, so it adds a constant complex-level offset
  (no mutation signal) while doubling the width dilutes the AA signal that worked —
  net *worse than neutral*. Plumbing landed: `gen_3di.py`, `gen_struct_embeddings.py`,
  `concat_aa_3di.py`, driver `scratch/run6c_driver.sh`.

- **run6d — cross-fusion. NOT RUN.** Was gated on run6c showing structure helps but
  the raw concat being noisy. run6c showed structure does not help at all, so a fusion
  refinement has nothing to refine.

- **run6e — multi-PLM concat. NOT RUN.** `AUGMENTATION.md` Tier 3.2 (Ankh ⊕ ProstT5
  AA, 2560-d) — orthogonal to structure and still a viable "best of both" idea, but
  out of scope once the gate redirected effort to AIDO. Left as a possible future
  baseline (both caches exist).

**Gate — TRIPPED.** run6a (0.685) and run6c (0.663) both failed to clear run2's 0.740,
so per this gate ProstT5's ceiling in MuLAN's head is real and effort redirects to
**AIDO** (`docs/history/PLAN_AIDO.md` / `../RESULTS.md` §8). The single untried lever that could
still revive H-struct is *modeled mutant* structures (FoldX repair / AlphaFold) so the
3Di channel carries mutation-specific signal — a much larger lift, not pursued now.

**Post-run6c follow-ups (all negative).** Three further structure fusions were tried
after the gate, catalogued in `../PROSTT5_STRUCTURE_OPTIONS.md`: gated 3Di concat
(E1, 0.705, gate→0.46), pooled 3Di as head-level context outside the `mut−wt` difference
(B1, 0.673), and interface-contact attention biasing (C3, 0.662, learnable α→≈0). Every
learnable structure knob trained to neutral/off — MuLAN's head consistently declines
ProstT5 structure for single-point ΔΔG. The structure thread is fully exhausted.

---

## Integration design

Reuses the existing offline-embedding interface — MuLAN only ever consumes cached
`[L, D]` `.pt` tensors, so none of this touches the model code except the optional
run6a/6d learnable modules.

1. **3Di extraction (run6b/c/d).** Foldseek `structureto3di` (or `foldseek
   createdb`) over `scratch/skempi2/` PDBs → per-chain 3Di strings, aligned to the
   same `auth_seq_id` residue indexing `build_wt_fasta.py` already uses, so 3Di and
   AA sequences are index-matched per residue. Validate length parity (3Di len ==
   AA len) per chain — the structural analogue of the `<AA2fold>` off-by-one check.
2. **Structure-mode embedding.** Extend `gen_layer_embeddings.py` (or a sibling
   `gen_struct_embeddings.py`) to run ProstT5 in fold mode on 3Di input and AA mode
   on AA input, strip the mode-prefix token, and write `[L, 1024]` (or concat
   `[L, 2048]`) per id.
3. **All-layer cache (run6a).** Add a `--all-layers` path to `gen_layer_embeddings.py`
   that stores `[n_layers, L, 1024]`; a small `LayerMix` module (learnable softmax
   over the layer axis) sits in front of `LazyConv1d`.
4. **Train.** `mulan-train --embeddings_dir <dir> --plm_model_name prostt5_struct`
   (etc.); `LazyConv1d` adapts to whatever width results.

Mutant handling for structure runs: reuse WT 3Di for the mutant pass (first cut);
revisit with modeled mutant structures only if run6c proves structure matters.

---

## Risks / open questions

- **Mutant structure approximation** is the main scientific risk: holding 3Di fixed
  across the mutation means the structure channel contributes a *complex-level* prior,
  not a mutation-specific one. That may still help (context) or may wash out in
  `mut − wt`. run6b/6c will show which; if it washes out, modeled mutant structures
  are the next lever.
- **Width inflation overfits 769 examples.** 2048-d (6c) and 2560-d (6e) double the
  first-layer params; run3b already showed a wide model can under-train. Mitigate
  with full epochs, and lean on the augmentation/CV protocol once a config clears the
  bar.
- **Foldseek 3Di ↔ ProstT5 token vocabulary** must match exactly (ProstT5 was trained
  on Foldseek's 20-state 3Di alphabet, lowercase convention); a vocab/case mismatch
  would silently produce garbage embeddings. Verify against the ProstT5 model card
  before run6b.
- **Learned scalar-mix caching cost** (25 layers) is ~25× disk per seq; fine for 1444
  seqs (~tens of GB worst case), but check before generating.
- **Does fold-mode encoding need a real structure or can AA→3Di prediction
  substitute?** For WT we use real structures (no prediction). The "predicted 3Di
  from sequence" variant (fully sequence-only, no PDB) is a fallback worth noting but
  is expected weaker than real 3Di — out of scope unless WT structures prove a
  bottleneck.

---

## Accuracy & sources

Confidence: the *baseline numbers* and the two negative results are high-confidence
(transcribed from this repo's run artifacts). The *hypotheses* (structure mode will
help; scalar-mix recovers mid-stack signal) are **untested predictions** — that is
the point of the runs. The feasibility claim (real WT 3Di obtainable from the
on-disk SKEMPI 2.0 PDBs via Foldseek) is high-confidence on the inputs existing,
medium on the plumbing working first try.

- ProstT5 (Heinzinger et al. 2024), bilingual AA↔3Di, Foldseek 20-state alphabet:
  DOI 10.1093/nargab/lqae150 (confidence: high).
- Foldseek 3Di structural alphabet (van Kempen et al. 2024, *Nat Biotechnol*):
  DOI 10.1038/s41587-023-01773-0 (confidence: high).
- ELMo/SeqVec learned layer-mix precedent (Heinzinger et al. 2019):
  DOI 10.1186/s12859-019-3220-8 (confidence: high).
- In-repo: `RESULTS_PROSTT5.md`, `LAYER_PROBE.md` (probe tables), `AUGMENTATION.md`
  (Tier 3.1/3.2/3.3), `../RUNS_CONSOLIDATED_v2.md`, `scratch/results/run1_s1102_ankh/RESULTS.md`
  (SKEMPI 2.0 PDB provenance).
