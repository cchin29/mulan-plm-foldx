# A2 — MINT complex-context embeddings for MuLAN ΔΔG

> **Archived — executed.** MINT ran the full 10-fold CV and is a backbone in the scaling suite.
> The finding is negative in the way that matters: MINT does not beat the multi-PLM consensus, and
> it is the weakest arm on the electrostatic tail this plan set out to attack. The embedding
> generator ships as `experiments/mint/gen_mint_emb.py`, and `mulan/data.py` cites §2 of this
> document for the pair-cache contract.

Plan of record for the doc's **A2** alternative (`scripts_plots/S1102_CROSSFOLD_MISPREDICT.md`
§Alternatives): replace MuLAN's monomer per-chain embeddings with **MINT** partner-context
embeddings, to attack the cross-chain electrostatic blind spot (1MAH/2O3B/1BRS/1JTG/2PCC) that a
monomer PLM cannot see. Nothing here clobbers existing results — see §0.

MINT = *Multimeric INteraction Transformer* (Ullanat/Jing/Sledzieski/Berger, Nat. Commun. 2026),
an ESM-2-650M backbone with cross-chain attention, pretrained on 96M STRING PPIs. Repo
`github.com/VarunUllanat/mint`, weights `varunullanat2012/mint` (`mint.ckpt`, MIT), dim **1280**.

---

## 0. Isolation (already set up — nothing existing is touched)

| purpose | path | status |
|---|---|---|
| code + this plan | `experiments/mint/` | tracked, new dir |
| MINT source (vendored clone) | `scratch/mint_src/` | scratch (gitignored) |
| dedicated venv | `.venv-mint/` | **add to `.gitignore`** next to `.venv-esmc` |
| embedding cache | `scratch/embeddings_mint/` | scratch (gitignored) |
| CV results | `scratch/results/mint_a2/` | scratch (gitignored) — **NOT** `embedding_sweep_balanced/` |
| sensitivity probe outputs | `scratch/mint_probe/` | scratch (gitignored) |

`scratch/` is gitignored wholesale (`.gitignore:167`), so caches/results/probes cannot collide with
the tracked paper/balanced anchors. The isolated results dir means MINT never shares a directory
with the 6-PLM balanced sweep it will be compared against. Reuse the existing balanced **splits**
read-only (`experiments/embedding_sweep/splits/balanced_seed42/`) so PCCs are directly comparable to
the base/aug/FoldX arms in the doc.

---

## 1. Install (dedicated venv, like `.venv-esmc`)

MINT pins its own stack (`fair-esm`-style checkpoint + lightning ckpt loading) and must not perturb
`.venv` (transformers 4.44, all `mulan-train`). Mirror the ESM-C pattern: a separate venv used
**only** for embedding generation; training still runs in `.venv` off the finished cache.

```
git clone https://github.com/VarunUllanat/mint scratch/mint_src
python3 -m venv .venv-mint
.venv-mint/bin/pip install -e scratch/mint_src         # + torch (MPS build), huggingface_hub
# weights: huggingface_hub.hf_hub_download("varunullanat2012/mint", "mint.ckpt")
```

**RESOLVED (smoke_test.py, 2026-07-14 — all questions answered):**
- **Install:** minimal — `.venv-mint` (py3.12) needs only `torch einops numpy` + `pip install -e
  scratch/mint_src` (base install has **no** `install_requires`; deepspeed/lightning/openmm are
  training-only and confined to `utils/wrapper.py`, off the inference path). `load_config` is inlined
  to dodge extract.py's pandas import.
- **Per-residue API (the crux):** bypass `MINTWrapper` (it pools). Build `ESM2(..., use_multimer=True)`,
  load `ckpt["state_dict"]` stripping the `model.` prefix, then
  `model(tokens, chain_ids, repr_layers=[33])["representations"][33][0]` → `(T, 1280)` per token.
  Drop specials via `~eq(cls=0)&~eq(eos=2)&~eq(pad=1)`, split by `chain_ids` (0=A,1=B). Verified
  exact residue counts: 110-res chain → `(110,1280)`, 89-res → `(89,1280)`.
- **Encoding:** `<cls>` + seq (`J→L`) + `<eos>` per chain, concatenated along length; `chain_ids` 0 for
  chain A, 1 for chain B. `Alphabet.from_architecture("ESM-1b")`.
- **MPS:** works, no fallback needed. Load 6.4 s; forward ~1 s/pair. Model is **813M** params
  (multimer cross-attention on top of the 650M backbone).
- **Cross-chain is ACTIVE (mechanism confirmed):** a single chain-A point mutation shifts the
  chain-B embedding, `‖enc(B|A_mut) − enc(B|A_wt)‖` relative norm **0.013** ≠ 0. This is the
  non-canceling partner term the whole A2 arm rides on — non-zero before we even train.
- Open: max complex length (ESM-2 `max_positions`=1024; long two-chain complexes may need truncation —
  handle in the generator).

---

## 2. THE integration decision — pair-keyed cache (load-bearing)

**Confirmed: 100% of S1102's 1100 rows are single-chain mutations (245 A-only, 855 B-only, 0
both-chain).** This makes the existing per-label `.pt` cache scheme *silently wrong* for MINT:

MuLAN keys the four embeddings by label (`data.py::_fill_metadata`):
`(seq1_label, seq2_label, mut_seq1_label, mut_seq2_label)`, where for a chain-A-only mutation
`mut_seq2_label = f"{seq2_label}_"` — **a constant string per complex**, and its sequence = wt2
(unchanged). In the monomer pipeline that dedup is correct (wt2's embedding is partner-independent).
Under MINT it is fatal: `enc(wt2 | mut1)` **varies with the chain-A mutation**, but the constant
label maps every A-mutation of a complex to *one* cached file → the partner embedding collapses to
`b_mut ≡ b_wt`.

That is exactly the non-canceling term A2 depends on. The head's bilinear product factors as
`(a_mut−a_wt)⊙b_mut + a_wt⊙(b_mut−b_wt)`; the monomer pipeline zeros the second term because
`b_mut≡b_wt`, and **reusing the label cache would re-zero it**, silently reducing MINT to a
monomer-quality no-op (the FoldX/3Di trap, round 3). Since every S1102 row is single-chain, this is
100% of the signal, not an edge case.

**Decision: key the cache per (complex, full-mutation-string), not per chain label.** Each row gets
its own bundle of four per-residue tensors, generated from **two MINT complex forward passes**:

- WT complex `MINT([wt1, wt2])` → `enc(wt1|wt2)`, `enc(wt2|wt1)`  — shared across a complex's rows
- MUT complex `MINT([mut1, mut2])` → `enc(mut1|mut2)`, `enc(mut2|mut1)` — per row

Implementation options (pick at build time, prototype the simpler first):
- **(a) Row-keyed dir, minimal MuLAN change.** Store `scratch/embeddings_mint/<rowhash>/{wt1,wt2,mut1,mut2}.pt`. Add a thin MINT dataset variant (or a `mint_pair` flag on `MulanDataset`) whose `_load_embeddings` keys by row index instead of the four labels. WT bundle memoized by `s1__s2` so it's computed once per complex, not once per row.
- **(b) Encode context into the labels.** Redefine `mut_seq2_label`/`seq2_label` to include the partner mutation hash so the *existing* loader stays untouched. Less code churn in `data.py` but leaks MINT semantics into the shared label scheme — riskier for the other PLMs. Prefer (a).

Storage: 1100 rows × 4 tensors × ~[250,1280] fp32 ≈ 5–6 GB (WT bundles dedup by complex → less).
Fits the 359 GB free.

---

## 3. Generation driver

New `experiments/mint/gen_mint_emb.py`, modeled on `gen_esmc600m_emb.py` (out-of-band, id
enumeration copied verbatim from `MulanDataset._fill_metadata` + `mulan.utils` so training reads the
cache with no PLM load). Differences: it forwards **complexes** (chain pairs) not single sequences,
runs in `.venv-mint`, writes the row-keyed bundles of §2, memoizes the WT complex pass per `s1__s2`.
Idempotent skip-existing. MPS bf16 → cast fp32 (match the suite).

---

## 4. Probe-first gate (cheap, no training — from the doc's cancellation check)

**Before any CV run**, run `experiments/mint/probe_partner_sensitivity.py` →
`scratch/mint_probe/`: for the known hotspots (1MAH WA276R, 2O3B DB75E/DB75N, 1BRS RA81Q) compute
‖`enc(mut2|mut1) − enc(wt2|wt1)`‖ (partner-chain embedding shift induced by the single-residue
mutation). Decision rule:
- **Non-trivial norm** (partner clearly moves) ⇒ the cross-chain signal survives the Siamese diff
  ⇒ proceed to full generation + CV.
- **Near-zero** ⇒ MINT's cross-attention is insensitive to single-residue mutations here ⇒ A2 will
  not beat monomer through the bilinear head. Fall back to §6 bypass before spending CV compute.

This is the go/no-go. It costs one MINT load and a handful of forward passes.

**RESULT (2026-07-14, `probe_partner_sensitivity.py`): GO, with a caveat.** Mean partner relative
shift over the 4 hotspots = **0.0062** (> 0.005 gate). Per hotspot (partner = non-mutated chain):
2O3B DB75N 0.0084, 2O3B DB75E 0.0065, 1BRS RA81Q 0.0081, **1MAH WA276R 0.0018**. The shift is
**localized** (2O3B partner peaks at residues 12–15, 1BRS at 51–54, top-10 residues carry 16–32% of
it) → genuine interface cross-chain signal, not diffuse noise. **Caveat:** the single biggest miss,
1MAH WA276R (true 8.81, buried aromatic core), moves the partner *least* — MINT's context is
strongest on the charged/electrostatic hotspots (2O3B, 1BRS) and weakest on the buried-core one, so
A2 is likely to help the electrostatic tail more than 1MAH. Signal is real but modest → keep the §6
B1 bypass in reserve if the bilinear diff attenuates it.

---

## 5. Train / evaluate (isolated arm, reuse balanced splits)

Add a row to a **MINT-specific** models list (not the shared `models.tsv` unless we also point
`ES_RESULTS_DIR` at `scratch/results/mint_a2/`): `mint  mint  scratch/embeddings_mint`. Because
`mint` is not in `PLM_ENCODERS`, training only works off a **complete** cache — same guard as ESM-C.
Run the 10 balanced folds via `run_sweep.sh` with a `config_mint.sh` that sets
`ES_RESULTS_DIR=scratch/results/mint_a2` (leaving `ES_SPLIT_DIR` = balanced). Queue with **pueue**
(per memory), 300 ep / patience 30, to match the arms in the doc. Aggregate → pooled OOF PCC, then
re-run `analyze_hotspots.py` logic with MINT as a 7th arm and compare on the worst-40 tail
specifically, not just aggregate PCC.

### RESULTS — FINAL (2026-07-14, 10/10 folds; pueue #51 complete)

Full balanced 10CV (`analyze_mint.py` per-fold + worst-40, `diagnose_shrinkage.py` for the mechanism;
both auto-detect folds and were re-run verbatim at completion). Headline: A2 lifts the weak fold's
bulk PCC but does **not** rescue the extreme electrostatic tail it was built for — it is marginally
*worse* on the worst-40 — and the diagnostic explains why.

**Pooled OOF PCC (all 1100): MINT 0.856 vs 6-PLM consensus 0.870** (gap −0.014). Per-fold MINT beats
consensus on 4/10 folds; the standout is **fold_0 +0.058** (0.792 vs 0.734 — the doc's worst fold),
but fold_0 holds only one named hotspot (1MAH YA69N) so the gain is a **broad-fold** effect, not a
hotspot rescue. Selected folds:

| fold | MINT | base | aug | FoldX | ΔMINT−base |
|---|---|---|---|---|---|
| 0 | **0.792** | 0.734 | 0.734 | 0.761 | **+0.058** |
| 6 | 0.886 | 0.878 | 0.884 | 0.881 | +0.008 |
| 8 | 0.796 | 0.802 | 0.805 | 0.841 | −0.006 |
| — | pooled 0.856 | 0.870 | — | — | −0.014 |

**Baseline-fairness caveat (do not over-read the −0.014 gap).** `base` is the **6-PLM consensus**
(esm2/esmc600m/prostt5/saprot/ankh/esmc6b mean), so this pits **one 650M model against a 6-model
ensemble**. A single model trailing the ensemble pooled PCC by 0.014 *while beating it by +0.058 on
fold_0* is a strong showing, not a loss. The gap conflates **three** effects — (i) context benefit,
(ii) MINT-650M vs ESM2 backbone, (iii) single-model vs ensemble. **§8a now attributes it (10/10 folds):**
MINT-mono (chains embedded *alone* through the same backbone) pools to **0.821**, so context
(complex−mono) is **+0.035 pooled / +0.038 cross-fold** — a genuine positive lift — while the mono arm
itself sits −0.050 below the consensus (backbone + single-model-vs-ensemble). The −0.014 deficit is that
one 650M model can't match a 6-model ensemble; context claws back most of it. See §8a for the full
three-way table and the ESM2-650M-vs-3B baseline caveat.

**Worst-40 tail (the A2 target): MINT is the *worst* of the four arms.** MAE base 3.63 / aug 3.61 /
FoldX 3.38 / **MINT 3.71**; MINT beats base on only 50% of the worst-40 (coin flip). The two named
hotspots that were pending at 8-fold both confirm it: **2O3B DB75E** true 5.44 → MINT 0.20 (base 0.93)
and **1BRS RA81Q** true 5.42 → MINT 0.71 (base 1.37) — MINT shrinks them *further* toward zero.

**Why the tail is NOT de-shrunk (mechanistic, `diagnose_shrinkage.py` over 1100 rows):**

| quantity | value | reading |
|---|---|---|
| δ_partner (median ‖b_mut−b_wt‖) | 0.036 | the revived non-canceling term — real, non-zero |
| δ_mutated (median ‖a_mut−a_wt‖) | 0.603 | the primary term |
| **partner/mutated ratio** | **0.069** | revived term is ~7% (≈15×) smaller than the primary |
| corr(δ_partner, \|true\|) | +0.03 | partner shift barely tracks hotspot magnitude |

**Subset correlation (the global average dilutes the tail, so restrict to where the
partner actually moves).** corr(δ_partner, MINT-over-base improvement) by subset (charged = D/E/K/R/H
as wt or mut; interface = top-quartile δ_partner):

| subset | n | corr(δpar, impr) | med δ_partner | MINT MAE | base MAE |
|---|---|---|---|---|---|
| all rows | 1100 | +0.010 | 0.036 | 0.872 | 0.857 |
| charged mutation | 526 | −0.023 | 0.036 | 0.863 | 0.849 |
| **interface (top-25% δ_partner)** | 275 | **+0.025** | 0.093 | 0.845 | 0.823 |
| charged AND interface | 126 | −0.050 | 0.090 | 0.807 | 0.783 |
| tail \|true\|≥2 | 360 | −0.003 | 0.040 | 1.385 | 1.401 |
| **interface AND \|true\|≥2** | 106 | **+0.050** | 0.097 | 1.198 | 1.185 |

This makes the head-bottleneck argument **airtight, in the opposite direction of the hopeful read**:
even restricted to the interface subset where the partner genuinely moves (med δ_partner 0.093, ~3×
the global 0.036), the correlation with improvement is at most **+0.05** — negligible, nowhere near
"strongly positive" — and MINT's MAE is *worse* than the consensus on **every** subset. The revived
signal doesn't help even on its home turf.

The §4 probe's GO was *necessary* (`b_mut ≠ b_wt`) but not *sufficient*: the revived term is an order
of magnitude too small and **uncorrelated** with prediction error, so it washes out in the pooled
`enc(1)*enc(2)` product. MINT ≈ monomer + small noise **for the tail / error-ranking** (scoped: the
§8a same-backbone ablation shows a real **+0.035 bulk-PCC** lift — not a contradiction, because that
lift is a *different quantity* from this partner-shift term; see §8a). Shrinkage slope (pred~true, <1 = shrinks):
MINT de-shrinks only marginally — 0.744 vs base 0.718 overall — and the genuine extremes are still
catastrophic in **both** arms (1MAH WA276R 8.81→0.55, 2O3B DB75N 5.90→−0.08, 1BRS HA100G 6.82→3.90).

**Conclusion:** the bottleneck is not the embedding's partner-blindness (MINT fixed that) but the
**head + MSE loss**, which regresses rare high-ΔΔG rows to the mean regardless of embedding. The
**monomer-MINT ablation (§8a) is now complete and reinforces this**: complex-context (both chains) is a
*real* +0.035-pooled bulk-PCC lever (mono 0.821 → complex 0.856), but it does **not** rescue the tail — worst-40
MAE only 4.02 → 3.71 (both worse than base 3.63) and the extremes stay shrunk in every arm (slope
|true|≥4 ≤ 0.86). δ_partner ≪ δ_mut and its ~0 error-correlation (even on the interface subset) are
per-row structural facts over 1100 rows, consistent across the full 10CV.

---

## 6. Fallback if the bilinear diff attenuates (doc's second lever)

If probe (§4) is positive but the CV arm still collapses toward monomer, route the partner-context
embedding to the head **outside** the `mut − wt` Siamese difference, reusing the existing **B1
`struct_context` hook** (`modules.py:230`, already "outside the siamese mut-wt difference"): feed a
pooled MINT complex vector (or `b_mut`) through `ctx_proj` so the signal reaches the regression head
even if the bilinear diff washes it out. No new architecture — an existing concat slot.

**Re-assessed after the §5 diagnostic (2026-07-14): B1 alone is unlikely to help.** The arm did not
collapse from *cancellation* — MINT successfully un-cancels the partner term — but from *magnitude*:
the revived partner signal is ~7% of the primary term and uncorrelated with prediction error (§5).
Routing a 7%-magnitude vector through `ctx_proj` won't move a head dominated by the primary term. The
diagnostic relocates the bottleneck to the **head + MSE loss** (extremes regressed to the mean), so
the higher-value levers are: (i) **tail-reweighted / transformed loss** (weight rare high-ΔΔG rows,
or Huber/log-target), or (ii) **residue-level cross-chain attention in the head** rather than the
pooled bilinear product. Keep B1 only as a cheap ablation to confirm it adds little.

---

## 7. Sequence of work & gates  — progress 2026-07-14

1. ✅ Install + smoke-test (§1) — API/MPS resolved, recorded above. `smoke_test.py`.
2. ✅ `gen_mint_emb.py` row-keyed cache (§2–3); WT dedup verified.
3. ✅ **Probe gate (§4): GO** (partner shift 0.0062, localized; 1MAH caveat). `probe_partner_sensitivity.py`.
4. ✅ `mint_pair` load path in `MulanDataset` (§2a) + CLI flag + 3 call sites. `test_mint_pair.py`
   passes (loader↔generator keying identical, partner term preserved, clean error on incomplete cache).
   **Code review (2 agents, high effort): correctness = no bugs; cleanup = 1 fix applied** (generator
   `parse_mutations` now raises on non-A/B to match `mulan.utils`; key-format duplication and mint-mode
   `_fill_metadata` dead-work noted as acceptable given the venv split + one-time init cost).
5. ✅ **10-fold balanced CV complete (pueue #51).** Split membership verified identical to
   `splits_balanced_foldxdec` (110/110 all folds). Infra: `config_mint.sh` (isolated `results/mint_a2`,
   `--mint_pair`), `run_sweep.sh` `ES_EXTRA_TRAIN_ARGS` hook, `mint` row in `models.tsv`. **FINAL result
   in §5:** pooled OOF 0.856 vs 6-PLM consensus 0.870; A2 lifts weak fold_0 (+0.058) but is the *worst*
   arm on the worst-40 tail (MAE 3.71) — the revived partner term is ~15× too small and error-
   uncorrelated even on the interface subset (`diagnose_shrinkage.py`).
6. ✅ **§8a monomer-MINT ablation complete** (pueue #52 gen → #53 CV, 10/10 folds). Attributes the
   0.856-vs-0.870 gap: **context (complex−mono) = +0.035 pooled / +0.038 cross-fold** (mono pooled
   0.821, cross-fold 0.811±0.065) — a genuine lift; the −0.014 deficit is the single-650M-model-vs-
   ensemble/backbone term (mono −0.050 below consensus). Context helps the **bulk, not the tail**
   (worst-40 MAE mono 4.02 → complex 3.71, both > base 3.63) → **reinforces** the head+MSE bottleneck.
   Baseline caveat: no ESM2-650M single-model base exists (suite `esm2` is 3B), so the backbone term is
   approximate. Full three-way table in §8a.
7. §5 diagnostic reframes §6: **B1 bypass demoted** to a cheap confirm-only ablation; the real levers
   are tail-reweighted loss or head-level cross-chain attention. Decide after the §8a ablation.

**Artifacts (all in `experiments/mint/`):** `smoke_test.py`, `gen_mint_emb.py`,
`probe_partner_sensitivity.py`, `test_mint_pair.py`, `analyze_mint.py`, `diagnose_shrinkage.py`,
this plan.

## 8. Risks / open items
- **Per-residue extraction** from MINT (not the pooled wrapper) — unproven; §1 smoke-test settles it.
- **MPS support** — may be CPU-only; if so, generation is slow but one-time (cache is read-only after).
- **Length limits** — ESM-2 650M context; long complexes (both chains concatenated) may truncate.
- **`mut − wt` semantics change** — `mut1/mut2` now embedded *as a complex*; the diff is no longer
  identically-partnered (intended) → keep a monomer-MINT ablation (embed each chain alone through
  MINT) to isolate "context" from "MINT-vs-ESM2 backbone."
- Add `.venv-mint/` to `.gitignore`.

### 8a. Monomer-MINT ablation — PRIORITIZED (the load-bearing control)

The §5 pooled gap (MINT 0.866 vs 6-PLM consensus 0.878) conflates context / backbone / ensemble
(§5 fairness caveat). This ablation isolates **context** by embedding each chain **alone** through the
*same* MINT-650M backbone (single-chain forward: one chain's tokens, `chain_ids` all 0, no partner in
the window), keying the cache identically. Then a MINT-mono arm vs the MINT-complex arm differ **only**
in partner context (i, above), and MINT-mono vs the ESM2 base isolates backbone (ii). Attribution:

- `MINT-complex − MINT-mono` = pure context benefit (what A2 actually buys).
- `MINT-mono − ESM2-base` = backbone difference (MINT-650M vs ESM2-650M), ensemble aside.

Build (✅ done): a `--single-chain` flag on `gen_mint_emb.py` (skip the partner concat, guard against
writing into the complex cache; smoke-tested — shapes ok, complex−mono per-residue Δ = 0.51 so context
is large), `config_mint_mono.sh` → `scratch/results/mint_mono/`, and a `mint_mono` models.tsv row. Same
balanced splits / 300ep-p30 budget. Ran in the `mint` group after #51 (pueue #52 gen → #53 CV).

### RESULTS — §8a ablation (2026-07-14/15, 10/10 folds; pueue #53 complete)

Aggregated with `aggregate.py` (`ES_CONFIG=config_mint_mono.sh … aggregate.py mint_mono` →
`results_mint_mono/mint_mono.json`); pooled OOF PCC, worst-40 tail MAE and the shrinkage slope
recomputed with the same method as the complex arm (`analyze_mint.py` / `diagnose_shrinkage.py` logic,
mono preds from `scratch/results/mint_mono/mint_mono/fold_*/…`).

**Three-way, bulk:**

| arm | cross-fold PCC (±std) | pooled OOF PCC | overall MAE | slope(pred~true) |
|---|---|---|---|---|
| MINT-complex | **0.850 ± 0.039** | **0.856** | 0.872 | 0.744 |
| MINT-mono | 0.811 ± 0.065 | 0.821 | 0.977 | 0.682 |
| ESM2-3B base (suite `esm2`) | 0.833 ± 0.053 | — | — | — |
| 6-PLM consensus base | — | 0.870 | 0.857 | 0.718 |

**Three-way, tail (worst-40 hotspots + shrinkage slope at |true|≥4):**

| arm | worst-40 MAE | beats base on worst-40 | slope \|true\|≥4 |
|---|---|---|---|
| MINT-complex | 3.71 | 50% | 0.861 |
| MINT-mono | **4.02** | 32% | 0.782 |
| base (consensus) | 3.63 | — | 0.829 |
| aug / FoldX | 3.61 / 3.38 | — | — |

**Attribution (the point of this control):**
- **Complex-context (both chains) = MINT-complex − MINT-mono = +0.035 pooled (+0.038 cross-fold).** A
  *genuine, positive* lift — the embedding is **not** a no-op. **This is a different quantity from the
  §5 δ_partner term, which is why the two results don't conflict.** The ablation re-contextualizes
  *both* chains — `enc(mut1|mut2)` vs `enc(mut1)` sharpens the **mutated** chain, not just the partner
  — whereas δ_partner is the within-arm mut-vs-wt shift `‖enc(mut2|mut1) − enc(wt2|wt1)‖`. So the
  +0.035 bulk lift is dominated by contextualizing the *mutated* chain, while the specific partner-shift
  term stays ~7% of primary and tail-useless. Read together: complex-context is a real **bulk** lever;
  "MINT ≈ monomer + noise" holds only for the **tail / error-ranking**. (The earlier flat read compared
  complex against a *different-backbone* 6-PLM ensemble; this same-backbone control isolates it.)
- **Backbone/ensemble = MINT-mono − base.** Mono sits **−0.050** below the 6-PLM consensus (pooled) and
  **−0.021** below the single-model ESM2-3B suite arm (cross-fold). This term is where the complex arm's
  −0.014-vs-consensus deficit actually comes from: one 650M-backbone model can't match a 6-model
  ensemble. Context (+0.035) claws back most of that gap, leaving the −0.014.
- **⚠ Baseline caveat (size mismatch — not clean).** MINT's backbone is ESM2-**650M**, but **no
  single-model ESM2-650M base arm exists** in the suite — the only ESM2 base is ESM2-**3B** (`esm2`,
  cross-fold 0.833) and the pooled comparator is the 6-PLM *consensus* (0.870). So "MINT-mono − ESM2
  backbone" is confounded with model size (650M vs 3B) and with ensembling; treat the backbone term as
  **approximate**. A hypothetical ESM2-650M single-model base would plausibly sit at/below the 3B, so
  the *true* MINT-650M backbone deficit vs a like-for-like base is likely smaller than the −0.02/−0.05
  shown here.

**Does §8a change the A2 conclusion? It reinforces it, with one candid nuance.** Context does **not**
rescue the tail: worst-40 MAE only moves mono 4.02 → complex 3.71 (still worse than base 3.63), mono
beats base on just 32% of the worst-40 (vs complex's coin-flip 50%), and the extremes stay
catastrophically shrunk in **all three** arms (slope |true|≥4 ≤ 0.86 everywhere). The bottleneck for the
electrostatic tail is the **head + MSE loss**, exactly as §5 concluded. The nuance §8a adds:
complex-context is a *real* +0.035 bulk-PCC lever (not noise) — it just lands mostly in the bulk. Note
the tail is **not inert** to cross-chain signal — worst-40 MAE *does* move mono 4.02 → complex 3.71
(a 0.31 gain), it just doesn't clear base 3.63 through the pooled head. That is mild positive evidence
for **A1**: putting cross-chain signal at *residue resolution in the head* (rather than pooled in the
embedding) may clear the bar where pooling couldn't — but only if paired with **C1**, since the MSE
loss caps the extremes regardless (hence A1×C1 is the decision cell, not A1 alone).
**Outranked finishing extra folds for interpretation.**

---

## 9. Candidate follow-up — MINT + FoldX-MLP (`add_scores`) (proposed 2026-07-15)

Would layering the decomposed FoldX channel on the MINT arm buy anything? **Expected: a small *bulk*
lift (~+0.005–0.01), elevated redundancy risk, and no tail fix — but ~free to run, and worth it
because the redundancy is non-obvious.**

**Why it could help (complementary):**
- **FoldX is the one cross-modality channel.** Residual corr FoldX↔PLM ~0.46–0.55 vs PLM↔PLM
  ~0.78–0.92, and it stayed orthogonal even to structure-aware SaProt (§18.2's seq-vs-structure axis
  *failed*: SaProt still gained +0.018). MINT is a learned ESM-2 model, so FoldX's explicit interface
  **electrostatics** (its only term with orthogonal signal) is a different modality from MINT's
  learned cross-chain statistics.
- **It bypasses the bottleneck that sank MINT's partner term.** FoldX-MLP enters via `add_scores` at
  the head (like B1), *outside* the pooled `enc(1)⊙enc(2)` product where MINT's cross-chain signal
  washed out — so it reaches the linear head directly.

**Why the payoff is probably small (floor-raiser × strong, cross-chain base):**
- FoldX's lift anti-correlates with base strength (corr −0.61); on the strongest arm (ESM C 6B) it's
  only +0.004. MINT-complex (0.856 pooled) is a strong base whose strength is *specifically
  cross-chain* — the exact regime FoldX's orthogonal ~20% occupies — so redundancy risk is *higher*
  here than for a monomer PLM. Expect ~+0.005–0.01, not the +0.017 of weak sub-6B PLMs.
- **But run it anyway:** "MINT already has cross-chain info ⇒ FoldX redundant" is the same prediction
  shape as §18.2's SaProt call, which was **wrong**. The redundancy is non-obvious; measure it.

**It will not fix the tail.** FoldX is the best worst-40 arm (MAE 3.38 vs MINT's 3.71), but that's
tail *rank*, not magnitude — extremes stay shrunk (MSE bottleneck). So the interesting cell is
**MINT + FoldX + C1**, not MINT + FoldX. Keep C1 in the loop for magnitude.

**Recipe (near-zero new compute):**
- FoldX's 12 decomposed terms are already cached per row in `scratch/foldx_s1102/splits_balanced_foldxdec`
  (model-independent physics) and the MINT row-keyed cache is on the same balanced folds — so this is
  a config, not a re-gen.
- Point `cv10_foldx_mlp_balanced_driver.sh` at the MINT embeddings (reuse `--mint_pair`) with the
  `zs_mlp`/`add_scores` head, isolated to `scratch/results/mint_foldxmlp/`.
- **Test the scalar too, expect it ≥ MLP.** Per RESULTS §20.1, on strong bases / small sets (S1102 ≈
  1100 rows) the FoldX *scalar* matches or beats the 12-term MLP — MINT is exactly that regime, so the
  scalar is the safer default; run `mint_foldxscalar` + `mint_foldxmlp` (+ optionally `mint_foldx_c1`).
- **Judge on both bulk (pooled/cross-fold PCC vs MINT-complex 0.856) and the worst-40 tail** — a bulk
  lift with an unchanged tail would confirm "orthogonal-but-capped, still head-bottlenecked."
