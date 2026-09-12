# Beyond S1102 to full SKEMPI v2 — sequence-only first, then FoldX

> **Archived — executed.** The campaign ran to completion: the curated build, the three split
> protocols and both FoldX arms all ship, as `data/splits/splits_skempi_full_*` and the
> `SUMMARY_*.md` set under `experiments/full_skempi_seqonly/`. The builders and scorers in that
> directory cite this document by section, so it remains the design record for how the row set,
> the clustering and the arm definitions were chosen.

_Written 2026-07-16. Implementation plan. Realizes step 1 of
`REVIEW_gaps_and_landscape.md` §4b — the leakage-controlled, frontier-comparable comparison for a **sequence-only**
method — then stages the FoldX arm. Everything isolated from the S1102 work._

## 0. Objective & framing

Evaluate MuLAN on the **full curated SKEMPI v2** (~7,085 mutations / ~345 complexes, **single AND
multiple** point mutations) under **homology-aware splits** and **per-structure + AUROC** metrics — the
setup the modern DL frontier (RDE, Prompt-DDG, PPIformer, CATH-ddG, USP-ddG, ProtBFF) actually uses.
S1102 (1,102 single-point) is the *classical-lineage* subset; it is **not** frontier-comparable (§2.8).

**Why sequence-only first:** MuLAN's edge is that the **base arm needs no per-mutant structure** — just
PLM embeddings — so the full-SKEMPI base comparison is *cheap* and is the *fair* frame for a sequence
method (vs ESM-1v / MSA-Transformer / Tranception / MINT — not the structure-integrated SOTA). The
FoldX/biophysical arm (which does need structures) is staged second, with a concrete Linux-box compute
plan (§8). **Positioning to keep candid:** the goal is a defensible sequence-only frontier number + the
diagnostic — not to out-compete structure models on their turf (ProtBFF/USP-ddG already fuse FoldX).

## 1. Isolation (all new; S1102 work untouched)

| purpose | path |
|---|---|
| plan + builders + eval | `experiments/full_skempi_seqonly/` |
| curated data (CSV, fasta, mutant seqs) | `scratch/skempi_full/` |
| PLM embeddings | `scratch/embeddings_skempi_full/<plm>/` |
| FoldX outputs (step 2) | `scratch/foldx_skempi_full/` |
| results | `scratch/results/full_skempi/<plm>_{base,aug,foldx}/<split>/fold_*/` |

All under gitignored `scratch/` except the plan/scripts. Do **not** reuse the S1102 dirs.

## 2. Data acquisition (reuse the reference pipelines — don't re-curate)

> **🔒 MANDATORY for comparability (verified 2026-07-16):** define the entry set by **running RDE's
> `load_skempi_entries` verbatim**, NOT a hand-rolled curation. The exact mutation set is *just the
> deterministic output of that parser* on `skempi_v2.csv` — anything else won't line up with the
> published tables even after the FoldX compute. **Our raw CSV already matches the frontier's:** 7,085
> rows / 345 PDB codes / 348 `#Pdb` complexes / 1,973 multi-mut rows = the papers' "~7,085 muts / ~345
> complexes." RDE's curation is minimal & reproducible: exclude only `block_list={'1KBH'}`, require
> finite ddG + existing PDB; keep single **and** multiple; ddG = RT·ln(Kd_mut/Kd_wt) with
> RT=(8.314/4184)·298.15 (positive = destabilizing — **matches MuLAN**; spot-checked: 1CSE_E_I LI38S →
> +1.19, = our S1102 label). Complex id = full `#Pdb` (with chain groups), not the 4-letter code.

> **📌 1KBH — the sole block-list exclusion (not a FoldX failure).** `block_list={'1KBH'}` is RDE's one
> hardcoded manual drop, ported verbatim as `BLOCK={"1KBH"}` in `build_skempi_full.py` (`if code in BLOCK:
> continue`). 1KBH is the CBP-NCBD/ACTR complex — an **NMR ensemble of two intrinsically-disordered
> partners that fold on binding**, with a large high-numbered mutation block (~896–1172); RDE deems it
> unreliable for a structure/ΔΔG model. This began as a curation decision — the earlier benchmark arms,
> where RDE's block list is *not* applied, do hold valid `results_S4169/1KBH.json` /
> `results_S2003/1KBH.json`, 3 muts each, `n_ok:3, unresolved:[], validation_failed:[]`, log "Your file
> run OK". **It is also a compute problem, and the two stages differ.** `BuildModel` succeeds on an
> already-repaired 1KBH, which is what those benchmark results are. `RepairPDB` does not terminate on
> it — past 22.5 h on two machines — so the complex cannot be repaired in the first place, and any
> claim that FoldX handles 1KBH cleanly holds only for the stage that was never the obstacle. 1KBH is
> therefore excluded before any worklist is built, enforced at four points; see
> `docs/CODEBASE_OVERVIEW.md` and the `INTRACTABLE` registry in skempi-foldx. Because 1KBH is dropped *before* the FoldX/embedding stage in the curated build,
> there is deliberately **no** `foldx_skempi_full/work/1KBH` and **no** 1KBH row in the 112-complex
> `foldx_skempi_full/results/` FoldX set the ESM-C 6B +FoldX arm consumes — the block list doing its job,
> nothing to fix.

- **Curated set + labels:** reuse an established curation rather than re-cleaning SKEMPI2 by hand:
  - **RDE** (`../reference_ddg_splits/RDE-PPI/`): `data/get_skempi_v2.sh` fetches `skempi_v2.csv` +
    `SKEMPI2_PDBs.tgz`; `rde/datasets/skempi.py::load_skempi_entries` parses `#Pdb` → complex,
    `Mutation(s)_cleaned`, ddG. We already have `scratch/skempi_v2.csv` (identical source). Run their
    loader to get the exact entries; confirm the count (~6,900–7,000) and that every `iclr24` fold
    complex is present.
  - **CATH-ddG** curation (via USP-ddG) gives the **813-mutation CATH-superfamily hold-out** split — use
    for the leakage-controlled generalization number (§3d).
- **Convert to MuLAN inputs** (`build_skempi_full.py`, new): emit the two files MuLAN's `data.py`
  consumes — `wt_sequences.fasta` (per chain) + a `mutated_complexes.tsv`
  (`wt1_label  wt2_label  mutations`), for **single and multiple** mutations. Apply the mutation string
  to the WT sequence to form each mutant (MuLAN handles multi-point natively). Drop the **10 largest
  complexes** (ProtBFF convention, ProSST/ESM context limit) → ~335 complexes / ~6,631 muts, or keep
  them for PLMs without the limit and flag.
- **Chain / numbering (⚠ the real new work):** the full set has **multi-chain role groups** S1102
  *filtered out* — antibodies (`H_L_antigen`), TCRs, multimers. Role→chain mapping (and the interacting
  pair for the FoldX arm) must handle these. Reuse RDE's `#Pdb` parsing (group1/group2 → ligand/receptor)
  and validate against ATOM records. Unit-test WT-residue identity per mutation as in
  `interface_xattn/build_masks.py` §2.

## 3. Splits (reuse published fold files where possible)

Ordered comparable→defensible (mirror `retrain_split` but on the full set):
- **3a. RDE by-complex 3-fold** (seed 2022) — `rde/datasets/skempi.py` logic; the leaderboard baseline.
- **3b. PPIformer `iclr24` family-stratified** — the actual JSON in `../reference_ddg_splits/PPIRef/...`
  (proteases/antibodies/misc quarantined). Directly PPIformer-comparable.
- **3c. CD-HIT ≤60% clustered** — ProtBFF recipe; the leakage-controlled sequence-homology split. CD-HIT is a small
  install; cluster chain sequences, GroupKFold on clusters.
- **3d. CATH-superfamily hold-out (813 muts)** — CATH-ddG/USP-ddG; the strictest, most-cited. Reuse
  their published split.
Report **single / multiple / all** separately (RDE convention).

## 4. Arms & PLMs

- **Step 1 (this plan's core): `base` (+ optional `aug`)**, sequence-only. **PLMs:** the 6 already
  wired — Ankh-large, ESM2-3B, ESM C 600M, ProstT5, SaProt, **ESM C 6B**. Priority if compute-limited:
  **ESM C 6B, Ankh, SaProt** (ceiling / workhorse / structure-aware).
- **Step 2: `foldx` / `foldxscalar`** — the FoldX 12-term (+ scalar) `add_scores` arm (= ProtBFF-lite);
  needs §8 FoldX compute. Scalar first (§20.1: scalar ≥ MLP on strong/large sets).
- Also run **MINT** as a sequence-only multi-chain baseline (it published SKEMPI numbers) for a direct
  comparator.

## 5. Embeddings (the step-1 cost)

Per PLM, embed all WT chains + all mutant sequences (unique). ~7,775 sequences.
- **Reference throughput:** ESM2-3B-class base gen was **311 min / 5.2 h for 1,444 seqs on the 40-core
  box** (RESULTS §12) → **~28 h/PLM** at full scale on that box; faster on GPU/MPS. One-time, cached.
- Sub-6B PLMs: Mac MPS or the 40-core CPU box. **ESM C 6B: the RTX-3090 box** (as for S1102, RESULTS §20).
- Multi-point mutants: just longer mutation strings applied to the WT — no special handling.

## 6. Training & eval

- Train the MuLAN **base** head per PLM × split × fold with the existing `mulan-train` (small head over
  cached embeddings — minutes/fold). Isolated `config_skempi_full.sh` (new results/split dirs).
- **Eval:** the `rescore_perstructure` metric core (split-agnostic) + the RDE reference
  (`rde/utils/skempi.py`: `per_complex_corr`, `overall_auroc`, single/multiple modes). Report
  per-structure Spearman/Pearson, overall Pearson/RMSE(+corr), AUROC — **against ProtBFF's ~0.51 P /
  0.48 S leakage-controlled-split anchor** and the sequence-only frontier baselines.

## 7. Multiple-point mutations
MuLAN handles them natively (S1400 in the paper). Include them; report single/multiple separately. For
the FoldX arm, FoldX BuildModel also takes multi-mutation strings — no extra plumbing beyond §2 parsing.

## 8. Compute / hardware plan — the two free Linux boxes (⚠ FoldX is CPU-bound)

**FoldX does not use a GPU** — it's an empirical force field (RepairPDB / BuildModel / AnalyseComplex),
CPU-only. So the GPU doesn't accelerate FoldX; the right host is the **40-core CPU box**, and the GPU
box is for the 6B *embeddings*. Assignment:

| host | spec | use |
|---|---|---|
| **40-core CPU box** | 40 cores @ 2.2 GHz, 62 GiB RAM, no GPU, 3.6 TB disk | **FoldX-at-scale (step 2)**; sub-6B embedding gen |
| **RTX-3090 box** | RTX 3090 24 GB, CUDA 12.4 | **ESM C 6B embeddings**; any GPU PLM gen |
| Mac M4 Pro (MPS) | primary | head training (all arms); sub-6B embeddings |

**FoldX-at-scale runtime estimate (step 2, 40-core box):** RepairPDB once/complex (~45–60 s × ~345 ≈
**~5 h**, one-time) + BuildModel+AnalyseComplex per mutation (~1–2 min/mut on 1 core × ~7,085 ≈
120–240 core-h → **~3–6 h wall on 40 threads**; antibody complexes are larger/slower → budget **~1–2
days** total). ⚠ **RAM/parallelism:** heed the S4169 lesson (RESULTS §20) — big complexes balloon
per-process RAM; cap parallelism per-complex-size (e.g. large antibody complexes at lower PAR) to stay
under 62 GB. **Net: FoldX compute is ~1–2 days on the free 40-core box — cheap; the antibody chain-map
plumbing (§2) is the real time sink, not the FLOPs.**

## 8b. 🚦 Pre-flight gate — RUN BEFORE the multi-day FoldX (step 2)

FoldX is **only** needed for the FoldX arm. Do **not** spend the multi-day FoldX until these pass:
1. **Curation reproduces (cheap):** RDE `load_skempi_entries` yields ~6,900–7,000 entries; all `iclr24`
   fold complexes present; ddG spot-checks match (done for 1CSE). → §2.
2. **Step-1 sequence-only base is competitive first.** Run step 1 (no FoldX). If the sequence-only base
   isn't in range of the sequence frontier baselines, reconsider before FoldX.
3. **PDB coverage exists — ✅ VERIFIED COMPLETE (2026-07-16):** all **345/345** SKEMPI PDB codes are
   already present in `scratch/skempi2/PDBs/` (345 `.pdb` + 345 `.mapping`, byte-identical to the RDE
   download). The earlier "211/345" was a miscount of *FoldX-run* complexes, not raw structures. The
   canonical `SKEMPI2_PDBs.tgz` (RDE `get_skempi_v2.sh`) is archived at `scratch/skempi_full/` for
   provenance. RepairPDB is still needed per-complex at FoldX time, but no structures are missing.
4. **Antibody/multi-chain FoldX mapping validated on ~10 complexes** (H_L_antigen groups S1102 filtered
   out): confirm `AnalyseComplex` gets the right interacting-chain pair. **This is where FoldX-at-scale
   can partially fail** → a non-comparable FoldX arm. Validate the mapping on a sample *before* the full
   run, and report FoldX **coverage %** (frontier FoldX methods report ~100%).

## 8c. FoldX compute — current run state (2026-07-16, live)

The full-SKEMPI FoldX arm splits into **single-point** (build_ddg's native path) and
**multi-point** (comma-joined variants, dropped by `load_skempi()`'s line-88 filter). Both write to
`scratch/foldx_skempi_full/` (gitignored). PDB coverage is complete (§8b.3).

**Single-point remainder — running on the Mac.** 211 complexes were already done; the **112 remaining**
(1,760 unique cleaned single-point muts) are the "remainder". Driver: `build_ddg_remainder.py` (worklist
from `load_skempi()`, resumable, skips the 4 done-dirs). Because the Mac has ~10 idle perf cores while
the retrain MPS sweep is GPU-bound, the 112 are split into **two disjoint pueue workers** (group `foldx`,
shared `results/`, disjoint complex sets → no work-dir races):
- **#76 `foldx_remainder_A6`** — `--jobs 6`, first 66 complexes (1,237 muts).
- **#77 `foldx_remainder_B4`** — `--jobs 4`, last 46 complexes (523 muts).

Output → `scratch/foldx_skempi_full/results/<pdb>.json`. 211 done + these 112 = the **complete
single-point** FoldX set.

**Multi-point — for the 40-core Linux box.** Driver: `build_ddg_multipoint.py` (new; thin driver over
`build_ddg.py`). A variant = one comma-joined `Mutation(s)_cleaned` row; FoldX BuildModel takes
multi-mutation lines natively (`SUB1,SUB2,...;`), one line per variant, keyed by the full variant
string. Each sub-mut is already on a real PDB chain → no role→chain mapping; validation checks every
sub-mut's WT identity against the repaired PDB and keeps a variant only if **all** sub-muts pass.
Results go to a **separate** dir (`results_multipoint/`) so a pdb with both single- and multi-point rows
never collides on `<pdb>.json`. **Dry-run:** 153 complexes / 1,848 unique multi-point variants / 0
cached. Resumable. Launch on Linux:
```bash
cd scratch/foldx_s1102 && export FOLDX_BIN=<path to your FoldX 5 binary> && \
python3 build_ddg_multipoint.py --jobs 8 \
  --results-dir $MULAN_ROOT/scratch/foldx_skempi_full/results_multipoint
```
⚠ The multi-point set **includes the antibody/TCR (H_L_antigen) complexes S1102 filtered out** — so the
§8b.4 chain-mapping caveat is live here: AnalyseComplex chain groups come straight from `#Pdb`; validate
the interacting pair on a sample and **report FoldX coverage %** before trusting this arm.

> **✅ DELIVERED (2026-07-17) — multi-point FoldX pulled back from the Linux box.**
> `foldx_multipoint_results_20260717_102856.tar.gz` (from the shared drop point) unpacked into
> `scratch/foldx_skempi_full/results_multipoint/` (gitignored) + run log `mp_foldx_run.log`.
> **152 complexes / 1,765 multi-point variants / 0 validation failures** (every complex `ok(N/N)` in the
> log; largest: 1JTG 136, 3SGB 88, 1CHO 84, 1PPF 56). Slightly below the 153/1,848 dry-run (a handful of
> variants dropped on the all-sub-muts-must-pass WT check). Schema: `{variants: {"<sub1,sub2,…>": {12-term
> decomposition}}, meta: {n_variants, validation_failed}}` — keyed by the full comma-joined variant string,
> **distinct** from the single-point `results/` `muts` schema, so the two dirs never collide. No local
> collision on pull (dir was new). Schema `{variants:{...}, meta:{...}}`.

> **✅ DELIVERED (2026-07-17) — §8b.4 audit + multi-point add_scores merge (both owed items done).**
> New tracked code under `experiments/full_skempi_seqonly/`:
> - **`audit_multipoint_foldx.py`** → `MULTIPOINT_FOLDX_AUDIT.md`. **Verdict: trustworthy.** Effective
>   coverage **152/152 usable complexes, 100% of usable variants** (the only gap is block-listed 1KBH,
>   unused). The feared antibody/TCR failure did **not** occur — all **60** multi-chain-group complexes
>   are 100% in-group. The one real defect is **grouping ambiguity**: 2 complexes (2C5D, 3SE3) are listed
>   under two `#Pdb` groupings but `build_ddg_multipoint` scored each under the first-seen one, so rows
>   built under the other grouping saw the wrong interface. The authoritative criterion is grouping-match
>   (chain-in-group alone misses 3SE3, whose off-grouping rows mutate the shared chain). Quarantine emitted
>   to `multipoint_foldx_exclude.tsv`: **`2C5D.AB.CD` (19 rows) + `3SE3.B.C` (2 rows) = 21 rows.**
> - **`build_multipoint.py`** — multi-point MuLAN inputs mirroring `build_skempi_full.py`:
>   `scratch/skempi_full/multi_point.tsv` (**1,636 unique rows / 146 complexes**, 99.9% sub-mut-validated)
>   + `multi_point_foldx_key.tsv` (the join sidecar: row → raw SKEMPI variant string) +
>   `wt_sequences_multipoint.fasta` (292 labels). The build's own chain-in-group filter drops the same
>   off-grouping copies, keeping the built set consistent with the trustworthy FoldX values.
> - **`merge_foldx_multipoint.py`** — the variant-string → `add_scores` merge. Joins through the sidecar
>   (MuLAN A/B-contiguous muts ≠ FoldX raw-cleaned keys, so a text match is wrong — only 14% coincidental),
>   honours the quarantine, and standardizes per-fold train-only (clip ±4, unmapped→0) exactly like the
>   single-point `merge_foldx{,_decomposed}.py`. Assemble mode → `multi_point_foldx.tsv`:
>   **1,615/1,615 non-quarantined rows covered (100%)**, 21 quarantined→zeros, and **FoldX Interaction-Energy
>   vs experimental ΔΔG Pearson r = 0.515** (n=1,615) — a real signal in the ProtBFF-anchor range that
>   independently validates the join (a broken join → r≈0). `test_multipoint.py` asserts all of the above.
> **Remaining (downstream, not blocking):** build the multi-point clustered split + run
> `merge_foldx_multipoint.py --split-dir …` to emit standardized fold tsvs; 3Di for any multi-point-only
> labels (SaProt arm); then train the multi-point ±FoldX arm. A clean future FoldX re-run would key
> `build_ddg_multipoint` per (code, grouping) to retire the 21-row quarantine.

## 9. Staging & effort

1. **Step 1 — sequence-only, full SKEMPI, single+multi, 3c/3d splits, per-structure+AUROC.**
   Data + embeddings + reuse splits/metrics. **~1–2 weeks**, no FoldX. *Candid first deliverable.*
2. **Step 2 — FoldX arm** (scalar then 12-term) — **gated on §8b.** + FoldX on the 40-core box (~1–2 days
   compute) + antibody chain-mapping (the slow part). Note the FoldX **energy** channel (our 12-term
   add_scores) is *related but not identical* to ProtBFF's biophysical-feature channel (interface /
   burial / dihedral / SASA / lDDT scaling embeddings) — comparable *method*, not a reproduction.
3. **Step 3 — one antibody OOD set** (AbBiBench CR6261 or RBD-ACE2/antibody DMS) for the design claim.

**Comparability framing (keep candid):** MuLAN is a **sequence-supervised** method — its fair peer group
in the frontier tables is the *sequence* entries (ESM-1v, MSA-Transformer, Tranception, MINT), with the
structure SOTA (RDE / PPIformer / CATH-ddG / USP-ddG / ProtBFF) as the *ceiling*, not the like-for-like
comparator. To match the frontier's **anti-symmetry** convention (ProtBFF/USP-ddG average forward+reverse),
average fwd/reverse at inference — cheap, reuses the reverse-aug machinery.

## 10. Risks / open items
- **Antibody multi-chain role→chain mapping** (§2) — highest-effort item; S1102 dodged it.
- **Label noise / experimental-method heterogeneity** (ProtBFF Fig 2C) — consider filtering ELISA-only
  or weighting; at least report method composition.
- **Context length** — drop the 10 largest (ProtBFF) or per-PLM cap; ESM2/ProSST limits.
- **Coordination** — uses base `mulan-train` (no A1 module) → current committed code; queue behind the
  A1/C1 MPS jobs, but embedding/FoldX gen runs on the **Linux boxes in parallel** with the Mac queue.
- **Scope creep** — keep step 1 sequence-only and shippable before touching FoldX/antibody structures.

## 11. Sequence
1. ⬜ `build_skempi_full.py` (RDE-curation → MuLAN fasta+tsv, single+multi) + WT-residue unit test. `scratch/` only — safe now.
2. ⬜ Split builders / import published fold files (3a–3d); coverage report.
3. ⬜ Embedding gen per PLM (sub-6B on 40-core box/MPS; 6B on RTX-3090). Cache.
4. ⬜ Train base (+aug) per PLM × split × fold; eval via rescore harness + RDE metrics.
5. ⬜ Write the sequence-only frontier table (vs ESM-1v/MSA-T/Tranception/MINT; ProtBFF anchor). **Ship step 1.**
6. ⬜ Step 2: FoldX on the 40-core box → foldx/foldxscalar arms. Step 3: antibody OOD.

**Artifacts (planned):** `build_skempi_full.py`, `test_build_skempi.py`, split importers,
`config_skempi_full.sh`, `gen_emb_*.sh`, `evaluate.py` (wraps the rescore core), this plan.

---

## Appendix A — build details & gotchas (hard-won; read before editing the builders)

Everything below was learned building `build_skempi_full.py` (single-point) and
`build_multipoint.py` / `merge_foldx_multipoint.py` (multi-point). Each item cost real
debugging; the numbers are the empirical tells.

### A.1 The numbering convention (the single biggest trap)
- **SKEMPI's `Mutation(s)_cleaned` position is a 1-based index into the CONTIGUOUS per-chain
  ATOM sequence — NOT the PDB author resseq.** Using the `<CODE>.mapping` author-resseq→seqindex
  lookup for the mutation position gave **37.9%** WT-validation coverage; indexing the contiguous
  sequence **directly** (`num = int(mut[2:-1])`) gave **99.6%**. Verified against known-good S1102
  (1CSE chain I, `LI38S`: `seq[37]=='L'` at direct position 38). ⇒ `pos = group_offset[chain] + num`.
- The `.mapping` file is still needed — but only for the **chain sequences** and their **group
  offsets**, never to translate the mutation number.
- **Insertion codes** (`116A`) make `int(mut[2:-1])` raise → the row/variant is dropped
  (`icode_or_bad_resseq`). Rare; acceptable loss. For 3Di, keep the author-resid as a **string**
  (`line[22:27].strip()`) so it matches the PDB ATOM id incl. icode.

### A.2 Partner assignment & the join-key collisions
- **Fixed g1→partner 'A' (seq1), g2→partner 'B' (seq2).** The intuitive "mutated side = A" is WRONG:
  it makes a g1-side and a g2-side mutation of the same complex BOTH chain 'A' and **collide** on the
  `(complex, mutation)` join key. The mutation's chain letter tells you which side it's on; the
  partner labels are fixed by the `#Pdb` group order.
- **Complex id = the FULL `#Pdb`, dot-joined: `code.g1.g2`** (labels `code.g1.g2_<group>`). Two reasons:
  (1) `rescore._pdb_of` splits on `"_"`, so dots survive → a unique per-`#Pdb` structure id; (2) a
  code-only id would **merge distinct complexes** — 3 codes host two different `#Pdb` complexes each,
  which silently collided join keys until this fix.
- **Replicate measurements** of the same `(complex, mutation)` are **averaged to one row**. The rescore
  harness *asserts* unique `(pdb, mut)` keys, so this dedup is mandatory (single-point merged 757
  replicates; multi-point merged 121).

### A.3 FoldX ↔ MuLAN key mismatch (the multi-point merge's whole reason to exist)
- FoldX JSON is keyed by the **raw SKEMPI cleaned string** (real chain letter + cleaned resnum, e.g.
  `RI48A,RI46A`); our tsv mutation uses the **A/B-partner + contiguous-offset** convention. They match
  on text only **~14%** of the time (coincidences where the real chain is literally 'A'/'B' and the
  group offset is 0). A naive `(pdb, mut)`-keyed merge is therefore silently ~86% wrong.
- **Fix:** `build_multipoint.py` emits a **join sidecar** `multi_point_foldx_key.tsv`
  (`row → code, raw_variant`); `merge_foldx_multipoint.py` joins through it. Note the two pipelines
  interpret the *same* cleaned integer differently and both validate high: `build_*` treats it as a
  contiguous index; FoldX's `repaired_wt_residues` treats it as the author resnum (`int(cols23-26)`).
  Don't "reconcile" them — carry the raw string.
- **Sanity gate that proves the join is right:** FoldX Interaction-Energy vs experimental ΔΔG
  **Pearson r = 0.515** on the covered multi-point rows. A broken join collapses this to ≈0, so the
  correlation is the cheapest end-to-end check.

### A.4 Grouping ambiguity (the §8b.4 defect that isn't antibodies)
- A few complexes appear in SKEMPI under **>1 `#Pdb` grouping** (`2C5D`: `A_C` & `AB_CD`; `3SE3`:
  `B_A` & `B_C`). `build_ddg_multipoint.load_multipoint()` collapses each pdb to the **first-seen**
  grouping and runs ONE AnalyseComplex group-arg for all its variants → rows built under the *other*
  grouping were scored against the **wrong interface**.
- **The authoritative trust criterion is grouping-match, not chain-in-group.** Chain-in-group catches
  `2C5D.AB.CD` (mutates chains outside `A_C`) but **misses `3SE3.B.C`** (its off-grouping rows mutate
  the shared chain 'B', so they *look* in-group). `audit_multipoint_foldx.py` computes the first-seen
  grouping per code and quarantines the rest → `multipoint_foldx_exclude.tsv` (`2C5D.AB.CD` 19 rows +
  `3SE3.B.C` 2 rows). The merge nulls those (standardized 0). `build_multipoint.py`'s own
  chain-in-group filter independently drops the same off-grouping copies, so the built set and the
  trustworthy FoldX values stay consistent.
- **Clean fix (future):** key `build_ddg_multipoint` per `(code, grouping)` instead of per code, then
  the 21-row quarantine disappears.

### A.5 Operational
- **Always `./.venv/bin/python`** for the builders (numpy/pandas); system `python3` has neither.
- The stock `experiments/gen_3di.py` is **hardwired to the S1102 label scheme** (role A/B, single chain,
  author-resnum token placement) — full-SKEMPI needs the custom `gen_3di_skempi_full.py` (multi-chain
  group order, contiguous seq-index placement, icode-aware). Multi-point-only labels (in
  `wt_sequences_multipoint.fasta` but not the single-point fasta) need a 3Di regen before the SaProt arm.
- **Multi-point fasta is kept separate** (`wt_sequences_multipoint.fasta`, 292 labels) so single-point
  outputs are never clobbered — **union** the two fastas for a combined-arm training run.
- All generated data lives in **gitignored `scratch/skempi_full/`**; only the builders, the audit report,
  and the exclude sidecar are tracked.
