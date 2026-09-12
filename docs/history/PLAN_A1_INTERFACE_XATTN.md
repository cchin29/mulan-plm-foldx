# A1 — Interface cross-attention in the MuLAN head

> **Archived — implemented.** The head described here ships as `mulan/interface_xattn.py`, wired
> through `mulan/modules.py` and configured by `interface_xattn` / `xattn_heads` / `xattn_dim` in
> `MulanConfig`; the masks come from `experiments/interface_xattn/build_masks.py`. Retained as the
> design record for why the block is shaped as it is.

Plan of record for the doc's **A1** alternative (`scripts_plots/S1102_CROSSFOLD_MISPREDICT.md`
§Alternatives).

> **⛔ Errata 2026-07-16 — base-model correction (supersedes the "no code written" note above).** The A1
> grid was subsequently built and run (pueue #55, then the fixed #60 — **both now killed**). It ran on
> **`ankh3_large`**, *not* the **Ankh v1 Large** named as the pilot base below (the MuLAN-paper reference,
> tag `ankh`, `cv10_ankh_converge`). The two are not interchangeable — Ankh3 was deprioritised for weak
> CV10 — they only *look* swappable because both are 1536-dim, so the wrong base loaded silently. **No
> executed A1 result carries the reference model yet.** A v1 reference arm is now configured
> (`config_a1_ankh_large.sh` + `lightatt_a1_ankh_large.json`, tag `ankh`); run it per
> `experiments/ANKH_PROVENANCE_AUDIT.md`. Salvaged ankh3 arms (ankh3 datapoint only):
> `scratch/results/a1_ankh3_large/{real,dense}` (10/10 folds), `shuffled` 6/10.

## Why A1, and why now

The MINT/A2 result ([`PLAN_MINT_A2.md`](PLAN_MINT_A2.md), 10-fold CV done) settled the direction:
feeding cross-chain context through the **embedding** does not fix the tail — MINT was the *worst*
arm on the worst-40 (MAE 3.71 vs FoldX 3.38; 2O3B DB75E 5.44→0.20, 1BRS RA81Q 5.42→0.71), because
the revived partner term is ~14× smaller than the primary term and error-uncorrelated even on the
interface subset. The bottleneck is the **head**: `LightAttModel` pools each chain to a vector and
combines them by a **global product + abs-difference** (`modules.py:214-228`), with **no
residue-level cross-chain pairing** — the interface is gone before the regression head sees it. A1
puts the cross-chain interaction *inside the head, at residue resolution*, which is the complementary
bet to A2 (context in the embedding) and C1 (tail-weighted loss, queued).

**Thesis A1 tests:** if the head can attend chain-1 interface residues to their chain-2 partners
*before* pooling, the pooled reps (and therefore the `mut − wt` difference) carry the cross-chain
electrostatic signal the product currently discards — and the tail de-shrinks. A1 runs on ordinary
cached monomer PLM embeddings (no MINT dependency), so it isolates the *head* change.

---

## 0. Isolation (mirror the MINT plan; nothing existing is touched)

| purpose | path | status |
|---|---|---|
| this plan | `docs/history/PLAN_A1_INTERFACE_XATTN.md` | — |
| mask builder + alignment test | `experiments/interface_xattn/build_masks.py`, `test_build_masks.py` | **DONE (step 2), safe during freeze** |
| interface masks (per complex) | `scratch/interface_masks/` | **BUILT: 110/110, heavy-8Å (gitignored)** |
| CV results | `scratch/results/a1_xattn/` | scratch — **NOT** `embedding_sweep_balanced/` |
| model change | `mulan/modules.py`, `mulan/config.py`, `mulan/data.py`, `mulan/train_utils.py` | **deferred — do NOT edit while jobs run** |

Reuse the balanced splits read-only (`experiments/embedding_sweep/splits/balanced_seed42/`) so PCCs
compare directly to base/aug/FoldX/MINT. Train in the main `.venv` (`mulan-train`) — unlike MINT/ESM-C
this is a model change, not an embedding-gen step, so no new venv. **Coordination:** A1 touches the
shared `mulan/` package (the model), so it must land only after the in-flight `mint` CV finishes and
on a branch, to avoid perturbing running/paused `mulan-train` jobs and their checkpoints.

---

## 1. The change, in one diagram

Current head (per fold, per row), `a=`chain-1, `b=`chain-2 per-residue embeddings `[L,D]`:

```
enc = AttentionMeanK          # conv + light-attention POOL -> vector, per sequence, independently
x_wt  = [ enc(a_wt) ⊙ enc(b_wt),  |enc(a_wt) − enc(b_wt)| ]     # pooled product: no residue pairing
x_mut = [ enc(a_mut)⊙ enc(b_mut), |enc(a_mut)− enc(b_mut)| ]
out   = linear( x_mut − x_wt )
```

A1 inserts a **residue-level interface cross-attention** stage *before* the pooling encoder, so the
per-residue reps already carry the partner interaction:

```
a_wt', b_wt'   = XAttn(a_wt,  b_wt,  M)      # M = [L1,L2] interface contact mask (WT geometry)
a_mut', b_mut' = XAttn(a_mut, b_mut, M)      # same M (rigid backbone; mutation doesn't move contacts)
# then the existing enc + product/abs-diff/(mut−wt) run on the primed reps
```

`M` is the binding-interface contact mask between the two chains (below). Because the four inputs
share the complex's interface, `M` is one per complex — same keying as B1's `struct_ctx`.

---

## 2. Interface mask construction (reuse FoldX assets — no new structure prep)

Everything needed is already in the repo from FoldX Stage-2:

- **Bound complex PDBs:** `scratch/foldx_s1102/work/<PDB>/<PDB>.pdb` — the two interacting chains
  (e.g. 1PPF = chains E, I). 110/110 complexes present.
- **Role→chain map + cleaned numbering:** `scratch/skempi_v2.csv` — the SKEMPI pdb field
  (`1CSE_E_I`) gives `(pdb, g1, g2)`; role A = group g1, role B = group g2, exactly as
  `scratch/foldx_s1102/build_ddg.py` (`load_skempi`) reads it.
- **Contacts:** `build_masks.py` uses Biopython `NeighborSearch` directly on the two chains'
  **heavy atoms** (min heavy-atom distance ≤ 8 Å). (Note: `foldenv.contacts`
  exposes `build_contact_index(structure, *, chain_id=…, mode=…)` — a single-chain Cα/Cβ tree, NOT
  the `build_kdtree` named in an earlier draft — so for an interface (two chains, heavy-atom) it was
  cleaner to call NeighborSearch directly than to bend that helper.)

`build_masks.py` (**DONE**, in `experiments/interface_xattn/`): per complex, mark `M[i,j]=True` if
role-A residue i and role-B residue j have any heavy atoms within **8 Å** (knobs: `--mode heavy
--cutoff 8` default, `--mode cb --cb-cutoff 5` for the tighter Cβ variant). Boolean `[L1,L2]` tensor
keyed `{s1}__{s2}.pt` in `scratch/interface_masks/`, index i/j = **FASTA position** (0-based).

**The #1 correctness risk — alignment (PDB numbering ↔ FASTA index) — is now CLOSED empirically.**
A read-only probe over all **1100** single-point rows found: `fasta[pos-1]==wt` for 1100/1100
(FASTA numbering) **and** exactly one chain in the role group has PDB ATOM `resnum==pos` with the WT
aa for 1100/1100 (PDB numbering, unambiguous). So **PDB ATOM resnum = FASTA index + 1** directly — no
alignment step, no offset table. Crucially there are **0 multi-chain role groups** in the S1102 set
(the 1AHW/1DVF-style `AB`/`CD` complexes were filtered out), so a role is always one real chain and
resnums never collide. `resolve_chain` still self-checks each chain's ATOM-vs-FASTA agreement and
aborts a complex at <90% match; `test_build_masks.py` asserts the WT-residue identity for all 1100
rows (the §2 unit test). Result: **110/110 complexes built, 0 errors, no empty interfaces**
(contacts/complex min 111 / median 217 / max 396). No structure was missing, so the dense-M fallback
was not needed.

---

## 3. Module design — two variants, A1a first

### A1a — residual cross-attention (low risk, recommended first)

A small bidirectional cross-attention block, added residually with a **gate init 0** so the model
starts *exactly* at the AA baseline and can only turn the channel on if it helps (the codebase's
consistent design rule — cf. `struct_gate`, `interface_bias.iface_alpha`, all init 0):

```
class InterfaceCrossAttention(nn.Module):   # 1–2 heads, dim = PLM D
    def forward(self, h1, h2, M, pad1, pad2):        # h1[B,L1,D], h2[B,L2,D], M[B,L1,L2] bool
        a1 = mha(query=h1, key=h2, value=h2, attn_mask=~(M  & pad2[:,None,:]))   # chain1←chain2
        a2 = mha(query=h2, key=h1, value=h1, attn_mask=~(M.transpose(1,2) & pad1[:,None,:]))
        h1 = h1 + torch.sigmoid(self.gate) * a1 * pad1[...,None]   # zero padded queries
        h2 = h2 + torch.sigmoid(self.gate) * a2 * pad2[...,None]
        return h1, h2                                              # feed existing enc + combine
```

Rows with no mutation on a chain keep `M` identical between wt and mut passes, so the mut−wt
difference isolates how the mutation reshapes the *interface-primed* representation — the signal the
pooled product cannot form today. Preserves everything that works; the only new params are the
cross-attention block.

### A1b — replace the pooled product (follow-up if A1a underperforms)

Pool the **interface-attended** representation directly into the combined feature (drop the
`enc(1)⊙enc(2)` product entirely), so the head's chain-combination *is* the cross-attention readout
rather than a bilinear pool. Higher ceiling, more surgery, more overfit risk on 1100 rows — hold
until A1a's result is in.

### 3c. Cost / complexity vs. the existing head (added 2026-07-15)

Measured against the actual config (`lightatt_a1_ankh3_large.json`: `xattn_dim=1536`, `xattn_heads=2`)
and the real S1102 chain lengths (from `wt_sequences.fasta`: 220 chains, **median 132, mean 157,
p90 281, max 749** residues). The head *is* the whole trainable model — the PLM is frozen and cached —
so these are the training FLOPs.

| component | params | FLOPs/row (MAC), L≈157 | dominant term |
|---|---|---|---|
| Existing head (`AttentionMeanK` ×4 seqs + combine) | ~3.0 M | ~1.8 GMAC | 6× `Conv1d` over D=1536 input channels |
| **A1 xattn @ `xattn_dim=1536`** (MHA, 2 pairs, bidir) | **~9.4 M** | **~6.2 GMAC** | the 4 `D×D` projections (Q,K,V,O) |

So A1 as configured is **~3.4× the head's FLOPs and ~4× its parameters** (head ~3.0M → ~12.4M).

- **Complexity.** Existing: `O(L·D·H·Σk)` — linear in L, linear in D. A1: `O((L1+L2)·D² + h·L1·L2·D)`
  per pair; the **`D²` projection term dominates** because D=1536 ≫ L≈157. Cost is therefore
  **projection-bound, not attention-bound** — the opposite of long-sequence transformers.
- **The interface mask buys inductive bias, not speed.** The `L1·L2·D` score matrix is ~5% of A1's
  cost (~0.3 GMAC); the 4 `D×D` projections are paid whether or not the mask is sparse. Sparsifying
  attention wouldn't help — the only compute lever is `xattn_dim`.
- **Wall-clock is fine.** 1100 rows × 10 folds × 300 ep at ~6 GMAC/row is seconds–minutes/fold on
  MPS/GPU; A1 roughly triples head compute but the absolute stays small. Memory is a non-issue too —
  padded `L1·L2` tensors are tens of MB at batch 32 even at the max-749 tail (unlike the ESM C 6B
  conv-over-long-seq OOM, RESULTS §20).
- **The real cost is params vs. data.** ~9.4M new params on **1100 training rows** is the genuine
  hazard — hence gate-init-0 and the shuffled-mask control (§5) to catch "just added capacity."
- **Cheap lever (recommended default over full-1536):** bottleneck the attention — project D→d
  (d≈64–128), attend, project back. At d=64: **~0.2M params, ~0.12 GMAC** (2·D·d + 4·d²) — **~44×
  fewer params, ~50× less compute**, i.e. ~5–7% on top of the existing head instead of ~340%. On an
  1100-row set the bottleneck is likely the *better* first cut; `xattn_dim=1536` is the
  high-capacity end of a sweep, not an obvious default. One-line config change.

---

## 4. Threading (follows the `struct_ctx` precedent exactly — deferred until code freeze lifts)

The mask rides the same rails B1 already uses, so the diff is small and localized:

- `config.py`: add `interface_xattn: bool = False`, `xattn_heads: int = 2`, **and `xattn_dim: int = 0`**
  (neutral defaults; old checkpoints unaffected — `from_dict` ignores unknown keys). The extra
  `xattn_dim` is needed because `nn.MultiheadAttention` needs the PLM embed width **D at
  construction**, unlike the lazy `LazyConv1d` encoder that infers D at first forward. Carrying D in
  the config (rather than lazy-building the block in `forward`) keeps the xattn params non-lazy so the
  optimizer — built over `model.parameters()` *before* the dummy forward — captures them, and avoids
  any change to `train.py`'s init order. Since D is PLM-specific, the pilot uses a **dedicated A1
  config JSON per PLM** (e.g. `xattn_dim: 1536` for Ankh-large) instead of the shared
  `lightatt_default_config.json`.
- `data.py`: add `interface_mask_dir`; in `__getitem__` load `{s1}__{s2}.pt` → `"iface_mask"`
  (parallel to `_load_struct_context`, keyed per complex).
- `train_utils.py`: add `interface_mask_dir` field; pass `iface_mask=inputs.get("iface_mask")` in the
  two `model(...)` calls (`:199`, `:317`). **Custom collate:** unlike the fixed-dim `struct_ctx`
  vector, masks are variable `[L1,L2]` → pad to `[maxL1,maxL2]` in-batch (pad=False) consistent with
  how `inputs_embeds` pad to max length. This is the main new plumbing.
- `modules.py`: in `LightAttModel.forward`, if `config.interface_xattn and iface_mask is not None`,
  run `InterfaceCrossAttention` on `(emb[0],emb[1])` and `(emb[2],emb[3])` before the encoder. Derive
  `pad1/pad2` from the existing channel-0 pad convention so padded residues stay at `padding_value`.

---

## 5. Controls / ablations (the interpretability spine — decide these before running)

Aggregate PCC will look nearly flat (the tail is <5% of rows) — as with MINT, **judge on the
worst-40 tail MAE and the named hotspots (1MAH WA276R, 2O3B DB75E/N, 1BRS RA81Q, 1PPF LB18W)**, not
pooled PCC alone.

1. **A1a vs base** (same PLM, same folds) — does interface xattn help the tail at all?
2. **A1a vs A1-dense** (all-pairs cross-attention, `M`≡all-True) — isolates the *interface prior*
   from generic added cross-attention capacity.
3. **A1a vs A1-shuffled** (permute `M`'s columns / random contact set of equal density) — the decisive
   control that the gain is the *true* interface pairing, not extra parameters. If shuffled ≈ real,
   A1 is just capacity.
4. **A1a × C1** (interface head × tail-weighted loss) — the key cell. The MINT diagnosis says the
   head *and* the MSE loss both regress extremes to the mean; A1 alone may still be shrunk by MSE, so
   run A1-alone, C1-alone (already queued), and A1+C1. Expect the tail to move only when both land.
5. *(later)* **A1 on MINT embeddings** — does residue-level complex context (A2) + an interface-aware
   head compound? Only if A1a on monomer embeddings shows life.

PLM for the pilot: one strong, fully-cached sequence-only model — **Ankh-large** (the v1 reference,
tag `ankh`, `cv10_ankh_converge` base — **NOT `ankh3_large`**; see Errata at top) or **ESM2-3B** — both
cheap and both share the hotspot misses. Confirm on **ESM C 6B** (the ceiling) only if the pilot is
positive.

**Watch for gate collapse:** like C3/E1, `sigmoid(gate)` may train toward 0 (model declines the
channel). Under A1-alone that is itself informative — it would say an interface-aware head can't
exploit the signal *under MSE*, pointing all weight to the A1×C1 cell.

---

## 6. Evaluation

Reuse `scripts_plots/analyze_hotspots.py` logic with A1 as an added arm on the identical balanced
folds: pooled OOF PCC, per-fold PCC, and — the metric that matters — worst-40 MAE and per-hotspot
predictions vs base/aug/FoldX/MINT. Re-run `experiments/mint/diagnose_shrinkage.py` shrinkage-slope
on the A1 predictions (slope→1 at |true|≥4 is the win condition, where MINT stalled at 0.80).

---

## 7. Risks / open items

- **Mask↔sequence alignment** (numbering) — highest risk; unit-test against SKEMPI WT residue per §2.
- **Variable-shape mask collation** — needs a custom collate_fn; keep pad positions masked in xattn.
- **Overfitting** on 1100 rows — 1–2 heads, gate init 0, dropout; watch val curves and the shuffled
  control.
- **Interface cutoff sensitivity** — 8 Å vs 5 Å Cβ changes mask density; sweep as a small knob.
- **Interaction with the mut−wt algebra** — A1 primes both wt and mut with the same `M`; confirm (unit
  test) that for a chain-1 mutation the chain-2 primed rep still differs wt vs mut only through the
  attention over the mutated chain, i.e. the intended cross-term is non-zero.
- **Coordination with in-flight jobs** — implement on a branch after the `mint` CV clears; do not edit
  `mulan/` while `mulan-train` jobs are queued/paused.

## 8. Sequence of work & gates

1. ⬜ **Hold** until `mint` group (#53 mono CV → #54 C1 grid) finishes and `default` is safe to
   touch. No `mulan/` edits before then.
2. ✅ **DONE** — `build_masks.py` + `test_build_masks.py` (§2). 110/110 masks built (heavy-8Å) in
   `scratch/interface_masks/`; alignment risk closed (1100/1100). Ran entirely during the freeze
   (writes only `scratch/`, no `mulan/` import). Re-run `--mode cb` if the 5 Å variant is wanted.
3. ✅ **DONE (branch `a1-interface-xattn`)** — `mulan/interface_xattn.py` (A1a, gated init-0
   *additive* scalar = exact no-op at init, cf. C3 `iface_alpha`, NOT the `sigmoid(gate)` of the
   earlier §3 sketch) + wiring: `config.py` (interface_xattn/xattn_heads/xattn_dim), `modules.py`
   (build + forward hook before the encoder), `data.py` (`interface_mask_dir` loader + collator 2D
   mask pad), `train_utils.py` (DatasetArguments field + `iface_mask` in both `model()` calls),
   `scripts/train.py` (from_table passthrough). Validated: `mulan` imports clean (so the paused #42
   is resumable), collator pads `[L1,L2]`→`[B,maxL1,maxL2]` / passes `None` through, disabled config
   ignores the mask, and with `interface_xattn=True` the init-gate=0 forward is an **exact no-op**
   vs the no-mask forward, then moves (no NaN) when the gate is flipped.
4. ⬜ **Next (still needs prep before a run):** a dedicated A1 config JSON (`interface_xattn:true`,
   `xattn_dim` per PLM) + a `config_a1_*.sh`/driver (à la C1) that sets `--interface_mask_dir
   scratch/interface_masks` and writes to `scratch/results/a1_xattn/`. Then pilot A1a on Ankh-large,
   10 balanced folds. Controls (§5) also need `--dense`/`--shuffle` mask variants from `build_masks.py`.
5. ⬜ Gate on the **worst-40 tail**, then run controls 2–4 (§5). A1×C1 is the decision cell.
6. ⬜ If positive: confirm on ESM C 6B; consider A1b and A1-on-MINT. If gate collapses / shuffled ≈
   real: A1 is not the lever, fold the finding back into the doc and lean on C1.

**Note (step 2 is safe now):** the mask builder and its unit test write only to `scratch/` and read
the FoldX PDBs + `skempi_v2.csv`; they import nothing from a running training job, so they can be
prototyped during the freeze. Everything under §3–4 waits.
