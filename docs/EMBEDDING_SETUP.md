# Embedding & Sweep Setup — HOWTO

Canonical guide for generating PLM embeddings and running the base / +aug / variation
sweeps across **all** models, so we stop re-discovering the per-model quirks. Pairs with
`CODEBASE_OVERVIEW.md`, `PLM_BACKEND.md` and `RESULTS.md`.

> **Mental model.** Embeddings are pre-generated **once per PLM** into a cache dir of
> `{id}.pt` per-residue tensors. Training then reads them **read-only** — if the cache is
> complete, `mulan-train` loads **no PLM at all** (it only loads one to fill *missing* ids).
> A "run" = (PLM cache) × (split partition) × (budget) × (optional variation channel).

---

## 0. Environments (venvs)

**Generation and training are separate jobs with incompatible dependency sets, and they are
meant to stay separate.** This is the single most important operational fact in this document,
and it is not a workaround — it is the architecture:

- **Generation** needs the EvolutionaryScale `esm` SDK, which pins `transformers<4.48.2`.
- **Training** needs neither. It reads `{id}.pt` tensors from a cache directory and **never
  imports `esm`**. The training environments do not have it installed at all.

Because the two worlds never have to coexist in one interpreter, the incompatible pins never
collide. That is also why **no single environment, and no single pip extra, can express this** —
`transformers` cannot simultaneously satisfy the SDK's ceiling and a training pin above it.

### What actually ran

Every ESM-C number in this repository was produced by these environments. Versions are recorded
because a `transformers` change can alter what an embedding *means* without any error:

| environment | machine | `esm` | `transformers` | `torch` | role |
|---|---|---|---|---|---|
| `~/miniforge3/envs/esmgen` | Linux, RTX 3090 | 3.2.1.post1 | 4.48.1 | 2.6.0+cu124 | **ESM-C 6B embedding generation** |
| `~/.venvs/mulan-esmc` | Linux, CPU | 3.2.3 | 4.48.1 | 2.12.1+cpu | ESM-C embedding generation, scoring, plotting |
| `~/miniforge3/envs/mulan` | Linux, RTX 3090 | *not installed* | 4.44.2 | 2.6.0+cu124 | MuLAN training |
| `~/.venvs/mulan`, `~/.venvs/mulan-aido` | Linux, CPU | *not installed* | 4.44.2 | 2.12.1+cpu | MuLAN training |
| `.venv` | macOS | *not installed* | 4.44 | — | training; generic + SaProt embedding generation |
| `.venv-esmc` | macOS | SDK | — | — | ESM-C 600M / 6B generation only |
| `.venv-structctx` | macOS | — | — | — | structural context (DSSP, esmc_600m embedder) |

**`transformers` 4.48.1 is not an arbitrary choice** — it is the highest version `esm` 3.2.3's
`transformers<4.48.2` ceiling permits. Treat it as the pinned generation version, not a floor to
drift upward from.

### Building the generation environment

```bash
python -m venv ~/.venvs/mulan-esmc
~/.venvs/mulan-esmc/bin/pip install -r requirements-esmc.txt
```

[`requirements-esmc.txt`](../requirements-esmc.txt) pins the SDK and `transformers` exactly. Do
**not** install this repository's own package into that environment, and do not install `esm`
into a training environment — the point is that they stay apart.

> **There is no `[esmc]` pip extra, deliberately.** One briefly existed in `pyproject.toml`
> requiring `transformers>=4.57`; it was added during packaging, never produced any number here,
> and could not resolve — the SDK caps `transformers` nine minor versions below that floor. It
> was removed rather than repaired, because a satisfiable version of it would still be wrong: it
> would merge two environments that are correct only while separate.

⚠️ **`../.venv` does NOT exist on this checkout.** Use repo-local `.venv/bin/python` for
everything, including `scripts_plots/*` figures. (Older notes citing `../.venv` are stale.)

---

## 1. Per-PLM embedding generation

`--plm_model_name` is the training tag; the encoder it maps to lives in
`mulan/constants.py::PLM_ENCODERS`. **ESM C 600M and 6B are registered there**, but their
embeddings are still generated out-of-band in a separate environment and read from cache at
train time, because the SDK pins `transformers` below what training uses.

| model | `--plm_model_name` | cache dir (sweep) | generator | venv | dim |
|---|---|---|---|---|---|
| Ankh-large | `ankh` | `scratch/embeddings` | `experiments/gen_emb_generic.py` | `.venv` | 1536 |
| ProstT5 | `prostt5` | `scratch/embeddings_prostt5` | `gen_emb_generic.py` | `.venv` | 1024 |
| ESM2-3B | `esm` | `scratch/emb_esm2_aug` | `gen_emb_generic.py` | `.venv` | 2560 |
| SaProt-650M | `saprot` | `scratch/emb_saprot` (+ `_seqonly`) | `experiments/gen_saprot_emb.py` | `.venv` | 1280 |
| ESM C 600M | `esmc_600m` | `scratch/embeddings_esmc600m` | `experiments/embedding_sweep/gen_esmc600m_emb.py` | **`.venv-esmc`** | 1152 |
| ESM C 6B | `esmc_6b` | (Linux) | `scratch/esmc6b_gen.py` (hand-built from snapshot) | `.venv-esmc`, transformers ≥4.57 | 2560 |
| AIDO.Protein-16B | `aido` | (Linux, offline) | offline embeddings, CPU bf16 | Linux | 2304 |
| Ankh3-large / -xl | `ankh3_large` / `ankh3_xl` | MPS caches | `gen_emb_generic.py` | `.venv` | 1536 / 2560 |

**Cache-dir names are NOT uniform** — they're hardcoded in each driver's `emb_of()`.
Notably `esm2 → emb_esm2_aug` (the empty `embeddings_esm2` is a decoy) and `ankh → embeddings`
(bare). Do **not** assume `embeddings_<tag>`.

### Generic PLMs (ESM2 / Ankh / ProstT5 / Ankh3)
```
.venv/bin/python experiments/gen_emb_generic.py <plm> <emb_dir> <table.tsv> <wt.fasta>
# e.g. ankh S1102:
.venv/bin/python experiments/gen_emb_generic.py ankh scratch/embeddings \
    experiments/embedding_sweep/data/S1102_filtered.tsv \
    experiments/embedding_sweep/data/wt_sequences.fasta
```
Generates only missing ids (idempotent). MPS by default; `MULAN_FORCE_CPU=1` for CPU.

### SaProt (structure-aware — needs 3Di)
SaProt embeds a **structure-aware (SA) token per residue = AA(upper)+3Di(lower)**. The 3Di
channel is the **WT** structure (mini3di via `experiments/gen_3di.py`); **a mutant reuses its
parent WT chain's 3Di** (single-point ⇒ backbone ~unchanged) — the SA token still flips
because its AA half changes.
```
MULAN=$PWD PYTHONPATH=$PWD .venv/bin/python experiments/gen_saprot_emb.py \
  --tables <S1102.tsv> --fasta <wt.fasta> --tdi scratch/3di/wt_3di.fasta \
  --real-dir scratch/emb_saprot --seqonly-dir scratch/emb_saprot_seqonly \
  --device cpu --mode both
```
`--mode`: `real` (WT 3Di) → `emb_saprot/`, `seqonly` (structure half = `#`) → `emb_saprot_seqonly/`,
`both` = write both in one model load (the structure ablation). CPU keeps MPS free.

### ESM C 600M (SDK-only, `.venv-esmc`)
```
ESMC_EMB=scratch/embeddings_esmc600m \
ESMC_WT=<wt.fasta> ESMC_TBL=<table.tsv> \
  .venv-esmc/bin/python experiments/embedding_sweep/gen_esmc600m_emb.py
```
Skip-existing, MPS bf16 (~1.4 s/seq), cast to fp32. Ids enumerated **exactly** as
`mulan.data.MulanDataset` so training reads them with no PLM load.

---

## 2. BASE runs (forward mutations)

**Splits** (`gen_splits.py`, `ES_CONFIG`-driven), tracked:
`experiments/embedding_sweep/splits/{paper_seed42,balanced_seed42}/fold_{0..9}/`.

- **Generic 4 models:** `bash scratch/embsweep_base_driver.sh {ankh|prostt5|esm2|saprot}`
  (honors `ES_CONFIG`; reuses the read-only cache; resumable per-fold).
- **ESM C 600M:** `bash scratch/embsweep_esmc600m_driver.sh` ← **use this, not the generic
  driver** (see Gotcha #1). Trains from the SDK cache via `--plm_model_name esmc_600m`.
- **50ep/patience10 from-scratch twins:** `scratch/cv10_<plm>_driver.sh` on `scratch/cv10_splits`.

Config: `lightatt_default_config.json`, batch 32, lr 5e-4; epochs/patience from `ES_CONFIG`.

---

## 3. +AUG runs (Tier-1 reverse + identity)

`experiments/augment.py` builds an augmented **train** table (val/test untouched, so the
metric stays comparable): (1) **reverse mutation** ΔΔG(wt→mut) = −ΔΔG(mut→wt); (2) **identity
anchors** wt→wt = 0. The reverse row adds **new reverse-mutant ids** (double notation, e.g.
`1A22_A_CA171A_AA171C`) that **need their own embeddings** — the base cache alone can miss them.

**Build aug split tables first** (PLM-independent, deterministic):
```
bash scratch/build_balanced_aug_splits.sh    # -> scratch/embsweep_aug_splits_balanced/
#   fold_{0..9}/{aug_train.tsv,aug.fasta}  +  union_train.tsv / union.fasta
```
The **union = augment(FULL dataset)** — partition-independent, so one union's embeddings
cover every fold. (Paper splitter → `scratch/embsweep_aug_splits/`; 50ep cv10 →
`scratch/cv10_aug_splits/`.)

**Run:**
- **Generic 4 models:** `bash scratch/embsweep_aug_driver.sh {ankh|prostt5|esm2|saprot}`.
  Its **`prewarm`** step loads the PLM once (MPS) to fill aug ids missing from the cache,
  then trains folds race-free. (2026-07-12: saprot/esm2 fully cached; ankh/prostt5 each
  missed 366 reverse-mutant tensors — their 50ep aug ran on Linux.)
- **ESM C 600M:** `bash scratch/embsweep_aug_esmc600m_driver.sh` ← dedicated driver. The
  generic `prewarm` can't load `esmc_600m` in `.venv`, so Stage A generates any missing
  aug-union embeddings in `.venv-esmc` (no-op when cached — the balanced aug union's 2725
  ids were already present from the 50ep aug run), Stage B trains from cache.

---

## 4. VARIATIONS (extra input channels / structure ablations)

These change the **splits/config/flags**, usually **not** the base embedding cache.

### 4a. FoldX `add_scores` channel (zero-shot binding ΔΔG)  — `experiments/foldx_s1102/`
Feed FoldX binding ΔΔG as an extra input alongside the embedding. **Embeddings unchanged**
(Ankh S1102 cache); the FoldX signal rides in the split tables + a wider head.

Pipeline:
1. **FoldX energies.** One JSON per complex in skempi-foldx's results format; `data/foldx/results_sp/`
   ships them for every SKEMPI complex. (The §18 runs used an earlier, S1102-only campaign of the
   same pipeline, ~110 complexes, whose energies differ from the shipped store because each
   complex ran a different mutation list — see `docs/FOLDX.md` on determinism. That campaign and
   its runner, `build_ddg.py`, do not ship; the shipped store is the one to build from now.)
2. **Stage 1 (scalar):** `python experiments/foldx_s1102/merge_foldx.py --src <splits> --out ankh`
   appends the *total* ΔΔG as a **5th column**, **standardized per-fold on train-only mean/std
   (leakage-free), clipped ±4, missing=0** → `./splits_ankh_foldx/` (defaults: the S1102 paper
   split and `data/foldx/results_sp`; `--out-dir` moves the output). Train with
   `models/config/lightatt_addscores_config.json` (`add_scores:true`) + **`--add_zs_scores True`**.
3. **Stage 2 (decomposed MLP):** `python experiments/foldx_s1102/merge_foldx_decomposed.py` writes
   the 12-term vector → `./splits_ankh_foldxdec/`; train with
   `models/config/lightatt_addscores_mlp_config.json` (`zs_mlp:true, zs_input_dim:12,
   zs_mlp_hidden:16`). The paired-arm CV drivers the §18 runs used do not ship; `scripts/train.py`
   with those configs is the same training.

⚠️ **Flag/field name mismatch:** the CLI flag is **`--add_zs_scores`** (→ `data_args`), but the
JSON config key is **`add_scores`** (+ `zs_mlp`). Both are required together. Aggregate with
`aggregate_foldx.py` / `aggregate_foldx_mlp.py`. Findings: `RESULTS.md` §18/§18.2,
`history/FINDINGS_FOLDX_MUTANT_3DI.md` (Stage-1 ceiling ≈ +0.017 on Ankh, a floor-raiser).

### 4b. SaProt structure ablation — real 3Di vs seq-only
Two caches, same id coverage: `emb_saprot` (real WT 3Di) vs `emb_saprot_seqonly` (structure
half masked `#`). Same driver, swap `--embeddings_dir`. Isolates the structural contribution.

### 4c. ProstT5 AA vs structure tokens
ProstT5 is bilingual (AA ↔ 3Di). The suite uses **AA mode** ("ProstT5 (AA)" in the plots). The
3Di-token structure variant is scoped in `history/PLAN_PROSTT5_STRUCTURE_v2.md` /
`history/PROSTT5_STRUCTURE_OPTIONS.md`.

### 4d. SKEMPI benchmarks (S1131 / S2003 / S4169) — distinct from S1102 CV
Live under `scratch/benchmarks/<SET>/` with **pre-defined `cv_splits`** (NOT the
balanced/random S1102 splitter — "balanced" is an S1102-CV knob only). WT fasta
`scratch/benchmarks/wt_sequences.fasta`; `S4169.tsv` is the **superset** table (covers
S1131/S2003), so one embedding pass fills all three. Base gen: `gen_emb_generic.py` (standard
PLMs) or the SDK bench-gen scripts (`esmc600m_bench_gen.sh`, `esmc6b_bench_gen.sh`). Aug bench:
`scratch/mac_aug_bench_fill.sh {esm|ankh}`.

### 4e. Structural-context module (Phase-2 §4)
Separate project ([foldenv](https://github.com/cchin29/foldenv), formerly `mulan/structural_context/`; `pip install ".[struct]"`): DSSP + per-protein context with **L2 disk
persistence** (`persist.py`, `MULAN_*` cache). Independent of the base embedding cache above;
the design notes for it were withdrawn before publication and are not recoverable from this
tree; `docs/history/README.md` records which documents that applies to.

---

## 5. Splitters, budgets & aggregation

- **Splitter** (`mulan/data.py`): `balanced` (fork default, equal folds) vs `random` (paper
  `rng.integers`). Set via `ES_SPLIT_METHOD` in the `ES_CONFIG`.
- **Budget:** 50ep/patience10 (from-scratch suite) vs 300ep/patience30 (paper-faithful sweep).
- **`ES_CONFIG` selects everything else** and keeps result trees separate so the paper anchor
  survives: `config.sh` (rng) → `results/embedding_sweep/`, `results/`; `config_balanced.sh`
  → `results/embedding_sweep_balanced/`, `results_balanced/`, `embsweep_aug_splits_balanced/`.

**Aggregate → tracked JSON/summary:**
```
ES_CONFIG=experiments/embedding_sweep/config_balanced.sh \
  .venv/bin/python experiments/embedding_sweep/aggregate.py   # -> <tag>.json + summary.md
```

**Plots (three protocol "sets"):** each figure script keys off `MULAN_PLOT_SERIES`:
`50ep` (50/10 balanced) · `300ep_balanced` (300/30 balanced) · `300ep` (300/30 rng.integers).
```
MULAN_PLOT_SERIES=300ep_balanced .venv/bin/python scripts_plots/plot_ddg_scaling.py
MULAN_PLOT_SERIES=300ep_balanced .venv/bin/python scripts_plots/plot_variability.py
```
Scaling reads a mean/std CSV (`ddg_scaling_data_<series>.csv`); variability reads a **per-fold**
CSV (`benchmarks_folds_<series>.csv`, cols `model,aug,dataset,fold,pcc`).

---

## 6. GOTCHAS (the re-discovery tax)

1. **Generic drivers silently no-op on `esmc600m`.** `embsweep_base_driver.sh` /
   `embsweep_aug_driver.sh` `plm_of`/`emb_of` only know `ankh|prostt5|esm2|saprot`; `esmc600m`
   → `!! unknown tag`, the task **exits in ~1 s reporting pueue "Success" with ZERO folds**.
   This ate the balanced base (#19) and aug (#24). Use the dedicated `embsweep_esmc600m_driver.sh`
   / `embsweep_aug_esmc600m_driver.sh`.
2. **ESM C 600M/6B are registered in `PLM_ENCODERS`**, but generate in `.venv-esmc`: the SDK
   and training cannot share an interpreter. Training only reads the cache.
3. **Non-uniform cache dirs** — `esm2→emb_esm2_aug`, `ankh→embeddings`, `saprot→emb_saprot`.
4. **SaProt needs 3Di** (`gen_3di.py`); mutants reuse parent WT 3Di. Two caches for the ablation.
5. **Aug adds reverse-mutant ids** (double-notation) that need embeddings; the aug **union =
   augment(full dataset)**, partition-independent. Always coverage-check before an aug run.
6. **FoldX flag vs config key:** CLI `--add_zs_scores True` **+** config `add_scores`/`zs_mlp` —
   both needed; the split tables must carry the score column(s).
7. **`../.venv` is absent** — use `.venv` (repo-local) everywhere, plots included.
8. **Benchmarks ≠ S1102 CV** — SKEMPI uses fixed `cv_splits`; the balanced/random knob does
   not apply.

---

## 7. CHECKS (run before/after every sweep)

**Coverage check (no model load)** — count aug/benchmark ids missing from a cache:
```
# generic (ESM2/Ankh/ProstT5): enumerate via MulanDataset, count missing .pt
.venv/bin/python - <<'PY'
import os; from mulan.data import MulanDataset
tbl,wt,emb,plm="<table.tsv>","<wt.fasta>","<emb_dir>","<plm>"
ds=MulanDataset.from_table(tbl,wt,emb,plm)   # generates missing; run gen separately to just count
print(len(ds),"rows")
PY
# esmc600m: import gen_esmc600m_emb.enumerate_ids() in .venv-esmc and diff against the cache
```

**Detect the silent no-op** — a 1-second pueue "Success" is suspect; verify fold count:
```
ls $ES_RESULTS_DIR/<tag>/fold_*/training_run/all_results.json | wc -l   # expect 10
pueue log <id> | grep -i "unknown tag"                                   # smoking gun
```

**Other checks:**
- Coverage + dim: the generators now fail loudly on a gap — `gen_emb_generic.py` asserts every
  dataset id produced a `.pt` and spot-checks tensors are finite `[L, dim]` with one consistent
  width; `gen_esmc600m_emb.py`/`gen_struct_embeddings.py`/`gen_layer_embeddings.py` assert
  per-embedding shape; `gen_saprot_emb.py` has an AA↔3Di length/desync guard.
- CV speed: set `MULAN_EMB_CACHE=1` (in-RAM cache) to kill the 10CV `torch.load`
  re-deserialization bottleneck — see `CV_SPEEDUP.md`.
- After results land: re-run `aggregate.py` (right `ES_CONFIG`), rebuild the plot CSVs, and
  regenerate the affected `MULAN_PLOT_SERIES`.
