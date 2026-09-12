# ESM C 6B on S1102, against Ankh

> Consolidated entry point for all results: [`../docs/RESULTS.md`](../docs/RESULTS.md). This is
> a deep-dive sub-doc.

**Headline:** ESM Cambrian (ESM C) 6B, run sequence-only through MuLAN, is the
**best PLM tested** for single-mutation ΔΔG on S1102 — **10-fold CV PCC 0.859 ±
0.069**, beating Ankh-large (0.832) by a paired **+0.0267 PCC** (t = 4.15, p ≈
0.0025, df 9), winning **9/10 folds** (sign test p ≈ 0.02). It also wins every
secondary metric (RMSE 1.176 vs 1.310, MAE 0.853 vs 0.944, Spearman 0.777 vs
0.731) and gets the closest any from-scratch config has to the paper's reference
0.868 (MuLAN-Ankh-large 10-fold CV).

Date: 2026-06-30/07-01. Raw artifacts in the gitignored `scratch/` tree.

## Results (S1102 mutation-based 10-fold CV, same config as the Ankh/ProstT5 cv10)

| PLM | Embed dim | Test PCC (mean ± std) | RMSE | MAE | SCC |
|---|---|---|---|---|---|
| **ESM C 6B** | 2560 | **0.859 ± 0.069** | **1.176** | **0.853** | **0.777** |
| Ankh-large | 1536 | 0.832 ± 0.057 | 1.310 | 0.944 | 0.731 |
| ProstT5 (AA) | 1024 | 0.805 ± 0.055 | 1.426 | 1.023 | 0.674 |

Per-fold PCC (same 10 folds, `split_data(num_folds=10, random_state=42)`):

| fold | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| ESM C 6B | 0.672 | 0.923 | 0.891 | 0.851 | 0.883 | 0.932 | 0.895 | 0.837 | 0.853 | 0.850 |
| Ankh | 0.691 | 0.892 | 0.859 | 0.812 | 0.863 | 0.894 | 0.877 | 0.827 | 0.806 | 0.802 |

ESM C 6B wins every fold except fold 0 (the hardest fold for both PLMs). The
margin over Ankh (+0.027) is essentially identical to Ankh's own margin over
ProstT5 — ESM C 6B sits one clear step above Ankh, as Ankh sits above ProstT5.

## Why this matters

Confirms the pre-registered forecast (~55% that the 6B beats Ankh; the free 600M
was predicted to only tie ESM-2 and sit below Ankh, so it was skipped). The win
is driven by **model scale + newer pretraining**, not by any structure channel:
this is pure sequence-mode ESM C. It's the first result to show a modern PLM
clearing Ankh in MuLAN's Light-Attention head — the prior negative results
(ProstT5 layer-swaps, 3Di structure fusions) were all about *representation
choice within a fixed model*, whereas this is a genuinely stronger base model.

## Method — loading the 6B (non-obvious)

ESM C is **not in any released `transformers`** (the HF weights declare
`model_type: "esmc"`, arch `ESMCForMaskedLM`, but no release — 4.48/4.57/5.12 —
registers it; the config's `transformers_version: 4.57.6` is EvolutionaryScale's
own build string). The EvolutionaryScale `esm` SDK's local registry only ships
the 300M/600M (6B was Forge-API-gated). So neither the documented `AutoModel`
path nor the SDK's `from_pretrained` loads the local 6B directly.

What works: the downloaded `EvolutionaryScale/esmc-6b-2024-12` safetensors are in
**native esm-package format** (keys prefixed `esmc.`), and after stripping that
prefix they map **exactly** (808/808 tensors, 0 shape mismatches) onto a
hand-built `esm.models.esmc.ESMC(d_model=2560, n_heads=40, n_layers=80,
use_flash_attn=False)` module. The only non-overlap is the unused masked-LM head
(`sequence_head.*`), loaded `strict=False`. **No second download** of the
"HF-compatible" biohub re-host was needed.

Gotchas hit and fixed:
- **NaN embeddings** from building on `meta` + `to_empty` — that leaves non-saved
  buffers (rotary frequencies etc.) uninitialized. Fix: build with **real init**,
  then overwrite params from the shards (shard-by-shard `load_state_dict`,
  peak RAM ~29 GB, fits the 62 GB box).
- **Broken `transformers` import** in the ESM C venv — `pip install esm` pulled a
  torchvision incompatible with torch 2.12 (`operator torchvision::nms does not
  exist`), which breaks `transformers`' lazy import. Fix: the embedding generator
  is **fully standalone** (imports only `esm` + `torch`, inlines mulan's
  `parse_fasta`/`parse_mutations`/id-label logic so cache filenames match), and
  torchvision was removed.

Per-residue embedding = final-layer hidden states (`ESMCOutput.embeddings`),
BOS/EOS stripped → `[L, 2560]`, saved one `{id_}.pt` per sequence, named
identically to mulan's `MulanDataset` enumeration.

## Environment split (keeps the established pipeline intact)

- **Embedding generation** runs in an isolated venv `~/.venvs/mulan-esmc`
  (torch 2.12 + `esm` 3.2.3, transformers unused/broken there) — writes the cache
  and exits.
- **cv10 training** runs in the original `~/.venvs/mulan` (transformers 4.44,
  untouched) **read-only from cache**: `data.py` only loads a PLM when embeddings
  are missing (`data.py:73`), so with all 1444 pre-cached, no ESM C code is
  touched during training and the Ankh/ProstT5 results stay reproducible.

## Runtime & footprint (CPU-only)

**Machine:** 2× Intel Xeon Silver 4114 (40 cores @ 2.20 GHz), 62 GiB RAM, no GPU,
3.6 TB disk (3.3 TB free). fp32 throughout.

| Resource | ESM C 6B |
|---|---|
| Weights download (safetensors, fp32) | 25.4 GB (6 shards) |
| Encoder params | 6 B (d_model 2560, 80 layers, 40 heads) |
| Embedding generation (1444 seqs) | 69.7 min (~2.9 s/seq, 40 threads) |
| Peak RAM, model load + inference | ~29 GB (of 62 GB) |
| cv10 wall-clock (10 folds, MAXPAR=4, 10 thr/job) | ~7h58m (23:49→07:47) |
| Embedding cache on disk | 1.7 GB (`scratch/embeddings_esmc6b/`, 1444 × 2560-d) |
| Trained `model.ckpt` (per fold) | 19 MB |

## Reproduce

- Weights: `hf download EvolutionaryScale/esmc-6b-2024-12`
- Loader: `scratch/esmc6b_lib.py` (build 6B, load local shards)
- Embeddings: `~/.venvs/mulan-esmc` → `python scratch/esmc6b_gen.py`
- cv10: `~/.venvs/mulan` → `scratch/cv10_esmc6b_driver.sh` (read-only from cache)
- Aggregate: `scratch/esmc6b_aggregate.py`; orchestration `scratch/esmc6b_orchestrate.sh`
- Code wiring: `esmc_6b` in `mulan/constants.py`; ESM C branch in
  `mulan/utils.py` (`load_pretrained_plm` + `embed_sequence`, both additive).

## Open follow-ups

- **10-fold CV on the standard SKEMPI benchmarks** (S1131/S4169/S2003) to confirm
  the win generalizes beyond S1102, as was done for Ankh-vs-ProstT5 (§10).
- **Tier-1 augmentation** with ESM C (helped Ankh +0.006 under CV).
- ESM C 6B (0.859) vs the still-pending **AIDO.Protein-16B** (§8) — the two live
  bets to beat Ankh; ESM C got there first.
