# AIDO.Protein as a MuLAN embedding — hypothesis & plan

> **Archived — executed.** AIDO.Protein-16B was integrated and trained; the result is
> [`../RESULTS.md`](../RESULTS.md) §14, where the largest encoder ties Ankh — scale alone does not
> win. The backbone is registered at `mulan/plm/registry.py` with a backend in
> `mulan/plm/backends/aido.py`, not by the `constants.py` edit this plan proposes.

Forward-looking plan for swapping **AIDO.Protein** in as the sequence embedder for
MuLAN ΔΔG prediction. **Status: weights staged, compute pending** — both models
downloaded to the local HF cache (2026-06-21); still pending a usable GPU (only a
2 GB Quadro P400 here). This captures the hypothesis, the integration design, and
concrete next steps so the work can start the moment a GPU is available.

### Downloaded (2026-06-21, local HF cache `~/.cache/huggingface/hub`)
- **`genbio-ai/AIDO.Protein-16B`** — 13 `pytorch_model-*.bin` shards, **~60–64 GB
  on disk** (repo is **fp32**, not the ~32 GB bf16 the older estimate assumed; load
  in bf16/fp16 at inference to halve memory). Public / non-gated, no HF token needed.
- **`genbio-ai/AIDO.Protein-RAG-3B`** — 3 shards, **~11 GB**. Smaller variant for
  CPU load/smoke-testing; note it is **RAG** (expects an MSA of homologs per query),
  so *not* a clean single-sequence drop-in for MuLAN — use only to exercise the API.

### Config confirmed (16B `config.json`)
`model_type: fm4bio`, hidden_size **2304**, 36 layers / 36 heads, ctx **2048**,
SwiGLU + RMSNorm + RoPE, **MoE** (`num_experts: 8`, `experts_per_token: 2`),
`torch_dtype: float32`, `use_lm_head: false`. **Resolves the open question below:**
the repo ships an `auto_map` (`AutoConfig`/`AutoModel`/`AutoModelForCausalLM` →
`modeling_fm4bio`), so a **`transformers`-native `AutoModel(..., trust_remote_code=True)`
path exists** — ModelGenerator is *not* mandatory, and `output_hidden_states` is
supported (layer probe for H3 is reachable). Approach A is feasible.

Companion to `RESULTS_PROSTT5.md` and `../docs/history/PLM_COMPARISON_S1102.md` (current
baselines: Ankh 0.757, ProstT5 0.740 on the S1102 single split).

## What AIDO.Protein is

[AIDO.Protein-16B](https://huggingface.co/genbio-ai/AIDO.Protein-16B) (GenBio AI):
- Encoder-only, **Mixture-of-Experts**, **16B parameters**, masked-LM pretraining.
- Hidden size **2304**, 36 layers, 36 heads, vocab 44, context length **2048**.
- Loads via GenBio's **ModelGenerator** package (`from modelgenerator.tasks import
  Embed`), *not* stock `transformers.AutoModel`. Trained on 256×H100.
- Smaller relatives exist but aren't clean drop-ins: **AIDO.Protein-RAG-3B** is
  retrieval-augmented (needs an MSA of homologs per query), not a single-sequence
  embedder.

## Hypothesis

**H1 (primary).** AIDO.Protein-16B's larger-scale MoE pretraining yields richer
per-residue embeddings than Ankh-large (1.2B) or ProstT5, raising MuLAN's S1102
test PCC above the current best (Ankh 0.757; Ankh+aug 0.769).

**H2.** Because MuLAN is frozen-PLM + a small trainable head, embedding *quality*
is the dominant lever — so a stronger PLM should help even with the same head and
data. (Consistent with our probe finding that PLM choice tracks downstream PCC.)

**H3 (layer behavior).** Like Ankh (and unlike ProstT5), AIDO's MLM objective
should keep its final layer near-optimal for the local ΔΔG signal — so the
default `last_hidden_state` is likely the right layer, and a mid-layer swap is
*not* expected to help. A layer probe (below) will test this cheaply.

**Null / risk.** With only 769 training mutations and a frozen PLM, a 2304-d
embedding may not beat 1536-d Ankh — the head may already be the bottleneck, or
the extra width may overfit. This is exactly what the experiment would settle.

## Why it can't run here yet

This box is CPU-only, 62 GB RAM. AIDO-16B needs ~32 GB (bf16) just for weights,
MoE/GPU kernels, and would take ~10–60 s/sequence on CPU → many hours/days for the
~1444 S1102 sequences. **GPU required.** (See `../docs/history/PLM_COMPARISON_S1102.md` runtime
table for the CPU baselines this is extrapolated from.)

## Integration design (two approaches)

**Approach B — offline embeddings (recommended).** MuLAN only consumes cached
`[L, D]` `.pt` tensors, so AIDO never has to live inside MuLAN:
1. `experiments/gen_layer_embeddings.py::build_id_seqs` already emits the exact
   id→sequence list MuLAN needs (wt + mutant chains) for the split.
2. On the GPU, a ~30-line ModelGenerator `Embed` script writes each as
   `<id>.pt` of shape `[L, 2304]` (strip CLS/SEP so length == residue count).
3. Copy the dir back; `mulan-train --embeddings_dir <aido_dir> --plm_model_name aido`
   (no AIDO load needed — all embeddings present). `LazyConv1d` adapts to 2304.
- Cleanest: avoids adding the `modelgenerator` dependency (which may pin a
  different `torch`/`transformers` than this venv) to the MuLAN package itself.

**Approach A — full in-MuLAN integration.** For native `plm-embed aido`:
- `mulan/constants.py`: add `"aido": "genbio-ai/AIDO.Protein-16B"`.
- `mulan/utils.py::load_pretrained_plm`: AIDO branch — try `AutoModel`/`AutoTokenizer`
  with `trust_remote_code=True`; else fall back to `modelgenerator.tasks.Embed`.
- `mulan/utils.py::embed_sequence`: AIDO branch with its own call path
  (`model.transform({"sequences":[...]})` → `model(batch)`) and CLS/SEP stripping,
  returning `[1, L, 2304]` aligned to residues.
- Run AIDO generation in a **separate venv** to isolate dependencies.

## Next steps (when GPU is available)

1. **Provision/verify GPU** (≥40 GB to hold bf16 weights comfortably; A100/H100).
2. **Pick the model**: AIDO.Protein-16B unless a smaller plain (non-RAG) variant
   is preferred for cost; confirm the ModelGenerator `Embed` API + dtype (bf16).
3. **Generate embeddings (Approach B)** for the S1102 id set; **validate shape**
   `[L, 2304]` with special tokens stripped (the check that caught ProstT5's prefix
   off-by-one).
4. **run5**: train MuLAN on AIDO embeddings, same split, default config —
   head-to-head vs run1 (Ankh 0.757) / run2 (ProstT5 0.740).
5. **AIDO layer probe** (`layer_probe.py --plm aido`, once a transformers/MoE
   forward with `output_hidden_states` is reachable) to test H3.
6. **run5 + Tier-1 augmentation** (the lever that helped Ankh).
7. **10-fold CV** for the winner if a defensible ranking is needed.
8. Fold results into `../docs/history/PLM_COMPARISON_S1102.md`.

## Risks / open questions

- **GPU memory**: 16B bf16 ≈ 32 GB; MoE expert weights all resident. Need ≥40 GB
  or sharding.
- **Dependency conflict**: `modelgenerator` vs the pinned env → isolate (separate
  venv / Approach B).
- **`trust_remote_code=True`** runs vendor modeling code (known org, acceptable).
- **MoE determinism**: routing/precision sensitivity — pin dtype and seed for
  reproducible embeddings.
- ~~**Does a transformers-native (`AutoModel`+`trust_remote_code`) path exist**, or is
  ModelGenerator mandatory?~~ **Resolved (2026-06-21):** yes — `config.json` has an
  `auto_map` for `fm4bio`, so `AutoModel(..., trust_remote_code=True)` works.

Sources: [AIDO.Protein-16B](https://huggingface.co/genbio-ai/AIDO.Protein-16B),
[GenBio ModelGenerator](https://github.com/genbio-ai/ModelGenerator),
[AIDO.Protein-RAG-3B](https://huggingface.co/genbio-ai/AIDO.Protein-RAG-3B).
