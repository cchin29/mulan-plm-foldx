# AIDO.Protein-16B on CPU — execution instructions (run5, offline embeddings)

> **Target machine:** the linux box (`$MULAN_ROOT`,
> 2× Xeon Silver 4114, 40 cores, 62 GiB RAM, no GPU). **Goal:** produce MuLAN ΔΔG
> numbers for **AIDO.Protein-16B** (the pending **run5**) *without a GPU*, by
> generating per-residue embeddings once on CPU and training MuLAN read-only from
> that cache — exactly the pattern already used for **ESM C 6B** (§11 of
> `RESULTS.md`).
>
> This document is self-contained but assumes the repo, the S1102 data, and the
> AIDO weights are already present (see Phase 0). Read `docs/history/PLAN_AIDO.md`
> and `experiments/RESULTS_ESMC.md` for background; this is the concrete runbook.

---

## 0. TL;DR

1. AIDO.Protein-16B is a **MoE (8 experts, top-2), 16B-param, encoder-only** PLM
   (`fm4bio`, hidden **2304**, 36 layers, ctx 2048). We only need a **forward pass**
   to cache `[L, 2304]` per-residue embeddings — no training of AIDO, no serving.
2. **fp32 weights (~64 GB) do NOT fit 62 GB RAM.** Load in **bf16 (~32 GB)** with
   shard streaming. bf16 (not fp16) is mandatory: same exponent range as fp32, so
   **no overflow**, only minor mantissa precision loss — fine for embeddings.
3. Compute is MoE top-2/8 → **~5–6 B active params/token**, comparable to ESM C 6B
   (which took ~70 min for 1444 seqs at 40 threads). Expect **a few hours →
   overnight**; it's a **one-time** job, then cached.
4. Do everything in an **isolated venv** for generation; the main `~/.venvs/mulan`
   (transformers 4.44) stays untouched and trains **read-only from the cache**.
5. Deliverable: a new `cv10_aido` result and a master-table row in `RESULTS.md`,
   head-to-head vs **ESM C 6B 0.859 / Ankh 0.832 / ProstT5 0.805** (10-fold CV).

**Acceptance criterion:** a completed 10-fold CV for AIDO with a sane PCC
(≳0.80 — if it collapses, the embeddings are wrong; see §7). Whether it *beats*
Ankh/ESM C is the scientific question, not a pass/fail.

---

## 1. Environment & key paths (verify all exist in Phase 0)

| What | Path |
|---|---|
| Repo root | `$MULAN_ROOT` |
| Main venv (training; **do not modify**) | `~/.venvs/mulan` (transformers 4.44) |
| **New** isolated gen venv (you create) | `~/.venvs/mulan-aido` |
| AIDO weights (HF cache) | `~/.cache/huggingface/hub/models--genbio-ai--AIDO.Protein-16B` |
| AIDO RAG-3B (family smoke test) | `~/.cache/huggingface/hub/models--genbio-ai--AIDO.Protein-RAG-3B` |
| WT sequences (FASTA) | `scratch/results/run1_s1102_ankh/wt_sequences.fasta` |
| S1102 table (1100 muts) | `scratch/results/run1_s1102_ankh/S1102_filtered.tsv` |
| 10-fold splits | `scratch/cv10_splits/fold_{0..9}/S1102_filtered_{train,val,test}.tsv` |
| MuLAN training config | `models/config/lightatt_default_config.json` |
| **New** embedding cache (you create) | `scratch/embeddings_aido/` |
| **New** results | `scratch/results/cv10_aido/fold_{0..9}/` |

All paths below are **relative to the repo root** unless absolute. Model
identifier: `genbio-ai/AIDO.Protein-16B`, embedding dim **2304**.

---

## 2. Guardrails (read before touching anything)

1. **Never modify or reinstall into `~/.venvs/mulan`.** The Ankh/ProstT5/ESM C
   numbers must stay reproducible. All AIDO/`transformers`/`flash-attn` work happens
   in `~/.venvs/mulan-aido`.
2. **Pre-cache *all* embeddings before training.** `mulan/data.py` (~line 73) only
   loads a PLM when an embedding id is **missing**. If even one is missing during
   `mulan-train`, it will try to load AIDO-16B inside the main venv — which you do
   **not** want. Confirm the cache is complete first (§6).
3. **bf16, not fp16, and never full fp32.** fp32 OOMs the box; fp16 risks overflow.
4. **One model instance only.** 32 GB weights → you cannot fit two copies in 62 GB.
   Generation is a **single serial process** with 40 intra-op threads (like ESM C).
5. **Embedding filenames must match `MulanDataset` enumeration byte-for-byte.** The
   generator below copies that logic verbatim from `esmc6b_gen.py`; do not change it.
6. **Cache tensors in float32** (cast after the bf16 forward) so they're identical in
   dtype to the other PLMs' caches and to what the training loader expects.

---

## 3. Why CPU works here (background for the executor)

- **Memory is the only hard wall**, and bf16 clears it (32 GB weights + short-seq
  activations ≈ 35–40 GB peak < 62 GB; ESM C peaked at 29 GB on this box).
- **MoE residency:** all 8 experts stay in RAM regardless of top-2 routing — no
  memory saving from sparsity, but compute *is* sparse (~2/8 of experts per token).
- **No native bf16 compute on Skylake-SP** (no AVX512-BF16): bf16 ops upcast to fp32
  internally, so you get the *memory* win but not a speed win, plus minor conversion
  overhead. This is expected; it does not affect correctness.
- We need **offline embeddings only** (Approach B in `docs/history/PLAN_AIDO.md`): AIDO never
  lives inside MuLAN. Cache `[L, 2304]` tensors, then `mulan-train --embeddings_dir`.

---

## 4. Phase 0 — preflight (do first, ~5 min)

```bash
cd $MULAN_ROOT
# Weights present and sized as expected (~60–64 GB, 13 shards)?
du -sh ~/.cache/huggingface/hub/models--genbio-ai--AIDO.Protein-16B
ls  ~/.cache/huggingface/hub/models--genbio-ai--AIDO.Protein-16B/snapshots/*/ | grep -E "pytorch_model|config.json|tokenizer"
python - <<'PY'
import json, glob
cfg = glob.glob("~/.cache/huggingface/hub/models--genbio-ai--AIDO.Protein-16B/snapshots/*/config.json")[0]
c = json.load(open(cfg))
print("model_type:", c.get("model_type"), "| hidden:", c.get("hidden_size"),
      "| layers:", c.get("num_hidden_layers"), "| experts:", c.get("num_experts"),
      "| experts/tok:", c.get("experts_per_token"), "| auto_map:", "auto_map" in c)
PY
# Data + splits + config present?
ls scratch/results/run1_s1102_ankh/{wt_sequences.fasta,S1102_filtered.tsv}
ls scratch/cv10_splits/fold_0/
ls models/config/lightatt_default_config.json
# Resources
free -g; nproc; df -h /home | tail -1
```

**Stop if:** weights are missing/short (re-`hf download genbio-ai/AIDO.Protein-16B`),
or `config.json` doesn't show `model_type: fm4bio` + `hidden_size: 2304` +
`auto_map`. Expected: `num_experts: 8`, `experts_per_token: 2`.

---

## 5. Phase 1 — isolated venv + go/no-go smoke test (~20–40 min)

### 5.1 Build the generation venv

```bash
python3 -m venv ~/.venvs/mulan-aido
source ~/.venvs/mulan-aido/bin/activate
pip install --upgrade pip
# CPU-only torch + the transformers stack fm4bio's remote code needs.
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "transformers>=4.44" accelerate safetensors sentencepiece einops numpy
# DO NOT install flash-attn (CUDA-only; will fail to build and isn't needed on CPU).
```

> **Note on `transformers` version:** fm4bio's remote code may pin a range. If import
> fails (§9-A), read the traceback and adjust the pin; genbio's ModelGenerator tracks
> a 4.4x-era transformers. Keep this venv separate so it never contaminates the main one.

### 5.2 Smoke test — this is the go/no-go for CPU feasibility

Save as `scratch/aido_smoke.py` and run it. It loads the real 16B in bf16 and embeds
3 short sequences, shaking out the three known risks (flash-attn import, MoE dispatch,
bf16-on-CPU) and giving a real per-seq timing to extrapolate the full-run ETA.

```python
# scratch/aido_smoke.py  — CPU go/no-go for AIDO.Protein-16B
import time, torch
from transformers import AutoModel, AutoTokenizer

MODEL = "genbio-ai/AIDO.Protein-16B"
torch.set_num_threads(40)

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

log("loading tokenizer"); tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
log("loading model (bf16, low_cpu_mem_usage) — expect a few min & ~32 GB RAM")
t0 = time.time()
model = AutoModel.from_pretrained(
    MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    # If this kwarg errors, delete it and force eager attention another way (see §9-A):
    attn_implementation="eager",
).eval()
log(f"loaded in {(time.time()-t0)/60:.1f} min")

# --- tokenizer format probe: we need exactly len(seq)+2 tokens (CLS + residues + SEP)
seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
for spaced in (False, True):
    s = " ".join(seq) if spaced else seq
    n = tok(s, return_tensors="pt")["input_ids"].shape[1]
    log(f"spaced={spaced}: {n} tokens for len {len(seq)} (want {len(seq)+2})")
# pick the format that yields len(seq)+2; use it below and in aido_gen.py

def embed(seq, spaced):
    s = " ".join(seq) if spaced else seq
    enc = tok(s, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)
    h = getattr(out, "last_hidden_state", None)
    if h is None: h = out.hidden_states[-1]
    return h[0, 1:-1, :].float().contiguous()   # strip CLS/SEP -> [L, 2304], fp32

SPACED = False   # <-- set from the probe above
for i, s in enumerate([seq, "ACDEFGHIKLMNPQRSTVWY", "GGGSGGGSGGGS"]):
    t = time.time(); e = embed(s, SPACED)
    assert e.shape == (len(s), 2304), (i, e.shape, len(s))
    assert torch.isfinite(e).all(), f"non-finite emb for seq {i}"
    log(f"seq{i} len {len(s)} -> {tuple(e.shape)} | finite OK | {time.time()-t:.1f}s")
# determinism + locality sanity
e1 = embed(seq, SPACED); e2 = embed(seq, SPACED)
log("deterministic:", torch.allclose(e1, e2))
log("SMOKE OK — extrapolate ETA: (s/seq above) × ~1444 seqs")
```

**Go/no-go:**
- ✅ **Go** if it loads, prints `SMOKE OK`, shapes are `[L, 2304]`, all finite,
  deterministic, and per-seq time is sane (≲15 s/seq → full run ≲6 h). Record the
  chosen `SPACED` value and the CLS/SEP layout (`[1:-1]` strip).
- ❌ If it **fails to import** → §9-A (flash-attn / version). If **MoE op errors** →
  §9-B. If **OOM** → §9-C. If **token count ≠ len+2** → §9-D. Fix, re-run smoke.

> Optional cheaper probe: the **RAG-3B** (11 GB) exercises the same `fm4bio` family
> code faster, but it's retrieval-augmented (expects an MSA), so use it only to debug
> import/attention/MoE mechanics — **not** to produce embeddings.

---

## 6. Phase 2 — generate the full embedding cache (~a few hours, one-time)

Save as `scratch/aido_gen.py`. It is a **copy of `scratch/esmc6b_gen.py`** with the
model call swapped to AIDO; the `enumerate_ids` / preprocessing logic is **verbatim**
so filenames and residue handling match every other PLM cache.

```python
# scratch/aido_gen.py — AIDO.Protein-16B per-residue embeddings for S1102 -> [L,2304]
import os, re, sys, time, torch
from transformers import AutoModel, AutoTokenizer

MODEL = "genbio-ai/AIDO.Protein-16B"
EMB   = "scratch/embeddings_aido"
WT    = "scratch/results/run1_s1102_ankh/wt_sequences.fasta"
TBL   = "scratch/results/run1_s1102_ankh/S1102_filtered.tsv"
SPACED = False          # <-- set from the smoke-test tokenizer probe (§5.2)
torch.set_num_threads(40)

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

def parse_fasta(path):                      # == mulan.utils.parse_fasta
    d, key = {}, None
    for line in open(path):
        if line.startswith("#") or not line.strip(): continue
        if line.startswith(">"):
            key = line.strip().split()[0][1:].split("|")[0]; d[key] = ""
        else: d[key] += line.strip().upper()
    return d

def parse_mutations(mutations, seq1, seq2): # == mulan.utils.parse_mutations
    seq1, seq2 = list(seq1), list(seq2)
    for m in mutations:
        if m[1] == "A": seq1[int(m[2:-1]) - 1] = m[-1]
        else:           seq2[int(m[2:-1]) - 1] = m[-1]
    return "".join(seq1), "".join(seq2)

def enumerate_ids():                        # == esmc6b_gen.enumerate_ids (id set is PLM-agnostic)
    wt = parse_fasta(WT); seqs = dict(wt)
    for line in open(TBL):
        if not line.strip(): continue
        f = line.split(); s1l, s2l, muts = f[0], f[1], tuple(f[2].split(","))
        m1, m2 = parse_mutations(muts, wt[s1l], wt[s2l])
        m1l = f"{s1l}_{'-'.join(m for m in muts if m[1] == 'A')}"
        m2l = f"{s2l}_{'-'.join(m for m in muts if m[1] == 'B')}"
        seqs[s1l], seqs[s2l], seqs[m1l], seqs[m2l] = wt[s1l], wt[s2l], m1, m2
    return seqs

def build_model():
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, attn_implementation="eager").eval()
    return model, tok

def embed_residues(model, tok, seq):
    s = " ".join(seq) if SPACED else seq
    enc = tok(s, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc)
    h = getattr(out, "last_hidden_state", None)
    if h is None: h = out.hidden_states[-1]
    return h[0, 1:-1, :].float().contiguous()          # strip CLS/SEP -> [L,2304], fp32

def main():
    os.makedirs(EMB, exist_ok=True)
    seqs = enumerate_ids()
    todo = [i for i in seqs if not os.path.exists(os.path.join(EMB, i + ".pt"))]
    log(f"total ids {len(seqs)} | cached {len(seqs)-len(todo)} | to do {len(todo)}")
    if not todo: log("nothing to do"); return
    model, tok = build_model(); t0 = time.time()
    for n, id_ in enumerate(todo, 1):
        seq = re.sub(r"[UZOB]", "X", seqs[id_].upper())  # == embed_sequence preprocessing
        emb = embed_residues(model, tok, seq)
        assert emb.shape == (len(seq), 2304), (id_, emb.shape, len(seq))
        assert torch.isfinite(emb).all(), f"non-finite embedding for {id_}"
        torch.save(emb, os.path.join(EMB, id_ + ".pt"))
        if n % 25 == 0 or n == len(todo):
            el = time.time() - t0
            log(f"  {n}/{len(todo)} | {el/n:.1f}s/seq | ETA {(len(todo)-n)*el/n/60:.0f} min")
    log(f"DONE: {len(todo)} embeddings in {(time.time()-t0)/60:.1f} min -> {EMB}")

if __name__ == "__main__": main()
```

Run it **resumably** in the isolated venv, logged and detached:

```bash
source ~/.venvs/mulan-aido/bin/activate
cd $MULAN_ROOT
nohup python scratch/aido_gen.py > scratch/aido_gen.log 2>&1 &
echo $! > scratch/aido_gen.pid
# monitor
tail -f scratch/aido_gen.log
# in another shell, watch peak RAM (should stay < ~45 GB):
while kill -0 $(cat scratch/aido_gen.pid) 2>/dev/null; do free -g | awk 'NR==2{print strftime("%H:%M:%S"),"used",$3"G"}'; sleep 60; done
```

The script is **idempotent** — re-running skips already-cached ids, so if it dies
(OOM, etc.) just fix and restart.

### 6.1 Validate the cache before training

```bash
source ~/.venvs/mulan-aido/bin/activate; cd $MULAN_ROOT
python - <<'PY'
import os, torch, glob, random
EMB="scratch/embeddings_aido"
fs=glob.glob(EMB+"/*.pt"); print("cached tensors:", len(fs))
for f in random.sample(fs, min(8,len(fs))):
    t=torch.load(f)
    assert t.ndim==2 and t.shape[1]==2304 and torch.isfinite(t).all(), (f,t.shape)
    assert t.float().std()>1e-4, ("degenerate emb", f)   # not all-constant
print("shape/finite/non-degenerate checks OK; dim 2304")
PY
```

**Expected count:** the same unique-id count ESM C produced (`ls scratch/embeddings_esmc6b | wc -l`).
Cross-check they match:
```bash
diff <(ls scratch/embeddings_esmc6b | sort) <(ls scratch/embeddings_aido | sort) && echo "ID SETS MATCH"
```
If the id sets differ, **stop** — the enumeration diverged and training will try to
load AIDO in the main venv (guardrail #2).

---

## 7. Phase 3 — minimal wiring + run5 10-fold CV (training, read-only from cache)

### 7.1 One-line additive wiring in the **main** repo

Add AIDO to the PLM registry so `--plm_model_name aido` is accepted (it is **not**
loaded, because all embeddings are cached — but the arg is validated). Edit
`mulan/constants.py`, in the `PLM_ENCODERS` dict, mirroring the `esmc_6b` line:

```python
    "esmc_6b": "EvolutionaryScale/esmc-6b-2024-12",  # existing
    "aido": "genbio-ai/AIDO.Protein-16B",            # ADD — 2304-dim, offline-embeddings only (CPU bf16)
```

That's the only change to the tracked code for Approach B. (An in-`mulan` load path
— Approach A in `docs/history/PLAN_AIDO.md` — is **not** needed here and would pull the fm4bio
remote code into the main venv; skip it.)

### 7.2 CV driver

Save as `scratch/cv10_aido_driver.sh` — a copy of `scratch/cv10_esmc6b_driver.sh`
with the PLM name, embedding dir, and result dir swapped. Runs in the **main** venv,
read-only from cache (no PLM load).

```bash
#!/usr/bin/env bash
# 10-fold CV for AIDO.Protein-16B on S1102, same config as the Ankh/ProstT5/ESM C cv10.
# Main venv (transformers 4.44); embeddings pre-cached in scratch/embeddings_aido,
# so NO PLM is loaded (data.py loads a PLM only for MISSING embeddings).
set -uo pipefail
source ~/.venvs/mulan/bin/activate
cd $MULAN_ROOT

SPLITS=scratch/cv10_splits
EMB=scratch/embeddings_aido
WT=scratch/results/run1_s1102_ankh/wt_sequences.fasta
CFG=models/config/lightatt_default_config.json
COMMON="--model_name_or_config_path $CFG --save_model True --num_epochs 50 \
  --batch_size 32 --learning_rate 5e-4 --early_stopping_patience 10 \
  --report_to none --disable_tqdm True"

MAXPAR=4
export OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 NUMEXPR_NUM_THREADS=10
log(){ echo "[$(date +%H:%M:%S)] $*"; }
rm -f scratch/.cv10_aido_complete

run_fold(){
  local i=$1 R F
  R=scratch/results/cv10_aido/fold_${i}; F=$SPLITS/fold_${i}
  mkdir -p "$R"
  mulan-train --train_data "$F/S1102_filtered_train.tsv" \
    --eval_data "$F/S1102_filtered_val.tsv" --test_data "$F/S1102_filtered_test.tsv" \
    --train_fasta_file "$WT" --test_fasta_file "$WT" \
    --embeddings_dir "$EMB" --plm_model_name "aido" \
    --output_dir "$R/training_run" $COMMON > "$R/training_run.log" 2>&1
  log "DONE fold $i -> $(grep -o '\"test_pcc\":[0-9.]*' $R/training_run/all_results.json 2>/dev/null)"
}

log "=== AIDO CV10 START (MAXPAR=$MAXPAR, ${OMP_NUM_THREADS} threads/job) ==="
for i in $(seq 0 9); do
  while [ "$(jobs -rp | wc -l)" -ge "$MAXPAR" ]; do wait -n; done
  log "launch fold $i"; run_fold "$i" &
done
wait
log "=== all folds done ==="; touch scratch/.cv10_aido_complete
```

```bash
chmod +x scratch/cv10_aido_driver.sh
nohup ./scratch/cv10_aido_driver.sh > scratch/cv10_aido_driver.log 2>&1 &
tail -f scratch/cv10_aido_driver.log
```

**Sanity gate:** after **fold 0** finishes, check its `test_pcc`. If it's ≳0.7 you're
on track; if it's near 0 / NaN, **stop** — the embeddings are bad (re-check §6.1). No
point running 10 folds on a broken cache.

### 7.3 Aggregate + compare

Save as `scratch/aido_aggregate.py` (copy of `esmc6b_aggregate.py`, add the AIDO row):

```python
import json, glob, statistics as st
METRICS = ["test_pcc", "test_rmse", "test_mae", "test_scc"]
def agg(tag, pat):
    per={m:[] for m in METRICS}
    for f in sorted(glob.glob(pat)):
        d=json.load(open(f))
        for m in METRICS:
            if m in d: per[m].append(d[m])
    return tag, per
def fmt(v): return "-" if not v else (f"{v[0]:.3f}" if len(v)==1 else f"{st.mean(v):.3f} ± {st.pstdev(v):.3f}")
rows=[agg("aido","scratch/results/cv10_aido/fold_*/training_run/all_results.json"),
      agg("esmc_6b","scratch/results/cv10_esmc6b/fold_*/training_run/all_results.json"),
      agg("ankh","scratch/results/cv10_ankh/fold_*/training_run/all_results.json"),
      agg("prostt5","scratch/results/cv10_prostt5/fold_*/training_run/all_results.json")]
print("\n| PLM | folds | Test PCC | RMSE | MAE | SCC |\n|---|---|---|---|---|---|")
for tag,per in rows:
    print(f"| {tag} | {len(per['test_pcc'])} | **{fmt(per['test_pcc'])}** | "
          f"{fmt(per['test_rmse'])} | {fmt(per['test_mae'])} | {fmt(per['test_scc'])} |")
# paired AIDO vs ESM C and vs Ankh (same folds)
a=rows[0][1]["test_pcc"]
for name,idx in (("esmc_6b",1),("ankh",2)):
    b=rows[idx][1]["test_pcc"]
    if len(a)==len(b)==10:
        d=[x-y for x,y in zip(a,b)]; wins=sum(1 for x in d if x>0)
        print(f"\nPaired Δ(aido − {name}) PCC = {st.mean(d):+.4f} (std {st.pstdev(d):.4f}); aido wins {wins}/10")
        print("per-fold aido:", [f"{x:.3f}" for x in a]); print(f"per-fold {name}:", [f"{x:.3f}" for x in b])
```

```bash
source ~/.venvs/mulan/bin/activate
python scratch/aido_aggregate.py | tee scratch/aido_cv10_summary.txt
```

---

## 8. Phase 4 (optional) — Tier-1 augmentation

Only if the base run5 looks competitive. Reuse the augmented-CV machinery (§5a of
`RESULTS.md`): `experiments/augment.py` builds each fold's augmented train table +
FASTA; the identity/reverse anchors add new sequence ids, so **pre-warm those extra
embeddings with `scratch/aido_gen.py`** (it will only generate the missing ones)
before running an augmented CV driver modeled on `scratch/cv10_aug_driver.sh`.
Augmentation helped Ankh (+0.006) and ProstT5 (+0.014) under CV.

---

## 9. Failure modes & fixes

**A. Import error / `flash_attn` on load.** If the traceback is a top-level
`from flash_attn import ...` (or a transformers-version mismatch): (i) ensure
`flash-attn` is **not** installed; (ii) pass `attn_implementation="eager"` (already in
the scripts) — if fm4bio ignores/rejects it, open the cached `modeling_fm4bio.py`
under `.../snapshots/<hash>/` and set the attention path to eager / guard the
flash-attn import; (iii) if it's a version error, adjust the `transformers` pin in the
`mulan-aido` venv per the traceback. This is an **isolated** venv, so iterate freely.

**B. MoE op / custom kernel error on CPU.** If the forward calls a CUDA-only fused
MoE kernel (grouped-GEMM / megablocks): find the expert-dispatch code in
`modeling_fm4bio.py` and switch to the plain **loop-over-experts** path (most fm4bio
builds ship one). This is the main portability risk; the smoke test (§5.2) surfaces it
early on a 3-seq run, not 8 h in.

**C. OOM during load or run.** Confirm `torch_dtype=torch.bfloat16` **and**
`low_cpu_mem_usage=True` (streams shards; without it, transformers may materialize
fp32 → 64 GB → OOM). Close other processes; check `free -g`. Do **not** raise to
fp32. If still tight, reduce to a single generation process (it already is) and ensure
no cv10 training runs concurrently with generation.

**D. Token count ≠ len(seq)+2.** Re-run the tokenizer probe in §5.2. If the tokenizer
needs space-separated residues, set `SPACED=True` in **both** scripts. If it adds a
different special-token pattern (e.g., only a leading CLS, or an `<AA2fold>`-style
prefix like ProstT5), adjust the `[1:-1]` slice so the returned length equals the
residue count — this is the exact check that caught ProstT5's off-by-one.

**E. NaN / non-finite embeddings.** The `assert torch.isfinite` will halt on the
offending id. bf16 has fp32's exponent range so overflow is unlikely; a NaN usually
means an uninitialized buffer (this bit ESM C via `meta`/`to_empty`) — ensure the
model is built with real weights (it is, via `from_pretrained`), not `meta` + manual
copy. Re-run smoke first.

**F. Can't do a full fp32 reference on this box.** True — 64 GB won't fit, so a strict
bf16-vs-fp32 parity check isn't possible here. Rely on the internal sanity checks
(finite, shape, non-degenerate, deterministic, WT vs mutant differing only near the
mutation site) **plus** the downstream PCC gate (§7.2). If a bigger-RAM or GPU box
ever becomes available, spot-check a handful of sequences in fp32 there.

---

## 10. Deliverables (write back when done)

1. **`RESULTS.md` master-table row** for run5, e.g.:
   `| cv10_aido | AIDO.Protein-16B | last (2304) | none | 10-fold CV | <PCC ± std> | <RMSE> | <MAE> | <one-line verdict vs ESM C/Ankh> |`
   and update the §8 "AIDO (planned)" section into a results section (paired Δ vs
   ESM C and Ankh, folds won, significance à la the ESM C write-up).
2. **New sub-doc** `experiments/RESULTS_AIDO.md`, modeled on `RESULTS_ESMC.md`:
   headline, results table, per-fold PCC, the **CPU/bf16 method** (this is the novel
   part — how the GPU-pending model was run on the CPU box), runtime/footprint, and
   the loading gotchas actually hit (§9). Link it from the `RESULTS.md` sub-doc index.
3. **Artifacts stay in `scratch/`** (gitignored): `aido_gen.py`, `aido_smoke.py`,
   `cv10_aido_driver.sh`, `aido_aggregate.py`, `aido_gen.log`, `aido_cv10_summary.txt`,
   `embeddings_aido/`, `results/cv10_aido/`.
4. **Commit only** the additive `mulan/constants.py` line (+ the new sub-doc &
   RESULTS.md edits).

---

## 11. Provenance & confidence

- Hardware, ESM C runtime baseline (~70 min / 1444 seqs, 29 GB peak), split/config,
  and the read-only-cache training pattern: `RESULTS.md` §7/§11,
  `experiments/RESULTS_ESMC.md`, `scratch/esmc6b_gen.py`,
  `scratch/cv10_esmc6b_driver.sh` (this doc's scripts are direct adaptations).
- AIDO architecture (`fm4bio`, 2304-d, 36 layers, MoE 8×top-2, fp32 repo, `auto_map`,
  `AutoModel`+`trust_remote_code`): `docs/history/PLAN_AIDO.md` (config confirmed
  2026-06-21) and [HF model card](https://huggingface.co/genbio-ai/AIDO.Protein-16B).
- **Confidence:** *high* that the memory math and the read-only-cache training flow are
  correct (bf16 32 GB fits, fp32 64 GB doesn't; ESM C proved the cache pattern). *Medium*
  on (a) full-run wall-time (extrapolated from ESM C + MoE active-param count + the
  Skylake bf16-upcast overhead — could be 2–3× either way) and (b) whether
  `modeling_fm4bio` runs clean on CPU or needs the §9-A/§9-B patches — the **smoke test
  (§5.2) is the go/no-go that settles this cheaply**. Not yet executed anywhere; this is a
  runbook, not a completed run.
