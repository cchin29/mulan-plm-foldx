# Apple Silicon (MPS) compatibility

MuLAN runs on Apple Silicon GPUs via PyTorch's Metal Performance Shaders (MPS)
backend, on CUDA GPUs, and on CPU — from the **same source, with no per-machine
edits**. "MPS compatible" here means two things:

1. **One device source of truth.** A single helper, `get_device()`, picks the
   best available backend (MPS → CUDA → CPU). Every entry point and every model
   loader routes through it, so a Mac run lands on the GPU instead of silently
   falling back to CPU.
2. **A safety net for missing kernels.** Some PyTorch ops still have no Metal
   kernel (for example `aten::linalg_eig`). The `PYTORCH_ENABLE_MPS_FALLBACK=1`
   environment variable routes any such op to CPU instead of raising
   `"… not currently implemented for the MPS device"`, so a run completes rather
   than crashing partway through. It is enabled by default as a safety net for the
   larger PLM front-ends (ProstT5 / T5), which are the most likely to reach one.

Neither mechanism does anything harmful on CUDA or CPU machines — `get_device()`
just returns `cuda`/`cpu`, and the fallback flag is a no-op where there is no MPS
backend. The code is portable, not Mac-only.

---

## 1. Device selection: `get_device()`

Defined in `mulan/utils.py`:

```python
def get_device() -> torch.device:
    """Return the best available device, preferring Apple Silicon (MPS), then CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
```

Precedence is **MPS → CUDA → CPU**. It is re-exported from the package
(`mulan/__init__.py`) so callers can use `mulan.get_device()`.

A subtlety the docstring calls out: **Python evaluates argument defaults once at
function-definition time**, so a loader cannot write `device=get_device()` in its
signature (that would freeze the choice at import). Instead the loaders default
`device=None` and resolve it *in-body*:

- `mulan/utils.py` `load_pretrained_plm(model_name, device=None)` → `if device is None: device = get_device()`
- `mulan/modules.py` `LightAttModel.from_pretrained(..., device=None)` → same pattern, then `torch.load(..., map_location=device)` and `cls(config).to(device)`

### Where it is wired

| File | How `get_device()` is reached |
| --- | --- |
| `mulan/utils.py` (`load_pretrained_plm`) | in-body default when `device is None` |
| `mulan/modules.py` (`LightAttModel.from_pretrained`) | in-body default when `device is None` |
| `scripts/predict.py` | `device = mulan.get_device()` |
| `scripts/compute_landscape.py` | `device = mulan.get_device()` |
| `scripts/extract_attentions.py` | `device = mulan.get_device()` |
| `scripts/generate_embeddings.py` | `device = mulan.get_device()` |
| `experiments/layer_probe.py` | `load_pretrained_plm(args.plm, device=get_device())` |
| `experiments/gen_layer_embeddings.py` | `load_pretrained_plm(args.plm, device=get_device())` |
| `experiments/gen_struct_embeddings.py` | `load_pretrained_plm("prostt5", device=get_device())` |
| `scratch/probe_layers.py` | `load_pretrained_plm("prostt5", device=get_device())` |
| `scratch/eval_s1102.py` | `device = mulan.get_device()` |

The scripts import the package symbol (`mulan.get_device()`); the
experiment/scratch helpers import it directly (`from mulan.utils import get_device`).
Making the device explicit at these call sites replaces the older pattern where a
device was hardcoded or left implicit, so all paths now share the same
MPS→CUDA→CPU logic.

---

## 2. The MPS fallback env var — and why placement matters

```python
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
```

### What it does

When an op has no Metal kernel, PyTorch's MPS fallback dispatcher runs it on CPU
(copying tensors across as needed) instead of raising. `aten::linalg_eig` is one
such op (used here to verify the behavior). The flag is enabled by default as a
safety net for the larger PLM front-ends (ProstT5 / T5); without it, a run that
reaches a missing kernel errors out on MPS, and with it the run finishes (at some
perf cost — see Limitations).

### The load-time-vs-dispatch-time gotcha

**PyTorch reads `PYTORCH_ENABLE_MPS_FALLBACK` at `import torch`**, when the MPS
fallback dispatch is registered — **not** at the moment an unsupported op is
dispatched. This was verified empirically on this machine (torch 2.12.0):

- Setting the var **before** `import torch` → fallback is active.
- Setting it **after** `import torch` → fallback is **not** active; the
  unsupported op still raises.

That is the entire reason for the placement rules below: the assignment must run
**before torch is first imported** in a given process.

### Why `setdefault`

`os.environ.setdefault(...)` only sets the value if the user hasn't already set
it. That means:

- An explicit user override is respected — `export PYTORCH_ENABLE_MPS_FALLBACK=0`
  disables the fallback (e.g. to surface an unsupported op as a hard error).
- On CUDA/CPU machines it is a harmless no-op — the flag exists but there is no
  MPS backend to consult it.

### Where it is set (two kinds of places)

**(a) The package**, in `mulan/__init__.py`, as the very first statements — before
the torch-importing submodules (`.config`, `.modules`, `.utils`) are imported.
This covers `import mulan`, library use, and interactive sessions.

**(b) Per-script guards**, placed immediately **before that file's `import torch`**,
in every entry point that imports torch directly (and imports torch before
`mulan`). Verified to sit before `import torch` in each of:

- `scripts/predict.py`
- `scripts/compute_landscape.py`
- `scripts/extract_attentions.py`
- `scripts/generate_embeddings.py`
- `scripts/train.py`
- `experiments/gen_layer_embeddings.py`
- `experiments/gen_struct_embeddings.py`
- `experiments/layer_probe.py`
- `experiments/gen_interface.py`
- `experiments/gen_struct_context.py`
- `scratch/eval_s1102.py`
- `scratch/probe_layers.py`

Both kinds are needed: importing `mulan` first would trigger (a), but a script
that does `import torch` before `import mulan` would already have registered the
dispatcher without the flag — so each such script carries its own guard. To
re-check placement, grep the marker `PYTORCH_ENABLE_MPS_FALLBACK` and confirm the
line precedes `import torch` in each file.

---

## 3. Training specifics (`scripts/train.py`)

Two MPS-relevant details in the HuggingFace `Trainer` path:

- **`dataloader_pin_memory=False`** in `TrainingArguments`. Pinned host memory is
  a CUDA concept and is unsupported on MPS; leaving it on produces a per-epoch
  warning. Disabling it is harmless elsewhere.
- **Model load device vs. Trainer device.** The model is loaded with
  `device="cpu"` (`mulan.load_pretrained(..., device="cpu")`). The `Trainer` then
  moves it to the device it auto-detects. With transformers 4.44.2 on Apple
  Silicon, `TrainingArguments.device` resolves to `mps` (`use_cpu` defaults to
  `False`), so training runs on the GPU even though the model was loaded on CPU.
  There is no need to pre-place the model on MPS yourself.

---

## 4. Same-device correctness (`scripts/compute_landscape.py`)

MPS enforces strict same-device checks: a CPU tensor cannot be `cat`-ed with an MPS
tensor. In `compute_landscape.py` the per-mutation ZS score is moved onto the
model's device when it is read:

```python
zs_score = None if scores_file is None else zs_scores[i, utils.aa2idx[aa]].to(device)
```

so the later concatenation that folds `zs_score` into the model input stays on a
single device. (`scripts/predict.py` similarly guards with
`zs_score.to(device) if zs_score is not None else None`.)

---

## 5. Intentional CPU exception: `scratch/esmc6b_smoke.py`

`scratch/esmc6b_smoke.py` is a scratch smoke test for **ESM-C 6B**. It is
deliberately **not** MPS-wired — it carries no fallback guard and loads on CPU:

```python
with torch.device("meta"):
    model = ESMC(d_model=2560, n_heads=40, n_layers=80, ...)
model = model.to_empty(device="cpu").eval()
# then load_state_dict shard-by-shard from safetensors
```

It builds the model on the `meta` device (no allocation), materializes empty
weights on CPU with `to_empty`, then streams the checkpoint shards in — a
memory-safe pattern for a very large model. Two reasons it stays on CPU:

- Its checkpoint path is hardcoded to a Linux box
  (`~/.cache/huggingface/hub/models--EvolutionaryScale--esmc-6b-...`),
  so it is not meant to run as-is on the Mac.
- Whether 6B even fits on MPS is **machine-dependent**. Apple Silicon uses
  unified memory, and Metal caps the GPU working set at roughly 70–75% of RAM.
  The weights alone are ~12 GB in bf16 / ~24 GB in fp32, before activations — so
  it will not fit on small Macs but can fit on a large-RAM machine. This is a
  practical note, not an absolute prohibition.

---

## 6. Limitations and opting out

- **Perf hit from CPU fallback.** Every op without a Metal kernel is executed on
  CPU, with tensor copies across the MPS↔CPU boundary. For PLMs that lean on
  those ops (some ProstT5 / T5 paths), embedding generation can be noticeably
  slower than a fully-resident GPU run. Functionally correct, not always fast.
- **ESM-C 6B on Mac** is memory-bound (see §5) and not wired for MPS in the
  scratch smoke test.
- **Opting out of the fallback.** Because the guards use `setdefault`, it is possible to
  override them from the shell:

  ```bash
  export PYTORCH_ENABLE_MPS_FALLBACK=0
  ```

  This makes an unsupported op raise instead of silently falling back to CPU —
  useful for finding exactly which ops are missing a Metal kernel, or
  to force a fully-on-device run. It must be set **before** the Python process
  starts (i.e. before `import torch`), for the same load-time reason as §2.

## 7. Numerical parity — MPS vs CPU/Linux

The device routing above is only trustworthy if MPS gives the *same numbers* as CPU. Measured
directly on **Ankh-large** (S1102), MPS end-to-end vs the Linux CPU reference on identical folds
(`scratch/ankh_parity.sh`; forces the CPU arm via the `MULAN_FORCE_CPU=1` hook in `get_device()`):

- **Embeddings (MPS vs CPU forward, same Mac), all 1444 seqs:** mean |Δ| **3.0e-8** (≤ fp32
  epsilon), max |Δ| 1.3e-6, per-residue cosine **1.000000** (median and min) — bit-for-bit
  equivalent up to single-precision rounding.
- **Downstream CV10 (paired, 10 folds):** Linux 0.8321 ± 0.058 vs MPS-emb+MPS-train 0.8301 ± 0.060
  → paired **Δ −0.0020**, inside the ±0.01–0.02 split-noise band (9/10 folds |Δ| ≤ 0.014; one
  fold −0.049 from seed-invariant MPS/CPU op nondeterminism, not device bias). Isolating just the
  embedding device (MPS-emb vs CPU-emb, train fixed on MPS): mean **+0.0035**.

**Takeaway:** MPS-measured results (Ankh3, SaProt, ankh3 +aug) are directly comparable to the
CPU/Linux-measured rows. See RESULTS.md §16 for the full table and interpretation.
