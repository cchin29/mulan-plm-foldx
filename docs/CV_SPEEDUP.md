# Speeding up MuLAN 10-fold CV (cached embeddings)

**Goal:** reduce wall-clock of the MuLAN 10CV runs. Scope assumes **PLM embeddings are
already computed and cached to disk** — the PLM front-end is never invoked during CV.

## The bottleneck (measured)

With embeddings cached, a CV run only exercises the tiny Light-Attention head plus
**data loading**. Loading — not GPU compute — dominates. `MulanDataset._load_embeddings`
(`mulan/data.py`) ran `torch.load` on **every `__getitem__`**: 4 files per row, once per
sample *per epoch per fold*, on the main thread (`num_workers=0`).

Measured on S1102 / Ankh-large:

| Per **epoch** | Value |
|---|---|
| `torch.load` calls | 4,408 (1,102 rows × 4 files) |
| Bytes deserialized | ~3.0 GB |
| Unique files touched | 1,449 (~1.0 GB working set) |
| In-epoch redundancy | 3.0× (WT chains shared across a complex's mutations) |

That ~3 GB re-deserialization repeats **every epoch × every fold** — order of a terabyte
over a full ~150-epoch × 10-fold run — to feed a ~1 GB unique input that fits in RAM.
Even with the OS page cache warm, `torch.load` still pays zip-unpack + unpickle + tensor
rebuild on every call, blocking the GPU.

## Options

| # | Option | Impact | Effort | Status |
|---|--------|--------|--------|--------|
| 1 | In-memory embedding cache | High | ~15 lines | **In the code** |
| 2 | Larger `per_device_train_batch_size` on MPS | Medium | config knob | Not adopted |
| 3 | Cross-fold cache reuse (mmap bundle / thread folds) | Low–Med | Medium | Not adopted |
| 4 | fp16 embeddings on disk (halve I/O + load) | Low | Medium | Not adopted — redundant while (1) holds |
| — | MLX / Core ML / ANE port | ~None for CV | Large | **Rejected** |

### 1. In-memory embedding cache — the one that shipped

Memoize deserialized tensors by file path on the dataset instance; the first `torch.load`
for an id is the only one, subsequent epochs reuse the resident tensor. Numerically
identical (cached tensors are read-only downstream — collator `pad_sequence` and the model
allocate new tensors; nothing mutates inputs). Toggle off with `MULAN_EMB_CACHE=0`.

- **Implemented in** `mulan/data.py`: per-instance `_tensor_cache`, new `_load_tensor(path)`
  helper; `_load_embeddings` and `_load_struct_context` route through it.
- **Verified:** bit-identical cache-on vs cache-off (`torch.equal`, 60 samples × 4 chains),
  against a green test suite (82 passed / 6 skipped at the time of this measurement).
- **Measured:** **6.0×** faster over 3 epochs of real `__getitem__` (11.6× on the isolated
  warm-page load path); per-epoch load ~0.46 s → ~0.04 s on S1102/Ankh.

### 2. Larger batch size on MPS — not adopted

The head is tiny, so at small batch the run is kernel-launch-bound and under-fills the GPU.
Raising `per_device_train_batch_size` in `TrainingArguments` (`scripts/train.py`) is a free
knob, but picking a value needs a before/after benchmark that was never run, so the default
stands.

### 3. Cross-fold cache reuse — not adopted

Folds run as separate `mulan-train` processes (fold_0/, fold_1/…), so the per-process cache
from (1) re-warms each fold. To kill that too: bundle the 1,449 `.pt` into one
memory-mapped tensor file loaded once, or run folds as threads under one parent holding the
cache. Secondary — (1) already captures the epochs×N multiplier, which is the bulk.

### 4. fp16 embeddings on disk — not adopted

Halves I/O and load cost, but mostly moot once (1) keeps the working set in RAM. Only worth
it if the working set ever outgrows RAM. Would need a parity check before adoption.

## Rejected: MLX / Core ML / ANE

These target GPU compute of the Light-Attention head, which is **not** the CV bottleneck
once embeddings are cached — the stall is I/O/deserialization on the main thread. MLX is a
separate framework (not an MPS-style backend), so it would require reimplementing the head
(and PLMs, for embedding runs) in `mlx.nn` + weight conversion + numerical-parity validation
— large effort aimed at the wrong constraint. MPS already runs the head with measured
CPU/Linux parity (RESULTS.md §16). Revisit only if the target shifts to accelerating
*embedding generation* for new-sequence workloads (landscape scans), not CV.

## Notes

- `num_workers>0` deliberately not pursued first: with (1) the loader is no longer the stall,
  and worker fork/IPC of tensors on MPS tends to add overhead rather than remove it.
- Data artifact (pre-existing, unrelated): `examples/S1102.tsv` has 2 WT labels
  (`2I9B`-family) absent from every FASTA and the embedding cache — a full-1102 run would
  error on those rows.
