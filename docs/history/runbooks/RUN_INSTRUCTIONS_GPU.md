# Full-SKEMPI ESM-C 6B on the GPU box — base vs +FoldX

Runs the ESM-C 6B arm of the full-SKEMPI single-point sweep (docs/history/PLAN_FULL_SKEMPI.md step 1/§8b.2)
on the RTX-3090 box. Reuses the **same driver + config the Mac used** for the smaller PLMs
(`run_bycomplex.sh` + `config_skempi_full.sh`); the only additions are the `esmc6b` model row,
the FoldX-merged splits (already built into this bundle), and env-overridable paths.

## What's in this bundle (extract at a repo-like root, that root = `MULAN_ROOT`)

```
experiments/full_skempi_seqonly/   config, models tsv (incl. esmc6b), evaluate.py, merge script, PLAN
experiments/retrain_split/         run_bycomplex.sh (the arm-generic driver)
experiments/embedding_sweep/       config.sh (training knobs, sourced)
experiments/rescore_perstructure/  rescore.py (scoring core for evaluate.py)
models/config/                     lightatt_{default,addscores,addscores_mlp}_config.json
scratch/skempi_full/               wt_sequences.fasta (630 labels), single_point.tsv
scratch/splits_skempi_full_clustered_id60_kfold/     base 4-col splits (3 folds)
scratch/foldx_skempi_full/splits_skempi_full_foldx/     +FoldX scalar (5-col)
scratch/foldx_skempi_full/splits_skempi_full_foldxdec/  +FoldX 12-term (16-col)
scratch/foldx_skempi_full/results/  112 per-complex FoldX JSON (only needed to re-run the merge)
```

## Prerequisites (already satisfied on the GPU box)

- A **CUDA torch + mulan** env. On the GPU box this is the conda env `~/miniforge3/envs/mulan`
  (torch 2.6.0+cu124, transformers 4.44.2; `mulan` is pip-installed *editable* from
  `~/mulan_esmc6b_gpu/code`, so that directory must stay in place even though the rest of that
  tree was reclaimed). Confirm with
  `"$MULAN_VENV_BIN/python" -c "import transformers,torch; print(transformers.__version__, torch.cuda.is_available())"`.
- **Two envs, distinct roles — do not conflate them:**

  | env | transformers | `esm` | role |
  |---|---|---|---|
  | `~/miniforge3/envs/mulan` | 4.44.2 | ✗ | **training** — this is `MULAN_VENV_BIN` |
  | `~/miniforge3/envs/esmgen` | 4.48.1 | 3.2.1 | **ESM-C 6B embedding generation only** (Step 1) |

  Ignore the old "`esmc_6b` needs `transformers>=4.57`" claim — neither env has ≥4.57, and 6B
  embedding generation bypasses `transformers` entirely (see Step 1). Note `esmgen`'s bundled
  `plm-embed` is broken (`torchvision::nms`); nothing uses it.
- Set two env vars to point at the box's layout:
  ```bash
  export MULAN_ROOT="$PWD"                                   # where this bundle was extracted
  export MULAN_VENV_BIN="$HOME/miniforge3/envs/mulan/bin"    # the working esmc_6b env's bin/
  ```
  > ⚠️ Older revisions of this doc pointed `MULAN_VENV_BIN` at `~/mulan_esmc6b_gpu/.venv/bin`.
  > That venv is a 28 KB stub and has never been what the queues actually used — don't restore it.

## Step 1 — pre-seed the ESM-C 6B embeddings (one-time, the only 6B-model-loading step)

Generating embeddings loads the 6B model; do it once, serially, so the 3 folds don't race on an
empty cache (after this, training only reads the `.pt` cache — no model reload):

> ⚠️ **`plm-embed … esmc_6b` does not work on this box** — and never did. The HF `esmc` model_type
> is in no released `transformers` and the ESM SDK registry lacks the 6B (see the docstring of
> `scratch/esmc6b_lib.py`); `esmgen`'s `plm-embed` additionally dies on `torchvision::nms`.
> Earlier revisions of this doc prescribed it; that instruction was never executed successfully.
> What actually produced every `esmc6b/*.pt` is `scratch/esmc6b_lib.py` +
> `scratch/esmc6b_gen_fasta.py`, which hand-build an `esm.models.esmc.ESMC` 6B module and load the
> native-format safetensors shard-by-shard.
>
> **This step runs in `esmgen`, not `MULAN_VENV_BIN`** — `esmc6b_lib` imports `esm`, which the
> `mulan` training env does not have.

```bash
ESMC_EMB=scratch/embeddings_skempi_full/esmc6b \
ESMC_SNAP="$HOME/esmc6b_weights/esmc-6b" ESMC_DEVICE=cuda ESMC_DTYPE=bfloat16 \
  "$HOME/miniforge3/envs/esmgen/bin/python" \
      scratch/esmc6b_gen_fasta.py scratch/skempi_full/wt_sequences.fasta
ls scratch/embeddings_skempi_full/esmc6b/*.pt | wc -l   # expect 630
```

`ESMC_SNAP` must be the **`esmc-6b/` subdirectory** (6 `model-*.safetensors` shards), not
`~/esmc6b_weights` itself — the loader globs `$ESMC_SNAP/model-*.safetensors` and silently
produces a randomly-initialised model if the glob is empty.

`esmc6b_gen_fasta.py` takes a **label→sequence fasta** whose headers already carry MuLAN's exact
`_fill_metadata` naming; it skips labels already cached, so it is resumable. For the other PLMs
(`ankh3_*`, `aido`, …) the ordinary `plm-embed`/`gen_emb_generic.py` path is fine.

## Step 2 — train the arms (base + both FoldX heads)

```bash
ES_CONFIG=experiments/full_skempi_seqonly/config_skempi_full.sh \
  ARMS="base foldx foldx_scalar" MAXPAR=3 \
  bash experiments/retrain_split/run_bycomplex.sh esmc6b
```
- `base` → `lightatt_default` on the 4-col clustered splits.
- `foldx` → `lightatt_addscores_mlp` + `--add_zs_scores` on the 16-col (12-term) splits.
- `foldx_scalar` → `lightatt_addscores` + `--add_zs_scores` on the 5-col scalar splits.
- 300 ep / patience 30 / bs 32 / lr 5e-4, 3 clustered folds. Resumable (skips a fold whose
  `all_results.json` exists). `MAXPAR=3` is safe post-seed (head training is light; the 6B model
  is no longer loaded). Skipping Step 1 requires `MAXPAR=1` so only one fold generates the cache.
- Outputs: `scratch/results/full_skempi/esmc6b_{base,foldx,foldx_scalar}/fold_{0,1,2}/training_run/`.

## Step 3 — score

```bash
"$MULAN_VENV_BIN/python" experiments/full_skempi_seqonly/evaluate.py
# per-structure Spearman/Pearson + AUROC per fold; writes experiments/full_skempi_seqonly/results/
```

## Gotchas learned on the Mac run — apply BEFORE the GPU run (2026-07-18)

1. **Multi-point needs a collator patch to `mulan/data.py` (REQUIRED for any multi-point ESM-C 6B
   run; not needed for single-point).** Base MuLAN cannot batch multi-point mutations at
   `batch_size>1`. `MulanDataset.__getitem__` returns a `data` field =
   `MutatedComplex(s1, s2, tuple(mutations))`; the collator has no case for it, so it hits
   `default_collate`, which stacks the mutations tuple. Single-point rows have exactly 1 mutation
   (fine); multi-point rows have 2–27 and *variable* counts →
   `RuntimeError: each element in list of batch should be of equal size` on the first batch, every
   arm/fold, ~2 s crash. `data` is pure metadata (`compute_loss` in `mulan/train_utils.py` does
   `inputs.pop("data")` before the model), so the fix is a passthrough branch added just before the
   `default_collate` fallback in `MulanDataCollator._collate_fn`:
   ```python
   elif isinstance(elem, MutatedComplex):
       return list(batch)   # metadata; popped in compute_loss, never stacked
   ```
   Provably zero-impact on single-point (the field is discarded downstream either way).

   **The branch is committed in this tree**, in `MulanDataCollator._collate_fn`
   (`mulan/data.py`), so multi-point collation works from a clean checkout with nothing to apply
   by hand. Confirm with:
   `python -c "import inspect,mulan.data as d; print('MutatedComplex' in inspect.getsource(d.MulanDataCollator._collate_fn))"`

2. **Never edit a script (or any sourced file) while its pueue task is mid-execution.** Bash reads
   scripts incrementally; editing `run_bycomplex.sh` while a task runs shifts byte offsets and
   corrupts the read at EOF → `syntax error near unexpected token '}'` and a non-zero exit code
   *even though the training finished and wrote valid results*. That poisoned exit code then
   cascade-fails every dependent pueue task. To change arms/config for a not-yet-started
   job, stash it and edit only while nothing is executing the file.

3. **Restart, don't rebuild, after a false failure.** A task marked `Failed`/`DependencyFailed`
   whose `all_results.json` files exist actually finished — `run_bycomplex.sh` is resumable and
   skips any fold with `all_results.json`, so `pueue restart --in-place <ids>` flips it green fast
   without retraining. Crashed folds leave a `training_run.log` but **no** `all_results.json`, so a
   rerun correctly redoes only those.

## Caveats / interpretation

- **FoldX coverage** (single-point). **Superseded 2026-07-30: the fix below landed. Do not quote
  the 87.8% / 98.7% / 90.8% triple in the next paragraph — it is the pre-fix state, and the
  dual-key merge that produced it was itself retracted.** Current figures follow the paragraph.

  The pre-fix state, kept for lineage: **87.8%** (3,655/4,165 rows) as of the 2026-07-27 dual-key
  merge — *not* the 28% this doc used to claim, which predated both the full-SKEMPI FoldX compute
  and the dual-key fix in `load_foldx`. Uncovered rows get the per-fold standardized-mean 0 (same
  convention as the S1102 merge), so the +FoldX arms test FoldX's lift on the covered subset — the
  base arm is unaffected. MP is 98.7%; combined SP+MP is 90.8%.
  Measured now — SP 99.2% (12393/12495 non-zero
  scalars), combined SP+MP 99.1% (5746/5801 per fold). Do not quote the 87.8% / 90.8% pair as
  current; they are the pre-fix state this bundle was built against, and the dual-key merge that
  produced them was itself retracted — its second key landed on other real rows and handed them a
  different mutation's ddG. Arms trained under it carry a `__cov87` archive.
  The remaining SP gap is a **residue-numbering join bug, not missing compute** — 494 of the 510
  uncovered rows are recoverable with zero new FoldX runs (→ 99.6% SP / 99.4% ALL). Losses
  concentrate on multi-chain antibody/TCR interfaces, so measured FoldX lifts are a **floor**.
  See `SP_FOLDX_COVERAGE_AUDIT.md` for the diagnosis and the fix spec; re-run
  `merge_foldx_full_skempi.py` after applying it.
- **Interpretation gate (§8b.2):** is sequence-only `base` in range of the sequence frontier, and
  does `+FoldX` move toward the ProtBFF leakage-controlled anchor (~0.51 P / 0.48 S)?
- fold_0 test = the protease-inhibitor mega-family — report per-fold, not just the mean.

## Re-building the FoldX splits (only if FoldX results change)

```bash
"$MULAN_VENV_BIN/python" experiments/full_skempi_seqonly/merge_foldx_full_skempi.py \
    --src scratch/splits_skempi_full_clustered_id60_kfold \
    --results scratch/foldx_skempi_full/results \
    --outdir scratch/foldx_skempi_full
```
