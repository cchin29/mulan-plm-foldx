# Codebase review — 2026-07-12

Four-partition read of the MuLAN codebase (protein ΔΔG prediction). Each partition
was reviewed independently for correctness, silent-failure, and cache-staleness bugs,
then consolidated and ranked by real-world impact (results/figure integrity weighted
highest). **No CRITICAL bug fires in any mainline flow.** Every real risk is an
edge/less-exercised path or a silent-failure/staleness footgun.

Partitions: **(1)** `mulan/` core + `scripts/` entrypoints · **(2)** `mulan/structural_context/`
+ `experiments/structctx_sweep/` · **(3)** embedding-gen + sweep scripts (`experiments/gen_*.py`,
`experiments/embedding_sweep/*`) · **(4)** `scripts_plots/*.py`.

Status legend: ✅ fixed 2026-07-12 · ⬜ open. Verdict: CONFIRMED (reproduced) / PLAUSIBLE (suspected).

---

## Did any of this affect committed results?

**Only Tier-1 B, and only one figure.** Verified empirically:

- **B (aug connector)** — YES. The committed `scripts_plots/ddg_scaling.png/.svg` (50ep scaling,
  commit `dd90dc7`) drew the `Ankh3-large +aug` lift from the wrong base (0.832 `Ankh-large`
  instead of its own 0.823). Underlying CSV numbers were all correct; only the drawn connector
  was wrong. **Fixed + re-rendered.**
- **A / D (aggregate.py)** — NO. Every committed sweep JSON is a genuine 10/10 folds with real
  per-fold PCCs. The one run A *would* have blessed (esmc600m `#19` no-op) was caught manually
  and is in no summary. D is a crash, not silent corruption — it never fired.
- **C (`strict=False`)** — NO. Not on the training path: `scripts/train.py:110-111` builds the
  model fresh (`LightAttModel(config)`), never loads a checkpoint. The `strict=False` site is only
  reached by `predict.py`/inference, which produced none of the committed CV/sweep results.

---

## Tier 1 — can silently corrupt a result or paper figure

| id | ✓ | partition | file:line | issue | verdict |
|----|---|-----------|-----------|-------|---------|
| A | ✅ | embedding | `embedding_sweep/aggregate.py:55` | Averaged over whatever folds existed with no `n==ES_NUM_FOLDS` check; a 0-fold model vanished from `summary.md` entirely — a truncated/no-op sweep could be committed as a valid paper number. | CONFIRMED |
| B | ✅ | plotting | `plot_ddg_scaling.py:228` | Aug→base lift connector matched by **param count**, which is non-unique (Ankh-large & Ankh3-large both 1.15e9) — `Ankh3-large +aug` anchored to the wrong base on the 50ep figure. | CONFIRMED |
| C | ✅ | core | `modules.py:252` | `from_pretrained` loaded with `strict=False` — a checkpoint predating a head (struct_context/ctx_proj/zs_mlp) loads with that head at random init, no warning, confidently wrong ΔΔG. | CONFIRMED |
| D | ✅ | embedding | `embedding_sweep/aggregate.py:107` | A fold with `all_results.json` but null `test_pcc` → `f"{None:.4f}"` **TypeError aborts the whole aggregation** (no summary written for the healthy models either). | CONFIRMED |

**Fixes applied**

- **A** — `fold_metrics` now returns `(rows, missing)`; `main` flags any run where
  `folds-with-test_pcc != expected` as PARTIAL (stderr + a `⚠ n/expected` cell + a warnings
  footer in `summary.md`), and a 0-fold model is reported as SKIPPED instead of silently dropped.
  JSON gains `expected_folds` + `complete` provenance. *(Verified: the fix immediately caught a
  genuinely partial 4/10 rng-`saprot` run the old code would have averaged.)*
- **D** — None-safe `f4`/`f3` formatters for every summary cell; `Δ PCC` guards `pcc_mean is None`.
- **B** — connector now matches on **model identity** via `canon_model()` (normalizes the literal
  `\n` used in some names, strips the ` +aug` marker), with a params-match fallback that *warns*.
  Two genuinely-inconsistent aug labels corrected in the CSV so all 10 rows canonicalize to their
  base: `Ankh +aug`→`Ankh-large +aug`, `ProstT5 +aug`→`ProstT5 (AA) +aug`. 50ep figure re-rendered.
- **C** — `from_pretrained` now captures `load_state_dict`'s missing/unexpected keys and
  `warnings.warn(RuntimeWarning)` on any mismatch (also switched to the lenient
  `MulanConfig.from_dict` — see T2).

---

## Tier 2 — latent footguns (guarded today, break silently when conditions shift)

| id | ✓ | partition | file:line | issue | verdict |
|----|---|-----------|-----------|-------|---------|
| T2-1 | ✅ | plotting | `plot_variability.py:82,171` | A model in the CSV but absent from the hardcoded `MODEL_ORDER` is silently dropped (no cell, no warning). `MODEL_ORDER` and `aggregate_folds.py`'s MAP are maintained separately. | CONFIRMED |
| T2-2 | ✅ | embedding | `embedding_sweep/run_sweep.sh:12-13` | Hardcoded `source config.sh` + absolute `cd` — running it with the balanced config silently wrote into the paper-anchor dir; only tracked runner that ignored `ES_CONFIG`. | CONFIRMED |
| T2-3 | ✅ | structctx | `persist.py:84` | DSSP disk-cache key omitted the mkdssp **executable** (inconsistent with its own in-memory key) and the **structure content** — switching mkdssp builds or force-refetching a structure served stale RSA/SS silently. | CONFIRMED |
| T2-4 | ✅ | structctx | `fetch.py:134-162` | `fetch_structure` dropped `version`/`fragments` on cache hits — the multi-fragment warning fired only on first download, so a cached multi-fragment protein silently used F1; `version` provenance was `None` after the first run. | CONFIRMED |
| T2-5 | ✅ | core | `data.py:303` | `data[(split_index==1) & (data[split_index==2])]` — `ndarray & DataFrame`, raises. Only hit with `num_folds==1, add_validation_set=False`. | CONFIRMED |
| T2-6 | ✅ | core | `modules.py:247` | `from_pretrained` used `MulanConfig(**config)` — a cross-version checkpoint with a renamed field crashes here but loads fine via the lenient `from_dict`. | CONFIRMED |

**Fixes applied**

- **T2-1** — `plot()` warns to stderr listing any CSV model missing from this series' `MODEL_ORDER`.
- **T2-2** — derives repo root from `BASH_SOURCE` (portable) and `source "${ES_CONFIG:-…/config.sh}"`,
  matching `gen_splits.py`/`aggregate.py`.
- **T2-3** — DSSP key now includes `_dssp_exe_tag(cfg)` (executable basename) and the mmCIF
  `_structure_fingerprint`; docstring updated.
- **T2-4** — provenance sidecar `<ACC>.meta.json` written at download, restored on cache hit; the
  multi-fragment warning moved out of the download branch so it fires on cached reads too.
- **T2-5** — `train_data = data[split_index != 0]` (everything not held out for test).
- **T2-6** — switched to `MulanConfig.from_dict` (folded into the C fix).

---

## Tier 3 — hardening / hygiene (fixed 2026-07-12)

| ✓ | partition | file:line | issue → fix |
|---|-----------|-----------|-------|
| ✅ | embedding | `gen_emb_generic.py`, `scratch/gen_plm_emb.py` | No coverage assert. → both now assert every dataset id produced a `.pt` and spot-check tensors are finite `[L, dim]` with one consistent width; `EMBEDDING_SETUP.md` §7 corrected. |
| ✅ | plotting | `aggregate_folds.py` | No fold-count/dedup guard; leaked handles. → `folds()` keys by fold id (dup warned+ignored), returns sorted; all reads/writes use `with`. |
| ✅ | core | `utils.py:60` | Non-A/B chain char aliased mutant onto WT. → `parse_mutations` now raises `ValueError` on any chain code other than 'A'/'B'. |
| ✅ | core | `modules.py:228` | `add_scores=True` + `zs_scores=None` gave a cryptic shape error. → explicit `ValueError` naming the cause before `self.linear`. |
| ✅ | core | `data.py:157` | `add_zs_scores` column semantics were a footgun. → documented the full column contract (4-col+flag = inference-only; >4-col = FoldX label+zs, flag ignored). |
| ✅ | core | `predict.py:90` | Multi-point zs-score key mismatch. → `parse_zs_scores` splits on `:` to match `parse_input`. |
| ✅ | structctx | `dssp.py:108` | `run_dssp` returned `{}` silently on empty output. → raises `RuntimeError` on empty keys, and on no residues for the selected chain. |
| ✅ | structctx | `context.py:180`, `dssp.py:62` | Poisoned disk embedding never evicted; silent version default. → length-guard failure on a disk-loaded tensor now `persist.evict_embedding`s it (self-heal); `_detect_version` warns on fallback. |
| ✅ | plotting | 3 diagram scripts | Bare-filename `savefig`; dead var. → all anchored to `HERE`; `ANKH3_SECOND` removed. |
| 📝 | core | `modules.py:124` | Pad mask could false-mask an exactly-0.0 channel-0 residue. → **documented, not changed**: threading an explicit pad mask would alter the mask on existing runs and invalidate trained checkpoints / reported results; probability ~0 for float embeddings. |

---

## Partition-health notes

- **Core** — mainline CV/training path correct (both splitters assign each sample to test once,
  val=`(test-1)%k`, train excludes both; no leakage/off-by-one). In-RAM embedding cache is
  numerically safe.
- **structural_context** — unusually well-tested (contact self-exclusion, Ambler numbering,
  NaN/JSON discipline, structure-fingerprint invalidation for SaProt all have real regression
  tests; 77 pass / 6 skipped). Residual risk was cache **invalidation** under non-default ops (T2-3/4).
- **Embedding** — id/sequence enumeration matches `mulan/data.py` verbatim; reverse-mutant
  double-notation + SaProt parent-3Di correct. Weak spot was the reporting layer (aggregate.py,
  run_sweep.sh).
- **Plotting** — aggregation math correct; every measured point ties out across the two
  pipelines. The one defect reaching a figure was the params-keyed connector (B).

## Verification

`ast.parse` clean on all edited files; `pytest mulan/structural_context/tests/` → **77 passed,
6 skipped**; all three variability series + the 50ep scaling figure re-render without warnings.
