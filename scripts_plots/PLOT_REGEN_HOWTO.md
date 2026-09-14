# Plot regeneration HOWTO

Canonical reference + regen steps for **every** script under `scripts_plots/`. Read this before
hand-editing any figure CSV or re-running a plot. The second half (the numbered `§0–§5`
pipeline) is the detailed, non-rediscoverable procedure for the `ddg_scaling` / `variability`
series; the first half is a complete inventory of what each script does, reads, writes, and
whether it still looks Active.

## 0. The interpreter

Always use the repo venv. It lives at repo-root `mulan/.venv`, so **from the repo root**:

```bash
./.venv/bin/python scripts_plots/<script>.py       # canonical
# or, from inside scripts_plots/:
cd "<repo>/scripts_plots" && ../.venv/bin/python <script>.py
```

## 0b. `results_matrix.py` is the data engine most figures read

`scripts_plots/results_matrix.py` (stdlib only) walks every result tree, reads `test_pcc` from
`fold_*/training_run/all_results.json`, and (re)writes two CSVs that several figures consume:

- `results_matrix.py` → `results_matrix.csv` (pooled Pearson matrix across PLM × arm × regime)
- `results_matrix.py --ps` → `results_matrix_ps.csv` (**per-structure Spearman T≥10** honest ladder)
- `results_matrix.py --res` → runtime (min/fold) + inferred platform (no CSV rewrite)
- `results_matrix.py --csv-only` → rewrite the pooled CSV without printing
- `results_matrix.py <substr>` → print-only, filtered by regime name (does **not** rewrite the CSV)

**General regen flow:** land folds → re-run `results_matrix.py` (and `--ps`) to refresh the CSVs →
re-run the figure scripts that read them. Independently, the `ddg_scaling`/`variability` series has
its own `aggregate_folds.py` → `benchmarks_folds*.csv` / hand-curated `ddg_scaling_data*.csv`
pipeline (see §1–§5).

⚠️ **The two CSVs have different input requirements, and only one of them can be rebuilt from a
checkout of this repository.** `--ps` reads the committed per-structure rung CSVs under
`experiments/{rescore_perstructure,retrain_split,full_skempi_seqonly}/`, all of which are in the
tree, so it rebuilds `results_matrix_ps.csv` byte-for-byte anywhere. The pooled view instead walks
`scratch/results/**/fold_*/training_run/all_results.json`, and `scratch/` is not published —
without it every row still appears, with `status=empty` and blank metrics, and the CSV is
overwritten in place. Run the pooled rebuild only where the fold JSONs exist; `results_matrix.csv`
is committed as an artefact of such a run, not as something a reader regenerates.

`experiments/gen_benchmark_matrix_data.py` has the same requirement and for the same reason: it
recomputes the MuLAN cells of `BENCHMARK_MATRIX.md` from the per-fold prediction TSVs under
`scratch/results/`. Without them it reports that nothing scored and exits — it does not print a
comparison. Its output is committed in `BENCHMARK_MATRIX.md`; a reader does not regenerate it.

`results_matrix_ps.csv` carries **two** metrics per row. `ps_spearman_T10` is per-structure —
computed within each complex, then averaged over complexes — and is the conclusion metric.
`spearman_foldavg` is computed within each test fold, then averaged over folds; it is the form the
CD-HIT ≤60% frontier column is reported in, and the only one comparable to it. They are not
interchangeable and neither is the pooled correlation. Blank means the rung predates the column,
which is "not scored", never zero.

`scripts_plots/artifacts/` is where dated decks are built, and it is **not tracked**. The published
slides are one self-contained file, `docs/slides/mulan-plm-foldx-slides.html`, and the figures
inside it are snapshots of the state on its build date rather than live output — regenerating a
script here does not update them. Rebuild the deck and re-bundle it (`DECK_HOWTO.md`).

---

## 1. Full script inventory

Statuses are an **inference from git activity** (newest commit in the repo is 2026-07-22; the active
window is roughly the last ~2 weeks), not a guarantee. "Superseded by X" means X is a newer sibling
that covers the same ground. Git range = first→last commit date (`--follow`, first-parent).

### Active — flagship figures & data engine

| Script | Purpose (short) | Output file(s) | Git range | Status |
|---|---|---|---|---|
| `results_matrix.py` | Coverage/data matrix; writes the CSVs most figures read | `results_matrix.csv`, `results_matrix_ps.csv` | 2026-07-18 → 2026-07-21 (6) | **Active (engine)** |
| `plot_ppS_scaling.py` | Per-structure Spearman vs PLM scale, 6-panel (ΔΔG + 3 splits + CD-HIT fold-avg & CATH-AUROC companions) | `ppS_scaling.{png,svg}` | 2026-07-21 → 2026-07-22 (4) | **Active (flagship)** |
| `plot_ppS_generalization.py` | Per-structure Spearman bars across embeddings, 3 leakage splits | `ppS_generalization.{png,svg}` | 2026-07-21 → 2026-07-22 (2) | **Active (flagship)** |
| `plot_ddg_scaling.py` | ΔΔG performance/cost vs model scale (the "scaling" scatter) | `ddg_scaling*.{png,svg}` | 2026-07-02 → 2026-07-14 (14) | **Active (flagship)** |
| `plot_variability.py` | JMP-style per-fold variability chart (PLM→aug→benchmark) | `ddg_variability*.{png,svg}` | 2026-07-04 → 2026-07-15 (23) | **Active (flagship)** |
| `aggregate_folds.py` | Builds the variability CSV from fold JSONs (series-aware) | `benchmarks_folds*.csv` | 2026-07-04 → 2026-07-13 (14) | **Active (feeds variability)** |
| `plot_common.py` | Shared theme/palette/CSV-loader module (imported by figures) | — (library) | 2026-07-04 (1) | **Active (shared lib)** |

### Active — schematics (data-free; re-run only on code change)

| Script | Purpose (short) | Output file(s) | Git range | Status |
|---|---|---|---|---|
| `plot_augmentation.py` | Augmentation schematic (inline, no data) | `augmentation.{png,svg}` | 2026-07-07 → 2026-07-12 (2) | Active (schematic) |
| `plot_model_timeline.py` | PLM release-timeline schematic (inline) | `model_timeline.{png,svg}` | 2026-07-07 → 2026-07-12 (2) | Active (schematic) |
| `plot_structure_fusion.py` | Structure-fusion architecture schematic (inline) | `structure_fusion.{png,svg}` | 2026-07-07 → 2026-07-12 (2) | Active (schematic) |

### Active — utility (not a figure)

| Script | Purpose (short) | Output file(s) | Git range | Status |
|---|---|---|---|---|
| `queue_watch.py` | Pueue watcher / auto-heal daemon | `scratch/queue_watch.*` | 2026-07-18 (1) | Active (utility) |

### Likely superseded — recent one-off frontier/ladder figures

These are recent (07-18/07-19) but conceptually **superseded by the `plot_ppS_*` flagship pair**
(07-22), which draws the same leakage-controlled ladder with frontier reference bands from a single
source (`results_matrix_ps.csv`). Inference from activity + overlap, not certainty.

| Script | Purpose (short) | Output file(s) | Git range | Status |
|---|---|---|---|---|
| `plot_generalization_ladder.py` | Headline figure: base collapses / FoldX holds across splits | `generalization_ladder.png` | 2026-07-19 (1) | Superseded? by `plot_ppS_*` |
| `plot_frontier_comparison.py` | Two-panel figure: MuLAN honest arms vs published SKEMPI frontier | `frontier_comparison.png` | 2026-07-18 (1) | Superseded? by `plot_ppS_generalization` |

### Diagnostics / one-off analyses (not paper figures)

Exploratory scripts with **hard-coded Linux-snapshot paths**
— `ROOT` now resolves from the script's own location, and the `analyze*` pair writes under
`scratch/outputs/`.
Kept for provenance; not part of the regen flow.

| Script | Purpose (short) | Output file(s) | Git range | Status |
|---|---|---|---|---|
| `plot_error_vs_ddg.py` | Per-model mispredict-error vs true ΔΔG scatter matrix (shrink-to-mean) | `error_vs_ddg_matrix.png` (env `OUT` overrides) | 2026-07-16 (1) | One-off diagnostic |
| `analyze_hotspots.py` | Cross-arm mispredicted-ΔΔG hotspot hunt (balanced OOF) | `hotspots_table.json` + `hotspot_report.txt` | 2026-07-15 (1) | One-off diagnostic |
| `analyze2.py` | Pass-2: pooled PCC per arm + FoldX-term residual diagnostics | `hotspot_report2.txt` | 2026-07-15 (1) | One-off diagnostic |

---

## 2. Per-script detail

### Flagships & engine

**`results_matrix.py`** — Status matrix across `{PLM} × {arm} × {regime}`. Reads `test_pcc` from
`fold_*/training_run/all_results.json` under `scratch/results/`, `scratch/foldx_s1102/`,
`scratch/gpu_results/`; the `--ps` view instead reads pre-scored per-structure CSVs under
`experiments/{rescore_perstructure,retrain_split,full_skempi_seqonly}/`. Arms are normalized to
`base | aug | fx_scalar | fx_mlp`. Every row carries test-set counts so an S1102 row (~366/fold)
can't be confused with a full-SKEMPI row (thousands). **Writes** `results_matrix.csv` (unfiltered
runs only) and, under `--ps`, `results_matrix_ps.csv`. Usage: `./.venv/bin/python
scripts_plots/results_matrix.py [--ps|--res|--csv-only|<regime-substr>]`.

**`plot_ppS_scaling.py`** *(flagship)* — 2×3 figure organized by split tier vs PLM
scale (shared log-x). **Panel 1** redraws the `ddg_scaling_300ep_balanced` performance panel
(Pearson r vs params). **Panels 2 / 3a / 4a** = per-structure Spearman ρ (T≥10) on the three
leakage-controlled full-SKEMPI splits (by-complex / clustered / CATH-superfamily). **Panels 3b / 4b**
= the frontier-comparable *companion* metrics sitting directly above their per-structure panel:
**3b** clustered **CD-HIT ≤60% fold-averaged Spearman** (the pooled-style metric ProtBFF/ProMIM/ProSST
report) and **4b** **CATH AUROC** (sign-of-effect, all-muts). Each PLM shows base + FoldX-scalar
(diamond) + FoldX-12-term-MLP (triangle). **Inputs:** `results_matrix_ps.csv` (the per-structure
panels 2/3a/4a) and `ddg_scaling_data_300ep_balanced.csv` (panel 1). The companion panels 3b/4b and
all **frontier reference bands** (violet) are **hard-coded in the script from
`experiments/BENCHMARK_MATRIX.md`** (CD-HIT Sp column → `CDHIT_FOLDAVG_S`; AUROC companion table →
`CATH_AUROC`; `FRONTIER*` dicts) — update those dicts when the matrix changes; they are NOT read from
a CSV. **Outputs:** `ppS_scaling.{png,svg}`. Usage: `./.venv/bin/python
scripts_plots/plot_ppS_scaling.py`. No env/CLI args.

**`plot_ppS_generalization.py`** *(flagship)* — Grouped bar version: per-structure
Spearman (T≥10) across embeddings on the same three leakage-controlled splits, base vs
+FoldX-scalar vs +FoldX-MLP. **Input:** `results_matrix_ps.csv` (single source of truth). Same
`experiments/BENCHMARK_MATRIX.md` frontier bands as `plot_ppS_scaling.py` (kept in sync).
**Outputs:** `ppS_generalization.{png,svg}`. Usage: `./.venv/bin/python
scripts_plots/plot_ppS_generalization.py`. No env/CLI args.

**`plot_ddg_scaling.py`** — ΔΔG performance (Pearson r, S1102 10-fold CV) and cost (peak RAM /
runtime) vs encoder params, in the "emergent property vs scale" style. Data-driven from
`ddg_scaling_data*.csv`; add a model = append a CSV row, no code change (pending rows render as
placeholders). **Series-aware** via `MULAN_PLOT_SERIES` (`50ep` default / `300ep` /
`300ep_balanced`). **Outputs:** `ddg_scaling*.{png,svg}`. See §2–§5 for the series mechanics.

**`plot_variability.py`** — JMP-style variability chart: per-fold Pearson r nested PLM → aug →
benchmark, with box plots, mean diamonds, and paper-target reference lines. **Input:**
`benchmarks_folds*.csv` (aggregator-generated — never hand-edit). Series-aware via
`MULAN_PLOT_SERIES`. Has a **loud `MODEL_ORDER` guard** (see §4). **Outputs:**
`ddg_variability*.{png,svg}`.

**`aggregate_folds.py`** — Builds the variability CSV (`benchmarks_folds*.csv`) by scanning fold
`all_results.json` and emitting per-fold `pcc`. Series-aware via `MULAN_PLOT_SERIES`: `50ep` uses
a curated `MAP`; `300ep*` auto-discover `<tag>[_aug]` dirs under `scratch/results/embedding_sweep*`
via a `TAG2NAME` dict. The run report prints each cell's `mean`/`pstdev` (the exact `pcc`/`pcc_std`
for the hand-curated scaling CSV). See §2–§3.

**`plot_common.py`** — Shared theme module (no plotting): `apply_theme()` rcParams, the
platform/per-model/per-dataset color families, `STATUS_STYLE`, and the `load_csv()` convention.
Imported by the figure scripts so the set shares one visual identity. *(Note: its docstring also
references a `plot_benchmarks.py`, which is not present in this directory — a stale mention.)*

### Schematics (data-free)

**`plot_augmentation.py`**, **`plot_model_timeline.py`**, **`plot_structure_fusion.py`** — Inline,
hand-laid schematics with no data inputs. Re-run only when their code changes:
`./.venv/bin/python scripts_plots/plot_<name>.py` → `<name>.{png,svg}`.

### Utility

**`queue_watch.py`** — Pueue watcher: every `WATCH_INTERVAL` (default 600s) it reads
`pueue status --json` and auto-restarts recoverable `Failed`/`DependencyFailed` tasks (up to
`WATCH_CAP`, default 2) in `WATCH_GROUPS`, writing `scratch/queue_watch.{status,log,state,ALERT}`.
Usage: `python3 scripts_plots/queue_watch.py --once`
(single poll) or run as a pueue daemon (see its docstring). Env: `WATCH_GROUPS`, `WATCH_INTERVAL`,
`WATCH_CAP`.

### Likely superseded

**`plot_generalization_ladder.py`** — Headline figure: Panel A = mean base per-structure Spearman
collapsing across leaky→by-complex→clustered while FoldX holds (widening gap); Panel B = clustered
ρ decomposed into base + FoldX lift. **Input:** `results_matrix_ps.csv`. **Output:**
`generalization_ladder.png`. Its story is now carried by
`plot_ppS_scaling.py`/`plot_ppS_generalization.py`.

Two breakages were fixed on 2026-08-04, both of which made it look superseded when it was merely
unrunnable. `ROOT` was a hard-coded Linux path, now resolved from `__file__`; and its three `split`
lookups were the pre-prefix names (`"leaky (S1102)"`, `"by-complex"`, `"clustered"`), which
`results_matrix.py` has since renamed to `"S1102 leaky (per-mut CV)"` and siblings. Unprefixed they
matched no row, so every model lookup raised `KeyError` rather than drawing an empty panel. All
three rungs are S1102_filtered, so the 2026-08-04 FoldX coverage step — single-point full-SKEMPI
only — leaves the figure's numbers where they were.

**`plot_frontier_comparison.py`** — Two-panel figure placing MuLAN's honest (CD-HIT ≤60%)
arms against the published SKEMPI-v2 frontier (Panel A pooled Spearman vs ProtBFF peers; Panel B
per-structure vs USP-ddG/CATH-ddG). **Input:** `experiments/retrain_split/results_clustered.csv`
(+ frontier numbers hard-coded from `reference_ddg_splits/*.md`). **Output:**
`frontier_comparison.png`. Overlaps the frontier-band story now in `plot_ppS_generalization.py`.

### Diagnostics / one-off (Linux-snapshot paths; not in the regen flow)

**`plot_error_vs_ddg.py`** — Per-model signed mispredict-error vs true ΔΔG scatter matrix on the
balanced 300/30 OOF folds (shows the shared shrink-to-mean blind spot). Reads
`test_predictions.tsv` from `scratch/results/embedding_sweep_balanced/` + esmc6b incoming.
**Output:** `error_vs_ddg_matrix.png` (`OUT` env overrides). Hard-coded Linux `ROOT`.

**`analyze_hotspots.py`** — Cross-arm mispredicted-ΔΔG hotspot hunt over pooled balanced OOF folds
(PLMs with base/aug/foldxmlp). **Outputs:** `hotspots_table.json` + `hotspot_report.txt` (to the
`scratch/outputs/`).

**`analyze2.py`** — Pass-2 diagnostics: pooled PCC per arm, FoldX-term-vs-residual correlations,
hard-complex identity lookup from `scratch/skempi_v2.csv`. **Output:** `hotspot_report2.txt`.
Hard-coded Linux paths.

---

## The `ddg_scaling` / `variability` series — detailed regen flow

(Everything below is the non-rediscoverable procedure for the two data-driven flagship series and
their aggregator. Supersedes the balanced/300ep gaps in `PLOT_UPDATE.md`, which only documents the
old 50ep auto pipeline.)

## 2. The three series (this is the part people forget)

Both `plot_ddg_scaling.py` and `plot_variability.py` render **three training protocols**, selected
by the `MULAN_PLOT_SERIES` env var (default `50ep`). Each series = its own pair of CSVs and its own
output basename:

| `MULAN_PLOT_SERIES` | scaling CSV | variability CSV | output suffix | how the CSVs are maintained |
|---|---|---|---|---|
| `50ep` (default) | `ddg_scaling_data.csv` | `benchmarks_folds.csv` | `ddg_scaling`, `ddg_variability` | scaling = **hand**; variability = **auto** (`aggregate_folds.py`, curated MAP) |
| `300ep` | `ddg_scaling_data_300ep.csv` | `benchmarks_folds_300ep.csv` | `*_300ep` | scaling = **hand**; variability = **auto** (aggregator auto-discovers) |
| `300ep_balanced` | `ddg_scaling_data_300ep_balanced.csv` | `benchmarks_folds_300ep_balanced.csv` | `*_300ep_balanced` | scaling = **hand**; variability = **auto** (aggregator auto-discovers) |

The **variability CSV of every series is now aggregator-generated** — never hand-edit it.
`aggregate_folds.py` is series-aware via the *same* `MULAN_PLOT_SERIES` env var:

```bash
# 1) (re)build the variability CSV for the series:
../.venv/bin/python aggregate_folds.py                                   # 50ep  -> benchmarks_folds.csv
MULAN_PLOT_SERIES=300ep_balanced ../.venv/bin/python aggregate_folds.py  # -> benchmarks_folds_300ep_balanced.csv
MULAN_PLOT_SERIES=300ep          ../.venv/bin/python aggregate_folds.py  # -> benchmarks_folds_300ep.csv
# 2) then render (both scripts read the series' CSVs):
MULAN_PLOT_SERIES=300ep_balanced ../.venv/bin/python plot_variability.py
MULAN_PLOT_SERIES=300ep_balanced ../.venv/bin/python plot_ddg_scaling.py
```

**How the two paths differ:** for `50ep` the aggregator uses the curated `MAP` (multi-dataset:
S1102 + the S1131/S4169/S2003 benchmark sweeps) scanning `scratch/results/cv10_*` / `bench_*`.
For `300ep`/`300ep_balanced` it **auto-discovers** every `<tag>[_aug]` dir under
`scratch/results/embedding_sweep[_balanced]/` (S1102-only), mapping tag→display name via the
`TAG2NAME` dict at the top of the script. New PLMs appear automatically once their folds land —
add the tag to `TAG2NAME` (and the model to `plot_variability._ORDER_*` to actually draw it).
Non-model dirs (e.g. `<tag>_foldxmlp` FoldX arms) have no `TAG2NAME` entry and are skipped +
listed in the run report. Partial cells (<10 folds) are emitted and render as an "n/10" cell.

The **scaling CSV is still hand-curated** (its `params`/`emb_dim`/`runtime`/layout columns
aren't in the fold JSONs) — but the aggregator run **prints each cell's `mean` and `pstdev`**,
which are exactly the `pcc` and `pcc_std` columns, so filling a scaling row is a copy of two
numbers (§3).

## 3. Adding a completed model to the `300ep_balanced` series

This is the recurring task (e.g. an `embsweep_bal_<plm>[_aug]` pueue job finishing). It is now
almost entirely automatic:

1. **Variability CSV — automatic.** Just re-run the aggregator; it discovers the new
   `<tag>[_aug]` dir and (re)writes every cell. If it's a brand-new PLM, first add its tag to
   `TAG2NAME` in `aggregate_folds.py` and the display name to `plot_variability._ORDER_300EP_BAL`
   (+ `MODEL_ABBR` / `MODEL_META`):
   ```bash
   MULAN_PLOT_SERIES=300ep_balanced ../.venv/bin/python aggregate_folds.py
   ```
   The run report prints `<Model> <base|aug> <n> folds  (<tag>)  mean=… pstdev=…` per cell —
   note the `mean`/`pstdev` for the next step. (Std convention is **population std**,
   `pstdev`; verified against existing rows, e.g. Ankh aug mean 0.8379 / pstdev 0.0574 → `0.057`.)
2. **Scaling CSV — one hand row.** Add (or replace) the model's row in
   `ddg_scaling_data_300ep_balanced.csv` using the printed `mean`→`pcc` and `pstdev`→`pcc_std`.
   Copy the column layout from an existing row: `+aug` rows use `series=aug_cv10` (diamond),
   base rows `series=seq_cv10` (dot). `params`/`emb_dim` come from the model's base row;
   `runtime` (h:mm) is the pueue wall-clock (`pueue status`, start→end) of the job. E.g.:
   ```
   ProstT5 +aug,1.21e9,0.8322,0.049,aug_cv10,measured,0,0,0,,,,1024,,,,mps,3:01,,
   ```
3. **Render both figures:**
   ```bash
   MULAN_PLOT_SERIES=300ep_balanced ../.venv/bin/python plot_variability.py
   MULAN_PLOT_SERIES=300ep_balanced ../.venv/bin/python plot_ddg_scaling.py
   ```

Adding a **base** model instead of `+aug`: same flow but the tag has no `_aug` suffix, the
benchmarks rows use `,base,`, and the scaling row uses `series=seq_cv10` (a plain dot, not a
diamond) — mirror an existing base row's columns.

## 3b. Adding a new PLM to the `ppS_scaling` / `ppS_generalization` figures

**This is a genuinely multi-file edit — the ppS figures do NOT auto-pick up a new PLM just because
its folds landed or it's in `results_matrix_ps.csv`.** Verified end-to-end 2026-07-23 adding
AIDO-16B (worked example below). Do all of these or the PLM renders partially / invisibly:

1. **Per-structure CSVs must carry the PLM first.** The panels read `results_matrix_ps.csv` +
   the `results_{sp,mp,cath}_*.csv`. Those come from the scorers, which have **hardcoded TAGS
   tuples** — add the tag to **both** `experiments/full_skempi_seqonly/score_multipoint.py` and
   `score_cath.py` (`score_sp_all.py` auto-discovers, no edit). Then re-run all three scorers +
   `results_matrix.py --ps` (see §5). Without this the PLM has **no ppS rows at all**.
2. **`plot_ppS_scaling.py` gates the per-structure panels (2/3a/4a) on a hardcoded `PARAMS`
   dict** (`models = [m for m in PARAMS if …]`, ~line 281) — a PLM present in the ps CSV is
   **invisible** until it is added to `PARAMS` **and** `EMB_DIM` **and** `LABEL` (label side/offset;
   `LABEL.get(m, ("bottom",14))` defaults, but set it explicitly to avoid overlaps). No lineage
   entry needed unless it has a same-family sibling.
3. **Companion panels 3b/4b are hardcoded dicts fed from `gen_benchmark_matrix_data.py`, NOT a
   CSV.** Add `("<Model>","<arm>"): <val>` to `CDHIT_FOLDAVG_S` (panel 3b, CD-HIT ≤60% fold-avg
   Spearman) and `CATH_AUROC` (panel 4b, CATH AUROC = `auroc_destab`) for base / fx_scalar /
   fx_mlp. A missing entry is silently **omitted** (no crash) — so 3b/4b will just drop the PLM if
   this is skipped. Source the numbers by running the generator (next point). Keep these dicts in
   sync between `plot_ppS_scaling.py` and `plot_ppS_generalization.py` (they duplicate the
   `FRONTIER` per-structure dict; the companion dicts live only in `plot_ppS_scaling.py`).
4. **`experiments/gen_benchmark_matrix_data.py` has its OWN hardcoded roster** — the `DISP` list
   (~line 55; tuples `(display, display2, tag)`). Add the PLM there, then run
   `./.venv/bin/python experiments/gen_benchmark_matrix_data.py`. Since 2026-08-06 it takes `--write`, which
   patches both `BENCHMARK_MATRIX.md` and `benchmark_matrix.html` in place, and `--verify`, which
   fails on any published cell that disagrees with the predictions. Both files are tracked and
   ship; neither is gitignored, and neither is hand-mirrored any more. In its stdout: first array = CD-HIT block, **CD-HIT fold-avg
   Spearman is column index 3**; the commented AUROC-companion array has **CATH AUROC at index 2**.
   Arm rows follow the model row with an empty first field (`["","+ FoldX scalar",…]`).
5. **Panel 1 of `ppS_scaling` = the same `ddg_scaling_data_300ep_balanced.csv` panel** — so §3's
   scaling-CSV rows also make the PLM appear on ppS panel 1. Model name there is the CSV `model`
   (e.g. `AIDO.Protein-16B`); on panels 2–4b it's the `results_matrix.py` display name (e.g.
   `AIDO-16B`). The two need not match (separate data sources) but pick sane names.
   - **Panel 1 glyphs now MATCH panels 2-4a** (2026-07-23): base = platform circle, **aug = ×**,
     **FoldX-scalar = orange diamond** (`#dd6b20`), **FoldX-MLP = blue triangle** (`#3182ce`) — the
     same `ARM_STYLE` scalar/MLP colors. This needs a **`foldxscalar_cv10` series row** per PLM in
     the balanced CSV (in addition to the `foldxmlp_cv10` row) — `draw_ddg_panel` draws each series
     separately. **`+FoldX scalar` rows must be named `"<Model> +FoldX scalar"`**; `_canon()` strips
     `+aug` / `+FoldX[-mlp|-scalar]` so the diamond/triangle stem links back to the model's base
     point — a name it can't canonicalize floats free. (This differs from the **standalone**
     `plot_ddg_scaling.py`, which still uses its old scheme: aug = diamond, FoldX-MLP = orange
     triangle, and ignores `foldxscalar_cv10`. Only `ppS_scaling` panel 1 was changed.)
6. **Render:** `./.venv/bin/python scripts_plots/plot_ppS_scaling.py` +
   `plot_ppS_generalization.py` (no env/args). `XLIM=(8e7, 2.4e10)` already fits a 16B model.

**AIDO-16B worked example (2026-07-23):** `PARAMS 1.6e10`, `EMB_DIM 2304`;
`CDHIT_FOLDAVG_S` base/scalar/mlp = 0.182 / 0.260 / 0.232; `CATH_AUROC` = 0.680 / 0.738 / 0.719;
ddg CSV rows base 0.8374 / aug 0.8391 / +FoldX(MLP) 0.8573 / +FoldX-scalar 0.8488.

**Label placement (`LABEL` dict, drives `_place_label` in all per-structure panels 2–4b):** entry
is `('bottom'|'top', dy)` = above/below the marker stack (ha=center), **or `('west', dx)`** = due
west of it (ha=right, vertically centred at the stack midpoint) — use `west` to pull crowded
low-param labels out into the empty left margin (SaProt + ESM-C 600M, which sit on top of each other
+ Ankh3-large at ~6e8 params). `dx`/`dy` are point-offsets (negative `dx` = further left).

**"Queue the missing FoldX-scalar for completeness":** panel 1 shows a scalar diamond only where a
`foldxscalar_cv10` row exists. As of 2026-07-23 the on-panel PLMs still missing balanced
FoldX-scalar are **ProstT5 + ESM-C 600M** (queued, pueue `foldx` #131/#132, MPS, via
`scratch/foldx_s1102/cv10_foldx_scalar_balanced_driver.sh <tag>` → `embedding_sweep_balanced/<tag>_foldxscalar`)
and **ESM-C 6B** (GPU-only, not yet run — the `foldx_esmc6b_balanced_chain.sh` † caveat). When they
land: add each PLM's `foldxscalar_cv10` row to the balanced CSV (mean/pstdev from the fold JSONs) and
re-render.

**Non-standard result tree → symlink into the canonical location** (don't special-case the code):
AIDO's S1102 balanced 10CV landed as `scratch/results/s1102_aido_balanced/{base,aug,foldx,foldx_mlp}`
instead of the usual `embedding_sweep_balanced/<tag>[_…]`. Mirror the esmc6b convention with
**relative symlinks** so the standard `s1102_300` path + `aggregate_folds.py` auto-discovery find it
(arm-dir naming differs — `foldx`=scalar, `foldx_mlp`=MLP):
```bash
cd scratch/results/embedding_sweep_balanced
ln -sfn ../s1102_aido_balanced/base      aido
ln -sfn ../s1102_aido_balanced/aug       aido_aug
ln -sfn ../s1102_aido_balanced/foldx     aido_foldxscalar
ln -sfn ../s1102_aido_balanced/foldx_mlp aido_foldxmlp
```
(Symlinks live under `scratch/` = gitignored, so they travel by tarball only, like the esmc6b ones.)

## 4. Gotchas

- **`plot_variability.py` has a loud MODEL_ORDER guard**: every `model` value in the CSV must
  appear in that script's hardcoded `MODEL_ORDER`, or it aborts. Base and `+aug` share the same
  `model` name (e.g. both `ProstT5`), so adding an aug block for a model that already has a base
  block needs no order edit. A **brand-new PLM** does: add it to `MODEL_ORDER` / `MODEL_ABBR` /
  `MODEL_META` (and to the ddg_scaling label positioning) before regenerating.
- **Population std, not sample std** (scaling CSV `pcc_std`). The aggregator already prints
  `pstdev`; if computing by hand, use `pstdev` — sample std (`stdev`) is ~5% larger and will
  silently disagree with every existing row.
- **ddg_scaling label collisions** (scaling CSV `dx`/`dy`, points-offset for the bold model
  label; `label=1` rows only). A new **rightmost / high-param** PLM's label easily overlaps the
  **right-aligned reference-line text** (`MuLAN-Ankh 0.868` / `MuLAN-ESM2-3B 0.854`). Place it
  **below the marker** (negative `dy`) in the empty region rather than above. The **same `dx`/`dy`
  drives `ppS_scaling` panel 1** (it reads this CSV), so one fix covers both figures. Worked
  example: AIDO-16B base used `dx=-6, dy=-56`.
- **Never hand-edit any `benchmarks_folds*.csv`** — all three are aggregator-generated. Re-run
  `aggregate_folds.py` with the matching `MULAN_PLOT_SERIES`. For a new PLM in a 300ep* series,
  add its tag to `TAG2NAME` (else the aggregator skips it and lists it under "skipped").
- **Verify before regenerating**: `ls scratch/results/embedding_sweep_balanced/<tag>/fold_*/training_run/all_results.json | wc -l`
  should be `10`. In-flight folds have only `checkpoint-*/trainer_state.json`, not `all_results.json`.
  The aggregator emits partial cells (they render "n/10"); re-run once the job finishes for the
  clean 10/10 cell.
- **Don't batch prematurely**: the balanced aug column is filled model-by-model as pueue jobs
  land (Ankh → ProstT5 → SaProt → ESM2 …). Each completed model is a clean append; add
  one at a time (precedent: commit 86e7799 added Ankh-large +aug alone).
- **The `plot_ppS_*` / ladder figures are separate** — they read `results_matrix_ps.csv`
  (regenerated by `results_matrix.py --ps`), not the `benchmarks_folds*`/`ddg_scaling_data*` CSVs.
  Their **per-structure** frontier comparator numbers are hard-coded from
  `experiments/BENCHMARK_MATRIX.md`; update the shared per-structure `FRONTIER` dict in **both**
  `plot_ppS_scaling.py` and `plot_ppS_generalization.py` together (they duplicate it). The
  **companion panels are `plot_ppS_scaling.py`-only**: its `CDHIT_FOLDAVG_S` + `FRONTIER_CDHIT`
  (panel 3b) and `CATH_AUROC` + `FRONTIER_CATH_AUROC` (panel 4b) are likewise hard-coded from
  BENCHMARK_MATRIX (CD-HIT Sp column / AUROC companion table) — refresh them there when the matrix
  changes. NB the CD-HIT metric is labeled **fold-averaged** in the figure; BENCHMARK_MATRIX still
  calls that column "pooled" (its `gen_benchmark_matrix_data.py` concatenates folds) — reconcile the
  wording when the doc changes.

## 5. Quick reference — regenerate everything currently in use

```bash
cd "<repo>/scripts_plots"
V=../.venv/bin/python
# --- per-structure ladder (flagships) ---
$V results_matrix.py --ps          # refresh results_matrix_ps.csv
$V plot_ppS_scaling.py
$V plot_ppS_generalization.py
# --- ddg_scaling / variability series ---
# 50ep (aggregate -> variability -> scaling):
$V aggregate_folds.py && $V plot_variability.py && $V plot_ddg_scaling.py
# 300ep balanced (aggregate CSV, then render; update scaling row by hand per §3 if new model):
MULAN_PLOT_SERIES=300ep_balanced $V aggregate_folds.py
MULAN_PLOT_SERIES=300ep_balanced $V plot_variability.py
MULAN_PLOT_SERIES=300ep_balanced $V plot_ddg_scaling.py
# 300ep rng (if used):
MULAN_PLOT_SERIES=300ep $V aggregate_folds.py
MULAN_PLOT_SERIES=300ep $V plot_variability.py
MULAN_PLOT_SERIES=300ep $V plot_ddg_scaling.py
# schematics (only on code change):
$V plot_augmentation.py && $V plot_model_timeline.py && $V plot_structure_fusion.py
```

---

## Appendix — how this inventory was generated / how to refresh it

The per-script metadata above was produced by:

1. `ls -la scripts_plots/` to enumerate `*.py` (and `*.sh`) scripts.
2. Reading each script's module docstring + skimming the code for **outputs** (`savefig`,
   `f"...png"`, `to_csv`, `print("wrote"...)`), **inputs** (`read_csv`, `open(`, `Path(...)`),
   and **args/env** (`argparse`, `sys.argv`, `os.environ`).
3. Git first/last commit dates per file (rename-following):
   ```bash
   git log --follow --no-patch --format=%as -- scripts_plots/<file>
   # last printed line = FIRST (oldest) commit; first line = MOST RECENT
   ```
   (Use `--no-patch`; this repo's git config otherwise injects diff output that corrupts the
   date parse.)
4. Classifying Active vs Likely-superseded from those ranges (newest repo commit ≈ 2026-07-22;
   active window ≈ last 2 weeks) plus obvious newer-sibling relationships.

To refresh: re-run the loop over the file list, diff against the table above, and update the
Status column. Statuses are inferences from git activity, not guarantees.
