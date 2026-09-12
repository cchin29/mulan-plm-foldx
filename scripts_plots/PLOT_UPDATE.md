# Plotting update — 2026-07-04

> **See `PLOT_REGEN_HOWTO.md` for the canonical regeneration steps** (the three
> `MULAN_PLOT_SERIES` series and the hand-maintained `300ep`/`*_balanced` CSVs). This doc
> below only covers the older **50ep auto** pipeline (`aggregate_folds.py`).

> **DEPRECATION (2026-07-05):** the **heatmap / faceted-bar** figures were removed —
> `plot_benchmarks.py`, `benchmarks_data.csv`, and `ddg_benchmarks_{heatmap,bars}.{png,svg}`
> are gone (recoverable from git history). Their content is already covered by the
> **variability chart** (per-fold benchmark PCC, `plot_variability.py`) and the **scaling
> figure** (`plot_ddg_scaling.py`), so they were redundant. The `plot_benchmarks.py` /
> `benchmarks_data.csv` lines below are retained for historical context only — **do not
> re-run them.** Live pipeline is now just `aggregate_folds.py → plot_variability.py` and
> `plot_ddg_scaling.py`.

Update to `scripts_plots/` of 2026-07-04. Adds the benchmark **variability chart**
and **heatmap/faceted-bar** figures, plus a shared theme module.

## Bundle contents

| File | Status | Purpose |
|---|---|---|
| `plot_common.py` | **new** | shared theme, palettes, helpers — imported by every `plot_*.py` |
| `aggregate_folds.py` | **new** | scans `../scratch/results` → writes `benchmarks_folds.csv` (per-fold) |
| `plot_variability.py` | **new** | JMP-style variability chart (PLM × aug × benchmark) |
| ~~`plot_benchmarks.py`~~ | **removed 2026-07-05** | heatmap + faceted bars — superseded by variability/scaling |
| `plot_ddg_scaling.py` | **updated** | now imports `plot_common` (behavior-identical) |
| ~~`benchmarks_data.csv`~~ | **removed 2026-07-05** | was the heatmap/bars curated summary |

## Apply

```bash
cd mulan/scripts_plots
tar xzf plot_update_20260704.tar.gz      # overwrites the files above + this doc
python aggregate_folds.py                # (re)build benchmarks_folds.csv from local results
python plot_variability.py               # -> ddg_variability.{png,svg}
python plot_ddg_scaling.py               # -> ddg_scaling.{png,svg} (uses the local data CSV)
# (plot_benchmarks.py removed 2026-07-05 — see deprecation banner above)
```

Deps: `matplotlib pandas numpy`.

## Do NOT clobber / notes

- **`ddg_scaling_data.csv` is box-owned** — it is *not* in the bundle. Keep the local
  copy; `plot_ddg_scaling.py` reads it unchanged.
- **`benchmarks_folds.csv` is generated** by `aggregate_folds.py` — never hand-edit; not
  shipped (each box regenerates from its own `scratch/results`).
- **`aggregate_folds.py` `MAP`** hardcodes result-dir names under `scratch/results`. Missing
  dirs render as `running` (harmless). ESM C 6B benchmark dirs are pre-wired as
  `bench_S1131_esmc6b` / `bench_S4169_esmc6b` / `bench_S2003_esmc6b` and `cv10_aug_esmc6b`
  — **if `bench_cv_driver` writes a different tag, edit those names** and re-run.
- **`benchmarks_data.csv`** (heatmap/bars summary) carries manual `status`
  (`measured|pending|held|dropped`); reconcile rows with the box's completed runs.

## Box-specific

- **Linux box** — owns the ESM C 6B benchmark + `+aug` runs. When they finish, just re-run
  `aggregate_folds.py` then `plot_variability.py`; the four ESM C `running` cells fill in.
- **Mac box** — SaProt / ESM2 / Ankh3 MPS runs; same two-command regen.

## What changed since the last figure code

- Variability chart restructured to **PLM (outer) → aug base/+aug (middle) → benchmark
  (inner)**; overlaid box plots (q25/median/q75 + 1.5·IQR whiskers) on per-fold dots;
  color = benchmark; base & +aug **subgroup means** reported (model/global means dropped);
  model labels show **params · embedding-dim**; **data-driven y-range** (no clipping).
- Shared `plot_common.py` extracted from `plot_ddg_scaling.py` (theme/palette/helpers).
- `aggregate_folds.py` added (per-fold long CSV, the variability-chart input).

Current data state (this snapshot): all benchmark cells complete **except ESM C 6B**
(S1131/S4169/S2003 + its `+aug`), still in flight. ESM2-3B S2003 completed this round
(10/10, r≈0.807).

## Cells still in flight at this snapshot (the `mac_queue2.sh` matrix, not shipped)

The Mac benchmark round-2 queue (`scratch/mac_queue2.sh`) generates data the bundle's current
layout does **not** yet surface:
**aug-benchmarks** (`bench_<DS>_<plm>_aug`) and **Ankh3-large/-xl benchmarks**. As those land,
extend the pipeline (do NOT hand-edit `benchmarks_folds.csv` — it's regenerated):

1. **`aggregate_folds.py` MAP** — add the new result-dir cells (tags from `mac_queue2.sh`):
   - SaProt `aug`: `S1131:bench_S1131_saprot_aug, S4169:bench_S4169_saprot_aug, S2003:bench_S2003_saprot_aug`
   - ESM2-3B `aug`: `bench_S1131_esm2_aug, bench_S4169_esm2_aug, bench_S2003_esm2_aug`
   - **New PLM `Ankh3-large`**: base `{S1102:cv10_ankh3_large_nlu, S1131:bench_S1131_ankh3large, S4169:bench_S4169_ankh3large, S2003:bench_S2003_ankh3large}`,
     aug `{S1102:cv10_ankh3_large_aug_nlu, S1131:bench_S1131_ankh3large_aug, S4169:bench_S4169_ankh3large_aug, S2003:bench_S2003_ankh3large_aug}`
   - **New PLM `Ankh3-xl`**: same with the `ankh3xl` / `cv10_ankh3_xl(_aug)_nlu` tags.
2. **`plot_variability.py`** — the aug subgroup is hardcoded to S1102 only:
   - widen `SUBGROUPS` aug slot → `("aug", ["S1102","S1131","S4169","S2003"])`
   - add to `MODEL_ORDER` / `MODEL_ABBR` / `MODEL_META`: `Ankh3-large ("1.15B",1536)`, `Ankh3-xl ("3.48B",2560)`
   - update the footnote ("Tier-1 aug was run on S1102 only …") once aug-benchmarks exist.
3. ~~`benchmarks_data.csv` (heatmap/bars)~~ — **deprecated/removed 2026-07-05**; no longer applicable.
4. **ESM C 6B** (Linux-owned) — its `running` cells fill in from a fresh Linux snapshot; no Mac action.
5. Re-run: `python aggregate_folds.py && python plot_variability.py` (heatmap/bars step dropped).
