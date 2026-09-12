# Deck and figure provenance — the map

**Start here.** This is the entry point for "a result changed, what do I re-run, and how do I
know the deck now agrees with it?" It answers three questions and delegates everything else:

| Question | Answered by |
|---|---|
| What is the order of operations from new folds to a sent deck? | **this doc, §1** |
| Which script drew this figure, and where did its numbers come from? | **this doc, §3** |
| How do I know a published number is still true? | **this doc, §4** |
| What exactly does script X read/write, and how do I add a model to it? | `PLOT_REGEN_HOWTO.md` |
| How do I lay out, bundle, verify and share the reveal.js deck? | `DECK_HOWTO.md` |

Those two are the reference manuals and are not duplicated here. `PLOT_REGEN_HOWTO.md` is the
per-script inventory plus the non-rediscoverable `ddg_scaling`/`variability` series procedure;
`DECK_HOWTO.md` is the reveal.js layout manual (the `vh` trap, the `.slidebody` scroll recipe,
the bundler, hosting). This doc is the thing that ties a *figure* to the *data* behind it and
to the *command that fails* when the two drift apart.

---

## 1. Order of operations

Do these in order. Steps 2–4 are cheap; skipping one is how a deck ends up publishing a number
that no longer exists anywhere upstream.

```bash
cd "<repo>/mulan"

# 1. New folds have landed. Refresh the two CSVs most figures read.
./.venv/bin/python scripts_plots/results_matrix.py            # -> results_matrix.csv
./.venv/bin/python scripts_plots/results_matrix.py --ps       # -> results_matrix_ps.csv

# 2. Recompute MuLAN's own rows in the benchmark matrix and patch both published copies.
./.venv/bin/python experiments/gen_benchmark_matrix_data.py --write
#    Writes experiments/benchmark_matrix.html AND experiments/BENCHMARK_MATRIX.md.
#    Exits 1 (and names the line) if a hand-written sentence still carries a stale number —
#    fix the sentence, re-run, and only then continue.

# 3. Re-render every figure that reads the matrix or the CSVs.
./.venv/bin/python scripts_plots/plot_ppS_scaling.py
./.venv/bin/python scripts_plots/plot_ppS_generalization.py
./.venv/bin/python scripts_plots/plot_fishbone.py --png
./.venv/bin/python scripts_plots/plot_frontier_comparison.py

# 4. Audit. Every one of these must exit 0 before the deck is built (§4).
./.venv/bin/python experiments/gen_benchmark_matrix_data.py --verify
./.venv/bin/python scripts_plots/check_fig1_numbers.py
./.venv/bin/python scripts_plots/comparators.py

# 5. Copy the refreshed assets into the deck folder, then bundle (see DECK_HOWTO.md).
python3 scripts_plots/bundle_deck.py scripts_plots/artifacts/<date>/deck/index.html --strict
```

**Step 2 before step 3, always.** The `plot_ppS_*` pair reads its comparator and AUROC values
back out of `BENCHMARK_MATRIX.md`. Rendering before writing the matrix draws the old numbers,
and the figure will not complain.

### The interpreter

Use the repo venv (`./.venv/bin/python`, matplotlib 3.11.0). This matters beyond imports:
matplotlib's raster and SVG output changes between minor versions, so a figure rendered by a
different matplotlib differs byte-for-byte from its siblings even when the data is identical.
Regenerating one figure with a different interpreter means regenerating the whole set, or
accepting a mixed-renderer deck. The stdlib-only scripts (`results_matrix.py`, `plot_fishbone.py`,
`gen_benchmark_matrix_data.py`, `comparators.py`, `check_fig1_numbers.py`, `bundle_deck.py`)
run under any `python3` and have no such constraint — `plot_fishbone.py` writes its SVG as text
and reproduces byte-for-byte across interpreters (verified 2026-08-06). Its `--png` flag is the
exception: rasterising needs `cairosvg` or `playwright`, so drop the flag if neither is present
and the SVG alone is enough.

---

## 2. The three ways a number gets into a figure

Every figure here uses exactly one of these patterns. Which one applies determines what breaks
when the underlying result moves — and whether anything breaks at all.

**Derive the drawing.** The script recomputes the figure from the data every run, so a stale
number cannot survive: there is nowhere for it to live. `plot_fishbone.py` is the reference
implementation — it reads its twelve numbers from two independent sources, cross-checks them
against each other, and exits 1 without writing anything if they disagree. Use this for new
figures whenever the drawing is generatable.

**Derive the check.** The artwork stays hand-authored, but every data-bearing number in it is
declared alongside a locator for its source, and a checker fails when figure and source
disagree. `check_fig1_numbers.py` does this for `fig1_panels.html` — 186 KB of hand-tuned SVG
that would be the wrong thing to rewrite as a generator. The rule that makes it honest: *a
number with no writable locator is a number the figure should not be asserting.*

**Derive the publication.** The numbers live in a published file that humans read directly
(`BENCHMARK_MATRIX.md`, `benchmark_matrix.html`), and a generator owns those cells:
`--write` patches them, `--verify` fails if they have drifted. `gen_benchmark_matrix_data.py`
does this for the benchmark matrix.

**And the fourth case, which is the one to fix.** A hand-authored figure whose numbers are
string literals with no declared source. Nothing breaks when they go stale, because nothing
is looking. That is how the retracted `CATH tier · per-struct ρ 0.468` survived the 2026-08-06
coverage correction and every text sweep of the deck. §5 lists the assets still in this state.

---

## 3. Figure → script → data source

### Flagship figures (data-driven)

| Deck asset | Produced by | Reads |
|---|---|---|
| `ppS_scaling.{png,svg}` | `scripts_plots/plot_ppS_scaling.py` | `results_matrix_ps.csv` (panels 2/3a/4a); `ddg_scaling_data_300ep_balanced.csv` (panel 1); `experiments/BENCHMARK_MATRIX.md` **via `comparators.py`** (panels 3b/4b + all frontier bands) |
| `ppS_generalization.{png,svg}` | `scripts_plots/plot_ppS_generalization.py` | `results_matrix_ps.csv`; `BENCHMARK_MATRIX.md` via `comparators.py` (frontier bands only) |
| `fishbone.{svg,png}` | `scripts_plots/plot_fishbone.py` | `experiments/rescore_perstructure/foldx_alone_baseline_allplm_cov99.csv`; `BENCHMARK_MATRIX.md`. Writes `fishbone_provenance.txt` listing all 12 drawn numbers against the file and row each came from |
| `frontier_comparison.png` | `scripts_plots/plot_frontier_comparison.py` | Panel A `experiments/retrain_split/results_clustered.csv`; Panel B `results_matrix_ps.csv` (`fullSK CATH-all`) + `foldx_alone_baseline_allplm_cov99.csv` for CIs; frontier values from `data/benchmarks/frontier.tsv` via `benchmarks.py` |
| `ddg_scaling*.{png,svg}` | `scripts_plots/plot_ddg_scaling.py` (`MULAN_PLOT_SERIES`) | `ddg_scaling_data*.csv` — **hand-curated**; adding a model means appending a row |
| `ddg_variability*.{png,svg}` | `scripts_plots/plot_variability.py` (`MULAN_PLOT_SERIES`) | `benchmarks_folds*.csv`, written by `aggregate_folds.py` — never hand-edit |
| `s1102_interface.{png,svg}` | `experiments/attention_interface/plot_s1102_interface.py` | `s1102_<tag>.csv` + `s1102_<tag>_perchain.csv` from `score_s1102.py`; also writes `RESULTS_s1102.md` |
| `skempi_full_interface.{png,svg}` | `experiments/attention_interface/plot_skempi_full_interface.py` | `s1102_skempi_full_<tag>.csv` + `_perchain.csv`; also writes `RESULTS_skempi_full.md` |

### Interactive widgets (iframed into the deck)

| Deck asset | Canonical copy | Numbers come from |
|---|---|---|
| `benchmark_matrix.html` | `experiments/benchmark_matrix.html` | **Generated.** `gen_benchmark_matrix_data.py --write` owns all three JS arrays (`fullsk`, `ladder`, `aurM`). The page derives its own group headline rows at render time, so patching the arrays fixes the whole page |
| `fig1_panels.html` | `images/foldx_figs/fig1_panels.html` | Hand-authored SVG; ~25 data-bearing nodes, each declared in `check_fig1_numbers.py` against `RESULTS.md` / `FINDINGS_FOLDX_MUTANT_3DI.md`. Sidecar: `scripts_plots/fig1_provenance.txt`, rewritten on every check |
| `metric_rationale.html` | `experiments/metric_rationale.html` | Hand-authored, **undeclared** — see §5 |
| `leakage_explainer.html` | deck asset folder only | Hand-authored, **undeclared** — see §5 |
| `split_ladder.html`, `coverage_treemap.html` | deck asset folder only | Hand-authored, **undeclared** — see §5 |

### Schematics (no data)

`plot_augmentation.py`, `plot_model_timeline.py`, `plot_structure_fusion.py` and the
`images/foldx_figs/fig{1a,1b,1c,1d,2,A1}*.py` panel generators draw hand-laid diagrams with no
data inputs. Re-run only when their code changes. `images/foldx_figs/build_panels.py`
assembles the four Figure-1 panel SVGs into `fig1_panels.html` and must run *after* them; its
output directory resolves from `__file__`, so it needs no editing.

### Where the raw numbers actually live

Everything above ultimately traces back to one of these:

- `scratch/results/full_skempi{,_bycomplex,_mp_bycomplex}/<arm>/fold_{0,1,2}/training_run/test_predictions.tsv` — the predictions
- the truth, three directories with *deliberately inconsistent* names — only the clustered one
  carries the `_kfold` suffix, so a `_kfold*` glob silently matches one split out of three:
  `scratch/splits_skempi_full_clustered_id60_kfold/`, `scratch/splits_skempi_full_bycomplex_seed42/`
  and `scratch/splits_skempi_full_mp_bycomplex_seed42/`, each `fold_{f}/skempi_{sp,mp}_test.tsv`
- `experiments/full_skempi_seqonly/results_cath.csv` — CATH-superfamily AUROC
- `experiments/retrain_split/results_clustered.csv` — S1102 clustered pooled Pearson/Spearman
- `experiments/rescore_perstructure/results.csv`, `…/foldx_alone_baseline_allplm_cov99.csv`
- `scripts_plots/results_matrix{,_ps}.csv` — the two engine-written CSVs
- `experiments/BENCHMARK_MATRIX.md` — the published comparator + MuLAN table, itself generated

---

## 4. The audit suite

Run all of these before building a deck. Each exits non-zero with a file-and-line message.
All five were run against the repo as it stands on 2026-08-06 and all five exit 0; the numbers
quoted below (314 cells, 0 typed comparators) are that run's actual output, not an estimate.

```bash
./.venv/bin/python experiments/gen_benchmark_matrix_data.py --verify
./.venv/bin/python scripts_plots/check_fig1_numbers.py
./.venv/bin/python scripts_plots/plot_fishbone.py          # the guard runs before it writes
./.venv/bin/python scripts_plots/comparators.py
python3 scripts_plots/bundle_deck.py <deck>/index.html --strict
```

**`gen_benchmark_matrix_data.py --verify`** compares all 314 published cells in
`benchmark_matrix.html` and `BENCHMARK_MATRIX.md` against what the prediction files actually
say, plus the `<details>` caption, the headline AUROC row, and the § footnote paragraph.
Comparison is on the *published three-decimal spelling*, not a float tolerance — the matrix
publishes three decimals, so three decimals is the thing to check. (A tolerance test looks
more careful and is wrong at the boundary: a computed 0.2295 prints as `0.229` and sits
exactly 5e-4 away from it.)

**`check_fig1_numbers.py`** anchors each number on the *label* it sits with in the SVG, not on
the bare digits, so `0.740` cannot pass by matching an unrelated `0.740` elsewhere in the file.
`--html PATH` checks a copy, which is how a deck asset is audited rather than the canonical file.

**`plot_fishbone.py`** cross-checks MuLAN's CATH per-structure ρ computed from the CSV against
the CATH cell of the `MuLAN SKEMPI best` row in `BENCHMARK_MATRIX.md`. Disagreement at 3 dp
prints both, names both files, and exits 1 **without writing anything** — the failure mode it
defends against is one source being regenerated without the other.

**`comparators.py`** prints every value the figures will draw and where each came from, ending
with a count of how many are typed (currently zero). Running it shows what a figure is *about*
to publish, before it is rendered.

**`bundle_deck.py --strict`** audits the deck's own CSS for viewport units and checks for
missing assets and residual local refs. It cannot catch layout problems; `DECK_HOWTO.md` §3 has
the headless render checks for those.

### What `--write` will not do

`gen_benchmark_matrix_data.py --write` owns table cells, the `<details>` caption and the
headline AUROC row. It deliberately does **not** rewrite the § footnote paragraph under the
AUROC companion table: re-wrapping a human sentence is a worse trade than retyping three
digits. Instead it *checks* that paragraph and reports it for a hand edit, and exits 1 while
the finding stands — so a stale sentence cannot be mistaken for a clean run.

---

## 5. Known gaps

Assets still in the fourth case from §2 — typed numbers, no declared source, nothing that
breaks when they go stale. Listed worst-first.

| Asset | Typed data-bearing numbers | Note |
|---|---|---|
| `leakage_explainer.html` | ~21 | No producer, no checker |
| `metric_rationale.html` | ~14 | No checker. One value (`0.881`) could not be traced to any declared source at all |
| `split_ladder.html` / `.png` | — | No producing script exists in the repo. The `.png` in the 2026-07-26 deck came from the hollowed ladder script and is stale |
| `coverage_treemap.html` / `.png` | — | No producing script in the repo |
| `point3_scatter.png`, `fig1_2x2.png` | — | No producing script in the repo |

`honest_ladder.png` was in this list and was deleted on 2026-08-07 rather than kept: it had no
producing script, was referenced by no file, and drew a bare "ProtBFF honest anchor ~0.48" on a
per-structure axis when ProtBFF's 0.477 is a *pooled* full-SKEMPI figure. With no generator it
could not be corrected, and with no referent nothing needed it. `generalization_ladder.png` draws
the same reference line correctly, sourcing 0.477 and naming the metric in the annotation.

Two of these are worth different treatments. The explainer widgets are text-heavy pages where
"derive the check" is the right pattern — declare each number with a locator, as `fig1_panels`
does. The orphaned PNGs are worse: an asset with no producer cannot be regenerated at all, so
the honest options are to rebuild a generator or to drop the slide.

---

## 6. Shared plumbing

Three small modules exist so that the programs above cannot disagree with each other.

**`scripts_plots/mdtable.py`** — locating, parsing and re-rendering the pipe tables in
`BENCHMARK_MATRIX.md`. A table is found by *the substrings its header row contains*, never by
line number, so inserting a paragraph above a table shifts nothing; zero or two matches is
fatal rather than resolved by taking the first hit. It exists because both the reader
(`comparators.py`) and the writer (`gen_benchmark_matrix_data.py`) need this rule, and a reader
and a writer that disagree about where a table starts is the same class of failure as a typed
copy, one level down.

**`scripts_plots/comparators.py`** — the single parsed source of every comparator and MuLAN
value the figures draw. `plot_ppS_scaling.py` and `plot_ppS_generalization.py` used to each
carry their own typed copies of the frontier dicts; two places to update, one place to forget.
Which comparator gets a labelled line and where a label hangs stays in the figure scripts —
that is layout, not data.

**`scripts_plots/plot_common.py`** — shared theme, palette and CSV-loader conventions, so the
figure set has one visual identity.

---

## 7. Changelog

- **2026-08-06** — This doc created. `mdtable.py` extracted; `comparators.py` rewired onto it;
  `gen_benchmark_matrix_data.py` rewritten to `--write`/`--verify` instead of printing arrays
  for a human to paste. Closing that paste seam immediately corrected 38 stale published cells
  in `BENCHMARK_MATRIX.md`'s AUROC section, which `plot_ppS_scaling.py` panel 4b had been
  drawing. `check_fig1_numbers.py` added for `fig1_panels.html`. `plot_fishbone.py` replaced the
  hand-authored `fishbone.html`.

  Same day, `PLOT_REGEN_HOWTO.md` was corrected in four places: its `plot_ppS_scaling.py` /
  `plot_ppS_generalization.py` entries, its §3b "adding a new PLM" checklist and its §4 gotcha
  all still instructed the reader to hand-update duplicated `FRONTIER` / `CDHIT_FOLDAVG_S` /
  `CATH_AUROC` dicts and to paste the generator's stdout into the matrix by hand. None of that
  had been true since the same morning. Worth naming plainly: a doc describing a paste step that
  no longer exists is the same failure this whole exercise is about, one level up — the fix
  changed the code and left the instructions asserting the old world.
