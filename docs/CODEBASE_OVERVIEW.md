# Codebase overview

A module-by-module tour, for someone who has just cloned this repository. The [root
README](../README.md) says what the fork asks and why; this says where the code that asks it
lives, and — more usefully — names the handful of things a newcomer gets wrong.

## What this is

Upstream [MuLAN](https://github.com/GianLMB/mulan) (Lombardi & Carbone) predicts the effect of a
mutation on protein–protein binding affinity (ΔΔG) **from sequence alone**: a frozen protein
language model produces per-residue embeddings, and a small "light attention" head reads them.
The upstream library (`mulan/`) and its five command-line entry points (`scripts/`) are retained
here essentially intact.

This fork adds three things, and they are the reason most of the rest of the tree exists:

1. **A multi-backbone PLM front end** — one registry of 17 backbones instead of a single
   hard-coded encoder, so "does the language model matter?" is answerable.
2. **A FoldX ΔΔG score channel** — cheap biophysics computed once per mutation and reused across
   every backbone, so "does structure-derived signal help a sequence model?" is answerable
   without confounding it with the choice of encoder.
3. **Leakage-controlled evaluation** — by-complex, sequence-clustered and CATH-superfamily
   hold-outs scored per structure, so "what survives leakage-controlled evaluation?" is answerable against
   published tables.

The exact statement of modifications, and the third-party data this depends on, is in
[`NOTICE`](../NOTICE).

## What ships, and what does not

Read this before going looking for something. This repository was extracted from a two-machine
research working tree, and the boundary between "published" and "was on a disk somewhere" is not
guessable from the file listing.

| | Present | Notes |
|---|---|---|
| `mulan/`, `scripts/`, `experiments/`, `scripts_plots/`, `tests/`, `docs/` | ✅ | All code and all written records. |
| `data/splits/` | ✅ 3.4 MB | Eight partitions, plus a build recipe for CATH, whose partition is not ours to redistribute. The most reusable artifact here. |
| `data/foldx/` | ✅ 2.7 MB | 322 single-point complexes / 4238 mutations, 152 multi-point complexes / 1765 variants, twelve energy terms each. |
| `models/config/` | ✅ | The ten small head-configuration JSONs. |
| `images/`, figures under `scripts_plots/` | ✅ | Generated PNG/SVG plus the scripts that generate them. |
| `results/` | ✅ 30.2 MB | Per-fold predictions and metrics, 666 folds over nine tiers; see [`results/README.md`](../results/README.md). With `data/splits/`, enough to recompute every reported metric without retraining. |
| `data/benchmarks/` | ✅ | `frontier.tsv` — the published comparator numbers, one row per `(method, metric, protocol)`, transcribed from the source papers and read through `scripts_plots/benchmarks.py`. Most plotting scripts read it; `plot_ppS_scaling.py` still carries its own literals for the fold-averaged panel, and one of them (`ProSST 0.354`) has no row in the table at all. |
| `models/pretrained/` | ❌ | The upstream MuLAN checkpoints are the original authors' artifacts and are **not** redistributed. Download them from upstream for `mulan-predict` / `mulan-att` / `mulan-landscape`; training and every evaluation protocol run from scratch without them. |
| `scratch/` | ❌ | The machine-local working tree — training outputs, embedding caches, FoldX scratch, queue drivers. Gitignored, and **it ships nothing**. |
| PLM embeddings, the FoldX binary, PDB inputs | ❌ | Tens of gigabytes, licensed, or externally distributed. |

**The `scratch/` point is worth dwelling on**, because paths under it appear throughout
`experiments/`, in `scripts_plots/results_matrix.py`, and all over
[`docs/history/`](history/README.md). Those references are real — that is where the work ran —
but a public reader cannot resolve any of them. When a script's default output path starts with
`scratch/`, it means "this regenerates locally", not "look here for the shipped copy". The
shipped copies live under `data/` and `results/`.

## Layout

```
mulan/            the library: light-attention model, dataset/collator, trainer, and the
                  plm / foldx / metrics / splits subpackages added by this fork
scripts/          the five CLI entry points (installed as the package `mulan.scripts`), plus
                  audit_local_results.py — reads a results tree and reports folds that crashed
                  silently, degenerate metrics, truncated arms and un-pulled re-runs
tests/            three suites, 164 tests — registry equivalence, FoldX contracts, metrics
experiments/      the research apparatus and its written record, one directory per campaign
                  (a campaign is one run of experiments aimed at one question)
scripts_plots/    figures, the results aggregator, and the deck bundles
data/             the published payload: splits and FoldX results
results/          per-fold predictions and metrics -- 666 folds, nine tiers
models/config/    head configurations: base · FoldX scalar · FoldX 12-term MLP · the rest
docs/             documentation; docs/history/ is an explicit archive of superseded material
examples/         small sample inputs for the CLIs
images/           the visual abstract and the FoldX architecture figures
```

## The core library — `mulan/`

### Upstream core, lightly modified

- **`modules.py`** — the model. `AttentionMeanK` is the light-attention encoder: parallel conv1d
  "feature" and "attention" heads at several kernel sizes, softmax-masked attention pooling
  concatenated with max-pooling, then an MLP. `LightAttModel` wraps it **siamese** — encode
  wild-type A/B and mutant A/B separately, combine each pair by elementwise product and absolute
  difference, take `mutant − wildtype`, and a final linear layer predicts ΔΔG. Several optional
  heads are gated off by default: the zero-shot `add_scores` slot and its `zs_mlp` variant over a
  decomposed FoldX vector, a pooled wild-type `struct_context` branch, a `struct_gate` on the 3Di
  block of an AA⊕3Di embedding, an `interface_bias` on attention pooling, and `interface_xattn`.
  `from_pretrained` loads with `strict=False` but *warns loudly* on any key mismatch, because a
  silently-random new head is indistinguishable from a loaded one until the numbers are wrong.
- **`config.py`** — `MulanConfig`, a dataclass carrying the head flags above, with lenient
  `from_dict` / `from_json`.
- **`data.py`** — `MulanDataset` / `MulanDataCollator`: read a mutation table plus a wild-type
  FASTA, apply mutations, resolve or generate per-sequence embeddings, pad and collate siamese
  batches. Also `split_data`, the **offline** k-fold splitter (`split_method="random"` reproduces
  the upstream partition; `"balanced"` gives equal folds) — it writes fold TSVs and is never
  called during training. The in-RAM tensor cache (`MULAN_EMB_CACHE`, on by default) exists
  because without it a cached-embedding run re-runs `torch.load` once per sample per epoch per
  fold; see [`CV_SPEEDUP.md`](CV_SPEEDUP.md).
- **`train_utils.py`** — `MulanTrainer`, a HuggingFace `Trainer` subclass with MSE loss and a
  siamese-aware `prediction_step`, plus the CLI argument dataclasses.
- **`utils.py`** — FASTA parsing, mutation-string parsing (`AB23G` = wild-type · chain · position
  · mutant, chain ∈ {A, B}), device selection, and the legacy embedding entry points, which now
  route through `mulan.plm`.
- **`interface_xattn.py`** — `InterfaceCrossAttention`, a residue-level cross-attention block that
  primes each chain's representation with its partner's interface signal before pooling. Gated to
  init-zero, so an untrained model starts as the plain baseline.

### `mulan/plm/` — the backbone registry

The front end that turns a sequence into `[L, dim]`. **[`registry.py`](../mulan/plm/registry.py)
is the single source of truth**: one `PlmSpec` row per backbone carrying the model id, the
embedding width, which backend loads it, the input convention (prefix, space-joining), the
trimming rule that recovers exactly one vector per residue, the pip extra it needs, and whether
it runs on Apple Silicon. Seventeen rows: ESM-2 (3B / 650M / 35M), Ankh v1 (large / base), Ankh3
(large / xl), ProstT5, ProtT5-XL, ProtBERT, SaProt (650M / 1.3B), ESM-C (600M / 6B), ESM3, AIDO,
MINT. To add a backbone, add a row — do not add an `if`.

That instruction is the whole point. Before the registry, picking a PLM meant three places
agreeing: a name→hub-id map, a loader branching on substrings of the *model id*, and an embedder
branching again on substrings of the *tokenizer's* `name_or_path`. Three surfaces, three chances
to disagree — and several backbones actually used in this work were in none of them, having been
generated out of band by scripts that were never in the repository at all.

- **`backends/`** — `base.py` (the shared trim-and-validate path), `hf.py`, `esm_sdk.py` (ESM-C
  and ESM3 via the EvolutionaryScale SDK), `aido.py`. Dispatch resolves the device too, demoting
  to CPU for backbones whose `mps_ok` is false rather than failing deep inside a forward pass.
- **`ids.py`** — the naming contract between a generator and the trainer. An embedding cache is a
  flat directory of `<id>.pt`, and the trainer finds a tensor by reconstructing the same id
  string. That contract was previously re-implemented in eight places, each with a comment saying
  it must match `MulanDataset._fill_metadata`. Copies that must agree, don't.

**A correction worth stating**, because older notes in this repository still say otherwise:
`mulan.constants.PLM_ENCODERS` is no longer a hand-maintained map. It is a *view* —
`constants.py` imports `registry.hub_ids()`. Any claim that ESM-C 600M/6B are "SDK-only and not
in `PLM_ENCODERS`" describes the pre-registry state and is false now. If that sentence turns up
in [`EMBEDDING_SETUP.md`](EMBEDDING_SETUP.md), it is stale.

See [`PLM_BACKEND.md`](PLM_BACKEND.md) for the design, [`EMBEDDING_SETUP.md`](EMBEDDING_SETUP.md)
for which generator produces which cache, and
[`mulan/EMBEDDING_PREPROCESSING.md`](../mulan/EMBEDDING_PREPROCESSING.md) for the per-model
preprocessing table.

### `mulan/foldx/` — the score channel

**Most of this package now lives elsewhere.** The producer half — computing and curating the
energies — was extracted to [skempi-foldx](https://github.com/cchin29/skempi-foldx), which is a
base dependency. What remains here is the consumer half: encoding those energies as *this*
model's score channel.

| Module | Where | Role |
|---|---|---|
| [`merge.py`](../mulan/foldx/merge.py) | **here** | Joining scores onto splits: key mapping, the grouping guard, per-fold train-only standardization, both arms. Also `CLIP`, `MISSING_FILL` and the column layouts — they describe how this model encodes the energies, so they stayed with the consumer. |
| [`__init__.py`](../mulan/foldx/__init__.py) | **here** | A thin adapter: `merge.py` plus a re-export of `skempi_foldx`, so `from mulan.foldx import load_store` still works and callers need not know where the boundary fell. |
| `terms.py` | skempi-foldx | The twelve `AnalyseComplex` terms, in order. The order is load-bearing: it is the column order of a decomposed split file and therefore the input order of the model's score MLP. |
| `config.py` | skempi-foldx | `FoldxConfig` — binary, PDB directory, SKEMPI table, work and result directories, passed explicitly rather than assigned to globals. |
| `skempi.py` | skempi-foldx | SKEMPI parsing and the role→author chain mapping, validated against the residue actually present in the structure. |
| `run.py` | skempi-foldx | The compute engine: `RepairPDB → BuildModel → AnalyseComplex`, resumable per complex. |
| `store.py` | skempi-foldx | Consolidating campaign directories into one canonical store, with a **value** audit rather than a key-set audit. |
| `exclusions.py` | skempi-foldx | The `INTRACTABLE` registry. |

Full treatment in [`FOLDX.md`](FOLDX.md); the scientific arc — every route for structure that was
tried and why a score channel won — is in
[`FOLDX_STRUCTURE_ARC_AND_AUGMENTATION_PLAN.md`](FOLDX_STRUCTURE_ARC_AND_AUGMENTATION_PLAN.md).

### `mulan/metrics/` and `mulan/splits/` — the evaluation protocols, as data

- **`metrics/core.py`** — the scoring functions, numpy-only, tie-aware, deterministic. They used
  to live inside a *script*, which every scorer imported by inserting a directory onto
  `sys.path`. The headline is `per_structure` at `T=10`.
- **`metrics/bootstrap.py`** — cluster bootstrap resampling over **complexes, not mutations**.
  There were previously three divergent copies; one hard-coded an absolute path at module scope
  and so could not be imported anywhere, which is why the second copy existed.
- **`metrics/tiers.py`** — a tier is a protocol applied to a dataset. Ten are registered; nine
  are leakage-controlled, and the eight whose partitions are redistributable resolve against
  `data/splits/` and ship. CATH is the ninth: leakage-controlled but carried `shipped=False`,
  because its partition is not ours to redistribute.
- **`splits/protocols.py`** — the four protocols as rows, each declaring what unit it holds out,
  whether it is leakage-controlled, whether it is *homology*-controlled, and whether a number from
  it can be placed in a published table. `comparability_warning()` returns the sentence to print
  beside such a number.

See [`SPLITS_AND_METRICS.md`](SPLITS_AND_METRICS.md).

## The CLIs — `scripts/`

`mulan-train`, `mulan-predict`, `mulan-att`, `mulan-landscape`, `plm-embed`. All take `--help`;
worked examples are in [`USAGE.md`](USAGE.md). (`scripts/uniprot_visualize.py` is a sixth,
utility script with no console entry point.)

Packaging is [`pyproject.toml`](../pyproject.toml) — there is no `setup.py`. Two details that
will confuse a reader otherwise:

- The on-disk layout is inherited from upstream: the package is `mulan/` but the CLI modules live
  in `scripts/`, remapped by `[tool.setuptools] package-dir` to `mulan.scripts`. That is why the
  entry points read `mulan.scripts.train:main` for a file at `scripts/train.py`.
- The package list is **explicit**, not `packages.find`. Discovery scans for directories matching
  the include pattern, so `include = ["mulan*"]` silently omitted the remapped `scripts/` and
  every console script installed broken.

Optional extras: `[mint]`, `[struct3di]`, `[struct]`, `[plots]`, `[dev]`. **ESM-C / ESM3 is
deliberately not among them.** Their SDK pins `transformers<4.48.2` while training runs 4.44.x,
and an extra is ANDed with the base requirements — so no version of such an extra can be both
satisfiable and faithful to what actually ran. Generation lives in its own environment via
[`requirements-esmc.txt`](../requirements-esmc.txt), which is sound because training reads cached
`{id}.pt` tensors and never imports the SDK. The base requirement is open-ended rather than
upper-bounded, which means a fresh install resolves a `transformers` far newer than the 4.44.x
every run here used. That range is checked at runtime by
`mulan.plm.check_transformers`, which warns, instead of by a pin that breaks the extra. See
[`REPRODUCE.md`](REPRODUCE.md#environments).

## Four common misreadings

**1. The default evaluation tier is the strictest one.** `mulan.metrics.get_tier()` with no
argument returns the CATH-superfamily hold-out. The legacy random 10-fold protocol — the upstream
paper's, in which the same complex appears in train and test — still exists, because reproducing
the published number requires it, but it must be **named**: `rescore.py --tier legacy_cv10`.
Running the scorer with no `--tier` is an error, not a fallback. It used to be a fallback *to the
leaky tier*, which meant the default behaviour of the metric core was the one protocol no claim
rests on.

**2. A split file's column count is its schema.** Split TSVs are headerless and `mulan/data.py`
dispatches on width: 4 columns = base, 5 = FoldX scalar, 16 = FoldX 12-term. Passing
`--add_zs_scores True` against a 4-column file makes the ΔΔG label itself be read as the score,
leaving no label, and training fails inside the loss.

**3. FoldX is deterministic.** Do not repeat the earlier reading of this data, which is purged
from the code but survives in some archived notes. FoldX 5.1 `BuildModel` produces bit-identical
output from identical inputs — verified on 1722 pairs, zero differing. What varies is the
*input*: `BuildModel` walks `individual_list.txt` sequentially in one process and each entry
inherits the optimisation state left by its predecessors, so a mutation's ΔΔG is a function of
`(repaired structure, every entry preceding it in the list)`. Computing a subset of a complex's
mutations is therefore **not** interchangeable with computing the union — which is why
`run.py::worklist_single_point` computes the union, and `worklist_from_table` survives only to
reproduce the historical campaigns.

**4. One complex is excluded from FoldX outright.** `RepairPDB` does not terminate on **1KBH** —
left running past 22.5 h on two machines. Curation already drops it, so it is in no split, but
curation was never what stopped it being *computed*: a driver enumerating its scope from the
result store picks it up regardless, and a driver is a loop, so killing the hung job only
postpones it. The exclusion is applied before any worklist is built and enforced at four points
(`run_campaign`, `process_complex`, `store.consolidate`, `build_results_all.py`). The entry
records the FoldX version it was observed against, and
`MULAN_FOLDX_ALLOW_INTRACTABLE=1` re-admits it to the compute path — but deliberately not to
curation, so a newer binary can be tested without silently changing which rows every split
contains.

## The research apparatus — `experiments/`

Treat this as a **record**, not a library. Each directory is a campaign: its build scripts, its
config shells, its result CSVs and its `SUMMARY*.md`. Numbers here were produced against result
trees under `scratch/`, so the scripts will not re-run unmodified on a fresh clone; the
*conclusions* are in the SUMMARYs and in [`RESULTS.md`](RESULTS.md).

| Directory | What it is |
|---|---|
| `full_skempi_seqonly/` | The full-SKEMPI campaign — dataset construction (`build_skempi_full.py`, `build_multipoint.py`), the five split builders, the per-tier config shells, and the `score_*.py` scorers. The headline work. |
| `retrain_split/` | The earlier S1102 ladder, and `run_bycomplex.sh` — the shared runner used by *every* tier, with `ARMS="base foldx foldx_scalar"`. |
| `rescore_perstructure/` | `rescore.py`, the per-structure scoring driver over `mulan.metrics`, plus `--selftest` (checks the numpy metrics against scipy/sklearn to 1e-9) and the FoldX-alone baseline. |
| `embedding_sweep/` | Per-PLM 10-fold CV on S1102, with the S1102 data and fold TSVs checked in under `data/` and `splits/`. |
| ~~`foldx_repair_sweep/`~~ | **Not in this repository.** The repair-count sweep is producer-side and left with the FoldX half; it lives in [skempi-foldx](https://github.com/cchin29/skempi-foldx) as `experiments/repair_sweep/`. It repairs each complex four times and runs BuildModel from each round; `list_hashes.py` there records the `individual_list.txt` bytes *before* cleanup, because the energies are regenerable and the record of what produced them is not. |
| `attention_interface/` | Whether attention weights localize the binding interface, across eleven backbones on S1102 and nine on full SKEMPI. The upstream Ankh result reproduces; nothing else matches it. |
| `interface_xattn/`, `mint/` | The two attempts to feed cross-chain context into the model, at the head and at the embedding respectively. |
| `cath_leakage/`, `paper_fairness/`, `skempi_interface/` | Leakage and fairness audits. |
| `structctx_sweep/` | The remaining consumer of the extracted structural-context module — needs `pip install ".[struct]"`. |

Top-level scripts cover embedding generation (`gen_emb_generic.py`, `gen_saprot_emb.py`,
`gen_3di.py` + `concat_aa_3di.py`, `gen_interface.py` + `concat_aa_interface.py`,
`gen_layer_embeddings.py` / `layer_probe.py`), Tier-1 augmentation (`augment.py`), and benchmark
construction (`build_benchmarks.py`). The `.md` files at the top
level are findings, not plans — `BENCHMARK_MATRIX.md` (comparator numbers transcribed from the
source papers, and tracked in git), `METRIC_RATIONALE.md`, `FOLDX_COVERAGE_FINDINGS.md`,
`AUGMENTATION.md`, `HYPERPARAMETERS.md`, and the `RESULTS_*.md` series.

### Where the structural-context module went

Per-residue structural context — AlphaFold fetch, DSSP-derived secondary structure and RSA,
contact maps — lived here as `mulan.structural_context`. It was extracted to
**[foldenv](https://github.com/cchin29/foldenv)** and is maintained there; module names transfer
one to one, so `from mulan.structural_context import context` becomes `from foldenv import
context`. What remains in this repository is the *model-side* branch that consumes a pooled
context vector (`struct_context` in `MulanConfig`), not the code that computes one.

## Figures and aggregation — `scripts_plots/`

- **`plot_ppS_scaling.py`** and **`plot_ppS_generalization.py`** — the flagship pair. Both take
  their per-structure Spearman panels from `results_matrix_ps.csv` as the single source of truth
  (`plot_ppS_scaling.py`'s first panel is the legacy 10-fold Pearson, from
  `ddg_scaling_data_300ep_balanced.csv`), and both show each backbone as base + FoldX-scalar +
  FoldX-12-term. Frontier comparator bands are read from
  [`data/benchmarks/frontier.tsv`](../data/benchmarks/frontier.tsv) through
  `scripts_plots/benchmarks.py`, so both panels draw the same numbers from one file.
- **`results_matrix.py`** — the coverage aggregator: walks every result tree and writes
  `results_matrix.csv` (pooled `test_pcc`) and, with `--ps`, `results_matrix_ps.csv`
  (per-structure Spearman). It still walks `scratch/results/`, so on a fresh clone it will find
  nothing there; the shipped copies are under `results/`.
- Others: `plot_ddg_scaling.py`, `plot_variability.py` (fed by `aggregate_folds.py`), the
  `plot_augmentation.py` / `plot_structure_fusion.py` / `plot_model_timeline.py` schematics, and
  `bundle_deck.py` for the self-contained result decks under `artifacts/`.
- [`PLOT_REGEN_HOWTO.md`](../scripts_plots/PLOT_REGEN_HOWTO.md) is the authoritative per-script
  inventory and regeneration procedure.

## Tests and CI

Three suites, 164 tests, run by [`ci.yml`](../.github/workflows/ci.yml) on Python 3.10 and 3.12
against CPU-only torch wheels. They exist for failures that do not raise:

- **`test_plm_registry.py`** — asserts the registry reproduces the pre-refactor inline branches
  *bit for bit*, comparing against the original implementations copied verbatim into the test
  file. Tens of gigabytes of embedding caches were produced by the replaced code; a regenerated
  tensor that disagrees with a cached one would surface only as a metric moving for no reason.
- **`test_foldx.py`** — pins the merge numerics and the grouping guard. A merge failure produces
  a correctly-shaped split file with wrong numbers, and the coverage counter cannot catch it,
  because coverage counts keys and the damage is in values. That is not hypothetical: an extra
  join key once left coverage byte-identical while handing ~13 mutations another mutation's
  energies.
- **`test_metrics_and_splits.py`** — per-structure aggregation, ties and degenerate cases, and
  the bootstrap.

`pytest` runs `tests/` only, by configuration: the suites under `experiments/` need optional
extras or a local `scratch/` tree, so including them made a bare `pytest` fail at *collection*.
CI also runs `rescore.py --selftest` and a hygiene job that greps for machine-specific paths and
host identifiers — patterns that leaked before extraction and must not come back.

## Where to go next

| Goal | Read |
|---|---|
| Run a CLI | [`USAGE.md`](USAGE.md) |
| Understand the evaluation, or compare a number to a paper | [`SPLITS_AND_METRICS.md`](SPLITS_AND_METRICS.md) |
| Understand the FoldX channel | [`FOLDX.md`](FOLDX.md), then [`FOLDX_STRUCTURE_ARC_AND_AUGMENTATION_PLAN.md`](FOLDX_STRUCTURE_ARC_AND_AUGMENTATION_PLAN.md) |
| Add or regenerate embeddings | [`PLM_BACKEND.md`](PLM_BACKEND.md), then [`EMBEDDING_SETUP.md`](EMBEDDING_SETUP.md) |
| Rebuild the pipeline end to end | [`REPRODUCE.md`](REPRODUCE.md) |
| See the numbers | [`RESULTS.md`](RESULTS.md) |
| See what is still unsettled | [`OPEN_QUESTIONS.md`](../OPEN_QUESTIONS.md) |
| Know what shipped | [`data/README.md`](../data/README.md), [`results/README.md`](../results/README.md) |

[`docs/history/`](history/README.md) holds superseded plans, closed investigations and dated run
notes. It is kept because several of this project's conclusions were reversed on evidence, and
the reversals are part of the record — but nothing in it is current, and much of it references
`scratch/` paths that do not resolve here.

> Installing over an older checkout needs `pip uninstall mulan` first: the distribution was
> renamed from `mulan` to `mulan-plm-foldx`, and the two own the same `mulan` import package, so
> pip will not replace one with the other.

> The repository publishes as `mulan-plm-foldx`; that name is set in `pyproject.toml`, the README
> and `CITATION.cff`.
