# Documentation

Start with [`CODEBASE_OVERVIEW.md`](CODEBASE_OVERVIEW.md) for a module-by-module tour, or the
[root README](../README.md) for what this fork is and why.

## Using the code

| Doc | What it covers |
|---|---|
| [`USAGE.md`](USAGE.md) | The five CLIs — `mulan-train`, `mulan-predict`, `mulan-att`, `mulan-landscape`, `plm-embed`. Inputs, flags, worked examples. |
| [`CODEBASE_OVERVIEW.md`](CODEBASE_OVERVIEW.md) | Module-by-module tour of `mulan/`, `scripts/`, `experiments/`, `scripts_plots/`. The best single onboarding document. |
| [`REPRODUCE.md`](REPRODUCE.md) | What ships and what is rebuilt locally, the shape of the pipeline from raw SKEMPI to the headline table, and an exact path for verifying every model number without retraining. The commands for *regenerating* embeddings and retraining are an outline. |
| [`MPS_COMPATIBILITY.md`](MPS_COMPATIBILITY.md) | Apple Silicon / MPS support, device selection, known kernel gaps and fallbacks. |
| [`CV_SPEEDUP.md`](CV_SPEEDUP.md) | Why cross-validation is fast here: the in-RAM embedding cache and what it replaced. |

## The three things this fork adds

| Doc | What it covers |
|---|---|
| [`EMBEDDING_SETUP.md`](EMBEDDING_SETUP.md) | **(a)** Per-PLM generator, cache directory and driver guide. The operational authority for producing embeddings. |
| [`PLM_BACKEND.md`](PLM_BACKEND.md) | **(a)** The backbone registry: how a tag resolves to a model, an input convention and a trimming rule, and why it is a table rather than a chain of `if`s. |
| [`FOLDX.md`](FOLDX.md) | **(b)** The FoldX binding-ΔΔG channel: pipeline, result store, the two arms, the column contract, coverage and caveats. |
| [`FOLDX_STRUCTURE_ARC_AND_AUGMENTATION_PLAN.md`](FOLDX_STRUCTURE_ARC_AND_AUGMENTATION_PLAN.md) | **(b)** The full scientific arc — every route for structure that was evaluated, why FoldX-as-a-score-channel won, and the augmentation plan. The narrative account; [`RESULTS.md`](RESULTS.md) remains the source of truth for any individual number. |
| [`SPLITS_AND_METRICS.md`](SPLITS_AND_METRICS.md) | **(c)** The evaluation protocols (by-complex, sequence-clustered, CATH-superfamily), the metric, and how to read our numbers against the published frontier. |

## Results and honesty

| Doc | What it covers |
|---|---|
| [`RESULTS.md`](RESULTS.md) | The consolidated results record — the master narrative for every campaign. |

## Archive

[`history/`](history/README.md) holds superseded plans, closed investigations and dated run notes.
They are kept for provenance — decisions in this project were often reversed on evidence, and the
reversals are part of the record. Do not treat anything in `history/` as current.
