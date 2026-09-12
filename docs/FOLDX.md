# The FoldX ΔΔG channel

> This document covers how MuLAN *consumes* FoldX energies as a feature. The pipeline that
> produces them lives in [skempi-foldx](https://github.com/cchin29/skempi-foldx); `mulan/foldx/`
> is now the merge step plus a re-export of it.
> `docs/FOLDX_STRUCTURE_ARC_AND_AUGMENTATION_PLAN.md` carries the scientific arc.

## What it is

A zero-shot score channel carrying **FoldX binding free-energy change**, fed to MuLAN alongside the
protein-language-model embeddings.

The quantity is the change in *interaction* energy between the two binding partners:

```
ΔΔG_int  =  InteractionEnergy(mutant)  −  InteractionEnergy(wild-type)
```

computed by FoldX as `RepairPDB → BuildModel → AnalyseComplex`, at roughly 8 s per mutation, with
the expensive `RepairPDB` step cached once per complex.

The property that makes it worth having: **the channel is encoder-independent.** It is computed once
per (complex, mutation) from structure, then reused unchanged across every PLM backbone and every
evaluation split. So it isolates a clean question — *does adding cheap biophysics to a frozen
sequence model close the gap to structure-based methods?* — without confounding it with the choice
of encoder.

## The two arms

| Arm | Input | Head | Added parameters |
|---|---|---|---|
| `foldx_scalar` | the single `Interaction Energy` term | one extra input to the final linear layer | **+1** |
| `foldx` | all **12** `AnalyseComplex` terms | `Linear(12→16) → ReLU → Dropout → Linear(16→1)`, collapsing to one slot | **+226** |

Both arms produce the same head width — the 12-term MLP compresses to a single scalar before the
concatenation — so the two are directly comparable, and the comparison is a clean test of whether
the energy decomposition carries signal beyond the summary term.

The 12 terms, in the order they occupy columns 4–15 of a decomposed split file:

```
Interaction Energy · Backbone Hbond · Sidechain Hbond · Van der Waals · Electrostatics ·
Solvation Polar · Solvation Hydrophobic · Van der Waals clashes · entropy sidechain ·
entropy mainchain · torsional clash · backbone clash
```

Model configs: `models/config/lightatt_addscores_config.json` (scalar) and
`models/config/lightatt_addscores_mlp_config.json` (12-term). Both set `add_scores: true`; the MLP
config adds `zs_mlp: true, zs_input_dim: 12, zs_mlp_hidden: 16`.

## The column contract

Split files are headerless TSV. The number of columns *is* the schema — `mulan/data.py` dispatches
on it:

| Columns | Layout | Arm | Model config |
|---|---|---|---|
| 4 | `seq1 · seq2 · mutations · ΔΔG` | `base` | `lightatt_default_config.json` |
| 5 | `… ΔΔG · z(Interaction Energy)` | `foldx_scalar` | `lightatt_addscores_config.json` |
| 16 | `… ΔΔG · z₀ … z₁₁` | `foldx` | `lightatt_addscores_mlp_config.json` |

> ⚠ **Footgun.** Passing `--add_zs_scores True` with a *4-column* file makes the ΔΔG label itself be
> read as the score, leaving no label; training then fails inside the loss. The column count and the
> flag must agree.

Scores are **z-standardized with a train-only fit** (per fold — the mean and standard deviation come
from the training partition alone, never from validation or test), clipped to ±4, and missing
values are imputed as 0, i.e. the fold's training mean. Because the fit is per-fold, changing which
rows are covered nudges every standardized value in that fold slightly; that is expected, and is not
evidence of a join error.

## Where the energies come from

**The pipeline that computes them is no longer in this repository.** It lives in
[skempi-foldx](https://github.com/cchin29/skempi-foldx) — MIT-licensed, dependency-free, and
shipping the computed results for SKEMPI 2.0: 322 complexes / 4238 single-point mutations and
152 / 1765 multi-point variants at the v0.1.0 tag vendored here.

It was extracted because none of it derived from upstream MuLAN, yet inside this repository it
inherited CC BY-NC-SA — so energies that cost a FoldX licence and CPU-weeks to produce, and
that are useful to anyone working on binding ΔΔG, could not be used commercially or relicensed by
anyone. It also had zero coupling in either direction, so the split cost nothing structurally.

Three things there are worth reading before trusting a number from this channel:

| | |
|---|---|
| [`docs/DETERMINISM.md`](https://github.com/cchin29/skempi-foldx/blob/v0.1.0/docs/DETERMINISM.md) | FoldX is deterministic, but a mutation's ΔΔG depends on every entry preceding it in `individual_list.txt` — so a *subset* computation is not interchangeable with a full one. Also the repair-count ablation. |
| [`docs/STORE.md`](https://github.com/cchin29/skempi-foldx/blob/v0.1.0/docs/STORE.md) | The result store, its coverage, and why `1KBH` is excluded outright. |
| [`docs/PROTOCOL.md`](https://github.com/cchin29/skempi-foldx/blob/v0.1.0/docs/PROTOCOL.md) | How these settings compare with the published ΔΔG literature, which has no single convention. |

A snapshot of the results is still vendored here at `data/foldx/`, so this repository reproduces
end to end without fetching anything. It is pinned to skempi-foldx **v0.1.0**, and the links above
point at that project's `main`, which documents its 0.2.0 — a rebuilt store whose record counts
and 1850 of whose values differ from the snapshot here. Read them for the method, not for figures
that should match `data/foldx/`. skempi-foldx is the canonical home for the energies themselves.

## Where the code lives

| Module | Role |
|---|---|
| `mulan/foldx/merge.py` | Joining onto splits: key mapping, the grouping guard, per-fold standardization, both arms. Also the clip bound, the missing-value fill and the column layouts — these describe how *this* model encodes the energies, so they stayed when the producer left. |
| `mulan/foldx/__init__.py` | A thin adapter: re-exports skempi-foldx so `from mulan.foldx import load_store` still works, and adds the merge API. |
| `skempi_foldx` (external) | The compute engine, the SKEMPI parsing, the result store, the term contract, the exclusion registry. |

This replaces `build_ddg.py` plus four campaign drivers, and six separate mergers. The drivers
worked by assigning to the engine's module globals (`bd.WORK = ...`) and re-applying the
assignment inside a process-pool initializer; paths now travel in a config object instead, so
two campaigns can run in one interpreter.

### Why the merge is verified rather than reviewed

A merge failure does not raise. It produces a correctly-shaped split file whose numbers are
wrong, and the coverage counter that should catch it cannot — coverage counts keys, and the
damage is in values. That is not hypothetical: a change that added a third candidate join key
left coverage byte-identical while handing ~13 mutations another mutation's energies, and it
surfaced only because downstream accuracy fell.

So the unified merger was checked by reproducing what it replaces:

| | files | result |
|---|---|---|
| S1102 10-fold, scalar + decomposed | 60 | **byte-identical**, coverage 100% |
| full-SKEMPI 3-fold, scalar + decomposed | 18 | **byte-identical**, coverage 12393/12495 = 99.2% |
| grouping-guard denials | | 54 row-instances on `3SE4.B.A` — matching the audit exactly |

The store consolidation was checked the same way: the per-mutation union of the five campaign
directories reproduces the previous store with **zero** value differences. That store now holds
322 complexes / 4238 mutations — one fewer complex than when this check was first run, because
`1KBH` was removed as uncomputable (see skempi-foldx's `STORE.md`).

## Licensing

**FoldX is licensed software from the CRG — free for academic and non-profit use, paid for commercial — and is not distributed with this repository.** Running the compute
side requires a separately obtained licence and binary from <https://foldxsuite.crg.eu/>. FoldX 5.1 does not need
a `rotabase.txt` file; older major versions do.

The FoldX *outputs* computed for this work — the per-complex ΔΔG JSON — are our own results. A
snapshot is vendored at `data/foldx/` so this pipeline reproduces end to end without a FoldX
licence; their canonical home, with the terms that govern reuse, is
[skempi-foldx](https://github.com/cchin29/skempi-foldx) and its `NOTICE`.

Note the licence asymmetry: **this repository is CC BY-NC-SA** (inherited from upstream MuLAN and
non-relicensable), but **the vendored energies under `data/foldx/` do not inherit it**. They are
skempi-foldx's artifact — MIT for its code, CC BY 4.0 for the SKEMPI-derived data — and they carry
those terms here too. See [`../data/foldx/README.md`](../data/foldx/README.md). Nothing about
being copied into an NC repository can narrow a CC BY 4.0 grant.
