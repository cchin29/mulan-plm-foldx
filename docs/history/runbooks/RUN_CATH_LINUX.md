# CATH-superfamily retrain — GPU/Linux-box run instructions

_The apples-to-apples run that drops MuLAN literally into USP-ddG Table 1. The CATH split is
already built on the Mac; this doc is the Linux/GPU-box half, needed because the ESM-C 6B (and
other ≥3B) embeddings live **only** on the GPU box._

## Why the Linux box is needed at all

The CATH retrain is **head-training on cached WT embeddings — no embedding regeneration** (our
build already covers 50/53 test complexes + all train complexes). But the embeddings are split
across machines:

| PLM | full-SKEMPI embeddings | CATH retrain runs on |
|---|---|---|
| ankh, esm2, prostt5, saprot | **Mac-resident** | **Mac** (already queued/runnable) |
| **esmc6b**, esmc600m, ankh3_large, ankh3_xl, saprot13b | **GPU-box only** (Mac dirs are empty) | **Linux/GPU box** ← this doc |

Two of our three leakage-controlled-tier headline arms are GPU-resident: **`esmc6b_foldx_scalar`** (best overall)
and **`ankh3_xl_foldx`** (best per-structure). So the frontier row can't be completed without this run.

**No new ESM-C 6B embeddings are required** unless we want the 3 uncovered test complexes
(`1S0W`, `1XXM`, `2NOJ`) — optional; they cost ~126 of USP's 813 test rows in aggregate and can be
added later with the standard 6B embedder (`scratch/esmc6b_gen_fasta.py`).

## The split (already built — ships in the next Mac→Linux snapshot)

`build_split_cath.py` reproduced USP-ddG's **literal** CATH-superfamily hold-out by joining their
shipped `cath_fold` column (constant per complex) to our rows **by complex**:

- **TEST** `scratch/splits_skempi_full_cath_kfold/fold_0/skempi_all_test.tsv` — 687 rows
  (421 single / 266 multiple) over 50/53 val complexes. Multi coverage 266/270 = 98.5%.
- **TRAIN/VAL** `skempi_all_{train,val}.tsv` — 4213 / 878 (10% of complexes, seed 2024).
- Single/multiple test subsets + audit: `test_single.tsv`, `test_multiple.tsv`, `JOIN_REPORT.txt`.

Config: `experiments/full_skempi_seqonly/config_skempi_full_cath.sh` (num_folds=1, basename
`skempi_all`, fasta `wt_sequences_cath_all.fasta`).

## Prereqs on the Linux box (verify before running)

```bash
cd /path/to/mulan
# 1. the split artifacts synced in (from the Mac snapshot):
ls scratch/splits_skempi_full_cath_kfold/fold_0/skempi_all_{train,val,test}.tsv
ls scratch/skempi_full/wt_sequences_cath_all.fasta
ls experiments/full_skempi_seqonly/config_skempi_full_cath.sh
# 2. the GPU-resident embeddings are present (the reason we're here):
for p in esmc6b esmc600m ankh3_large ankh3_xl; do echo -n "$p: "; ls scratch/embeddings_skempi_full/$p | wc -l; done
# 3. collator patch present (multi-point batching) — required since the "all" set has multi rows:
grep -n "MutatedComplex" mulan/data.py | head
```

## Run — BASE arm (cheapest; the base floor + FoldX-lift denominator)

Cached embeddings → minutes/fold. Queue via pueue in an isolated `cath` group so it doesn't
disturb any in-flight jobs:

```bash
pueue group add cath 2>/dev/null; pueue parallel -g cath 1
pueue add -g cath -- bash -lc 'cd /path/to/mulan && \
  ES_CONFIG=experiments/full_skempi_seqonly/config_skempi_full_cath.sh \
  ARMS=base MAXPAR=1 bash experiments/retrain_split/run_bycomplex.sh \
  esmc6b esmc600m ankh3_large ankh3_xl'
```

## Run — FoldX arms (foldx_scalar / foldx = the headline lift)

**The CATH FoldX splits are already built on the Mac and ship in the snapshot** — no merge step on
the Linux box. `merge_foldx_cath.py` produced them (mixed single+multi, one shared train-fit
z-standardization); coverage 99.1% (5746/5801 per fold), measured 2026-08-01:

```bash
ls scratch/foldx_skempi_full/splits_cath_foldx/fold_0/skempi_all_test.tsv       # 5-col scalar
ls scratch/foldx_skempi_full/splits_cath_foldxdec/fold_0/skempi_all_test.tsv    # 16-col 12-term
# (only if regenerating:) .venv/bin/python experiments/full_skempi_seqonly/merge_foldx_cath.py
```
Config `config_skempi_full_cath.sh` already points `ES_FOLDXSCALAR_DIR`/`ES_FOLDXDEC_DIR` at these.
Run the FoldX arms:

```bash
pueue add -g cath -- bash -lc 'cd /path/to/mulan && \
  ES_CONFIG=experiments/full_skempi_seqonly/config_skempi_full_cath.sh \
  ARMS="foldx foldx_scalar" MAXPAR=1 bash experiments/retrain_split/run_bycomplex.sh \
  esmc6b esmc600m ankh3_large ankh3_xl'
```
_(NOTE: this line previously scoped the single-point FoldX arm to a 28.1%-covered subset. That
triple was the pre-fix `__cov_pre` lineage; the join defect behind it was fixed on 2026-07-30 and
coverage measured 99.1% on 2026-08-01. Both FoldX arms are effectively fully covered — no such
scoping is needed.)_

## Score → the frontier row

```bash
# per-structure + overall + AUROC, broken out single / multiple / all (USP-ddG Table 1 layout)
.venv/bin/python experiments/full_skempi_seqonly/score_cath.py
#   -> results_cath{,_single,_multiple}.csv  +  SUMMARY_cath.md
# format the leaderboard rows (all / single / multiple):
.venv/bin/python experiments/retrain_split/format_usp_row.py \
    experiments/full_skempi_seqonly/results_cath.csv --all --split cath --mut all
```

## Hand back to the Mac

Snapshot only the text deliverables (no `.pt`): `results_cath/results.csv`, `SUMMARY_cath.md`, the
`scratch/results/full_skempi_cath/**/test_predictions.tsv` + `all_results.json`. The Mac merges,
regenerates `results_matrix_ps.csv`, and fills the frontier figure/table.

## Mac side (runs in parallel — for reference)

```bash
pueue add -g cath -- bash -lc 'cd "$MULAN_ROOT" && \
  ES_CONFIG=experiments/full_skempi_seqonly/config_skempi_full_cath.sh \
  ARMS=base MAXPAR=1 bash experiments/retrain_split/run_bycomplex.sh ankh esm2 saprot prostt5'
```
ankh/esm2/saprot/prostt5 are Mac-resident → these run on the Mac; the two machines cover all PLMs.
