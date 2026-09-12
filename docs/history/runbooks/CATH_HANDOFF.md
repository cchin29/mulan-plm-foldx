# CATH-superfamily retrain — GPU-box handoff (targeted tarball)

Targeted drop for the **CATH-superfamily hold-out retrain** (USP-ddG's exact 813-mut test set) —
the apples-to-apples run that drops MuLAN literally into USP-ddG Table 1. Everything additive;
nothing the GPU box has is overwritten. Source: Mac, branch `a1-interface-xattn`.

## Why this run (30-second version)

MuLAN's leakage-controlled numbers so far are on the **CD-HIT ≤60%** split (= ProtBFF's split). USP-ddG /
CATH-ddG report on the stricter **CATH-superfamily** hold-out. USP-ddG **ships** that partition as a
static `cath_fold` column — so we reproduced their **literal** 813-mut test set (not a proxy) by
joining it to our full-SKEMPI rows **by complex** (`cath_fold` is constant per complex; verified 0
splits). This run gives the byte-exact frontier-comparable row.

## Unpack (from the mulan repo root on the GPU box)

```bash
cd /path/to/mulan
tar xzf mulan_cath_handoff_<TS>.tar.gz --strip-components=1   # paths are mulan/... ; lands in-tree
```

## What's inside (all pre-built — no split/merge step on the GPU box)

- `scratch/splits_skempi_full_cath_kfold/` — the split. TEST 687 rows (421 single / 266 multiple)
  over 50/53 val complexes; TRAIN 4213 / early-stop val 878. `test_single.tsv`, `test_multiple.tsv`,
  `JOIN_REPORT.txt`.
- `scratch/foldx_skempi_full/splits_cath_foldx{,dec}/` — FoldX add_scores splits (5-col scalar /
  16-col 12-term), single+multi combined, one shared train-fit z-standardization. Coverage
  99.1% (5746/5801 per fold), measured 2026-08-01.
- `scratch/skempi_full/wt_sequences_cath_all.fasta` — union WT fasta (covers all 670 split labels).
- `experiments/full_skempi_seqonly/config_skempi_full_cath.sh` — the run config (num_folds=1,
  basename `skempi_all`; sources config_skempi_full.sh which the GPU box already has).
- `experiments/full_skempi_seqonly/{build_split_cath,merge_foldx_cath,score_cath}.py` — builder /
  FoldX-merge / breakout-scorer (for reproducibility; the outputs above are already built).
- `experiments/retrain_split/format_usp_row.py` — frontier-row formatter.
- `experiments/full_skempi_seqonly/RUN_CATH_LINUX.md` — the full run guide (prereqs → run → score).
- `scratch/staging_usp/USP-ddG/data/SKEMPI2/skempi_v2.csv` — USP's labeled CSV (only needed to
  RE-generate the split via build_split_cath.py; not needed to run the retrain).

## What the GPU box already has (NOT shipped)

- The **embeddings** — this is the whole reason the run is here: `scratch/embeddings_skempi_full/`
  for esmc6b / esmc600m / ankh3_large / ankh3_xl live only on the GPU box. **No embedding
  generation is needed** (50/53 test complexes covered; only 1S0W/1XXM/2NOJ absent — optional).
- MuLAN source incl. the multi-point collator patch (the MP runs used it), `run_bycomplex.sh`,
  `config_skempi_full.sh`, `embedding_sweep/config.sh`, `rescore.py`.

## Run (see RUN_CATH_LINUX.md for full detail)

```bash
# BASE arm (base floor + FoldX denominator)
ES_CONFIG=experiments/full_skempi_seqonly/config_skempi_full_cath.sh \
  ARMS=base MAXPAR=1 bash experiments/retrain_split/run_bycomplex.sh esmc6b esmc600m ankh3_large ankh3_xl
# FoldX arms (headline lift; splits pre-built)
ES_CONFIG=experiments/full_skempi_seqonly/config_skempi_full_cath.sh \
  ARMS="foldx foldx_scalar" MAXPAR=1 bash experiments/retrain_split/run_bycomplex.sh esmc6b esmc600m ankh3_large ankh3_xl
# SCORE -> single/multiple/all + frontier row
.venv/bin/python experiments/full_skempi_seqonly/score_cath.py
.venv/bin/python experiments/retrain_split/format_usp_row.py \
    experiments/full_skempi_seqonly/results_cath.csv --all --split cath --mut all
```

Cached embeddings → minutes/fold on the GPU; expect ~30–90 min for all 4 PLMs base, similar per
FoldX arm. (Mac runs ankh/esm2/saprot/prostt5 separately — those embeddings are Mac-resident.)

## Hand back to the Mac (text only, no `.pt`)

`results_cath{,_single,_multiple}.csv`, `SUMMARY_cath.md`, and
`scratch/results/full_skempi_cath/**/{test_predictions.tsv,all_results.json}`. The Mac merges,
regenerates `results_matrix_ps.csv`, and fills the frontier figure/table.
