# By-complex COMBINED (single+multi) retrain — GPU/Linux-box run instructions

_The frontier-matched by-complex run: MuLAN per-structure Spearman on the **combined single+multi**
SKEMPI v2 set with a whole-PDB hold-out — the exact protocol RDE-Network / DiffAffinity / Prompt-DDG /
BA-DDG report (they train **one** model on single+multi, not SP-only and MP-only pooled). The split is
already built on the Mac; this doc is the Linux/GPU-box half, needed because the ≥3B / ankh3 / saprot13b
embeddings live **only** on the GPU box._

## Why the Linux box is needed

Head-training on cached WT embeddings — **no embedding regeneration** (the combined set is a subset of
the full-SKEMPI complexes already embedded; the SP and MP by-complex runs read the same caches). But the
embeddings are split across machines:

| PLM | full-SKEMPI embeddings | combined-by-complex run |
|---|---|---|
| ankh, esm2, prostt5, saprot, **esmc600m** | Mac-resident | **Mac** (already queued: pueue #133–137) |
| **esmc6b**, **saprot13b**, **ankh3_large**, **ankh3_xl** | GPU-box only (Mac dirs empty/partial) | **Linux/GPU box** ← this doc |

## The split (already built — ships in the handoff tarball)

Built fresh over the **combined** per-complex sizes (not SP∪MP pooled — that would leak, since SP and MP
were partitioned independently and a complex can land in different folds). `build_splits_bycomplex.py`,
seed 42, 3 folds, val-frac 0.15:

- `scratch/splits_skempi_full_bycomplex_all_seed42/fold_{0,1,2}/skempi_all_{train,val,test}.tsv`
- 5801 rows / 337 complexes → ~1934 test muts & **~40 complexes ≥10 muts per fold** (much richer T≥10
  coverage than SP-only's 24 or CATH's 13).
- **Integrity verified on the Mac:** each complex is a test complex in exactly one fold; train/val/test
  complex-disjoint within every fold.

Config: `experiments/full_skempi_seqonly/config_skempi_full_bycomplex_all.sh` (num_folds=3, basename
`skempi_all`, fasta `wt_sequences_cath_all.fasta` — the same union fasta the CATH tier uses; it covers
all 337 combined complexes).

## Prereqs on the Linux box (verify before running)

```bash
cd /path/to/mulan
# 1. split + data + config synced in (from the handoff tarball):
ls scratch/splits_skempi_full_bycomplex_all_seed42/fold_0/skempi_all_{train,val,test}.tsv
ls scratch/skempi_full/{all_point.tsv,wt_sequences_cath_all.fasta}
ls experiments/full_skempi_seqonly/config_skempi_full_bycomplex_all.sh
# 2. GPU-resident embeddings present (the reason we're here):
for p in esmc6b saprot13b ankh3_large ankh3_xl; do echo -n "$p: "; ls scratch/embeddings_skempi_full/$p 2>/dev/null | wc -l; done
# 3. multi-point collator patch present (the "all" set has multi rows):
grep -n "MutatedComplex" mulan/data.py | head
```

## Run — BASE arm (cached embeddings → minutes–hours/fold; base is all we need for the frontier bars)

Queue via pueue in an isolated group so it doesn't disturb in-flight jobs. `MULAN_ROOT`/`MULAN_VENV_BIN`
make `run_bycomplex.sh` portable to the Linux layout. One task per model (granular — reorderable):

```bash
pueue group add bycplx_all 2>/dev/null; pueue parallel -g bycplx_all 1
ROOT=/path/to/mulan
for tag in esmc6b saprot13b ankh3_large ankh3_xl; do
  pueue add -g bycplx_all -l "bycplxALL_${tag}" -- bash -lc "cd $ROOT && \
    MULAN_ROOT=$ROOT MULAN_VENV_BIN=$ROOT/.venv/bin \
    ES_CONFIG=experiments/full_skempi_seqonly/config_skempi_full_bycomplex_all.sh \
    ARMS=base MAXPAR=1 bash experiments/retrain_split/run_bycomplex.sh $tag"
done
```

Output → `scratch/results/full_skempi_bycomplex_all/<tag>_base/fold_{0,1,2}/training_run/`. Resumable
(skips a fold whose `all_results.json` exists).

## Hand back to the Mac

Snapshot only the text deliverables (no `.pt`): for each `<tag>_base/fold_*/training_run/`, the
`test_predictions.tsv` + `all_results.json`. The Mac scores them (per-structure Spearman T≥10 on
`skempi_all_test.tsv`), adds the `fullSK bycomplex-ALL` rows to `results_matrix_ps.csv`, and completes
the frontier-matched by-complex panel (comparators: BA-DDG 0.513, RDE-Network 0.401, DiffAffinity 0.397,
Prompt-DDG 0.426 — all computed on this same combined single+multi protocol).

## Mac side (runs in parallel — for reference)

Already queued (pueue #133–137, group `retrain`): ankh (first) → esmc600m → prostt5 → saprot → esm2, all
base arm. The two machines cover all PLMs.

## Optional later — FoldX arms & AIDO-16B

- FoldX arms aren't built for this combined tier yet. When wanted: `merge_foldx_full_skempi.py --src
  scratch/splits_skempi_full_bycomplex_all_seed42` into a distinct `bycomplex_all` foldx outdir, set
  `ES_FOLDXSCALAR_DIR`/`ES_FOLDXDEC_DIR` in the config, then `ARMS="foldx foldx_scalar"`.
- AIDO-16B is not in `models_skempi_full.tsv` (separate `aido_ladder_manifest.py` pipeline); run it via
  that manifest against `skempi_all_*` if an AIDO combined-by-complex bar is wanted.
