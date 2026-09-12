# ESM-C 6B base arm on the augmented combined clustered tier

GPU-box runbook. One arm, three folds, no PLM inference and no changes under `mulan/`.
Follow-up to the Ankh-large result recorded in `RESULTS.md` §24 and
`experiments/beyond_foldx/REDUNDANCY_RESULT.md`.

> **Executed 2026-08-06. The result is null, and it did not answer the question this runbook was
> written to answer.** ESM-C 6B's base arm on `clustered-ALL+aug` came back at 0.2058 → 0.1738,
> Δ −0.0320 [−0.0760, +0.0100]; AIDO-16B, run as the tie-breaker, at 0.1565 → 0.1104,
> Δ −0.0461 [−0.0981, +0.0057]. Both CIs exclude +0.056, so the base gain is not a property of the
> tier. Separately, the six-of-six FoldX result the question below rests on was measured on a
> defective augmented channel; Ankh-large's two arms have since been re-run and most of the harm was
> the defect. `RESULTS.md` §24 carries the current reading. The design and the pre-registered
> criteria below are kept as the record of what was asked in advance of the answer.

## 1. The question

Tier-1 augmentation loses on **six of six** FoldX arms on `fullSK clustered-ALL`, and Ankh-large —
the only backbone with a base arm there — *gains* on it. Paired over the 121 shared complexes:

| Ankh-large, clustered-ALL | plain | +aug | Δ paired |
|---|---|---|---|
| base | 0.109 | 0.165 | **+0.056 [+0.010, +0.104]** |
| + FoldX scalar | 0.418 | 0.353 | **−0.065 [−0.097, −0.034]** |
| + FoldX 12-term | 0.410 | 0.368 | **−0.042 [−0.080, −0.001]** |

Three significant deltas in opposite directions inside one backbone. That is what rules out
"augmentation fails at this scale" and leaves "augmentation conflicts with the FoldX channel".

One backbone is thin for a claim that general. ESM-C 6B is the discriminating second case because
its two augmented FoldX arms are already trained (0.414 → 0.377 scalar, 0.422 → 0.360 12-term:
two of the six negatives), and because it holds the largest non-redundant base-pathway content of
any backbone on this tier — `part_base` 0.127 against Ankh-large's 0.033. If the sign split is a
property of augmentation rather than of Ankh-large, this run reproduces it on the backbone where
there is most to lose.

## 2. Preflight

Four things must hold before the run. Three are cheap checks; one is a build.

**a. The config carries its own `ES_SPLIT_DIR`.** Commit `3f8d422` added the override to
`config_skempi_full_clustered_all_aug.sh`. Confirm the checkout has it:

    grep ES_SPLIT_DIR experiments/full_skempi_seqonly/config_skempi_full_clustered_all_aug.sh
    # expect: export ES_SPLIT_DIR=scratch/splits_skempi_full_clustered_all_aug

Without that line the aug config inherits `ES_SPLIT_DIR` from the unaugmented parent, and
`ARMS=base` trains the **plain** base into the `_aug` results directory and exits 0. A wrong answer
that reports success, not a failure. It would land within noise of the plain arm and read as
"augmentation does nothing" — the exact conclusion this run exists to test.

**b. The base aug split exists.** `scratch/` is gitignored, so it does not arrive with the
checkout. §3 builds it.

**c. ESM-C 6B embeddings cover the augmented backgrounds.** Reverse rows carry synthetic ids that
encode the mutated background (`1A22.A.B_A_CA171A`, `1A22.A.B_A_CA171A_AA171C`), and each needs its
own `.pt`. The augmented FoldX arms on this tier already ran on this box, so the cache should be
complete; verify rather than assume, because the driver's guard is `[ -d "$emb" ]` — directory
existence only, so an empty or partial cache passes it.

    awk '/^>/{gsub(/^>/,"");print $1}' scratch/skempi_full/wt_sequences_clustered_all_aug.fasta \
      | sort -u > /tmp/aug_ids.txt
    wc -l < /tmp/aug_ids.txt          # expect 5702
    missing=0; while read id; do
      [ -f "scratch/embeddings_skempi_full/esmc6b/$id.pt" ] || { echo "MISSING $id"; missing=$((missing+1)); }
    done < /tmp/aug_ids.txt; echo "missing=$missing"

`missing` must be 0. If it is not, the gap is embedding generation, not training, and this runbook
does not cover it.

**d. The test TSVs are untouched by augmentation.** This is what makes the contrast paired. It held
on the Mac by md5 across all three folds; re-check on this box after §3:

    for f in 0 1 2; do
      md5sum scratch/splits_skempi_full_clustered_all_id60_kfold/fold_$f/skempi_all_test.tsv \
             scratch/splits_skempi_full_clustered_all_aug/fold_$f/skempi_all_test.tsv
    done

The two md5s must match within each fold.

## 3. Build the base aug split

The 4-column base split is the 5-column FoldX split minus its last column. This is exact, not
approximate: `cut -f1-4` of `splits_cath_foldx_aug` reproduces
`splits_skempi_full_clustered_all_aug` byte-for-byte on train, val and test across all three folds
(verified 2026-08-05, 9 of 9 files).

    SRC=scratch/foldx_skempi_full/clustered_all/splits_cath_foldx_aug
    DST=scratch/splits_skempi_full_clustered_all_aug
    for f in 0 1 2; do
      mkdir -p $DST/fold_$f
      for s in train val test; do
        cut -f1-4 $SRC/fold_$f/skempi_all_$s.tsv > $DST/fold_$f/skempi_all_$s.tsv
      done
    done

Row counts to expect, and the shape of the augmentation:

| fold | train | val | test |
|---|---|---|---|
| 0 | 6058 | 537 | 2295 |
| 1 | 6852 | 695 | 1753 |
| 2 | 6877 | 689 | 1753 |

Train is roughly double the unaugmented tier (2969 / 3353 / 3359); val and test are unchanged.
Every row must have exactly 4 tab-separated fields:

    awk -F'\t' 'NF!=4{print FILENAME": "NR" NF="NF; bad=1} END{exit bad}' $DST/fold_*/*.tsv \
      && echo "4-col OK"

## 4. Run

    ES_CONFIG=experiments/full_skempi_seqonly/config_skempi_full_clustered_all_aug.sh \
        ARMS=base MAXPAR=1 bash experiments/retrain_split/run_bycomplex.sh esmc6b

300ep / patience 30, the same budget as every other arm in the matrix. Reference runtime: 3 h 29 m
for Ankh-large (D = 1536) on Apple MPS at `MAXPAR=2`. ESM-C 6B is D = 2560, so expect more work per
step and far more throughput per step on this box.

Results land in `scratch/results/full_skempi_clustered_all_aug/esmc6b_base/fold_{0,1,2}/training_run/`.
The plain twin — the comparator — is already on disk at
`scratch/results/full_skempi_clustered_all/esmc6b_base/`; do not overwrite it.

## 5. The standing driver check

`run_bycomplex.sh` logs one line per arm before it trains. Read it. If the split is wrong the run
is wrong regardless of what the config says:

    === esmc6b base  plm=esmc_6b emb=scratch/embeddings_skempi_full/esmc6b  splits=scratch/splits_skempi_full_clustered_all_aug  300ep/p30 ===

`splits=` **must** end in `_aug`. Ending in `_id60_kfold` is trap 2a firing — kill the run, fix the
config, start over. Nothing downstream detects it.

## 6. Scoring

    ./.venv/bin/python experiments/full_skempi_seqonly/score_sp_all.py clustered_all_aug

This writes `experiments/full_skempi_seqonly/results_clustered_all_aug.csv` and appends the
`fullSK clustered-ALL+aug` rows to `scripts_plots/results_matrix_ps.csv`. The conclusion metric is
**per-structure Spearman at T ≥ 10** (ppS), folds pooled before grouping by complex — not
`test_pcc`, which the training log prints per fold and which is a pooled Pearson over a different
population.

For the paired contrast and its CI, and for the redundancy decomposition:

    ./.venv/bin/python experiments/beyond_foldx/redundancy.py --root . \
        --tier clustered-ALL --model esmc6b \
        --out experiments/beyond_foldx/redundancy_clustered_all_esmc6b.csv

That prints `fullSK clustered-ALL` and `fullSK clustered-ALL+aug` side by side. Both tiers read the
same three test TSVs, so the rows are paired by construction. The script self-validates — a
single-feature FoldX blend is affine in rank(FoldX) within complex, so `blend_fx` must equal
`rho_fx` to 1e-6, and it raises `[BUG ]` otherwise. If that line appears, the numbers are not usable.

## 7. Reading the result

Comparator is **ESM-C 6B's own plain base arm on the same tier: ppS 0.2058** (5801 rows, 121
scored complexes). Not FoldX-alone (0.431 here), not Ankh-large.

| outcome | reading |
|---|---|
| base Δ significantly **positive**, ~+0.05 | the sign split is a property of augmentation. §24's conclusion generalizes past one backbone; write it up as such. |
| base Δ **flat**, CI spanning 0 | Ankh-large's gain is backbone-specific. §24's conclusion narrows to Ankh-large and the conflict reading loses its support — the FoldX-arm negatives go back to being ambiguous. |
| base Δ significantly **negative** | augmentation hurts ESM-C 6B on every arm, which is the scale hypothesis surviving on this backbone while failing on Ankh-large. Report the split; do not average the two. |

Also record `part_base` either way. On Ankh-large it doubled (0.033 → 0.081, paired
+0.047 [+0.004, +0.093]) while `base~fx` did not move, and the optimally weighted blend still
returned nothing (+0.003 → +0.002). Whether ESM-C 6B repeats that pattern — more non-redundant
signal, still unspendable — is the part that bears on the beyond_foldx diagnosis rather than on
augmentation.

A note on what this run cannot settle: the *mechanism* for the FoldX arms' decline is open. The
obvious explanation — that exact negation forces an odd-symmetric response to the FoldX features,
which the data would penalise — is not supported: the FoldX→ΔΔG slope is +1.16 on stabilizing rows
against +0.83 on destabilizing, but the asymmetry is +0.33 [−0.15, +0.73] under a cluster bootstrap
over the 238 training complexes.

**This does not change what this run measures, but it changes what it can be compared against.** On
2026-08-05 the FoldX channel of every synthesized row was found to carry a constant offset of about
one channel width, because the negation was applied after standardisation rather than before
(`FOLDX_ANTISYMMETRY_RESULT.md` §1). The base arm reads only the 4-column split and never touched
that channel, so **this run is unaffected and still decisive for the question it was written to
settle**. The FoldX arms it would be set beside are not: those must be re-run on the repaired splits
before any base-vs-FoldX sign split is read as a result.

## 8. Traps

1. `ES_SPLIT_DIR` inheritance — §2a. Silent, and produces the null result this run is testing for.
2. `[ -d "$emb" ]` is an existence check, not a completeness check. An empty cache directory passes
   it. §2c is the real gate.
3. The augmented ids are not PDB codes. `1A22.A.B_A_CA171A` is a mutated background, and the
   complex key is `chain1.split("_")[0]` = `1A22.A.B` — the same complex as the forward row. Any
   ad-hoc grouping that splits on `_` differently will mis-key the augmented rows.
4. `config_skempi_full_cath_aug.sh` carries its `ES_SPLIT_DIR` override as of 2026-08-05, so it no
   longer hits trap 1. The split it names, `scratch/splits_skempi_full_cath_aug`, is **not built** —
   `scratch/` is gitignored, and the config comment carries the `cut -f1-4` recipe. Both invariants
   were verified on that tier before the line was written: the cut is exact against the unaugmented
   4-col split (4213 / 878 / 687 rows on train / val / test), and the augmentation is train-only
   (val and test byte-identical, train 4213 → 8683). CATH is a single hold-out, so fold_0 only.
5. Do not re-score the plain tier from the aug results directory, or vice versa. Both live under
   `scratch/results/full_skempi_clustered_all{,_aug}/` and differ only by suffix.
6. `test_pcc` in the training log is not the conclusion metric and is not comparable to anything in
   `results_matrix_ps.csv`.
7. Report ppS to four decimals when merging back. The effect being measured is ~0.05 and the paired
   CI half-width on Ankh-large was ±0.047.

## 9. Optional extension

AIDO-16B is the other backbone with augmented FoldX arms on this tier and no base arm (plain base
0.1565; aug FoldX 0.4156 → 0.3845 scalar, 0.4040 → 0.3767 12-term). Same command, same split, same
preflight, `aido` in place of `esmc6b` — and §2c re-run against
`scratch/embeddings_skempi_full/aido`. It is worth queueing behind the primary run only if the
ESM-C 6B result is flat, since two backbones disagreeing is the case where a third breaks the tie.
