# Tier-1 augmentation on the by-complex-ALL FoldX arms — ESM-C 6B, GPU box

Builds the augmented FoldX splits for `full_skempi_bycomplex_all` and trains `esmc6b`'s two FoldX
arms against them. **6 runs** (2 arms × 3 folds). Setup is seconds; the training is the only cost.

Sibling of `RUN_BYCOMPLEX_ALL_LINUX.md`, which covers the unaugmented tier. Set `MULAN_ROOT` /
`MULAN_VENV_BIN` for the Linux layout exactly as that doc describes; every path below is
repo-relative.

## Motivation

Augmentation on top of the FoldX arms is measured on two protocols and is negative on both:

| protocol | arm | plain ppS | aug ppS | Δ |
|---|---|---|---|---|
| `clustered_all` (3 folds) | `foldx` | 0.4220 | 0.3598 | −0.0622 |
| `clustered_all` (3 folds) | `foldx_scalar` | 0.4144 | 0.3765 | −0.0379 |
| `cath` (1 fold) | `foldx` | 0.4310 | 0.4704 | +0.0394 |
| `cath` (1 fold) | `foldx_scalar` | 0.4378 | 0.4307 | −0.0071 |

`bycomplex_all` is the third protocol and the one the frontier comparison is built on
(RDE-Network / DiffAffinity / Prompt-DDG / BA-DDG), and it has no augmented counterpart. Adding it
either makes the negative result consistent across all three protocols or isolates it to the
clustered split — and `cath`, at a single fold, cannot settle that on its own.

## What the augmentation does, and why the setup is cheap

Tier-1 aug adds reverse-mutation and identity rows to **train only**. `merge_foldx_aug.py` reads the
canonical `foldxdec` split verbatim for the forward and val/test rows, so coverage and values stay
identical to the reference arms, and synthesizes only the new train rows.

A reverse row's 12-term vector is the standardised form of the **negated raw** value,
`clip((−x − mu)/sigma)`, and an identity row's is the standardised form of raw zero, `clip(−mu/sigma)`.
Negating the standardised value instead — which this script did until 2026-08-05 — puts a constant
`2·mu/sigma` on every synthesized row, about one channel width. `merge_foldx_aug.py` reconstructs the
canonical `mu`/`sigma` per fold per term and aborts if they fail to reproduce the committed forward
channel; see `experiments/beyond_foldx/FOLDX_ANTISYMMETRY_RESULT.md` §1. Note that the antisymmetry
this rests on is FoldX's own and is only approximate (§2 there) — reverting a mutation recovers about
0.78 of the forward effect, which no amount of arithmetic here can fix.

Val and test are byte-identical to the unaugmented tier's — verified per fold on the two existing
aug tiers. That is what makes the aug-vs-plain delta a paired comparison on one test set, and it is
why the scorer points both tiers at a single truth file.

Every augmented sequence is byte-identical to some WT chain (a reverse mutant of `X→Y` *is* the WT
of the reverse direction; identity rows are the WT). So `--materialize` symlinks rather than
embeds: **no PLM inference, no GPU time for embeddings.**

## 1. Build the augmented splits

```bash
python experiments/full_skempi_seqonly/merge_foldx_aug.py \
  --dec-src scratch/foldx_skempi_full/bycomplex_all/splits_cath_foldxdec \
  --base    skempi_all \
  --fasta   scratch/skempi_full/wt_sequences_cath_all.fasta \
  --out-scalar scratch/foldx_skempi_full/bycomplex_all/splits_cath_foldx_aug \
  --out-dec    scratch/foldx_skempi_full/bycomplex_all/splits_cath_foldxdec_aug \
  --out-fasta  scratch/skempi_full/wt_sequences_bycomplex_all_aug.fasta
```

`--dec-src` is the **16-column decomposed** split, not the 5-column scalar one; the scalar arm is
regenerated from term 0. `--fasta` is `wt_sequences_cath_all.fasta` — the same source fasta
`config_skempi_full_bycomplex_all.sh` already uses, not a bycomplex-specific file.

**Check before continuing.** Train must grow while val and test stay byte-identical:

```bash
A=scratch/foldx_skempi_full/bycomplex_all/splits_cath_foldx_aug
P=scratch/foldx_skempi_full/bycomplex_all/splits_cath_foldx
for k in 0 1 2; do for f in test val; do
  cmp -s $A/fold_$k/skempi_all_$f.tsv $P/fold_$k/skempi_all_$f.tsv \
    && echo "fold_$k $f IDENTICAL" || echo "fold_$k $f DIFFERS — STOP"
done; wc -l < $A/fold_$k/skempi_all_train.tsv; wc -l < $P/fold_$k/skempi_all_train.tsv; done
```

A `DIFFERS` on val or test means the aug pass touched an evaluation set and the comparison would be
invalid — stop and report rather than training. Train should come out roughly 2× (the clustered
tier went 2969 → 6058).

## 2. Materialize the ESM-C 6B embeddings

```bash
python experiments/full_skempi_seqonly/merge_foldx_aug.py --materialize \
  --base    skempi_all \
  --out-dec scratch/foldx_skempi_full/bycomplex_all/splits_cath_foldxdec_aug \
  --emb-dir scratch/embeddings_skempi_full/esmc6b
```

This reports residual misses. **A non-empty residual list is a stop condition** — it means an
augmented id did not resolve to a WT chain, and training would silently drop those rows per fold
rather than fail.

## 3. Write the config

`experiments/full_skempi_seqonly/config_skempi_full_bycomplex_all_aug.sh`:

```sh
# Tier-1 aug on the COMBINED single+multi by-complex tier — the frontier-matched protocol's
# augmented counterpart, and the third of the three aug tiers. Arms foldx/foldx_scalar here ==
# foldx_mlp_aug / foldx_scalar_aug. Splits built by merge_foldx_aug.py from the canonical
# bycomplex_all foldxdec split; val/test verbatim, train augmented. See RUN_BYCOMPLEX_ALL_AUG_GPU.md.
source experiments/full_skempi_seqonly/config_skempi_full_bycomplex_all.sh
export ES_WT_FASTA=scratch/skempi_full/wt_sequences_bycomplex_all_aug.fasta
export ES_FOLDXSCALAR_DIR=scratch/foldx_skempi_full/bycomplex_all/splits_cath_foldx_aug
export ES_FOLDXDEC_DIR=scratch/foldx_skempi_full/bycomplex_all/splits_cath_foldxdec_aug
export ES_RESULTS_DIR=scratch/results/full_skempi_bycomplex_all_aug
```

The parent already exports `ES_SCALAR_CONFIG` / `ES_MLP_CONFIG` / `ES_NUM_FOLDS=3` /
`ES_BASENAME=skempi_all`, so those four overrides are the whole file.

## 4. Train

```bash
ES_CONFIG=experiments/full_skempi_seqonly/config_skempi_full_bycomplex_all_aug.sh \
  ARMS="foldx foldx_scalar" MAXPAR=1 \
  bash experiments/retrain_split/run_bycomplex.sh esmc6b
```

Expect roughly the cost of the unaugmented `esmc6b` FoldX arms on this tier plus the ~2× train
rows. Price it off a completed arm rather than the first fold — arm-to-arm drift within a tier has
been the dominant ETA error on both boxes.

> ⚠️ **Pull the current `run_bycomplex.sh` before running.** The copy in the 08-04 snapshot predates
> the `.FAILED` marker and `rc` check, and without them a crashed fold logs `DONE` with an empty
> `test_pcc` and the driver exits 0. Three empty directories under a task reporting Success is
> exactly how `esmc600m_base` was lost, and four other result files were found in
> the same state.

## 5. Return

Send back `scratch/results/full_skempi_bycomplex_all_aug/esmc6b_{foldx,foldx_scalar}/` plus the two
new split dirs and the aug fasta, so the Mac can rebuild rather than trust the transfer. The scorer
here is already wired for the tier — `score_sp_all.py bycomplex_all_aug` reads
`scratch/results/full_skempi_bycomplex_all_aug` and writes `results_bycomplex_all_aug.csv`, which
`results_matrix.py --ps` picks up as `fullSK bycomplex-ALL+aug`. Until the results land it scores as
`(none found)`, which is expected rather than an error.

## Optional — the base arm, which is what attributes the effect

Every full-SKEMPI aug arm is a FoldX arm, so "augmentation stops helping at this scale" and
"augmentation fights the FoldX columns" currently fit the evidence equally well. The base arm
separates them, and it is 3 more runs.

The augmented four-column split needs no builder: the 4-column base split **is** the 5-column FoldX
split with its last column removed — verified byte-identical across train, val and test on every
fold of both `clustered_all` and `cath`.

```bash
A=scratch/foldx_skempi_full/bycomplex_all/splits_cath_foldx_aug
O=scratch/splits_skempi_full_bycomplex_all_aug
for k in 0 1 2; do mkdir -p $O/fold_$k
  for f in train val test; do cut -f1-4 $A/fold_$k/skempi_all_$f.tsv > $O/fold_$k/skempi_all_$f.tsv; done
done
```

Then add one line to the config and run `ARMS=base`:

```sh
export ES_SPLIT_DIR=scratch/splits_skempi_full_bycomplex_all_aug
```

**Without that line `ARMS=base` is actively misleading**: the aug configs inherit `ES_SPLIT_DIR`
from the unaugmented parent, so it would train the plain base into the `_aug` results directory and
exit 0 — a wrong answer that looks like a successful run.
