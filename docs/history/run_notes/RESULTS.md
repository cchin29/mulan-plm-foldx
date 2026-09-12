# MuLAN paper reproduction — findings (2026-06-22)

Goal: reproduce the MuLAN paper's (Lombardi & Carbone, bioRxiv 2024.08.24.609515)
S1102 benchmark numbers for the Ankh-large model, using only locally-runnable
code (no ESM2-3B download).

Paper's reference number (Table 1, mutation-based 10-fold CV):
**MuLAN-Ankh-large on S1102 → PCC = 0.868, RMSE = 1.185 kcal/mol**

All result artifacts for this round are in `results/run1_s1102_ankh/`.

## Data provenance

- `examples/S1102.tsv` (in repo) is the real S1102 benchmark: 1102 single-point
  mutations across 111 PDB complexes, ΔΔG in kcal/mol, sourced from SKEMPI v1
  via Geng et al. 2019.
- The repo does **not** ship wild-type sequences for these complexes. Naively
  fetching whole-chain FASTA from RCSB does not match SKEMPI's mutation
  numbering (confirmed mismatch on PDB 1A22 at position 171: RCSB SEQRES has
  extra engineered/disordered residues SKEMPI's numbering excludes).
- Fix: downloaded SKEMPI 2.0's own data from life.bsc.es/pid/skempi2/database/download/:
  - `skempi_v2.csv` — gives the actual partner-chain letters per complex (e.g.
    `1A22_A_B`, `1CSE_E_I` — chain letters are *not* always literally A/B)
  - `SKEMPI2_PDBs.tgz` — SKEMPI's cleaned PDB files, numbered consistently with
    mutation labels
  - `build_wt_fasta.py` extracts per-residue sequences from these PDB ATOM
    records (indexed by `auth_seq_id`, gap-filled with `X`), producing
    `wt_sequences.fasta`. Validated exact-match against the known-correct
    sequences embedded in `examples/sample_mut.txt` for both chains of 1A22.
  - 110/111 complexes resolved this way. `2I9B` is excluded — it's SKEMPI-v1-only
    and absent from the v2 structure bundle. Filtered table:
    `S1102_filtered.tsv` (1100 mutations).

## Bug found and fixed

`scripts/train.py` called `logging.get_logger()` (stdlib `logging` has no such
method) instead of `hf_logging.get_logger()` (the transformers alias) —
`mulan-train` was completely non-functional before this fix.
Committed locally: `043ac4a`.

## Experiment 1 — direct inference with the pretrained checkpoint

Ran the shipped `models/pretrained/mulan_ankh.ckpt` directly over all 1100
mutations in `S1102_filtered.tsv` (script: `eval_s1102.py`).

**Result: PCC = 0.967, RMSE = 0.723 kcal/mol** (N=1100)

**Not a valid reproduction.** This beats the paper's own CV number, which is
the tell: the released checkpoint was almost certainly trained on (some or
all of) S1102 itself, so this measures memorization, not generalization.
Useful only as a pipeline sanity check (correct sequences/embeddings/scoring
would not produce an improvement like this if something were broken).

Artifacts: `eval_s1102.log`, `s1102_predictions.tsv`.

## Experiment 2 — from-scratch training on a held-out split

To get a leakage-free number, trained a **fresh, randomly-initialized**
LightAttModel (`models/config/lightatt_default_config.json`, not the
pretrained checkpoint) on a single mutation-based 70/15/15 split of
`S1102_filtered.tsv` (`mulan.data.split_data`, seed 42 → 769/157/174 train/val/test,
saved in `splits/`).

Hyperparameters matched the paper: AdamW, batch size 32, LR 5e-4,
`ReduceLROnPlateau` (factor 0.5, patience 5), `early_stopping_patience=10`,
up to 50 epochs (ran the full 50, never triggered early stopping).
Reused the ~1444 Ankh embeddings already cached from Experiment 1 (no new PLM
calls). Training took ~48 min, CPU-only.

**Result on held-out test set (N=174): PCC = 0.757, RMSE = 1.531 kcal/mol**

This is the fairer comparison point, but it's a **single split**, not the
paper's 10-fold average, so variance is much higher and it trained on fewer
examples per run (769 vs. the paper's ~990/fold). Both effects plausibly
explain why this run underperforms the paper's 0.868/1.185.

Artifacts: `training_run.log`, `training_run/` (contains `model.ckpt`,
`all_results.json`, `test_predictions.tsv`, HF trainer checkpoints).

## Summary table

| Run | PCC | RMSE (kcal/mol) | Valid comparison to paper? |
|---|---|---|---|
| Paper (10-fold CV avg.) | 0.868 | 1.185 | — (reference) |
| Exp. 1: pretrained ckpt, direct inference | 0.967 | 0.723 | No — likely leakage |
| Exp. 2: from-scratch, single holdout split | 0.757 | 1.531 | Partially — right methodology, higher variance than a 10-fold average |

## Possible next step

Run the same from-scratch training 10x over different folds and average, to
get a number directly comparable to the paper's reported 0.868/1.185. Each
run takes ~48 min CPU-only → ~8 hours for all 10 folds.

## Reusable infrastructure (kept at `scratch/` top level, not moved)

- `skempi_v2.csv`, `skempi2/` — raw SKEMPI 2.0 download (~30MB + ~30MB PDBs)
- `pdb_ids.txt` — the 111 PDB IDs referenced in S1102
- `build_wt_fasta.py`, `eval_s1102.py` — scripts; re-running them regenerates
  `wt_sequences.fasta` / `S1102_filtered.tsv` fresh at the `scratch/` top level
  without touching the preserved copies in `results/run1_s1102_ankh/`
- `embeddings/` — cached Ankh-large embeddings, reused across experiments
