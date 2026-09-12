# Embedding sweep — paper-faithful S1102 reproduction + model comparison

Tracked, reproducible area for (1) **reproducing the MuLAN paper's S1102 result** (Ankh-large
PCC 0.868, Table 1) and (2) running the **embedding-model sweep on that exact paper-faithful
setup**. Kept separate from the pre-2026-07-09 runs so those stay reproducible; run outputs land
in gitignored `scratch/`, and only the small splits + aggregated metrics are committed.

## Layout

| path | tracked | what |
|---|---|---|
| `config.sh` | ✅ | single source of truth — CV setup + all paths (sourced by bash, parsed by python) |
| `models.tsv` | ✅ | model registry: `tag → plm_model_name → embeddings_dir` (read by bash + python) |
| `gen_splits.py` | ✅ | (re)generate / `--verify` the paper folds via `data.split_data(split_method="random")` |
| `run_sweep.sh` | ✅ | driver: trains each model × fold; heavy output → `scratch/` |
| `aggregate.py` | ✅ | scratch results → committed `results/<tag>.json` + `results/summary.md` |
| `data/` | ✅ | `S1102_filtered.tsv` (1100), `wt_sequences.fasta` (220), `PROVENANCE.md` |
| `splits/paper_seed42/fold_{0..9}/` | ✅ | the exact partition (small text; pins it against RNG drift) |
| `results/` | ✅ | aggregated metrics only (per-model JSON + summary.md) |
| `scratch/results/embedding_sweep/<tag>/fold_k/` | ❌ | checkpoints, logs, per-fold predictions (heavy/regenerable) |
| `scratch/embeddings*`, `scratch/emb_saprot` | ❌ | per-model embedding caches (reused as-is) |

## Paper-faithful setup (see `config.sh`)

- **Splitter**: upstream `rng.integers` folds, seed 42 (`split_method=random` — the paper method;
  the fork's balanced default is preserved for old runs). Unbalanced test sizes 88–125.
- **Budget**: "train until convergence" — `num_epochs=300`, `early_stopping_patience=30` (no
  50-epoch cap); LR halves on val-loss plateau patience 5 (built into the trainer, = paper).
- **Head**: `lightatt_default_config.json` (`add_scores=false`) — the paper's Table 1 MSE head.
- **Data**: `S1102_filtered` (1100; 2I9B's 2 muts absent locally — see `data/PROVENANCE.md`).

## Run

All commands from the repo root; uses `./.venv` and Mac/MPS.

```bash
python experiments/embedding_sweep/gen_splits.py            # (re)generate the folds
python experiments/embedding_sweep/gen_splits.py --verify   # committed splits == regeneration

# train one or more models (reuses each model's embedding cache)
bash experiments/embedding_sweep/run_sweep.sh ankh
MAXPAR=2 bash experiments/embedding_sweep/run_sweep.sh ankh prostt5 esm2 esmc6b aido \
                                                       ankh3_large ankh3_xl saprot

python experiments/embedding_sweep/aggregate.py             # refresh results/summary.md
```

Models: `ankh prostt5 esm2 esmc6b aido ankh3_large ankh3_xl saprot` (edit `models.tsv` to add
more, e.g. `saprot_1.3b` → `scratch/emb_saprot13b`). Override budget with `NUM_EPOCHS=`/`PATIENCE=`
env not needed — edit `config.sh` (the tracked source of truth).

## Reproduction gate

Ankh anchors the reproduction: it should approach the paper's 0.868 under this setup before the
other models are read as a fair comparison. The C1×C3 attribution (splitter vs budget) is measured
by the investigation in `scratch/results/cv10_ankh_{mps_parity,upstream_split,converge,upstream_converge}`;
Once Ankh reproduces, run the full sweep here.
