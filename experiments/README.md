# MuLAN experiments

Reproducible scripts and notes for the S1102 ΔΔG experiments in this fork.
Code and result *summaries* are tracked here; large artifacts (PLM weights,
per-residue embeddings, model checkpoints, feature `.npz`) live in the gitignored
`scratch/` tree and are regenerable from these scripts.

All commands assume the repo root as CWD and the `~/.venvs/mulan` venv active.

## Inputs (built once, in scratch)

- `scratch/results/run1_s1102_ankh/S1102_filtered.tsv` — 1100 single mutations
  across 110 SKEMPI complexes (S1102 minus `2I9B`, which is SKEMPI-v1-only).
- `scratch/results/run1_s1102_ankh/wt_sequences.fasta` — wild-type chain
  sequences from SKEMPI 2.0 cleaned PDBs (mutation-consistent numbering).

See `../docs/history/PLM_COMPARISON_S1102.md` and `scratch/results/run1_s1102_ankh/RESULTS.md`
for how these were produced.

## Scripts

| Script | Purpose |
|---|---|
| `layer_probe.py` | Per-layer linear probe — which PLM hidden layer best encodes ΔΔG. |
| `gen_layer_embeddings.py` | Write MuLAN-format embeddings from chosen layer(s) of a PLM (single layer or concat), for training on non-default layers. |
| `augment.py` | Tier-1 data augmentation: reverse-mutation (antisymmetry) + identity (ΔΔG=0) anchors for a training table. |

### Layer probe

```bash
python experiments/layer_probe.py --plm prostt5 \
    --data  scratch/results/run1_s1102_ankh/S1102_filtered.tsv \
    --fasta scratch/results/run1_s1102_ankh/wt_sequences.fasta \
    --features-out scratch/probe_features_prostt5.npz \
    --md-out experiments/results/probe_prostt5.md

python experiments/layer_probe.py --plm ankh \
    --data  scratch/results/run1_s1102_ankh/S1102_filtered.tsv \
    --fasta scratch/results/run1_s1102_ankh/wt_sequences.fasta \
    --features-out scratch/probe_features_ankh.npz \
    --md-out experiments/results/probe_ankh.md
```

Add `--limit 8` for a fast smoke test. Findings: `LAYER_PROBE.md`.

## Results notes

**All results are consolidated in [`../docs/RESULTS.md`](../docs/RESULTS.md)** (the canonical
entry point — new results go there). The docs below are deep-dive sub-docs:

- `LAYER_PROBE.md` — layer-probe findings and interpretation (ProstT5, Ankh).
- `results/probe_*.md` — raw per-layer tables emitted by `layer_probe.py`.
- `RESULTS_PROSTT5.md`, `RESULTS_CV10.md`, `AUGMENTATION.md`, `HYPERPARAMETERS.md` — see the
  sub-doc index in `../docs/RESULTS.md`.

The campaign plans that produced these results are archived in
[`../docs/history/`](../docs/history/README.md) — `PLAN_AIDO.md`, `PLAN_FULL_SKEMPI.md`,
`PLAN_RETRAIN_BYCOMPLEX.md`, `PLAN_MINT_A2.md`, `PLAN_A1_INTERFACE_XATTN.md`,
`RESCORE_PERSTRUCTURE_SPEC.md` and `SAPROT_ASSESSMENT.md`.
