# Ankh provenance audit — which "ankh" runs are v1 vs Ankh3

_Written 2026-07-16 (branch `a1-interface-xattn`). Audit of every Ankh run in the work area to
separate the **paper-reference Ankh v1 Large** from **Ankh3**, after the A1 grid was found running on
Ankh3-large while labelled "Ankh-large"._

## Why this matters

- **Ankh v1 Large (`ElnaggarLab/ankh-large`) is the MuLAN-paper reference model.** It must be carried
  as a reference arm through **every** experiment so results are comparable to the paper.
- **Ankh3 (`ankh3-large` / `ankh3-xl`) was deprioritised** — it underperformed in the original
  50 ep / patience-10 CV10 screen. It is a research variant, *not* the reference.
- The two are easy to confuse: **both `ankh-large` and `ankh3-large` are 1536-dim**, so an A1
  cross-attention head built for one loads and trains on the other with no error — a silent swap.

## The tag → model mapping (ground truth: `mulan/constants.py`)

| tag | HF model | which Ankh | embeddings dir | plm width |
|---|---|---|---|---|
| `ankh` | `ElnaggarLab/ankh-large` | **v1 Large (reference)** | `scratch/embeddings` | 1536 |
| `ankh_base` | `ElnaggarLab/ankh-base` | v1 Base | — | 768 |
| `ankh3_large` | `ElnaggarLab/ankh3-large` | Ankh3 Large | `embeddings_ankh3_large_nlu` | 1536 |
| `ankh3_xl` | `ElnaggarLab/ankh3-xl` | Ankh3 XL | `embeddings_ankh3_xl_nlu` | 2560 |

Runs do **not** record `plm_model_name` / `embeddings_dir` in their artifacts (`all_results.json` has
them `null`), so provenance below is traced through the **producing driver** (which sets
`--plm_model_name` + `--embeddings_dir`), not the run output.

## Inventory

### ✅ Genuine Ankh v1 Large — correctly labelled (`plm=ankh`, `scratch/embeddings`)

| result dir(s) | producer | verified by |
|---|---|---|
| `cv10_ankh_converge`, `cv10_ankh_upstream_split`, `cv10_ankh_upstream_converge` | `cv10_ankh_repro_mac.sh` | `EMB=scratch/embeddings`, `--plm_model_name ankh` |
| `cv10_ankh_mps_parity`, `cv10_ankh_cpu_parity` | `cv10_ankh_repro_mac.sh` (parity cells) | same |
| `cv10_ankh_reconciled`, `cv10_ankh_seed43/44/45`, `cv10_ankh` | `cv10_ankh_seedband.sh` | `--embeddings_dir scratch/embeddings --plm_model_name ankh` |
| `run1_s1102_ankh`, `run4a_ankh_aug`, `cv10_aug_ankh`, `cv10_aug_prewarm_ankh` | `batch_driver2.sh`, `cv10_aug_driver.sh` | `ankh) echo scratch/embeddings` |
| `bench_S1131_ankh`, `bench_S2003_ankh`, `bench_S4169_ankh` | `bench_cv_driver.sh` | `emb_dir ankh) echo scratch/embeddings` |
| `embedding_sweep_balanced/{ankh, ankh_aug, ankh_foldxmlp}` | sweep drivers, tag `ankh` | `models.tsv: ankh → ankh → scratch/embeddings` |

**These are the reference arms.** The committed `experiments/rescore_perstructure/results.csv` uses the
last row (`ankh_base = cv10_ankh_converge`, `ankh_aug`, `ankh_foldxmlp`) — **all Ankh v1, clean.**

### ✅ Genuine Ankh3 — correctly labelled

`cv10_ankh3_large_nlu / _aug_nlu / _s2s`, `cv10_ankh3_xl_nlu / _aug_nlu`,
`embedding_sweep_balanced/{ankh3_large, ankh3_large_foldxmlp, ankh3_xl}`, `a1_ankh3_large`,
`emb_ankh3_{large,xl}_bench` — via `cv10_ankh3_driver.sh` / `ankh3_prewarm.sh` (`embeddings_ankh3_*`).

### ⚠️ MISLABELLED — scaffolding says "ankh"/"Ankh-large" but the model is `ankh3_large`

| artifact | says | actually runs | fix |
|---|---|---|---|
| `config_a1_ankh.sh` | "ankh" (filename) | `ankh3_large` (`lightatt_a1_ankh3_large.json`) | rename → `config_a1_ankh3_large.sh` |
| `config_c1_ankh.sh` | "ankh" (filename) | `ankh3_large` | rename → `config_c1_ankh3_large.sh` |
| `run_a1_grid.sh` / `run_c1_grid.sh` | — | `TAG=${TAG:-ankh3_large}` default | make CFG overridable; add v1 arm |
| metrics dirs `results_a1_ankh`, `results_c1_ankh` | "ankh" | ankh3_large metrics | TAG-parameterise → `results_{a1,c1}_${TAG}` |
| `docs/history/PLAN_A1_INTERFACE_XATTN.md` prose "Ankh-large pilot (`cv10_ankh_converge`)" | v1 | ran on ankh3_large | note the base swap |
| queue: `#60 a1_xattn_grid` (running), `#61 foldx_mlp_bal_ankh3_large`, `#54 c1_ankh_grid` | "ankh" | ankh3_large | see §Impact |

**Note:** the *leaf result dirs* (`a1_ankh3_large`, `c1_ankh3_large`) are accurately named — no Ankh3 run
writes into a bare-`ankh` results dir, so there is **no silently corrupted data**. The mislabelling is
at the **config / metrics-dir / plan / queue-label** level, which is exactly where a reader sees
"ankh" and assumes the v1 reference.

## Impact

1. **The A1 (interface-xattn) and C1 (tail-loss) grids never carried the v1 reference.** They were built
   solely on the deprioritised `ankh3_large`. Their scientific value against the paper suite is limited
   until a v1 arm exists.
2. **Running job `#60` and queued `#54`** are ankh3-large. Whether to keep them depends on whether the
   want an ankh3 datapoint at all; the *reference* A1/C1 result still has to be produced on v1.
3. **Queued `#61` (`foldx_mlp_bal_ankh3_large`)** produces an ankh3-large foldx-MLP that is **not** the
   `ankh_foldxmlp` (v1) already in the rescore — do not treat them as the same arm.
4. **The rescore is unaffected** — its three ankh arms are all v1.

## The fix (✅ APPLIED 2026-07-16 via `experiments/fix_ankh_labels.sh`, after the queue was cleared)

v1 reference arms added (additive):
- `models/config/lightatt_a1_ankh_large.json` — v1 A1 head (1536, accurate name; identical geometry to
  the ankh3 one).
- `experiments/embedding_sweep/config_a1_ankh_large.sh`, `config_c1_ankh_large.sh` — v1 arms
  (tag `ankh`, accurate metrics/result roots).

Corrected renames applied (`fix_ankh_labels.sh`, run with `CONFIRM=1` once no task was Running/Queued):
- `git mv config_a1_ankh.sh → config_a1_ankh3_large.sh`, `config_c1_ankh.sh → config_c1_ankh3_large.sh`.
- Their internal `ES_METRICS_DIR` / `ES_RESULTS_DIR` retargeted `*_ankh → *_ankh3_large`.
- `run_{a1,c1}_grid.sh` now point at the renamed configs via `CFG=${CFG:-…_ankh3_large.sh}` (overridable),
  so the v1 configs above drive the reference arm with a `CFG=` override.
- Stale in-comment references to the old filenames swept from the live grid/config files (historical
  commit-log / changelog mentions left intact).

**Run the v1 reference arms** (after the fix, or now with a manual CFG override):
```bash
TAG=ankh CFG=experiments/embedding_sweep/config_a1_ankh_large.sh MAXPAR=2 \
  bash experiments/embedding_sweep/run_a1_grid.sh      # A1 grid on Ankh v1 → scratch/results/a1_ankh
TAG=ankh CFG=experiments/embedding_sweep/config_c1_ankh_large.sh MAXPAR=2 \
  bash experiments/embedding_sweep/run_c1_grid.sh      # C1 grid on Ankh v1 → scratch/results/c1_ankh
```
