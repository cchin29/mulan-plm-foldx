# MuLAN paper comparison: fairness, split provenance, leakage

Answers one question: *"Are the S1131/S2003/S4169 splits identical to the
original MuLAN paper? Verify no data leakage before claiming a fair comparison."* Two parts —
(1) is our comparison to the paper like-for-like, and (2) are our own splits leakage-free.

## 1. The comparison is NOT like-for-like — it's from-scratch vs pretrained (document the gap)

The MuLAN paper's headline (Lombardi & Carbone; `RESULTS.md` records **MuLAN-Ankh-large S1102
10-fold PCC 0.868 / RMSE 1.185**, Table 1) uses the **released pretrained `mulan-ankh`
checkpoint**. Every run in this fork trains **from scratch** (randomly-initialized
`LightAttModel`, PLM used only as a frozen offline feature extractor). So the paper number is a
**ceiling, not a like-for-like baseline** — our best from-scratch S1102 results (ESM C 6B 0.859;
+aug 0.878) straddle it, but the comparison is architecture-vs-checkpoint, not split-vs-split.

Two further mismatches with the paper's table:
- The paper reports **S1400** for the largest set; we sweep **S4169** (different set entirely).
- Our S1131/S2003/S4169 are **single-chain-per-partner subsets** (MuLAN's one-seq-per-partner
  design), not the verbatim published sizes — see §2.

**Decision: document the gap; do NOT attempt a released-checkpoint
repro.** A like-for-like reproduction would require loading the pretrained `mulan-ankh` weights
and re-running the paper's exact splits — a substantial lift that competes with the tool
work, for a comparison whose conclusion ("pretraining is a ceiling above
from-scratch") is already clear. Instead we state the from-scratch framing explicitly wherever
the 0.868 line appears (already done as dashed-grey target lines in the ddg_variability plots).

## 2. Split provenance — subsets, nested, sizes confirmed computationally

Full provenance is in `experiments/BENCHMARK_DATASETS.md`; `audit_leakage.py` re-derives the key
facts from the data itself (not by citation):

| set | rows | unique keys | complexes | nominal name-size | subset frac |
|---|---|---|---|---|---|
| S1131 | 1127 | 1127 | 110 | 1131 | 1.00 (essentially full) |
| S4169 | 2497 | 2497 | 211 | 4169 | 0.60 |
| S2003 | 1124 | 1122 | 174 | 2003 | 0.56 |

- **Nesting confirmed:** all 1127 S1131 keys and all 1122 S2003 keys are contained in S4169 —
  S4169 is the superset, S1131 ⊂ S4169, S2003 = its non-alanine half. Per-set PCCs are therefore
  **correlated, not independent** — report them as one nested family, not three data points.
- **Subset caveat:** the single-chain restriction drops ~40% of S4169 and ~44% of S2003, so those
  numbers are **not comparable to external S4169/S2003 leaderboards**. The *per-PLM contrast* on
  our sets is exact (identical splits, only the embedder changes) — that is the valid comparison.

## 3. Leakage audit — S1131/S4169 clean; S2003 a 0.18% disclosed artifact

`audit_leakage.py` checks every CV fold (and every augmentation fold) for exact-key and
reverse-mutation train/test contamination. Full report: `results/LEAKAGE_AUDIT.md`.

- **S1131 and S4169: CLEAN.** Across all 10 folds each: zero exact leakage, zero reverse-mutation
  leakage, train∪val∪test = the full set, and reverse-mutation **augmentation** (7k–14k aug rows)
  carries **no** held-out label into training. (Augmentation keys the reverse mutation to the
  *mutant* structure, so it can't collide with a forward held-out key — verified, not assumed.)
- **S2003: MINOR ARTIFACT (~0.18%).** The only contamination is **2 duplicated mutations**, both
  in complex **3SE4** (`PA27L`, `KA134R`) — repeat SKEMPI 2.0 measurements of the same mutation
  with *divergent* ΔΔG (Δ ≈ 0.82 / 1.08 kcal/mol) that were never averaged, so the random CV
  splitter can place the two copies on opposite sides of a fold. 2 of 1122 unique keys → negligible
  for the reported 0.852 PCC, but disclosed for transparency. S2003 was **derived locally** (no
  canonical published file), which is why it alone carries this; the GeoPPI sets are pre-deduplicated.

**Optional fix (not applied):** averaging the 2 divergent 3SE4 duplicates would make S2003
verbatim-clean; the effect on the PCC is far below fold noise (±0.04–0.07), so we disclose rather
than re-run the completed sweep. If S2003 is ever regenerated, dedup-by-averaging at build time.

## Bottom line

Our SKEMPI benchmarks are **leakage-free for S1131/S4169** and carry a **negligible, characterized
0.18% duplicate-measurement artifact in S2003**. The comparison to the paper's 0.868 is
**from-scratch vs pretrained** (and S4169 ≠ the paper's S1400, subsets ≠ published sizes), so it
is framed as a ceiling, not a matched baseline — the valid, exact comparison in this fork is the
**per-PLM contrast** on identical splits (ESM C 6B > Ankh ≈ AIDO > ESM2-3B > ProstT5).

Reproduce: `.venv-structctx/bin/python experiments/paper_fairness/audit_leakage.py`
