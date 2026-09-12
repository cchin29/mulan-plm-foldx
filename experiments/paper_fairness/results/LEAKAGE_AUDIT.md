# L4 — SKEMPI benchmark leakage + provenance audit

## Set integrity & subset sizes

| set | rows | unique keys | dups | complexes | nominal name-size | subset frac |
|---|---|---|---|---|---|---|
| S1131 | 1127 | 1127 | 0 | 110 | 1131 | 1.00 |
| S4169 | 2497 | 2497 | 0 | 211 | 4169 | 0.60 |
| S2003 | 1124 | 1122 | 2 | 174 | 2003 | 0.56 |

## Nesting (vs S4169 superset)

- **S1131_in_S4169**: 1127/1127 keys contained (✅ full subset).
- **S2003_in_S4169**: 1122/1122 keys contained (✅ full subset).

## CV-fold leakage (exact + reverse); partition integrity

- **S1131** (10 folds): exact train/val/test leakage = **0**, reverse-mutation-into-train = **0**, every fold's train∪val∪test = full set: **True**.
- **S4169** (10 folds): exact train/val/test leakage = **0**, reverse-mutation-into-train = **0**, every fold's train∪val∪test = full set: **True**.
- **S2003** (10 folds): exact train/val/test leakage = **7**, reverse-mutation-into-train = **0**, every fold's train∪val∪test = full set: **True**.

## Augmentation leakage (reverse-mut aug must not carry a held-out label)

- **S1131** (10 aug folds): reverse-aug rows = 7128, held-out (fwd/rev) labels leaked into aug_train = **0**.
- **S4169** (10 aug folds): reverse-aug rows = 13896, held-out (fwd/rev) labels leaked into aug_train = **0**.
- **S2003** (10 aug folds): reverse-aug rows = 6936, held-out (fwd/rev) labels leaked into aug_train = **6**.

## Verdict (per set)

- **S1131: CLEAN** — folds partition the set with zero exact/reverse train-test contamination; augmentation leaks nothing.
- **S4169: CLEAN** — folds partition the set with zero exact/reverse train-test contamination; augmentation leaks nothing.
- **S2003: MINOR ARTIFACT** — the only contamination traces to **2 duplicated mutation(s)** (1122 unique of 1124 rows = 0.18%): 3SE4 PA27L (Δ=0.8209), 3SE4 KA134R (Δ=1.0753). These are repeat SKEMPI measurements of the same mutation with *divergent* ΔΔG (never averaged), which the random CV splitter can place on both sides. Negligible for the reported PCC, but not verbatim-clean.

**Bottom line:** S1131 and S4169 (GeoPPI, deduplicated) are leakage-free; S2003 (derived locally from SKEMPI 2.0) carries 2 duplicate-measurement mutations in one complex — a ~0.18% artifact to disclose, not systematic leakage. See `leakage_audit.json` for per-fold detail.
