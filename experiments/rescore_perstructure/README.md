# FoldX-alone baseline — which file is current

Two generations of the FoldX-alone baseline live in this directory. They differ because the
FoldX join defect was fixed on 2026-07-30, moving single-point coverage from 87.8% to 99.2% and
the combined SP+MP figure to 99.1%. The baseline itself moved with it, so the two generations are
not interchangeable and their numbers must never be mixed in one table or figure.

## Current

| File | What it holds |
|---|---|
| `foldx_alone_baseline_allplm_cov99.csv` | All ten backbones at 99.x% coverage. **The only current data file here.** |
| `FOLDX_ALONE_BASELINE_RESULT.md` | The write-up. Its v3 section is current; everything below v3 is marked superseded in place. |

FoldX alone on the CATH-superfamily hold-out reads **0.383** in the current file. Nineteen of the
78 paired contrasts favour a MuLAN arm over it.

## Pre-fix lineage — retained, not current

| File | Why it is kept |
|---|---|
| `foldx_alone_baseline.csv` | Ankh-large only, 87.8% coverage |
| `foldx_alone_baseline_allplm.csv` | All ten backbones, 87.8% coverage |
| `foldx_alone_baseline.html` | Backbone-switchable figure built from the above |
| `foldx_alone_baseline.png` | Static render of the same |

These four carry **0.363** for FoldX alone on the single-point tiers and **0.468** for the ESM-C 6B
CATH scalar arm. Both are retracted. The corrected values are 0.418 and 0.438. The HTML and PNG
also state a win count of one significant positive contrast, which was Ankh-large's; at 99.2%
coverage Ankh-large has none.

They are retained because the coverage correction is part of the record and the lineage is cited
in `docs/RESULTS.md` §22. They do not carry the `__cov87` / `__cov_pre` archive suffix that marks
other superseded artifacts, which is why this file exists.

## Other files

`results.csv` / `results.png` are the per-structure rescoring output, unaffected by the coverage
fix. `bootstrap_ci.py`, `crosscheck_reference.py`, `foldx_gap_analysis.py`, `plot_results.py` and
`rescore.py` are the scripts; `SUMMARY.md` and `REVIEW_v1_gaps_and_assumptions.md` are the
accompanying notes.
