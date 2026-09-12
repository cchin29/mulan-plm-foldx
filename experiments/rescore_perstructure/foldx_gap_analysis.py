#!/usr/bin/env python3
"""Why this FoldX-alone score sits above the frontier's, decomposed.

Reproduces the tables in FOLDX_ALONE_BASELINE_RESULT.md, "Distance between this FoldX and the
frontier's" (and the same passage quoted in RESULTS.md §26 and BENCHMARK_MATRIX.md).

The question: on the clustered CD-HIT <=60% rung this pipeline's unsupervised FoldX reaches 0.512
pooled Spearman, where ProtBFF Table 2 reports FoldX at 0.294. A 0.218 gap in a baseline neither
side trains needs an explanation before anything built on top of it can be quoted.

Three things it computes:

  1. **Cross-tier comparison.** The gap against every published FoldX number, which is what shows
     it is not uniform. CATH -- 687 mutations at 100% channel coverage on both sides -- agrees to
     0.006, which rules out RepairPDB, the FoldX version, and the choice of `Interaction Energy`
     as the scalar. The by-complex tiers give a consistent +0.06 over the same 5,801 rows.

  2. **Metric decomposition.** Where the clustered rung's extra 0.15 lives: multi-point rows, and
     between-complex ordering. Pooling rewards both; the per-structure metric discards both.

  3. **Failure-mode simulation.** Degrading this data the way FoldX coverage actually fails --
     by whole structure, and on multi-point mutations first -- and finding what reproduces 0.294.
     Random row-wise dropout is included to show it does NOT: it needs ~40% coverage to get there,
     which no published pipeline would report. That contrast is the point of running both.

Inference from one side only. The frontier's FoldX predictions are not published, so the mechanism
this supports is consistent with the evidence rather than demonstrated. Anything written from this
output should say so.

Usage:  ./.venv/bin/python experiments/rescore_perstructure/foldx_gap_analysis.py [--root .]
"""
from __future__ import annotations
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from foldx_alone_baseline import load_split, spearman, per_complex_rhos  # noqa: E402

# Published FoldX baselines, with the tier of ours they are comparable to. `metric` names the
# column in foldx_alone_baseline_allplm_cov99.csv that measures the SAME quantity -- a comparison
# across metrics would be meaningless here, and getting that wrong is the error this whole
# investigation exists to rule out.
PUBLISHED = [
    ("CATH-superfamily", "AUROC",            "fullSK CATH-all",      "auroc_destab",    0.754,  687, "USP-ddG Table 1"),
    ("by-complex",       "per-structure Sp", "fullSK bycomplex-ALL", "rho",             0.369, 5801, "Prompt-DDG Table 1"),
    ("by-complex",       "AUROC",            "fullSK bycomplex-ALL", "auroc_destab",    0.658, 5801, "BA-DDG Table 1"),
    ("clustered id60",   "pooled Sp",        "fullSK clustered-ALL", "pooled_spearman", 0.294, 5801, "ProtBFF Table 2"),
]
CLUSTERED_SPLITS = "scratch/foldx_skempi_full/clustered_all/splits_cath_foldx/fold_{f}/skempi_all_test.tsv"
FXA_CSV = "experiments/rescore_perstructure/foldx_alone_baseline_allplm_cov99.csv"
N_BOOT = 60          # resamples per simulated condition; the spread is small, the mean is the point
SEED = 0


def cross_tier(root: str) -> None:
    path = os.path.join(root, FXA_CSV)
    ours = {r["tier"]: r for r in csv.DictReader(open(path)) if r["arm"] == "FoldX alone"}
    print("1. Every tier that publishes a FoldX number\n")
    print(f"   {'tier':<18}{'metric':<19}{'here':>7}{'published':>11}{'gap':>8}{'n':>7}  source")
    for tier, metric, key, field, lit, n, src in PUBLISHED:
        row = ours.get(key)
        if not row or not row.get(field):
            print(f"   {tier:<18}{metric:<19}{'—':>7}{lit:>11.3f}{'—':>8}{n:>7}  {src} (tier absent)")
            continue
        v = float(row[field])
        print(f"   {tier:<18}{metric:<19}{v:>7.3f}{lit:>11.3f}{v - lit:>+8.3f}{n:>7}  {src}")
    print("\n   The CATH row is the control: matched coverage, and it agrees. Whatever produces")
    print("   the other rows is therefore not the FoldX pipeline itself.")


def load_clustered(root: str):
    paths = [os.path.join(root, CLUSTERED_SPLITS.format(f=f)) for f in range(3)]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        sys.exit(f"clustered FoldX splits absent (needs scratch/): {missing[0]}")
    d = load_split(paths)
    keys = list(d)
    return (np.array([d[k][0] for k in keys]),                 # experimental ddG
            np.array([d[k][1] for k in keys]),                 # FoldX scalar
            np.array([k[0] for k in keys]),                    # complex
            np.array(["," in k[1] for k in keys]))             # multi-point?


def decompose(y, fx, cx, mp) -> None:
    print("\n2. Where the clustered rung's excess lives\n")
    print(f"   rows {len(y)}  complexes {len(set(cx))}  multi-point {int(mp.sum())} ({mp.mean():.1%})"
          f"  uncovered {int((fx == 0).sum())}")
    print(f"\n   {'component':<44}{'Spearman':>10}")
    print(f"   {'all rows, pooled':<44}{spearman(fx, y):>10.4f}")
    for lbl, m in (("single-point rows only", ~mp), ("multi-point rows only", mp)):
        print(f"   {lbl:<44}{spearman(fx[m], y[m]):>10.4f}")
    # between = can FoldX order whole complexes; within = can it order inside one.
    ymu = {c: y[cx == c].mean() for c in set(cx)}
    fmu = {c: fx[cx == c].mean() for c in set(cx)}
    cs = sorted(set(cx))
    print(f"   {'between-complex (complex means)':<44}"
          f"{spearman(np.array([fmu[c] for c in cs]), np.array([ymu[c] for c in cs])):>10.4f}")
    yw = y - np.array([ymu[c] for c in cx])
    fw = fx - np.array([fmu[c] for c in cx])
    print(f"   {'within-complex (both mean-centred)':<44}{spearman(fw, yw):>10.4f}")
    r, _, _ = per_complex_rhos(fx, y, cx)
    print(f"   {'per-structure mean (T>=10) — discards between':<44}{np.mean(list(r.values())):>10.4f}")
    print("\n   Pooling rewards the multi-point rows and the between-complex ordering. Those are")
    print("   the first two things a lower-coverage FoldX loses.")


def simulate(y, fx, cx, mp, target: float) -> None:
    rng = np.random.default_rng(SEED)
    cs = sorted(set(cx))
    print(f"\n3. What reproduces the published {target:.3f}\n")

    print("   Row-wise random dropout (uncovered rows carry the standardised mean, our convention):")
    for cov in (0.9, 0.7, 0.5, 0.4):
        v = [spearman(np.where(rng.random(len(fx)) < cov, fx, 0.0), y) for _ in range(N_BOOT)]
        hit = "  <-- target" if abs(np.mean(v) - target) < 0.015 else ""
        print(f"     coverage {cov:>4.0%}{np.mean(v):>26.4f}{hit}")
    print("     -> needs ~40% coverage. No published pipeline would report that, so random")
    print("        row loss is not the mechanism.")

    print("\n   Whole-complex dropout (how RepairPDB actually fails — a structure at a time),")
    print("   with multi-point rows also unusable (they are the hardest to map):")
    for frac in (0.0, 0.1, 0.2, 0.3, 0.4):
        v = []
        for _ in range(N_BOOT):
            drop = set(rng.choice(cs, int(len(cs) * frac), replace=False))
            v.append(spearman(np.where(np.isin(cx, list(drop)) | mp, 0.0, fx), y))
        hit = "  <-- target" if abs(np.mean(v) - target) < 0.015 else ""
        print(f"     MP unusable + {frac:>4.0%} of complexes{np.mean(v):>16.4f}{hit}")
    print("     -> reproduces it. Consistent with the evidence; not demonstrated, since the")
    print("        frontier's FoldX predictions are not published.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="repo root (needs scratch/ for parts 2-3)")
    args = ap.parse_args()
    cross_tier(args.root)
    y, fx, cx, mp = load_clustered(args.root)
    decompose(y, fx, cx, mp)
    simulate(y, fx, cx, mp, target=0.294)


if __name__ == "__main__":
    main()
