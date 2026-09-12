#!/usr/bin/env python3
"""Redundancy between the MuLAN base pathway and the FoldX scalar (queue item 2).

Question: how much of what the base arm knows is already in the FoldX number, and what is
the ceiling on any rule that combines the two? No training — everything comes off predictions
already on disk.

Conventions are inherited by import from experiments/rescore_perstructure/foldx_alone_baseline.py
(complex key = chain1.split("_")[0]; per-structure mean over complexes with n >= 10; test folds
pooled before grouping), so the `rho_base` and `rho_fx` columns here must reproduce that script
and results_matrix_ps.csv. They are printed for exactly that reason.

Everything is computed WITHIN complex and on ranks, matching the ppS metric:

  rho_base       Spearman(base, truth)            - reproduces the matrix
  rho_fx         Spearman(foldx, truth)           - FoldX alone, model-independent
  rho_base_fx    Spearman(base, foldx)            - THE REDUNDANCY NUMBER. The base arm never
                                                    sees FoldX, so this is convergence, not leakage.
  part_base      Spearman(base, truth | foldx)    - what the PLM adds that physics does not have
  part_fx        Spearman(foldx, truth | base)    - and the converse
  blend_fx       LOCO global linear fit on [foldx]
  blend_both     LOCO global linear fit on [foldx, base]
  gain           blend_both - blend_fx            - what an optimally weighted blend of exactly
                                                    these two signals returns over the physics alone

The blend is fitted GLOBALLY and evaluated leave-one-complex-out: features and target are
within-complex normalised ranks, the fit uses every row outside the held-out complex, and ppS is
recomputed on the held-out complex. That is the analogue of what training does (one rule for all
complexes), and it is self-validating - a single-feature FoldX fit is affine in rank(foldx) within
each complex, so `blend_fx` must equal `rho_fx` to numerical precision. It is checked below.

A per-complex leave-one-out fit was tried first and rejected: with n as low as 10 each point has
enough leverage over its own held-out prediction to scramble the within-complex ordering, and the
FoldX-only reference came back at 0.217 against a true 0.418. The bias is not shared equally by
the one- and two-feature fits, so differences taken across it are not interpretable.

Coverage: uncovered mutations carry foldx == 0.0 exactly (per-fold standardised mean), which is a
tie block that distorts residualisation. Primary analysis is on COVERED rows; --rows all switches.

Usage:
  python experiments/beyond_foldx/redundancy.py --root . --model esmc6b,ankh,...
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, os.pardir, "rescore_perstructure"))
from foldx_alone_baseline import (  # noqa: E402  - conventions come from there, deliberately
    _rank, pearson, spearman, boot_ci, load_split, load_preds, cplx,
)

T = 10
B = 10000
SEED = 0
ALL_MODELS = ("esmc6b", "ankh", "ankh3_xl", "ankh3_large", "saprot", "saprot13b",
              "prostt5", "esmc600m", "esm2", "aido")


def _resid(y, X):
    """Residual of y after least-squares regression on X (intercept added)."""
    A = np.column_stack([np.ones(len(y))] + [np.asarray(c, float) for c in X])
    beta, *_ = np.linalg.lstsq(A, np.asarray(y, float), rcond=None)
    return y - A @ beta


def partial_spearman(x, y, z):
    """Spearman(x, y | z): correlate the rank residuals of x and y on rank z."""
    rx, ry, rz = _rank(x), _rank(y), _rank(z)
    if np.std(rz) == 0:
        return spearman(x, y)
    return pearson(_resid(rx, [rz]), _resid(ry, [rz]))


def _nrank(a):
    """Within-complex rank scaled to [0, 1], so complexes of different size are comparable."""
    r = _rank(a)
    return r / (len(r) - 1) if len(r) > 1 else r


def loco_blend(per, feats):
    """Leave-one-complex-out ppS for a global linear blend over `feats`.

    `per` maps complex -> dict of within-complex normalised-rank arrays (plus "truth" raw, kept
    for the final Spearman). Returns {complex: rho}.
    """
    ks = list(per)
    out = {}
    for held in ks:
        A, y = [], []
        for k in ks:
            if k == held:
                continue
            d = per[k]
            A.append(np.column_stack([np.ones(len(d["truth"]))] + [d[f] for f in feats]))
            y.append(d["y"])
        A = np.vstack(A)
        y = np.concatenate(y)
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        d = per[held]
        Ah = np.column_stack([np.ones(len(d["truth"]))] + [d[f] for f in feats])
        out[held] = spearman(Ah @ beta, d["truth"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default=os.path.join(_HERE, "redundancy.csv"))
    ap.add_argument("--model", default=",".join(ALL_MODELS))
    ap.add_argument("--rows", choices=("covered", "all"), default="covered")
    ap.add_argument("--tier", default="", help="comma-separated substrings; restrict TIERS")
    ap.add_argument("--boot", type=int, default=B)
    args = ap.parse_args()
    R = args.root
    MODELS = [m.strip() for m in args.model.split(",") if m.strip()]

    def S(*p):   return os.path.join(R, "scratch", *p)
    def RES(*p): return os.path.join(R, "scratch", "results", *p)

    # Same tier map and same results-dir identification as foldx_alone_baseline.py.
    TIERS = {
        "fullSK clustered-SP": (
            [S(f"foldx_skempi_full/splits_skempi_full_foldx/fold_{f}/skempi_sp_test.tsv") for f in range(3)],
            "full_skempi", 3),
        "fullSK bycomplex-SP": (
            [S(f"foldx_skempi_full/splits_skempi_full_foldx/fold_{f}/skempi_sp_test.tsv") for f in range(3)],
            "full_skempi_bycomplex", 3),
        "fullSK MP-clustered": (
            [S(f"foldx_skempi_full/splits_mp_clustered_mp_foldx/fold_{f}/skempi_mp_test.tsv") for f in range(3)],
            "full_skempi_mp_clustered", 3),
        "fullSK MP-bycomplex": (
            [S(f"foldx_skempi_full/splits_mp_bycomplex_mp_foldx/fold_{f}/skempi_mp_test.tsv") for f in range(3)],
            "full_skempi_mp_bycomplex", 3),
        "fullSK CATH-all": (
            [S("foldx_skempi_full/splits_cath_foldx/fold_0/skempi_all_test.tsv")],
            "full_skempi_cath", 1),
        # The +aug tier reads the SAME test TSVs: augmentation touches train only, and the three
        # test files are byte-identical between splits_cath_foldx and splits_cath_foldx_aug
        # (md5 checked 2026-08-05). So aug-vs-plain is a paired contrast on one test set.
        "fullSK clustered-ALL": (
            [S(f"foldx_skempi_full/clustered_all/splits_cath_foldx/fold_{f}/skempi_all_test.tsv")
             for f in range(3)],
            "full_skempi_clustered_all", 3),
        "fullSK clustered-ALL+aug": (
            [S(f"foldx_skempi_full/clustered_all/splits_cath_foldx/fold_{f}/skempi_all_test.tsv")
             for f in range(3)],
            "full_skempi_clustered_all_aug", 3),
    }
    if args.tier:
        want = [t.strip() for t in args.tier.split(",") if t.strip()]
        TIERS = {k: v for k, v in TIERS.items() if any(w in k for w in want)}
        if not TIERS:
            raise SystemExit(f"--tier {args.tier!r} matched none of the known tiers")

    rows, log = [], []
    for tier, (splits, resdir, nf) in TIERS.items():
        truth_fx = load_split(splits)
        if not truth_fx:
            log.append(f"[skip] {tier}: no split rows found")
            continue
        keys = list(truth_fx)
        y_all = np.array([truth_fx[k][0] for k in keys])
        fx_all = np.array([truth_fx[k][1] for k in keys])
        cov = fx_all != 0.0
        log.append(f"[cov ] {tier}: {cov.mean():.1%} of {len(keys)} rows carry a FoldX value")

        for model in MODELS:
            paths = [RES(resdir, f"{model}_base", f"fold_{f}", "training_run",
                         "test_predictions.tsv") for f in range(nf)]
            pred = load_preds(paths)
            if pred is None:
                log.append(f"[miss] {tier} / {model}_base: predictions absent")
                continue

            sel = [i for i, k in enumerate(keys)
                   if k in pred and (cov[i] or args.rows == "all")]
            y = y_all[sel]
            fx = fx_all[sel]
            bp = np.array([pred[keys[i]] for i in sel])
            kk = np.array([keys[i][0] for i in sel])

            per, frame = {}, {}
            for k in np.unique(kk):
                m = kk == k
                if int(m.sum()) < T:
                    continue
                t, f_, b = y[m], fx[m], bp[m]
                if np.std(t) == 0 or np.std(f_) == 0 or np.std(b) == 0:
                    continue
                per[k] = dict(
                    rho_base=spearman(b, t),
                    rho_fx=spearman(f_, t),
                    rho_base_fx=spearman(b, f_),
                    part_base=partial_spearman(b, t, f_),
                    part_fx=partial_spearman(f_, t, b),
                )
                frame[k] = dict(truth=t, y=_nrank(t), fx=_nrank(f_), base=_nrank(b))
            if not per:
                log.append(f"[warn] {tier} / {model}: no complex reached T={T}")
                continue

            def col(name):
                return np.array([v[name] for v in per.values()], float)

            b_fx = loco_blend(frame, ["fx"])
            b_both = loco_blend(frame, ["fx", "base"])
            gain = np.array([b_both[k] - b_fx[k] for k in per], float)
            gm, glo, ghi = boot_ci(gain[~np.isnan(gain)], args.boot)
            pm, plo, phi = boot_ci(col("part_base"), args.boot)
            rec = dict(tier=tier, model=model, n_cplx=len(per), n_mut=len(sel),
                       rho_base=np.nanmean(col("rho_base")),
                       rho_fx=np.nanmean(col("rho_fx")),
                       rho_base_fx=np.nanmean(col("rho_base_fx")),
                       part_base=pm, part_base_lo=plo, part_base_hi=phi,
                       part_fx=np.nanmean(col("part_fx")),
                       blend_fx=np.nanmean(list(b_fx.values())),
                       blend_both=np.nanmean(list(b_both.values())),
                       gain=gm, gain_lo=glo, gain_hi=ghi)
            rows.append(rec)
            # self-check: a single-feature FoldX blend is affine in rank(fx) within each complex
            if abs(rec["blend_fx"] - rec["rho_fx"]) > 1e-6:
                log.append(f"[BUG ] {tier} / {model}: blend_fx {rec['blend_fx']:.6f} != "
                           f"rho_fx {rec['rho_fx']:.6f} - the LOCO blend is not order-preserving")

    hdr = ["tier", "model", "n_cplx", "n_mut", "rho_base", "rho_fx", "rho_base_fx",
           "part_base", "part_base_lo", "part_base_hi", "part_fx",
           "blend_fx", "blend_both", "gain", "gain_lo", "gain_hi"]
    with open(args.out, "w") as f:
        f.write(",".join(hdr) + "\n")
        for r in rows:
            f.write(",".join(f"{r[h]:.4f}" if isinstance(r[h], float) else str(r[h])
                             for h in hdr) + "\n")

    cur = None
    print(f"\nrows = {args.rows}   (per-complex, T>={T}; cluster bootstrap B={args.boot}, seed={SEED})")
    print(f"\n{'tier':<22}{'model':<12}{'cplx':>5}{'base':>8}{'foldx':>8}{'base~fx':>9}"
          f"{'part_base':>11}{'part_fx':>9}{'blend2':>8}{'gain':>9}  95% CI")
    print("-" * 122)
    for r in rows:
        t = r["tier"] if r["tier"] != cur else ""
        cur = r["tier"]
        sig = "*" if (r["gain_lo"] > 0 or r["gain_hi"] < 0) else " "
        print(f"{t:<22}{r['model']:<12}{r['n_cplx']:>5}{r['rho_base']:>8.3f}{r['rho_fx']:>8.3f}"
              f"{r['rho_base_fx']:>9.3f}{r['part_base']:>11.3f}{r['part_fx']:>9.3f}"
              f"{r['blend_both']:>8.3f}{r['gain']:>+9.3f}{sig} "
              f"[{r['gain_lo']:+.3f}, {r['gain_hi']:+.3f}]")
    print("-" * 122)
    print("base~fx   = within-complex Spearman between base predictions and the FoldX scalar")
    print("part_base = Spearman(base, truth | foldx) - the PLM's non-redundant contribution")
    print("blend2    = LOCO global linear blend of both signals; gain = blend2 - FoldX alone")
    print("*         = 95% CI on gain excludes 0")
    for l in log:
        print(l)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
