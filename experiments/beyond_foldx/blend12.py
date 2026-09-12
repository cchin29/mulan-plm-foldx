#!/usr/bin/env python3
"""Does the fx_mlp edge come from the FoldX decomposition or from the backbone?

Follow-up to REDUNDANCY_RESULT.md finding 5: on clustered-SP the observed `fx_mlp` arms reach
0.439-0.446, above the 0.427 ceiling of any linear blend of the base pathway with the FoldX
*scalar*. The excess must come from the twelve decomposed terms. This splits that cleanly, with
no training, by fitting global leave-one-complex-out linear blends over three feature sets:

  blend_1     [scalar]                 - must equal FoldX alone exactly (validation)
  blend_12    [12 decomposed terms]    - the decomposition on its own, NO PLM involved
  blend_13    [12 terms + base pred]   - the decomposition plus the backbone

  gain_12  = blend_12 - blend_1        - what the decomposition buys over the total
  gain_plm = blend_13 - blend_12       - what the backbone buys ON TOP of the full decomposition

`gain_plm` is the quantity of interest: if it is ~0 while `gain_12` is positive, then everything
the fx_mlp arms have over the scalar arms is physics the total ddG discards, and the backbone is
contributing nothing to that advantage.

Features are within-complex normalised ranks by default (`--feat rank`), which keeps the
validation identity from redundancy.py: a single-feature scalar fit is affine in rank(scalar)
within each complex, so blend_1 reproduces ppS to 1e-6. Energy terms are additive in kcal/mol,
so ranking each one separately does discard cross-term magnitude; `--feat z` re-runs on
within-complex z-scores as a check that the conclusion does not depend on that choice.

Split files: 16 columns = chain1, chain2, mutation, ddG_experimental, then 12 FoldX terms. The
scalar comes from the 5-column sibling split, joined on (complex, mutation).

Usage:
  python experiments/beyond_foldx/blend12.py --root . --model esmc6b,ankh,...
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, os.pardir, "rescore_perstructure"))
from foldx_alone_baseline import (  # noqa: E402
    _rank, spearman, boot_ci, load_split, load_preds, cplx,
)
from redundancy import _nrank, loco_blend  # noqa: E402

T = 10
B = 10000
SEED = 0
NTERM = 12
ALL_MODELS = ("esmc6b", "ankh", "ankh3_xl", "ankh3_large", "saprot", "prostt5", "esmc600m",
              "esm2", "aido")


def load_dec(paths):
    """16-column decomposed split TSVs -> {(complex, mut): (truth, [12 terms])}."""
    d = {}
    for p in paths:
        if not os.path.exists(p):
            continue
        for ln in open(p):
            a = ln.rstrip("\n").split("\t")
            if len(a) < 4 + NTERM:
                continue
            try:
                d[(cplx(a[0]), a[2])] = (float(a[3]), [float(x) for x in a[4:4 + NTERM]])
            except ValueError:
                continue
    return d


def _zs(a):
    a = np.asarray(a, float)
    s = a.std()
    return (a - a.mean()) / s if s > 0 else np.zeros_like(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default=os.path.join(_HERE, "blend12.csv"))
    ap.add_argument("--model", default=",".join(ALL_MODELS))
    ap.add_argument("--feat", choices=("rank", "z"), default="rank")
    ap.add_argument("--boot", type=int, default=B)
    args = ap.parse_args()
    R = args.root
    MODELS = [m.strip() for m in args.model.split(",") if m.strip()]
    enc = _nrank if args.feat == "rank" else _zs

    def S(*p):   return os.path.join(R, "scratch", *p)
    def RES(*p): return os.path.join(R, "scratch", "results", *p)

    # tier -> (scalar splits, decomposed splits, results dir, n_folds)
    TIERS = {
        "fullSK clustered-SP": (
            [S(f"foldx_skempi_full/splits_skempi_full_foldx/fold_{f}/skempi_sp_test.tsv") for f in range(3)],
            [S(f"foldx_skempi_full/splits_skempi_full_foldxdec/fold_{f}/skempi_sp_test.tsv") for f in range(3)],
            "full_skempi", 3),
        "fullSK bycomplex-SP": (
            [S(f"foldx_skempi_full/splits_skempi_full_foldx/fold_{f}/skempi_sp_test.tsv") for f in range(3)],
            [S(f"foldx_skempi_full/splits_skempi_full_foldxdec/fold_{f}/skempi_sp_test.tsv") for f in range(3)],
            "full_skempi_bycomplex", 3),
        "fullSK MP-clustered": (
            [S(f"foldx_skempi_full/splits_mp_clustered_mp_foldx/fold_{f}/skempi_mp_test.tsv") for f in range(3)],
            [S(f"foldx_skempi_full/splits_mp_clustered_mp_foldxdec/fold_{f}/skempi_mp_test.tsv") for f in range(3)],
            "full_skempi_mp_clustered", 3),
        "fullSK MP-bycomplex": (
            [S(f"foldx_skempi_full/splits_mp_bycomplex_mp_foldx/fold_{f}/skempi_mp_test.tsv") for f in range(3)],
            [S(f"foldx_skempi_full/splits_mp_bycomplex_mp_foldxdec/fold_{f}/skempi_mp_test.tsv") for f in range(3)],
            "full_skempi_mp_bycomplex", 3),
        "fullSK CATH-all": (
            [S("foldx_skempi_full/splits_cath_foldx/fold_0/skempi_all_test.tsv")],
            [S("foldx_skempi_full/splits_cath_foldxdec/fold_0/skempi_all_test.tsv")],
            "full_skempi_cath", 1),
    }

    rows, log = [], []
    for tier, (sc_paths, dec_paths, resdir, nf) in TIERS.items():
        sc = load_split(sc_paths)
        dec = load_dec(dec_paths)
        if not sc or not dec:
            log.append(f"[skip] {tier}: scalar={len(sc)} dec={len(dec)} rows")
            continue
        keys = [k for k in sc if k in dec and sc[k][1] != 0.0]   # covered rows only
        log.append(f"[join] {tier}: {len(keys)} covered rows joined "
                   f"({len(sc)} scalar, {len(dec)} decomposed)")
        # the two files must agree on the label, or they are not the same rows
        mism = sum(1 for k in keys if abs(sc[k][0] - dec[k][0]) > 1e-6)
        if mism:
            log.append(f"[BUG ] {tier}: {mism} rows disagree on ddG_experimental between the "
                       f"scalar and decomposed splits")
            continue

        y = np.array([sc[k][0] for k in keys])
        fxs = np.array([sc[k][1] for k in keys])
        trm = np.array([dec[k][1] for k in keys])
        kk = np.array([k[0] for k in keys])

        for model in MODELS:
            paths = [RES(resdir, f"{model}_base", f"fold_{f}", "training_run",
                         "test_predictions.tsv") for f in range(nf)]
            pred = load_preds(paths)
            if pred is None:
                log.append(f"[miss] {tier} / {model}_base: predictions absent")
                continue
            sel = np.array([i for i, k in enumerate(keys) if k in pred], int)

            frame = {}
            for k in np.unique(kk[sel]):
                idx = sel[kk[sel] == k]
                if len(idx) < T:
                    continue
                t = y[idx]
                if np.std(t) == 0 or np.std(fxs[idx]) == 0:
                    continue
                d = dict(truth=t, y=_nrank(t), scalar=enc(fxs[idx]), scalar_raw=fxs[idx],
                         base=enc(np.array([pred[keys[i]] for i in idx])))
                for j in range(NTERM):
                    d[f"t{j}"] = enc(trm[idx, j])
                frame[k] = d
            if not frame:
                log.append(f"[warn] {tier} / {model}: no complex reached T={T}")
                continue

            terms = [f"t{j}" for j in range(NTERM)]
            b1 = loco_blend(frame, ["scalar"])
            b12 = loco_blend(frame, terms)
            b13 = loco_blend(frame, terms + ["base"])
            ks = list(frame)
            g12 = np.array([b12[k] - b1[k] for k in ks], float)
            gpl = np.array([b13[k] - b12[k] for k in ks], float)
            m12, lo12, hi12 = boot_ci(g12[~np.isnan(g12)], args.boot)
            mpl, lopl, hipl = boot_ci(gpl[~np.isnan(gpl)], args.boot)
            rho_fx = np.nanmean([spearman(frame[k]["scalar_raw"], frame[k]["truth"]) for k in ks])
            rows.append(dict(tier=tier, model=model, n_cplx=len(frame),
                             blend_1=np.nanmean([b1[k] for k in ks]),
                             blend_12=np.nanmean([b12[k] for k in ks]),
                             blend_13=np.nanmean([b13[k] for k in ks]),
                             gain_12=m12, g12_lo=lo12, g12_hi=hi12,
                             gain_plm=mpl, gplm_lo=lopl, gplm_hi=hipl))
            if abs(rows[-1]["blend_1"] - rho_fx) > 1e-6:
                log.append(f"[BUG ] {tier} / {model}: blend_1 {rows[-1]['blend_1']:.6f} != "
                           f"FoldX alone {rho_fx:.6f}")

    hdr = ["tier", "model", "n_cplx", "blend_1", "blend_12", "blend_13",
           "gain_12", "g12_lo", "g12_hi", "gain_plm", "gplm_lo", "gplm_hi"]
    with open(args.out, "w") as f:
        f.write(",".join(hdr) + "\n")
        for r in rows:
            f.write(",".join(f"{r[h]:.4f}" if isinstance(r[h], float) else str(r[h])
                             for h in hdr) + "\n")

    cur = None
    print(f"\nfeatures = within-complex {args.feat}   (T>={T}; cluster bootstrap B={args.boot}, seed={SEED})")
    print(f"\n{'tier':<22}{'model':<12}{'cplx':>5}{'FoldX':>8}{'12-term':>9}{'+base':>8}"
          f"{'gain_12':>10}{'':<20}{'gain_plm':>10}")
    print("-" * 122)
    for r in rows:
        t = r["tier"] if r["tier"] != cur else ""
        cur = r["tier"]
        s12 = "*" if (r["g12_lo"] > 0 or r["g12_hi"] < 0) else " "
        spl = "*" if (r["gplm_lo"] > 0 or r["gplm_hi"] < 0) else " "
        print(f"{t:<22}{r['model']:<12}{r['n_cplx']:>5}{r['blend_1']:>8.3f}{r['blend_12']:>9.3f}"
              f"{r['blend_13']:>8.3f}{r['gain_12']:>+10.3f}{s12} "
              f"[{r['g12_lo']:+.3f}, {r['g12_hi']:+.3f}]{r['gain_plm']:>+10.3f}{spl} "
              f"[{r['gplm_lo']:+.3f}, {r['gplm_hi']:+.3f}]")
    print("-" * 122)
    print("gain_12  = 12 decomposed terms vs the FoldX total, NO PLM in either")
    print("gain_plm = base prediction added on top of the full decomposition")
    print("*        = 95% CI excludes 0")
    for l in log:
        print(l)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
