#!/usr/bin/env python3
"""FoldX-alone unsupervised baseline vs MuLAN arms, per-structure Spearman (T>=10).

Answers: how much does MuLAN add *on top of the FoldX number it is handed*, on each
leakage-controlled split tier?

Conventions match experiments/rescore_perstructure/rescore.py exactly:
  - complex key   = chain1_id.split("_")[0]
  - per_structure = mean within-complex correlation over complexes with n_mut >= T,
                    skipping complexes with zero variance in pred or truth
  - folds are POOLED (union of all test folds) before grouping by complex

Truth + FoldX scalar come from the FoldX-annotated split TSVs
(cols: chain1, chain2, mutation, ddG_experimental, foldx_scalar). The scalar is
standardised per fold, but each complex lives in exactly one test fold, so
within-complex ordering — all Spearman needs — is preserved. That also lets the
clustered-SP map be reused for the by-complex-SP re-partition of the same 4165 rows.

Uncovered mutations carry foldx == 0.0 exactly (per-fold standardised mean). Reported
both ways: `all` (what the model actually sees) and `cov` (covered rows only).

Those uncovered rows are the one place FoldX-alone per-structure rho depends on the partition.
Standardisation is monotone within a fold, so it cannot reorder covered rows inside a complex —
but an uncovered row sits at exactly 0.0 and is NOT rescaled, so where a complex mixes covered
and uncovered mutations, the 0.0 rows' rank against the covered ones shifts with the fold's
standardisation constants. It is 34 rows in 4165 and moves the SP tiers' FoldX-alone rho by
0.0002 between the by-complex and clustered partitions.

Metrics per row: `rho` = per-structure Spearman (T>=10, the conclusion metric); `sp_foldavg` =
Spearman computed within each test fold then averaged over folds — the form the CD-HIT <=60%
frontier column is reported in, and the only one comparable to it; `pooled_pearson` /
`pooled_spearman` / `auroc_destab` = the whole tier in one vector. Fold-averaged and pooled are
NOT interchangeable: the single-fold CATH tier makes them identical by construction (n_folds=1),
and on the 3-fold tiers they differ by more than several of the model gaps being compared.

Cluster bootstrap (resample the complex set) for CIs and for the paired contrast
MuLAN_arm - FoldX_alone, which is the quantity that decides whether MuLAN earns its keep.

Usage:  python foldx_alone_baseline.py [--root REPO_ROOT] [--out results.csv]
"""
import argparse
import os
from collections import defaultdict

import numpy as np

from mulan.metrics.core import degenerate   # spread-vs-scale, not np.std == 0; see its docstring

T = 10
B = 10000
SEED = 0


# --------------------------------------------------------------------------- stats
def _rank(a):
    a = np.asarray(a, float)
    order = a.argsort(kind="mergesort")
    r = np.empty(len(a), float)
    r[order] = np.arange(len(a), dtype=float)
    # average ties
    i = 0
    s = a[order]
    while i < len(a):
        j = i
        while j + 1 < len(a) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    return pearson(_rank(x), _rank(y))


def auroc_destab(pred, true):
    """AUROC for `destabilizing` (ddG > 0) — rank statistic, same definition as rescore.auroc."""
    lab = np.asarray(true, float) > 0
    s = np.asarray(pred, float)
    npos, nneg = int(lab.sum()), int((~lab).sum())
    if npos == 0 or nneg == 0:
        return np.nan
    r = _rank(s) + 1.0            # _rank is 0-based with tie averaging
    return float((r[lab].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def per_complex_rhos(pred, true, key, T=T):
    """-> dict complex -> rho, over complexes with n>=T and non-degenerate variance."""
    pred = np.asarray(pred, float); true = np.asarray(true, float); key = np.asarray(key)
    out, n_qual, n_degen = {}, 0, 0
    for k in np.unique(key):
        m = key == k
        if int(m.sum()) < T:
            continue
        n_qual += 1
        p, t = pred[m], true[m]
        if degenerate(p) or degenerate(t):
            n_degen += 1
            continue
        out[k] = spearman(p, t)
    return out, n_qual, n_degen


def boot_ci(rhos, B=B, seed=SEED):
    """Cluster bootstrap over complexes -> (mean, lo, hi)."""
    v = np.array(list(rhos), float)
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(B, len(v)))
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_boot(rhos_a, rhos_b, B=B, seed=SEED):
    """Paired cluster bootstrap of (a - b) over the shared complex set."""
    shared = sorted(set(rhos_a) & set(rhos_b))
    if not shared:
        return np.nan, np.nan, np.nan, 0
    d = np.array([rhos_a[k] - rhos_b[k] for k in shared], float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), len(shared)


# --------------------------------------------------------------------------- io
def cplx(chain1):
    return chain1.split("_")[0]


def load_split(paths):
    """FoldX-annotated split TSVs -> {(complex, mut): (truth, foldx)}."""
    d = {}
    for p in paths:
        if not os.path.exists(p):
            continue
        for ln in open(p):
            a = ln.rstrip("\n").split("\t")
            if len(a) < 5:
                continue
            try:
                d[(cplx(a[0]), a[2])] = (float(a[3]), float(a[4]))
            except ValueError:
                continue
    return d


def load_split_folds(paths):
    """Same rows as load_split, but kept one dict per fold instead of pooled.

    Per-structure rho does not need this — it groups by complex, and a complex sits in exactly
    one test fold, so pooling first is equivalent. The fold-averaged correlation does need it:
    that metric correlates *across* complexes within a fold, so which rows share a fold changes
    the answer, and the same 5801 rows give different values under the by-complex and the
    clustered partition.
    """
    return [{k: v for k, v in load_split([p]).items()} for p in paths]


def load_preds(paths):
    """test_predictions.tsv (chain1, chain2, mut, prediction) -> {(complex, mut): pred}."""
    d = {}
    for p in paths:
        if not os.path.exists(p):
            return None
        for ln in open(p):
            a = ln.rstrip("\n").split("\t")
            if len(a) >= 4:
                try:
                    d[(cplx(a[0]), a[2])] = float(a[3])
                except ValueError:
                    pass
    return d


def load_preds_folds(paths):
    """load_preds, one dict per fold (None if any fold is absent) — see load_split_folds."""
    out = []
    for p in paths:
        d = load_preds([p])
        if d is None:
            return None
        out.append(d)
    return out


def fold_avg_corr(fold_dicts, value_of, truth_of, keep=None, corr=spearman):
    """Mean (and SD) of the per-fold correlation between `value_of` and `truth_of`.

    `fold_dicts` is a list of {key: payload}; the two accessors pull the predicted and the true
    value out of one payload, so this serves both the FoldX-alone rows (payload = (truth, foldx))
    and the model arms (payload = pred, truth read from the split). Folds are weighted equally —
    that is the definition of the metric, and it is why the SD is reported beside the mean.

    `corr` selects the correlation: Spearman for the CD-HIT frontier column, Pearson for the
    S1102 CV10 panel, whose published points are themselves a mean±sd over the ten folds.
    """
    vals = []
    for fd in fold_dicts or []:
        ks = [k for k in fd if keep is None or keep(k)]
        if len(ks) < 3:
            continue
        r = corr([value_of(fd, k) for k in ks], [truth_of(fd, k) for k in ks])
        if r == r:
            vals.append(r)
    if not vals:
        return np.nan, np.nan, 0
    # ddof=0: the folds are the whole partition, not a sample of folds.
    return float(np.mean(vals)), (float(np.std(vals)) if len(vals) > 1 else np.nan), len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="foldx_alone_baseline.csv")
    ap.add_argument("--model", default="ankh",
                    help="PLM prefix(es) in the results dirs; comma-separated for several")
    ap.add_argument("--boot", type=int, default=B, help="bootstrap resamples")
    args = ap.parse_args()
    R = args.root
    MODELS = [m.strip() for m in args.model.split(",") if m.strip()]
    nB = args.boot

    def S(*p):  return os.path.join(R, "scratch", *p)
    def RES(*p): return os.path.join(R, "scratch", "results", *p)

    # tier -> (list of FoldX-annotated split TSVs, results dir, n_folds, mutation filter)
    TIERS = {
        "fullSK CATH-all": (
            [S("foldx_skempi_full/splits_cath_foldx/fold_0/skempi_all_test.tsv")],
            "full_skempi_cath", 1, "all"),
        "fullSK CATH-single": (
            [S("foldx_skempi_full/splits_cath_foldx/fold_0/skempi_all_test.tsv")],
            "full_skempi_cath", 1, "single"),
        "fullSK CATH-multiple": (
            [S("foldx_skempi_full/splits_cath_foldx/fold_0/skempi_all_test.tsv")],
            "full_skempi_cath", 1, "multi"),
        # results dirs identified by their per-fold test-row signature:
        #   full_skempi 1673/1246/1246 = clustered-SP ; full_skempi_bycomplex 1389/1388/1388 = bycomplex-SP
        #   full_skempi_mp_* 546/545/545 ; full_skempi_bycomplex_all 1934/1934/1933 = SP+MP combined
        "fullSK clustered-SP": (
            [S(f"foldx_skempi_full/splits_skempi_full_foldx/fold_{f}/skempi_sp_test.tsv") for f in range(3)],
            "full_skempi", 3, "all"),
        # Same 4165 rows as clustered-SP under a different partition. That distinction is invisible
        # to per-structure rho and decisive for the fold average, so this reads the by-complex
        # partition's own FoldX-annotated splits (1389/1388/1388) rather than reusing the
        # clustered ones — pointing both tiers at one directory made their fold averages identical,
        # which they are not.
        "fullSK bycomplex-SP": (
            [S(f"foldx_skempi_full/bycomplex/splits_skempi_full_foldx/fold_{f}/skempi_sp_test.tsv") for f in range(3)],
            "full_skempi_bycomplex", 3, "all"),
        # COMBINED single+multi (5801 rows = the 4165 SP + 1636 MP). These are the tiers the
        # frontier's own protocol matches, and the ones plot_ppS_scaling.py panel 2 draws, so
        # without them that panel had no FoldX-alone reference to sit against. The FoldX-annotated
        # splits live under a `splits_cath_foldx` basename because the ALL-tier builder is the CATH
        # one reused — the directory name is the builder's, the contents are this tier's (row
        # signatures 1934/1934/1933 and 2295/1753/1753 match the truth folds exactly).
        "fullSK bycomplex-ALL": (
            [S(f"foldx_skempi_full/bycomplex_all/splits_cath_foldx/fold_{f}/skempi_all_test.tsv") for f in range(3)],
            "full_skempi_bycomplex_all", 3, "all"),
        "fullSK clustered-ALL": (
            [S(f"foldx_skempi_full/clustered_all/splits_cath_foldx/fold_{f}/skempi_all_test.tsv") for f in range(3)],
            "full_skempi_clustered_all", 3, "all"),
        "fullSK MP-bycomplex": (
            [S(f"foldx_skempi_full/splits_mp_bycomplex_mp_foldx/fold_{f}/skempi_mp_test.tsv") for f in range(3)],
            "full_skempi_mp_bycomplex", 3, "all"),
        "fullSK MP-clustered": (
            [S(f"foldx_skempi_full/splits_mp_clustered_mp_foldx/fold_{f}/skempi_mp_test.tsv") for f in range(3)],
            "full_skempi_mp_clustered", 3, "all"),
        # The leaky per-mutation CV10 partition behind ddg_scaling_data_300ep_balanced.csv, whose
        # published points are a mean±sd of Pearson over these ten folds. Included only so that
        # panel 1 of plot_ppS_scaling.py can carry a FoldX-only reference on ITS metric and ITS
        # partition; pooled Pearson is not partition-invariant (per-fold standardisation is affine
        # per fold, so pooling folds that were scaled differently is not the same number), which is
        # why the reference cannot be borrowed from the S1102 by-complex or clustered rows.
        # No results dir: the CV10 arms live under several run trees, and panel 1 reads them from
        # its own CSV — only the FoldX-alone row is wanted here.
        "S1102 CV10-balanced": (
            [S(f"foldx_s1102/splits_balanced_foldx/fold_{f}/S1102_filtered_test.tsv") for f in range(10)],
            None, 10, "all"),
        "S1102 by-complex": (
            [S(f"foldx_s1102/splits_bycomplex_foldx/fold_{f}/S1102_filtered_test.tsv") for f in range(3)],
            "retrain_bycomplex", 3, "all"),
        "S1102 clustered": (
            [S(f"foldx_s1102/splits_clustered_id60_foldx/fold_{f}/S1102_filtered_test.tsv") for f in range(3)],
            "retrain_clustered", 3, "all"),
    }
    ARMS = [("base", "base"), ("fx_mlp", "foldx"), ("fx_scalar", "foldx_scalar")]

    rows = []
    log = []
    for tier, (splits, resdir, nf, mfilter) in TIERS.items():
        truth_fx = load_split(splits)
        if not truth_fx:
            log.append(f"[skip] {tier}: no split rows found")
            continue

        def keep(k):
            if mfilter == "all":   return True
            multi = "," in k[1]
            return multi if mfilter == "multi" else not multi

        keys = [k for k in truth_fx if keep(k)]
        y = np.array([truth_fx[k][0] for k in keys])
        fx = np.array([truth_fx[k][1] for k in keys])
        kk = np.array([k[0] for k in keys])
        cov = fx != 0.0

        # --- FoldX alone (model-independent: computed once per tier) ------
        r_fx_all, nq, nd = per_complex_rhos(fx, y, kk)
        r_fx_cov, _, _ = per_complex_rhos(fx[cov], y[cov], kk[cov])
        m, lo, hi = boot_ci(r_fx_all.values(), nB)
        mc, _, _ = boot_ci(r_fx_cov.values(), nB)
        # Fold-averaged Spearman — the form the CD-HIT ≤60% frontier column is reported in.
        fx_folds = load_split_folds(splits)
        _v, _t = (lambda d, k: d[k][1]), (lambda d, k: d[k][0])
        fa, fa_sd, fa_n = fold_avg_corr(fx_folds, _v, _t, keep)
        pfa, pfa_sd, _ = fold_avg_corr(fx_folds, _v, _t, keep, corr=pearson)
        rows.append(dict(tier=tier, model="—", arm="FoldX alone", rho=m, lo=lo, hi=hi,
                         n_cplx=len(r_fx_all), n_mut=len(keys),
                         coverage=float(cov.mean()), rho_covered_only=mc,
                         d_vs_foldx="", d_lo="", d_hi="", n_shared="",
                         sp_foldavg=fa, sp_foldavg_sd=fa_sd, n_folds=fa_n,
                         pr_foldavg=pfa, pr_foldavg_sd=pfa_sd,
                         # pooled forms + AUROC, so every panel of plot_ppS_scaling.py can draw
                         # its FoldX-only reference on its OWN metric instead of borrowing one
                         pooled_pearson=pearson(fx, y), pooled_spearman=spearman(fx, y),
                         auroc_destab=auroc_destab(fx, y)))

        if resdir is None:
            log.append(f"[note] {tier}: FoldX-alone only (no results dir mapped)")
            continue

        for model in MODELS:
            for arm_id, suffix in ARMS:
                paths = [RES(resdir, f"{model}_{suffix}", f"fold_{f}",
                             "training_run", "test_predictions.tsv") for f in range(nf)]
                pred = load_preds(paths)
                if pred is None:
                    log.append(f"[miss] {tier} / {model}_{suffix}: predictions absent")
                    continue
                sel = [i for i, k in enumerate(keys) if k in pred]
                if len(sel) < 0.5 * len(keys):
                    log.append(f"[warn] {tier} / {model}_{suffix}: only {len(sel)}/{len(keys)} rows joined")
                sel = np.array(sel, int)
                p = np.array([pred[keys[i]] for i in sel])
                r_arm, _, _ = per_complex_rhos(p, y[sel], kk[sel])
                m, lo, hi = boot_ci(r_arm.values(), nB)
                # paired vs FoldX alone on the shared complex set
                dm, dlo, dhi, ns = paired_boot(r_arm, r_fx_all, nB)
                # Fold-averaged Spearman for the arm, on the same folds the FoldX-alone row used,
                # so the two are directly differenceable on this tier.
                pf = load_preds_folds(paths)
                _av, _at = (lambda d, k: d[k]), (lambda d, k: truth_fx[k][0])
                _ak = lambda k: keep(k) and k in truth_fx
                afa, afa_sd, afa_n = fold_avg_corr(pf, _av, _at, _ak)
                apfa, apfa_sd, _ = fold_avg_corr(pf, _av, _at, _ak, corr=pearson)
                rows.append(dict(tier=tier, model=model, arm=arm_id, rho=m, lo=lo, hi=hi,
                                 n_cplx=len(r_arm), n_mut=len(sel), coverage="",
                                 rho_covered_only="", d_vs_foldx=dm, d_lo=dlo, d_hi=dhi,
                                 n_shared=ns, sp_foldavg=afa, sp_foldavg_sd=afa_sd,
                                 n_folds=afa_n, pr_foldavg=apfa, pr_foldavg_sd=apfa_sd,
                                 pooled_pearson=pearson(p, y[sel]),
                                 pooled_spearman=spearman(p, y[sel]),
                                 auroc_destab=auroc_destab(p, y[sel])))

    # ------------------------------------------------------------------ output
    hdr = ["tier", "model", "arm", "rho", "lo", "hi", "n_cplx", "n_mut", "coverage",
           "rho_covered_only", "d_vs_foldx", "d_lo", "d_hi", "n_shared",
           # appended, never interleaved: `rho` stays column 4 for every existing reader
           "sp_foldavg", "sp_foldavg_sd", "n_folds", "pr_foldavg", "pr_foldavg_sd",
           "pooled_pearson", "pooled_spearman", "auroc_destab"]
    with open(args.out, "w") as f:
        f.write(",".join(hdr) + "\n")
        for r in rows:
            f.write(",".join(
                f"{r[h]:.4f}" if isinstance(r[h], float) and not np.isnan(r[h])
                else ("" if (isinstance(r[h], float) and np.isnan(r[h])) else str(r[h]))
                for h in hdr) + "\n")

    # console table
    cur = None
    print(f"\n{'tier':<22}{'model':<13}{'arm':<11}{'rho [95% CI]':<26}{'n_cplx':>7}"
          f"{'n_mut':>7}   Δ vs FoldX-alone [95% CI]")
    print("-" * 124)
    for r in rows:
        tlabel = r["tier"] if r["tier"] != cur else ""
        cur = r["tier"]
        ci = f"{r['rho']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]" if not np.isnan(r["rho"]) else "n/a"
        if r["d_vs_foldx"] == "":
            d = f"(coverage {r['coverage']:.1%}; covered-only {r['rho_covered_only']:.3f})" \
                if r["coverage"] != "" else ""
        else:
            sig = "  *" if (r["d_lo"] > 0 or r["d_hi"] < 0) else "   "
            d = f"{r['d_vs_foldx']:+.3f} [{r['d_lo']:+.3f}, {r['d_hi']:+.3f}]{sig}"
        print(f"{tlabel:<22}{r['model']:<13}{r['arm']:<11}{ci:<26}"
              f"{r['n_cplx']:>7}{r['n_mut']:>7}   {d}")
    print("-" * 124)

    # leaderboard: which (model, arm) actually clear FoldX alone, and where
    print("\n==== (model, arm) pairs whose paired Δ vs FoldX-alone has a CI excluding 0 ====")
    wins = [r for r in rows if r["d_vs_foldx"] != "" and r["d_lo"] > 0]
    loss = [r for r in rows if r["d_vs_foldx"] != "" and r["d_hi"] < 0]
    for label, group in (("BEATS FoldX alone", wins), ("LOSES to FoldX alone", loss)):
        print(f"\n  {label}: {len(group)}")
        for r in sorted(group, key=lambda z: -abs(z["d_vs_foldx"])):
            print(f"    {r['tier']:<22}{r['model']:<13}{r['arm']:<11}"
                  f"{r['d_vs_foldx']:+.3f} [{r['d_lo']:+.3f}, {r['d_hi']:+.3f}]")
    n_contrast = sum(1 for r in rows if r["d_vs_foldx"] != "")
    print(f"\n  total contrasts: {n_contrast}  ·  positive-significant: {len(wins)}"
          f"  ·  negative-significant: {len(loss)}"
          f"  ·  indistinguishable: {n_contrast - len(wins) - len(loss)}")
    print(f"*  = 95pct CI excludes 0 (cluster bootstrap over complexes, B={B}, seed={SEED})")
    for l in log:
        print(l)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
