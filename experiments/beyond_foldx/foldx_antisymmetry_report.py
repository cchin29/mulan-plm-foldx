#!/usr/bin/env python3
"""Read foldx_antisymmetry.csv and apply the decision rule fixed in docs/history/runbooks/RUN_FOLDX_ANTISYMMETRY_PROBE.md.

Under perfect antisymmetry, ddG_reverse = -ddG_forward exactly, so:
    corr(fwd, -rev) = +1 ;  mean(fwd + rev) = 0 ;  sd(fwd + rev) = 0.

The third is the one that decides the augmentation question. A high correlation with a wide spread
still means individual augmented rows carry wrong values, and it is individual rows the model
trains on -- an aggregate agreement does not rescue them.

The decision rule is applied as written, not chosen after seeing the numbers.

Usage:  ./.venv/bin/python experiments/beyond_foldx/foldx_antisymmetry_report.py
"""
from __future__ import annotations
import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "foldx_antisymmetry.csv")
B, SEED = 10000, 0


def _rank(a):
    a = np.asarray(a, float)
    order = a.argsort(kind="mergesort")
    r = np.empty(len(a), float)
    r[order] = np.arange(len(a), dtype=float)
    s = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    return pearson(_rank(x), _rank(y))


def cluster_boot(vals, pdbs, stat, B=B, seed=SEED):
    """Bootstrap over COMPLEXES, matching every other contrast in this repository."""
    pdbs = np.asarray(pdbs)
    uniq = np.unique(pdbs)
    idx = {p: np.where(pdbs == p)[0] for p in uniq}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        sel = np.concatenate([idx[uniq[k]] for k in pick])
        v = stat(sel)
        if v == v:
            out.append(v)
    if not out:
        return float("nan"), float("nan")
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


# --- residue properties, for the systematics pass -------------------------------------------
# Kyte-Doolittle hydropathy, van der Waals volume (A^3), and formal charge at pH 7. The question
# each answers: does the negation fail worse when the reversal changes burial (volume), packing
# environment (hydropathy), or electrostatics (charge)?
VOL = {"G": 60, "A": 89, "S": 89, "C": 109, "D": 111, "P": 113, "N": 114, "T": 116, "E": 138,
       "V": 140, "Q": 144, "H": 153, "M": 163, "I": 167, "L": 167, "K": 169, "R": 174, "F": 190,
       "Y": 194, "W": 228}
HYD = {"I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8, "G": -0.4, "T": -0.7,
       "S": -0.8, "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2, "E": -3.5, "Q": -3.5, "D": -3.5,
       "N": -3.5, "K": -3.9, "R": -4.5}
CHG = {"D": -1, "E": -1, "K": 1, "R": 1, "H": 0.1}


def systematics(rows, fwd, rev, s, pdb):
    """What predicts the failure of the negation. Reported as |sum| -- the absolute error in the
    fabricated value -- because sign cancels across a heterogeneous set and would hide the effect."""
    import numpy as np
    wt = np.array([r["wt"] for r in rows])
    mt = np.array([r["mt"] for r in rows])
    err = np.abs(s)

    def show(name, mask):
        if mask.sum() < 5:
            return
        print(f"    {name:<30} n={int(mask.sum()):<5} mean|err| {np.mean(err[mask]):>6.3f}"
              f"   median {np.median(err[mask]):>6.3f}   corr(fwd,-rev) {pearson(fwd[mask], -rev[mask]):>6.3f}")

    print("\n  By residue-property change (reversal direction: mt -> wt):")
    dv = np.array([VOL.get(b, 0) - VOL.get(a, 0) for a, b in zip(mt, wt)])   # volume restored
    for lbl, m in (("restores volume (>+40 A^3)", dv > 40), ("near-neutral volume", np.abs(dv) <= 40),
                   ("removes volume (<-40 A^3)", dv < -40)):
        show(lbl, m)
    dh = np.array([HYD.get(b, 0) - HYD.get(a, 0) for a, b in zip(mt, wt)])
    for lbl, m in (("restores hydrophobicity >+2", dh > 2), ("near-neutral hydropathy", np.abs(dh) <= 2),
                   ("restores polarity <-2", dh < -2)):
        show(lbl, m)
    dc = np.array([CHG.get(b, 0) - CHG.get(a, 0) for a, b in zip(mt, wt)])
    show("charge change", np.abs(dc) >= 1)
    show("no charge change", np.abs(dc) < 1)

    print("\n  By special residue involved:")
    for aa, nm in (("G", "glycine"), ("P", "proline"), ("C", "cysteine")):
        show(f"{nm} either side", (wt == aa) | (mt == aa))
    show("alanine scan (mt == A)", mt == "A")
    show("non-alanine substitution", mt != "A")

    nres = np.array([float(r["n_res"] or 0) for r in rows])
    if (nres > 0).sum() > 10:
        print("\n  By complex size (residues in the repaired structure):")
        q = np.quantile(nres[nres > 0], [1 / 3, 2 / 3])
        for lbl, m in ((f"small (<={int(q[0])})", (nres > 0) & (nres <= q[0])),
                       (f"medium", (nres > q[0]) & (nres <= q[1])),
                       (f"large (>{int(q[1])})", nres > q[1])):
            show(lbl, m)

    ia = np.array([str(r.get("in_aug", "")) == "1" for r in rows])
    if ia.any() and (~ia).any():
        print("\n  Augmented-split membership (the rows whose reverse was actually fabricated):")
        show("in an augmented train split", ia)
        show("not augmented", ~ia)


def main() -> None:
    rows = [r for r in csv.DictReader(open(CSV)) if r["status"] == "ok" and r["sum"] != ""]
    fwd = np.array([float(r["ddg_fwd"]) for r in rows])
    rev = np.array([float(r["ddg_rev"]) for r in rows])
    pdb = [r["pdb"] for r in rows]
    s = fwd + rev
    print(f"{len(rows)} mutations over {len(set(pdb))} complexes\n")

    pr, sp = pearson(fwd, -rev), spearman(fwd, -rev)
    plo, phi = cluster_boot(None, pdb, lambda i: pearson(fwd[i], -rev[i]))
    mlo, mhi = cluster_boot(None, pdb, lambda i: float(np.mean(s[i])))
    dlo, dhi = cluster_boot(None, pdb, lambda i: float(np.std(s[i])))
    print(f"{'statistic':<34}{'value':>9}{'95% CI':>22}{'antisym':>10}")
    print(f"{'corr(fwd, -rev)  Pearson':<34}{pr:>9.4f}{f'[{plo:+.4f}, {phi:+.4f}]':>22}{'+1':>10}")
    print(f"{'corr(fwd, -rev)  Spearman':<34}{sp:>9.4f}{'':>22}{'+1':>10}")
    print(f"{'mean(fwd + rev)':<34}{np.mean(s):>9.4f}{f'[{mlo:+.4f}, {mhi:+.4f}]':>22}{'0':>10}")
    print(f"{'sd(fwd + rev)':<34}{np.std(s):>9.4f}{f'[{dlo:+.4f}, {dhi:+.4f}]':>22}{'0':>10}")
    print(f"{'sd(ddG) for scale':<34}{np.std(fwd):>9.4f}")

    # The directional statistic. Perfect antisymmetry puts -rev on fwd with slope exactly 1;
    # a slope below 1 means reverting a mutation does NOT recover as much as the forward move
    # spent, which is the signature of the structure having relaxed around the mutant. That is a
    # mechanism, not just an error bar, and it is what a mean absolute error would hide.
    slope = float(np.polyfit(fwd, -rev, 1)[0])
    slo, shi = cluster_boot(None, pdb, lambda i: float(np.polyfit(fwd[i], -rev[i], 1)[0]))
    print(f"\n{'regression -rev ~ fwd, slope':<34}{slope:>9.4f}{f'[{slo:+.4f}, {shi:+.4f}]':>22}{'1':>10}")
    print(f"  -> reverting recovers {slope:.0%} of the forward effect on average"
          f"{'; the remainder is hysteresis' if slope < 0.98 else ''}")
    print(f"  mean|ddG| forward {np.mean(np.abs(fwd)):.3f}   reverse {np.mean(np.abs(rev)):.3f}")

    # Does the error grow with the size of the structural change? Hysteresis should.
    print("\nBy |ddG_forward| tercile:")
    q = np.quantile(np.abs(fwd), [1 / 3, 2 / 3])
    for lbl, m in (("small", np.abs(fwd) <= q[0]),
                   ("medium", (np.abs(fwd) > q[0]) & (np.abs(fwd) <= q[1])),
                   ("large", np.abs(fwd) > q[1])):
        if m.sum() < 3:
            continue
        print(f"  {lbl:<8} n={int(m.sum()):<4} mean|fwd| {np.mean(np.abs(fwd[m])):>6.3f}"
              f"   mean(sum) {np.mean(s[m]):>+7.4f}   sd(sum) {np.std(s[m]):>6.4f}"
              f"   corr {pearson(fwd[m], -rev[m]):>6.3f}")

    # How wrong is a negated row, as a fraction of the value it claims?
    nz = np.abs(fwd) > 1e-9
    relerr = np.abs(s[nz]) / np.abs(fwd[nz])
    print(f"\nPer-row error of the negation, |fwd+rev| / |fwd|:")
    print(f"  median {np.median(relerr):.3f}   75th pct {np.percentile(relerr,75):.3f}"
          f"   90th pct {np.percentile(relerr,90):.3f}")
    print(f"  rows where the fabricated value is off by >50% of its own magnitude: "
          f"{int((relerr > 0.5).sum())}/{int(nz.sum())} ({(relerr > 0.5).mean():.0%})")

    systematics(rows, fwd, rev, s, pdb)

    print("\nDecision rule (docs/history/runbooks/RUN_FOLDX_ANTISYMMETRY_PROBE.md §5), applied as written:")
    if pr >= 0.95 and mlo <= 0 <= mhi:
        print("  -> correlation >= 0.95 and mean-sum CI covers zero.")
        print("     Antisymmetry holds well enough that the negation is not what degraded the")
        print("     FoldX arms. RESULTS.md 24's recorded reading stands; mechanism stays open.")
        print("     WEAK evidence: the probe skips RepairPDB on the mutant, which biases here.")
    elif pr <= 0.8 or np.std(s) >= np.std(fwd):
        print("  -> correlation <= 0.8, or sd(sum) comparable to the spread of ddG itself.")
        print("     The augmented FoldX column is substantially fabricated. 24's conclusion")
        print("     narrows from 'augmentation conflicts with the FoldX channel' to 'the")
        print("     antisymmetric construction of the FoldX column is invalid' -- which makes")
        print("     the augmentation repairable rather than abandoned.")
    else:
        print("  -> in between. Report the numbers, pick no reading; the ESM-C 6B base arm")
        print("     (docs/history/runbooks/RUN_AUG_BASE_ESMC6B_GPU.md) is the decisive experiment instead.")


if __name__ == "__main__":
    main()
