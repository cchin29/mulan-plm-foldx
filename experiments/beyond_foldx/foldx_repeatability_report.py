#!/usr/bin/env python3
"""How much of the measured asymmetry is FoldX run-to-run noise rather than hysteresis?

`foldx_antisymmetry.py --control` re-runs the SAME forward mutation on the SAME repaired wild type
the stored value came from. Any deviation is noise: same input, same command, different answer.

Why this is needed. sd(ddG_fwd + ddG_rev) cannot on its own distinguish a FoldX that is
antisymmetric but noisy from one that is quiet but genuinely path-dependent — both produce spread.
The control measures the first directly, so the second can be obtained by difference.

Both quantities carry noise from two FoldX computations: the reverse test compares a stored forward
against a fresh reverse, and the control compares a stored forward against a fresh re-run. So the
control's spread is directly subtractable, in variance:

    sd(test)^2  ~  sd(hysteresis)^2 + sd(control)^2

This treats the noise as additive and independent of the hysteresis, which is the standard
assumption and is checked here per magnitude tercile rather than assumed globally — if noise scaled
with |ddG| the same way hysteresis does, the decomposition would be misleading and the tercile table
would show it.

Usage:  ./.venv/bin/python experiments/beyond_foldx/foldx_repeatability_report.py
"""
from __future__ import annotations
import csv
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CTL = os.path.join(HERE, "foldx_repeatability.csv")
TEST = os.path.join(HERE, "foldx_antisymmetry.csv")


def load(path, dev_col=True):
    rows = [r for r in csv.DictReader(open(path)) if r["status"] == "ok" and r["sum"] != ""]
    fwd = np.array([float(r["ddg_fwd"]) for r in rows])
    other = np.array([float(r["ddg_rev"]) for r in rows])
    d = np.array([float(r["sum"]) for r in rows])
    return rows, fwd, other, d


def main() -> None:
    if not os.path.exists(CTL):
        raise SystemExit(f"control CSV not found: {CTL}")
    crows, cfwd, cagain, cdev = load(CTL)
    trows, tfwd, trev, tsum = load(TEST)

    print(f"control: {len(crows)} forward re-runs over {len({r['pdb'] for r in crows})} complexes")
    print(f"test   : {len(trows)} reversals over {len({r['pdb'] for r in trows})} complexes\n")

    exact = int((np.abs(cdev) < 1e-9).sum())
    print(f"FoldX reproduces the stored value exactly in {exact}/{len(cdev)} re-runs "
          f"({exact/len(cdev):.0%}) -- it is close to deterministic but not bitwise")
    print(f"  noise:      mean {cdev.mean():+.4f}   sd {cdev.std():.4f}   "
          f"max|dev| {np.abs(cdev).max():.3f}")
    print(f"  asymmetry:  mean {tsum.mean():+.4f}   sd {tsum.std():.4f}\n")

    nvar = cdev.var()
    tvar = tsum.var()
    hvar = max(tvar - nvar, 0.0)
    print(f"Variance decomposition of the reversal spread:")
    print(f"  total          sd {np.sqrt(tvar):.4f}   var {tvar:.4f}   100%")
    print(f"  FoldX noise    sd {np.sqrt(nvar):.4f}   var {nvar:.4f}   {nvar/tvar:.1%}")
    print(f"  hysteresis     sd {np.sqrt(hvar):.4f}   var {hvar:.4f}   {hvar/tvar:.1%}")

    print("\nPer |ddG_forward| tercile -- does noise scale with magnitude the way hysteresis does?")
    q = np.quantile(np.abs(tfwd), [1 / 3, 2 / 3])
    print(f"  {'tercile':<9}{'n(test)':>8}{'sd(test)':>10}{'n(ctl)':>8}{'sd(noise)':>11}{'noise share':>13}")
    for lbl, lo, hi in (("small", -np.inf, q[0]), ("medium", q[0], q[1]), ("large", q[1], np.inf)):
        mt = (np.abs(tfwd) > lo) & (np.abs(tfwd) <= hi)
        mc = (np.abs(cfwd) > lo) & (np.abs(cfwd) <= hi)
        if mt.sum() < 5 or mc.sum() < 5:
            continue
        st, sn = tsum[mt].std(), cdev[mc].std()
        print(f"  {lbl:<9}{int(mt.sum()):>8}{st:>10.4f}{int(mc.sum()):>8}{sn:>11.4f}"
              f"{(sn**2/st**2 if st > 0 else float('nan')):>12.1%}")

    print("\n  A noise share that stays small as |ddG| grows means the magnitude trend in the")
    print("  reversal test is hysteresis, not FoldX becoming less repeatable on large effects.")


if __name__ == "__main__":
    main()
