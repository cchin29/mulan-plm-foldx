#!/usr/bin/env python3
"""Negation and standardisation do not commute, and the augmented FoldX column negates the wrong one.

Second defect found while probing FoldX antisymmetry (2026-08-05), independent of it and of any
FoldX re-run -- this reads only files already on disk.

`merge_foldx_aug.py:112` builds each reverse row's channel as `[-x for x in vec]`, negating the
ALREADY-STANDARDISED vector, and justifies it as: "each term is already train-centered at ~0 by the
canonical standardization, so negation adds negligible offset".

That conflates two different statements. Train-centring makes the standardised values have mean 0,
which is true and irrelevant. The question is whether negating the standardised value equals
standardising the negated value, and it does not:

    z(x)  = (x - mu) / sigma            the pipeline's channel for a forward row
    z(-x) = (-x - mu) / sigma           what a REVERSE row should carry, if FoldX were antisymmetric
    -z(x) = (mu - x) / sigma            what the augmentation actually writes

    -z(x) - z(-x) = 2*mu/sigma          a constant offset on every reverse row

`mu` is the raw train mean of FoldX Interaction Energy. It is not near zero -- most mutations are
destabilising -- so the offset is large.

REPAIRED 2026-08-05: `merge_foldx_aug.py` now negates the RAW value and standardises with the same
constants, and the live `*_aug` splits are rebuilt. This script therefore reads the preserved
pre-repair splits (`*_aug__negz`); pointed at the live ones it finds zero exactly-negated rows, which
is the repair working rather than a failure. Pass another suffix as argv[1] to check a different
lineage.

Distinct from the hysteresis probe (`foldx_antisymmetry.py`), which asks whether FoldX is
antisymmetric at all. This defect would be present in full even if it were perfectly antisymmetric.

Usage:  ./.venv/bin/python experiments/beyond_foldx/foldx_aug_offset.py [aug-suffix]
"""
from __future__ import annotations
import glob
import json
import os
import sys

import numpy as np

AUG = sys.argv[1] if len(sys.argv) > 1 else "_aug__negz"

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "scratch", "foldx_skempi_full", "results*")
TIERS = {
    "clustered-ALL": "scratch/foldx_skempi_full/clustered_all/splits_cath_foldx{aug}/fold_{f}/skempi_all_train.tsv",
    "CATH": "scratch/foldx_skempi_full/splits_cath_foldx{aug}/fold_{f}/skempi_all_train.tsv",
}


def raw_ie() -> dict:
    """(pdb, mut) -> raw FoldX Interaction Energy, under both the role key and the `cleaned` alias."""
    out = {}
    for f in glob.glob(os.path.join(RESULTS, "*.json")):
        pdb = os.path.basename(f)[:-5]
        try:
            d = json.load(open(f))["muts"]
        except Exception:
            continue
        for k, v in d.items():
            ie = v.get("Interaction Energy")
            if ie is None:
                continue
            out[(pdb, k)] = ie
            if v.get("cleaned"):
                out[(pdb, v["cleaned"])] = ie
    return out


def load(path: str) -> dict:
    d = {}
    if not os.path.exists(path):
        return d
    for ln in open(path):
        a = ln.rstrip("\n").split("\t")
        if len(a) >= 5:
            d[(a[0], a[2])] = (float(a[3]), float(a[4]))
    return d


def reverse(m: str):
    return f"{m[-1]}{m[1]}{m[2:-1]}{m[0]}" if "," not in m and len(m) >= 4 else None


def main() -> None:
    raw = raw_ie()
    print(f"raw FoldX Interaction Energy recovered for {len(raw)} (pdb, mut) keys\n")
    for tier, tmpl in TIERS.items():
        for f in range(3):
            p_plain = os.path.join(ROOT, tmpl.format(aug="", f=f))
            p_aug = os.path.join(ROOT, tmpl.format(aug=AUG, f=f))
            plain, aug = load(p_plain), load(p_aug)
            if not plain or not aug:
                continue

            # Recover the standardisation constants the pipeline used, by regressing the committed
            # standardised channel on the raw value it came from. R^2 reports how well that holds
            # (it is below 1 because the canonical merge clips at +/-4).
            pr = [(raw[(c.split(".")[0], m)], z) for (c, m), (_, z) in plain.items()
                  if (c.split(".")[0], m) in raw]
            if len(pr) < 50:
                continue
            r = np.array([x[0] for x in pr])
            z = np.array([x[1] for x in pr])
            slope = float(np.polyfit(r, z, 1)[0])
            sigma = 1 / slope
            mu = float(r.mean() - z.mean() * sigma)
            r2 = float(np.corrcoef(r, z)[0, 1] ** 2)

            # Confirm the construction: every reverse row is EXACTLY the negated forward channel.
            exact = other = 0
            for (c, m), (_, zf) in plain.items():
                rv = reverse(m)
                if rv and (c, rv) in aug:
                    if abs(aug[(c, rv)][1] + zf) < 1e-6:
                        exact += 1
                    else:
                        other += 1
            synth = sum(1 for k in aug if k not in plain)
            off = 2 * mu / sigma
            print(f"{tier} fold_{f}")
            print(f"  standardisation recovered: mu {mu:+.4f}  sigma {sigma:.4f}  (R^2 {r2:.4f})")
            print(f"  reverse rows that are exactly -z(forward): {exact} (other: {other})")
            print(f"  synthesized rows: {synth}/{len(aug)} = {synth/len(aug):.0%} of the training set")
            print(f"  OFFSET on every one of them: {off:+.3f} sd  ({2*mu:+.3f} kcal/mol raw), "
                  f"against a channel sd of {z.std():.3f}")
            print(f"  -> the error is {abs(off)/z.std():.2f}x the channel's own spread\n")


if __name__ == "__main__":
    main()
