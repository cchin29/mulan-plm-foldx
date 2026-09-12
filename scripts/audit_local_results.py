#!/usr/bin/env python3
"""Integrity audit for a tree of trained result arms. Read-only, stdlib only, no network.

Run it after `run_bycomplex.sh` or any other driver that writes
``<tier>/<arm>/fold_<k>[/training_run]/all_results.json``::

    python3 scripts/audit_local_results.py                  # default root: scratch/results
    python3 scripts/audit_local_results.py --root results   # the published per-fold tree
    python3 scripts/audit_local_results.py --tier full_skempi

Every check below exists because the corresponding failure actually happened while producing the
results in this repository, and none of them were caught by anything else:

1. **Fold present but no `all_results.json`.** `full_skempi_bycomplex_all/esmc600m_base` sat as
   three empty directories for a week. `mulan-train` died on an SDK-only encoder missing from the
   legacy map, and `run_bycomplex.sh` logged DONE anyway and exited 0 -- so the queue runner
   recorded Success, and a supervisor that restarts only what reports Failed could never see it.
   **An integrity check cannot be built on exit codes**, which is why this reads the tree.

2. **`.FAILED` markers.** Dropped by the fixed `run_bycomplex.sh`. Catches the same class going
   forward, but only for runs made after that fix, which is why check 1 has to exist separately.

3. **Degenerate metrics.** A file can exist and still be useless: `test_pcc` absent, null, NaN,
   or outside [-1, 1]. An aggregator reading it will happily plot the result.

4. **Truncated arms**, against a fold count *derived per tier* rather than assumed. This matters:
   an earlier hand-check assumed 3 folds everywhere and flagged all 31 CATH arms as incomplete,
   when the CATH split is a single hold-out and 1 fold is correct.

5. **Live directory identical to an archive.** When an arm is re-run the old directory is
   archived in place as `__cov*` and a fresh one written under the original name, so a live
   directory equal to its archive means the re-run was never pulled. The archive must be **at
   least as complete** as the live directory for this to mean anything -- `__cov99_partial` is a
   mid-flight partial copy of a *current* run, and comparing only the folds they share reports a
   false positive. That mistake was made once already.

Exit status is 0 when clean, 1 when anything is found, so it can gate a queue step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
from collections import defaultdict

ARCHIVE_MARKERS = ("__cov", "__badkey")

#: Tiers whose split is a SINGLE hold-out, so one fold is correct and not a truncation. The CATH
#: partition is one superfamily-disjoint split, not a k-fold -- see docs/SPLITS_AND_METRICS.md.
#: Without this, the family check reports full_skempi_cath as truncated against its 3-fold
#: siblings, which is the same false positive a hand-check produced twice before this script
#: existed. Encoded here so it is stated once and cannot be rediscovered.
SINGLE_HOLDOUT_TIERS = ("_cath",)


def is_single_holdout(tier: str) -> bool:
    return any(m in tier for m in SINGLE_HOLDOUT_TIERS)
DEFAULT_ROOT = pathlib.Path(__file__).resolve().parent.parent / "scratch" / "results"


def is_archive(name: str) -> bool:
    return any(m in name for m in ARCHIVE_MARKERS)


def sha(p: pathlib.Path):
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except OSError:
        return None


def discover_arms(root: pathlib.Path):
    """-> {arm_dir: {fold_name: all_results.json path or None}}

    Layout varies by tier and must not be assumed. Three exist in this tree:

        cv10_ankh/fold_3/training_run/...                    tier IS the arm
        full_skempi_bycomplex/esm2_foldx/fold_1/...          tier/arm
        a1_ankh3_large/dense/ankh3_large/fold_2/...          tier/x/arm

    So find every ``fold_*`` directory at any depth and take its parent as the arm, rather than
    matching a layout: an earlier hand-written tier pattern matched nothing in one tier and
    reported *clean*, which is indistinguishable from actually being clean. Never glob a layout
    you believe in; glob the thing you are counting.
    """
    arms = defaultdict(dict)
    for fold in root.rglob("fold_*"):
        if not fold.is_dir():
            continue
        # Two layouts, both real: the working tree writes <fold>/training_run/all_results.json,
        # and the published tree flattens that level away. Believing in one of them reported
        # every fold of the other as a silent crash -- 666 findings against a clean tree, which
        # is exactly the "glob the thing you are counting" rule below, broken by this line.
        # next() already filters on exists(), so this is a Path or None -- do not re-test it.
        arms[fold.parent][fold.name] = next(
            (c for c in (fold / "training_run" / "all_results.json",
                         fold / "all_results.json") if c.exists()), None)
    return arms


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=pathlib.Path, default=DEFAULT_ROOT)
    ap.add_argument("--tier", action="append",
                    help="restrict to tiers whose name contains this (repeatable)")
    ap.add_argument("--quiet", action="store_true", help="only print findings")
    args = ap.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        print(f"!! results root not found: {root}")
        return 1

    all_arms = discover_arms(root)

    def tier_of(arm: pathlib.Path) -> str:
        return arm.relative_to(root).parts[0]

    def label(arm: pathlib.Path) -> str:
        return str(arm.relative_to(root))

    arms = {a: f for a, f in all_arms.items()
            if not is_archive(str(a.relative_to(root)))
            and (not args.tier or any(s in tier_of(a) for s in args.tier))}

    # Expected fold count is derived per tier from the modal non-zero count of its own arms --
    # never assumed. CATH tiers are legitimately 1 fold, bench tiers 10, full_skempi* 3.
    by_tier = defaultdict(list)
    for a, folds in arms.items():
        by_tier[tier_of(a)].append(sum(1 for j in folds.values() if j))
    expected = {}
    for tier, counts in by_tier.items():
        nz = [c for c in counts if c]
        expected[tier] = max(set(nz), key=nz.count) if nz else 0

    # A tier with ONE arm derives its expectation from that arm, so truncation can never fire --
    # bench_S4169_saprot_aug derived 4 while every sibling bench_S4169_* tier is 10. Compare each
    # tier against the modal count of the tiers sharing its family prefix to catch that.
    family = defaultdict(list)
    for tier, n in expected.items():
        if n:
            family["_".join(tier.split("_")[:2])].append((tier, n))
    tier_anomaly = []
    for fam, members in family.items():
        members = [(tier, n) for tier, n in members if not is_single_holdout(tier)]
        if len(members) < 3:
            continue
        counts = [n for _, n in members]
        modal = max(set(counts), key=counts.count)
        for tier, n in members:
            if n != modal:
                tier_anomaly.append((tier, n, modal, fam))

    no_results, failed, degenerate, truncated, stale = [], [], [], [], []
    n_folds = 0

    for arm, folds in sorted(arms.items()):
        got = {k: v for k, v in folds.items() if v}
        n_folds += len(got)
        tier = tier_of(arm)

        for fold, j in sorted(folds.items()):
            if (arm / fold / ".FAILED").exists():
                failed.append((label(arm), fold))
            if j is None:
                no_results.append((label(arm), fold))

        for fold, j in sorted(got.items()):
            try:
                v = json.loads(j.read_text()).get("test_pcc")
            except (OSError, json.JSONDecodeError) as e:
                degenerate.append((label(arm), fold, f"unreadable: {e}")); continue
            if v is None:
                degenerate.append((label(arm), fold, "test_pcc absent/null"))
            elif not isinstance(v, (int, float)) or math.isnan(v):
                degenerate.append((label(arm), fold, f"test_pcc={v!r}"))
            elif not -1.0 <= v <= 1.0:
                degenerate.append((label(arm), fold, f"test_pcc out of range: {v}"))

        exp = expected.get(tier, 0)
        if exp and 0 < len(got) < exp:
            truncated.append((label(arm), len(got), exp))

        # Stale live directory: identical to an archive that is AT LEAST as complete.
        for cand, cfolds in all_arms.items():
            if cand == arm or cand.parent != arm.parent:
                continue
            if not is_archive(cand.name) or not cand.name.startswith(arm.name):
                continue
            arch = {k: v for k, v in cfolds.items() if v}
            if len(arch) < len(got):
                continue          # partial archive of this very run -- not evidence
            shared = got.keys() & arch.keys()
            if shared and all(sha(got[k]) == sha(arch[k]) for k in shared):
                stale.append((label(arm), cand.name, len(shared)))

    # An empty tree would otherwise print "clean", which is the exact confusion `discover_arms`
    # warns about: nothing matched and nothing found look identical. Say so, and fail, because an
    # audit that inspected nothing has not passed. Reachable in a fresh clone -- `results/` is
    # populated only after a rescore, and `scratch/` is never distributed.
    if not arms:
        print(f"!! no fold directories under {root}\n"
              f"   Nothing was audited. This is NOT a pass. Point --root at a tree containing "
              f"<tier>/<arm>/fold_<k>[/training_run]/all_results.json, or train something first.")
        return 1

    if not args.quiet:
        print(f"scanned {len(by_tier)} tiers, {len(arms)} live arms, "
              f"{n_folds} folds with results")
        print("derived folds/tier: "
              + ", ".join(f"{t}={n}" for t, n in sorted(expected.items()) if n) + "\n")

    def block(title, rows, fmt):
        if rows:
            print(f"!! {title}: {len(rows)}")
            for r in rows[:40]:
                print("     " + fmt(r))
            if len(rows) > 40:
                print(f"     ... and {len(rows)-40} more")
            print()

    block("folds present but missing all_results.json (silent-crash signature)", no_results,
          lambda r: f"{r[0]} {r[1]}")
    block("folds carrying a .FAILED marker", failed, lambda r: f"{r[0]} {r[1]}")
    block("degenerate metrics", degenerate, lambda r: f"{r[0]} {r[1]}: {r[2]}")
    block("truncated arms (fewer folds than the tier's derived count)", truncated,
          lambda r: f"{r[0]}: {r[1]} of {r[2]}")
    block("live directory identical to a complete archive (re-run never pulled)", stale,
          lambda r: f"{r[0]} == {r[1]} ({r[2]} folds)")
    block("tiers whose fold count differs from their family", tier_anomaly,
          lambda r: f"{r[0]}: {r[1]} folds, but {r[3]}_* is usually {r[2]}")

    total = sum(map(len, (no_results, failed, degenerate, truncated, stale, tier_anomaly)))
    print("clean -- no integrity problems found in this tree" if not total
          else f"{total} finding(s). This checks one tree against itself. If results are produced "
               f"on more than one machine, it cannot tell you whether this copy is the current "
               f"one -- that needs a value-by-value comparison against the machine that trained "
               f"the arm, never a comparison of directory names.")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
