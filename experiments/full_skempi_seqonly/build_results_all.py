#!/usr/bin/env python3
"""Unify every already-computed single-point FoldX result into ONE directory of symlinks.

The full-SKEMPI SP FoldX merge (merge_foldx_full_skempi / merge_foldx_cath) reads a single
results dir, historically only `scratch/foldx_skempi_full/results` (112 complexes) — while the
SAME per-complex JSONs for ~200 more complexes already exist in the S1102 + benchmark runs
(`scratch/foldx_s1102/results{,_S1131,_S2003,_S4169}`). All are produced by the identical
build_ddg.py pipeline with SKEMPI `#Pdb`-derived chain groups, so a given (pdb, mut) is
identical wherever it was computed — *except where the campaigns passed different
mutation lists; see CAVEAT 1 below.*
Reading them all lifts SP coverage ~28% -> ~58% with NO new FoldX compute — a pure plumbing gap.

Rather than teach every merge about 5 dirs, we materialise a single unified dir
`scratch/foldx_skempi_full/results_all/` of RELATIVE symlinks — one `<pdb>.json` per complex,
pointing at the MOST COMPLETE source JSON for that complex. Verified property (build_results_all
--audit): for every complex present in multiple dirs, the largest JSON already contains the union
of all muts (0 muts recoverable only by merging), so a single link per complex loses nothing —
no JSON dict-merge is ever needed. Relative targets resolve on both the Mac and the Linux box
(both carry foldx_s1102/ + foldx_skempi_full/ in snapshots).

CAVEAT 1 — "identical wherever computed" holds per mutation LIST, not per mutation.
    FoldX 5.1 BuildModel is deterministic: 1722 pairs with an identical repaired structure AND an
    identical individual_list.txt prefix agree to the last decimal (docs/FOLDX.md). But BuildModel
    walks the list sequentially in one process, so a mutation's ddG is a function of (repaired
    structure, every entry preceding it). Two campaigns that asked for different SUBSETS of a
    complex's mutations put a given mutation at a different position, and get a different number.
    Measured 2026-07-30 against `results_remainder/` (an orphaned 159-complex recompute):
    2430 of 2473 shared mutations agree on all 12 terms, **40 differ**, across five complexes
    (1GC1 28, 2AJF 4, 3KBH 4, 3NVN 2, 3SE4 2). This is NOT a chain-assignment or grouping error —
    the FoldX logs show every run used the same SKEMPI group (e.g. 1GC1 -> `Group 0 = G`,
    `Group 1 = C`, identical in work_remainder and all four foldx_s1102 work dirs). It is the
    the list-position effect above, not randomness. Magnitude on Interaction Energy:
    median |d| 0.039, mean 0.192, max 2.22 kcal/mol against a population sigma of 1.63 — ~0.12
    sigma typical, 1.36 sigma worst case, affecting 40 of 12495 SP split rows (0.32%).
    So: a value mismatch between two source dirs is EXPECTED when their mutation lists differ,
    not evidence of a plumbing bug. Do not "fix" it by merging dirs or re-picking sources. The
    real remedy is upstream: always compute the UNION of a complex's mutations (mulan/foldx/run.py
    worklist_single_point does this by construction).

CAVEAT 2 — `--audit` cannot see this. It compares `set(muts)` — mut KEYS only (see audit()).
    It verifies the superset property (no mut is lost by linking one file per complex) and
    nothing about the numbers behind those keys. Same blind spot that let the 2026-07-29
    dual-key bug through: a count/keyset check cannot validate values. To compare values, diff
    the 12-term dicts per (pdb, mut) explicitly.

CAVEAT 3 — on-disk state can diverge from what build() produces. build() writes RELATIVE
    SYMLINKS, but this box's `results_all/` currently holds **323 regular files, 0 symlinks**
    (materialised at some point by a dereferencing copy). That is harmless and arguably more
    robust — the dir is self-contained, so pruning a source dir cannot break it — but note that
    re-running build() replaces those files with symlinks again, re-coupling the dir to
    foldx_s1102/. Check `find results_all -type l | wc -l` before assuming either layout.

Point the mergers at this dir (default in merge_foldx_full_skempi.py / merge_foldx_cath.py):
    --results / --sp-results  scratch/foldx_skempi_full/results_all

Idempotent: safe to re-run; it clears and rebuilds the symlink dir. Text/link only, no compute.

    python experiments/full_skempi_seqonly/build_results_all.py           # build
    python experiments/full_skempi_seqonly/build_results_all.py --audit   # report overlap, no write
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

from skempi_foldx.exclusions import INTRACTABLE, is_excluded  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

# Ordered only for deterministic tie-breaks; selection is by max mut-count (superset).
SOURCE_DIRS = [
    "scratch/foldx_skempi_full/results",        # canonical full-SKEMPI SP run
    "scratch/foldx_s1102/results",              # S1102 SP FoldX
    "scratch/foldx_s1102/results_S1131",        # benchmark subsets
    "scratch/foldx_s1102/results_S2003",
    "scratch/foldx_s1102/results_S4169",
]
DEST = "scratch/foldx_skempi_full/results_all"


def n_muts(path):
    try:
        return len(json.load(open(path)).get("muts", {}))
    except Exception:
        return -1


def collect():
    """pdb -> (best_source_abspath, n_muts) picking the JSON with the most muts.

    Excluded complexes are dropped HERE rather than downstream. This dir is what scope
    enumeration reads, so a stale JSON for an intractable complex re-arms the compute loop every
    round -- see mulan.foldx.exclusions."""
    best, dropped = {}, set()
    for d in SOURCE_DIRS:
        for f in glob.glob(os.path.join(ROOT, d, "*.json")):
            pdb = os.path.splitext(os.path.basename(f))[0]
            if is_excluded(pdb):
                dropped.add(pdb)
                continue
            n = n_muts(f)
            if n < 0:
                continue
            if pdb not in best or n > best[pdb][1]:
                best[pdb] = (os.path.abspath(f), n)
    for pdb in sorted(dropped):
        print(f"[build_results_all] not linked -- {INTRACTABLE[pdb]}")
    return best


def audit():
    """Confirm the superset property: for every multi-dir complex, does the largest JSON == union?"""
    from collections import defaultdict
    muts = defaultdict(dict)
    for d in SOURCE_DIRS:
        for f in glob.glob(os.path.join(ROOT, d, "*.json")):
            pdb = os.path.splitext(os.path.basename(f))[0]
            try:
                muts[pdb][d] = set(json.load(open(f)).get("muts", {}))
            except Exception:
                muts[pdb][d] = set()
    multi = {p: v for p, v in muts.items() if len(v) > 1}
    lose = 0
    for p, v in multi.items():
        union = set().union(*v.values())
        if max(len(s) for s in v.values()) < len(union):
            lose += 1
    print(f"[audit] {len(muts)} complexes; {len(multi)} in >1 dir; "
          f"{lose} where the largest JSON is NOT the union (would need a real merge)")
    return lose


def build():
    best = collect()
    dest = os.path.join(ROOT, DEST)
    # clean rebuild (remove stale links only; never touch real source files)
    if os.path.isdir(dest):
        for f in glob.glob(os.path.join(dest, "*.json")):
            os.unlink(f)
    os.makedirs(dest, exist_ok=True)
    for pdb, (src, n) in sorted(best.items()):
        link = os.path.join(dest, f"{pdb}.json")
        rel = os.path.relpath(src, dest)          # relative -> portable across boxes
        os.symlink(rel, link)
    tot_muts = sum(n for _, n in best.values())
    print(f"[build] {len(best)} complexes symlinked -> {DEST}  ({tot_muts} muts across all complexes)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", action="store_true", help="report overlap only, no writes")
    a = ap.parse_args()
    if a.audit:
        audit()
    else:
        audit()
        build()


if __name__ == "__main__":
    main()
