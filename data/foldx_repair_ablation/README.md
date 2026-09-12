# FoldX repair-count ablation — 1× vs 5× `RepairPDB`

The source data behind `OPEN_QUESTIONS.md` § "Repair count", and behind the sentence in
`README.md` that offers the single-`RepairPDB` default as the reason this repository's FoldX
baseline (0.383) sits below the published one (0.430).

Per-complex FoldX runs over the **13 CATH T≥10 test complexes**, repeated at two `RepairPDB`
counts so the twelve-term channel can be compared against itself with the repair count as the only
changed variable.

## What is here, and what is not

| | |
|---|---|
| `repair_1x/<pdb>.json` | 13 complexes, one repair round — the shipping default, and what `data/foldx/` was built at |
| `repair_5x/<pdb>.json` | the same 13 at five rounds |

Each file is `{"muts": {<mutation>: {<12 FoldX terms>}}, "meta": {...}}`, with
`meta.repair_iterations` recording the arm. These are computed FoldX **outputs**, which is the
category `NOTICE` permits redistributing — the same basis on which `data/foldx/` ships.

**The repaired structures are not here and will not be.** The FoldX working directories hold
`<pdb>_original.pdb` and the SKEMPI cleaned inputs verbatim alongside `<pdb>_Repair.pdb`, so they
are structures rather than energies. That is a licence boundary, not an oversight, and it is why
the RMSD convergence figures in `OPEN_QUESTIONS.md` cannot be recomputed from this directory while
the ΔΔG ones can.

## Three results, three denominators

`OPEN_QUESTIONS.md` § "Repair count" reports three things from this data and they do **not** share
a denominator, which is easy to miss when reading them as one block:

| result | complexes | rows |
|---|---|---|
| structural convergence (RMSD by round) | 13 | — (needs the `work/` trees; not reproducible here) |
| energy effect — 97.6% of mutations move on `Interaction Energy` | 13 | 334 |
| **metric effect — 0.398 → 0.459** | **11** | 323 |

The metric row drops `2KSO` and `2PCC`: they carry 5 and 6 single-point mutations respectively and
fall below the T≥10 threshold once multi-point rows are excluded, taking 11 of the 334 rows with
them.

## Reproducing the metric effect

The mutation keys here are **author-chain form** (`meta.mode` is `author`) — `AD88G` — while
`data/skempi_full/single_point.tsv` is role-chain form — `AB88G`. Joining without the remap
silently drops three complexes to zero rows and quietly changes the answer. `remap_mut()` in
`experiments/full_skempi_seqonly/merge_foldx_full_skempi.py` is the shipped implementation and the
chain order in `meta.groups` is what it keys on. Its docstring calls it a legacy fallback, correct
only where the mutated chain is first in its group and its author numbering matches the sequence
index — that holds for all 13 complexes here, and all 334 mutations join under it. The
authoritative `remap_mut_resnum` needs `scratch/skempi2/PDBs/*.mapping`, which does not ship, so
it is not the one to use from a clone.

**The published figures reproduce exactly from what is here** — no CATH split, no network, no
FoldX licence. All 334 mutations join, 11 complexes clear T≥10, and the two arms come out at
0.3984 and 0.4585 for a paired Δ of +0.0601, against the reported
**+0.0601 [+0.0204, +0.1000]**:

```python
import csv, json, glob, os, sys
sys.path.insert(0, "experiments/full_skempi_seqonly")
from merge_foldx_full_skempi import remap_mut
from mulan import metrics

truth = {(r[0].split(".")[0], r[2]): float(r[3])
         for r in csv.reader(open("data/skempi_full/single_point.tsv"), delimiter="\t")}

for arm in ("repair_1x", "repair_5x"):
    pred, true, key = [], [], []
    for f in sorted(glob.glob(f"data/foldx_repair_ablation/{arm}/*.json")):
        pdb, d = os.path.basename(f)[:-5], json.load(open(f))
        groups = tuple(d["meta"]["groups"].split(","))
        for mut, terms in d["muts"].items():
            k = remap_mut(mut, groups) or mut
            if (pdb, k) in truth:
                pred.append(terms["Interaction Energy"]); true.append(truth[(pdb, k)]); key.append(pdb)
    print(arm, metrics.per_structure(pred, true, key, T=10)[1])   # 0.3984, then 0.4585
```

The CI half of the published result is a paired cluster bootstrap over complexes; the point
estimates above are what the interval is built on.
