# Runbook: Rosetta flex-ddG on the S1102 fold-0 hotspots

> **Status: deferred / future work.** Scoped out for now (no Rosetta installed,
> license + compute lead time). This is the recipe to pick it up later. Context:
> [`S1102_FOLD0_OUTLIER.md`](S1102_FOLD0_OUTLIER.md) — FoldX showed the fold-0
> hotspot signal *is* in the structure but rigid-backbone magnitudes are
> unreliable (overshoots clashes, undershoots cavities). flex-ddG adds backbone
> flexibility to test whether that fixes the magnitudes.

## Goal

Recompute **binding** ΔΔG for the 18 fold-0 hotspots with a flexible-backbone
method and check the specific hypothesis: does relaxing the backbone **tame the
steric-clash overshoot** (e.g. 1PPF LB18W: FoldX 11.9 → target ~7.4) and
**improve the cavity-creating cases** (2FTL IB18A: FoldX 1.05 → target ~5.0)?
Not expected to rescue the electrostatic outliers (that's the FEP regime).

## Prerequisites (the missing pieces)

1. **PyRosetta** (recommended, scriptable) or full **Rosetta** (`rosetta_scripts`
   binary). Both are free for academics but **license-gated** — request a token
   (few-day turnaround). PyRosetta installs via conda with the token.
2. **flex_ddG protocol files** — canonical implementation:
   [`Kortemme-Lab/flex_ddG_tutorial`](https://github.com/Kortemme-Lab/flex_ddG_tutorial)
   (RosettaScripts XML + resfile driver + `analyze_flex_ddG.py`). Not in this repo.

## Reusable inputs (already resolved)

From the FoldX work in `scratch/foldx_fold0/` — **do not re-derive**:
- SKEMPI complex PDBs: `scratch/skempi2/PDBs/<PDB>.pdb`.
- Mutation → PDB-chain mapping and the two-chain interface groups, validated by
  matching WT residue identity at each position (`run_all.py`).

### Target table (18 hotspots)

| PDB | resfile mut | mutated chain | chains (group A / B) | true ΔΔG | FoldX | notes |
|---|---|---|---|---|---|---|
| 1PPF | `18 I PIKAA W` | I | E / I | 7.44 | 11.89 | clash — overshoot to fix |
| 1PPF | `18 I PIKAA Y` | I | E / I | 6.54 | 12.59 | clash — overshoot to fix |
| 1PPF | `20 I PIKAA E` | I | E / I | 6.33 | 2.47 | charge (FEP regime) |
| 1PPF | `20 I PIKAA Q` | I | E / I | 4.63 | 1.92 | polar |
| 2FTL | `18 I PIKAA A` | I | E / I | 4.97 | 1.05 | cavity — undershoot to fix |
| 1KTZ | `28 B PIKAA L` | B | A / B | 4.48 | 8.03 | clash — overshoot to fix |
| 2O3B | `74 B PIKAA A` | B | A / B | 3.23 | 1.06 | loss-of-contact |
| 3SGB | `7 I PIKAA A`  | I | E / I | −2.54 | 0.79 | charge loss (Ala) |
| 1CSE | `38 I PIKAA S` | I | E / I | 1.17 | 2.76 | (PLM overpred) |
| 1R0R | `13 I PIKAA N` | I | E / I | 1.48 | 2.06 | (PLM overpred) |
| 1R0R | `10 I PIKAA P` | I | E / I | 3.30 | 0.29 | Pro backbone |
| 1A4Y | `435 A PIKAA A`| A | A / B | 3.48 | 1.76 | charge loss (Ala) |
| 1S0W | `142 C PIKAA F`| C | A / C | −2.10 | −1.16 | buried gain-of-bulk |
| 1XD3 | `8 B PIKAA A`  | B | A / B | 2.74 | 3.01 | loss-of-contact |
| 1XD3 | `74 B PIKAA L` | B | A / B | 2.43 | 5.84 | charge→hydrophobic |
| 1MAH | `69 A PIKAA N` | A | A / F | 5.22 | 0.19 | enzyme aromatic loss |
| 1MAH | `121 A PIKAA Q`| A | A / F | 3.00 | −0.57 | enzyme polar |
| 1FCC | `43 C PIKAA A` | C | A / C | 3.77 | 1.73 | aromatic hotspot (Ala) |

(Resfile line = `<PDB resnum> <chain> PIKAA <newAA>`; rest of pose gets `NATAA`
or `NATRO`. **Chains to move** for the binding calc = the *non-mutated* group,
e.g. move `E` when the mutation is on `I`.)

## Per-mutation recipe

1. **Prep PDB** — strip waters/hetatms, keep only the two interface chains. No
   FoldX-style repair (flex_ddG does its own constrained minimization).
2. **Resfile** — one mutation line (table above) + repack neighborhood defaults.
3. **Chains-to-move** — the partner group (unbound by 1000 Å translation to score
   the interface).
4. **Run flex_ddG XML** — per mutation it:
   - applies harmonic CA coordinate constraints (ensemble stays near crystal),
   - runs **backrub** (~35,000 moves) → ensemble of **~35 backbone models**,
   - in each model `PackRotamers` + `Min` for **both WT and mutant**,
   - scores bound − separated → ΔΔG_bind per model.
5. **Aggregate + calibrate** — average the ensemble and apply the published
   reweighting (`fa_talaris2014-gam` / REF2015 correction via
   `analyze_flex_ddG.py`). **Raw Rosetta ΔΔG is uncalibrated — this step is
   mandatory.**

## Suggested scaffold (generatable now, without Rosetta)

Everything except the Rosetta install can be pre-built into `scratch/flex_ddg/`:
- cleaned per-complex PDBs (2 chains each),
- 18 resfiles (from the table above),
- a `chains_to_move.tsv`,
- `run_flex_ddg.py` templated on the Kortemme tutorial XML,
- `collect.py` to parse ensemble scores → calibrated ΔΔG → merge with
  `scratch/foldx_fold0/results_extended.json` for a 3-way PLM/FoldX/flex-ddG table.

Then once the license lands it's a single `python run_flex_ddg.py`.

## Cost

Expensive unit = the **~35-model ensemble per mutation**. On these small
(~250–350-residue) interfaces, ~20–40 min/model/core → **~15–25 core-hours per
mutation**; 18 mutations ≈ **~300–450 core-hours**. Embarrassingly parallel: ~1–1.5
days wall on the 14 local cores, a few hours on a cluster.

## Gotchas

- **Numbering**: resfiles use PDB numbering + chain ID — wrong number silently
  mutates the wrong residue. (WT identities already validated in the FoldX run.)
- **Charge-changing mutations** (YB20E, 1XD3 RB74L, 3SGB/1A4Y →Ala): implicit
  solvent handles these worst — expect them to stay the weakest points even after
  backbone relaxation. Route to FEP, not flex-ddG.
- **Calibration**: must apply the GAM/linear reweighting; skipping it gives
  poorly-scaled ΔΔG.
- **1XD3 LB8A note**: both chains A and B carry Leu at position 8 — dataset chain
  is B, so use chain **B** (already resolved).

## Success criterion

Not overall accuracy (MAE was a wash between FoldX and the PLMs). The test is
**mechanism-specific**: flex-ddG should move the 3–4 gain-of-bulk **clash** cases
from FoldX's overshoot toward truth (r on that subset stays high, magnitude
improves) and lift the **cavity** cases (2FTL IB18A, 2O3B QB74A) that FoldX
undershot — while the electrostatic outliers remain hard. If so, it confirms the
prescription: a **flexible-backbone, complex-aware** structural signal is what the
sequence models lack on these points.

## References

- Barlow, Ó Conchúir, Thompson, Suresh, Lucas, Heinonen, Kortemme (2018),
  "Flex ddG: Rosetta Ensemble-Based Estimation of Changes in Protein–Protein
  Binding Affinity upon Mutation," *J. Phys. Chem. B*.
- `Kortemme-Lab/flex_ddG_tutorial` (GitHub) — protocol XML, resfile format,
  `analyze_flex_ddG.py`.
- SKEMPI 2.0 — source of the S1102 complexes and experimental ΔΔG.
