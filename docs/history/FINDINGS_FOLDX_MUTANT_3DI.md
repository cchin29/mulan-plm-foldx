# Finding — FoldX-modeled mutants do NOT yield mutation-aware 3Di

**Date:** 2026-07-03. **Context:** installing FoldX to get "3Di on the mutant" for SaProt.
**Verdict:** FoldX BuildModel cannot deliver mutation-differentiated 3Di. Do not build the
mutant-3Di pipeline on it.

## Evidence (SKEMPI 1KNE, mutation Thr→Lys chain P)

Chain: `RepairPDB` → `BuildModel` (mutation `TP2K`) → foldseek `structureto3didescriptor`
on the FoldX WT-remodel and the mutant PDB.

- foldseek 3Di, chain P: WT `DDDDDD` vs MUT `DDDDDD` — **0 positions differ** (AA correctly
  changed at pos 2: `QTARKS`→`QKARKS`).
- Atom-level check: **0 of 232 backbone (N/CA/C/O) atoms moved** between WT-remodel and
  mutant (chains A and P). FoldX BuildModel repacks **sidechains only**; backbone is fixed.

foldseek 3Di is a pure backbone (Cα-neighborhood) descriptor ⇒ fixed backbone ⇒ mutant
3Di ≡ WT 3Di, structurally guaranteed (not a peptide artifact).

## Why it matters

Feeding FoldX mutants to SaProt = `WT-3Di + mutant-AA` per residue = the same WT-structure
approximation that already failed for ProstT5 (run6c, PCC 0.663 < run2 0.740). This is the
backbone-invariance trap catalogued in `PROSTT5_STRUCTURE_OPTIONS.md` (A2) and `RESULTS.md`
§9 — now confirmed for FoldX specifically. And you don't need FoldX for it: foldseek on the
WT PDB gives the identical 3Di directly.

## Options (open — awaiting decision)

1. **SaProt with WT 3Di (cheap).** foldseek on WT PDBs → WT 3Di, use for both passes. A faithful
   test of structure-as-context for SaProt; low cost. No FoldX needed.
2. **ESMFold mutant re-fold (the real lever).** Re-fold each mutant *sequence* → moved
   backbone → genuinely different 3Di. Bigger lift (ESMFold; CPU-slow / GPU-ideal). The only
   route to mutation-aware 3Di.
3. **FoldX ΔΔG as score feature.** Repurpose FoldX: batch BuildModel over S1102 → per-mutation
   ΔΔG_fold → MuLAN `add_scores` zero-shot channel (`AUGMENTATION.md` Tier 2.1). Not 3Di.

## Tools installed (not wasted regardless of choice)

- FoldX 5.1: `~/tools/foldx/foldx_20270131` (→ `~/.local/bin/foldx`); no `rotabase.txt` dep.
- foldseek AVX2: `~/tools/foldseek/bin/foldseek` (→ `~/.local/bin/foldseek`).
- Smoke-test artifacts: a scratch directory, not committed (RepairPDB 51 s, BuildModel 12 s on 1KNE).

## SKEMPI numbering gotcha

For 1KNE the FoldX mutation string had to use the `Mutation(s)_cleaned` column (`TP2K`), NOT
`Mutation(s)_PDB` (`TP6K`) — cleaned matched the actual PDB ATOM numbering (THR at P2; P6 is
SER). Any batch pipeline must pick the column that aligns per-PDB, or map to ATOM records.
