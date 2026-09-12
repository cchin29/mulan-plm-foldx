# AIDO.Protein-16B — S1102 balanced 10CV (300ep/p30) results (2026-07-23)

Ran on an RTX 3090 GPU box, 4 arms × 10 folds, balanced_seed42, 300 ep /
patience 30, `--save_model True` (best-of-N restore), GPU_PAR=3. Completed,
40/40 folds, 0 failures. Metric = mean `test_pcc` over 10 folds. This is the AIDO entry
in the `embedding_sweep_balanced` matrix, comparable to the esmc6b runs done here.

## AIDO 4-arm results vs esmc6b

| arm | **AIDO-16B** | esmc6b | Δ (esmc6b − AIDO) |
|---|---|---|---|
| base | **0.8374** | 0.8812 | +0.0438 |
| aug (Tier-1) | **0.8391** | 0.8800 | +0.0409 |
| foldx (add_scores scalar) | **0.8488** | 0.8847 | +0.036 |
| foldx_mlp (12-term decomposed MLP) | **0.8573** | 0.8890 | +0.032 |

**esmc6b foldx is now balanced-split — the † cross-split caveat is RESOLVED (2026-07-24).**
The esmc6b *balanced* foldx chain (`cv10_esmc6b_balanced/{baseline,foldx,foldx_mlp}`,
balanced_seed42, 300ep/p30) has now been run — **baseline on the Linux CPU box**, **foldx /
foldx_mlp on the GPU box** (RTX 3090). Real balanced 10CV means (`test_pcc`, n=10):
**baseline 0.8749 → foldx 0.8847 → foldx_mlp 0.8890**, a clean monotonic FoldX lift of
**+0.010 (scalar) / +0.014 (MLP)** measured *within the chain* (identical rows/split), with
MLP > scalar by +0.004. This confirms the earlier prediction: balanced esmc6b foldx
(0.885–0.889) sits **above** the paper-split placeholder (0.871) that previously filled these
cells. Two honest caveats: (a) the chain's own baseline 0.8749 runs ~0.006 below the canonical
balanced base 0.8812 (they are separate 10CV partitions) — so read the lift *within* the chain,
not against the 0.881 base; (b) baseline is CPU while foldx/foldx_mlp are GPU, so the absolute
lift carries a small BLAS/CUDA-kernel environment component — the within-GPU MLP-over-scalar
+0.004 is the environment-clean part.

## Standings

**1. AIDO's disadvantage from the ladder HOLDS on S1102 — but the gap is small.**
On the directly-comparable balanced base/aug arms, esmc6b beats AIDO by **~0.04**
(base 0.881 vs 0.837; aug 0.880 vs 0.839). Same direction as the full-SKEMPI ladder
(esmc6b > AIDO everywhere), but the S1102 single-point gap (~0.04) is far tighter than the
ladder's clustered/multipoint rungs. The largest sequence model is still not the best head.

**2. FoldX genuinely helps AIDO on S1102 (single-point).**
AIDO climbs monotonically base 0.837 → foldx-scalar 0.849 → foldx-MLP 0.857 (**+0.020** base→MLP,
with MLP > scalar by +0.009). This mirrors the ladder's *single-point-SP* finding (FoldX lifts
AIDO) and contrasts with the ladder's *multipoint-clustered* rung, where AIDO's foldx-MLP arm
regressed. On S1102's single-point mutations the decomposed FoldX signal is clean and additive.

**3. FoldX narrows but does not close the gap.**
AIDO's best arm (foldx-MLP 0.857) still trails esmc6b's plain balanced base (0.881) by ~0.024,
and esmc6b's own balanced foldx arms (0.885 scalar / 0.889 MLP) remain well ahead (~+0.032).
Structural ΔΔG features help the weaker backbone but don't overturn the backbone ranking.

**4. 300ep/p30 vs the legacy 50ep/p10.**
The new balanced runs edge the old `cv10_aido` numbers up modestly: base 0.837 (was 0.828),
aug 0.839 (was 0.836) — the extra epochs help ~0.005–0.01, and the ordering is unchanged.

**Bottom line:** on S1102 as on the full-SKEMPI ladder, AIDO-16B is the weakest of the compared
backbones on the base arm; FoldX (esp. the decomposed MLP) is its friend on single-point data,
lifting it +0.02, but not enough to catch esmc6b. Consistent with the attention→interface
picture — raw 16B scale doesn't buy the interface/ΔΔG signal the smaller esmc6b already carries.

## Provenance
- AIDO: `scratch/results/s1102_aido_balanced/{base,aug,foldx,foldx_mlp}/fold_*/…all_results.json`
  (pulled text-only from GPU box; heads left on box).
- esmc6b base/aug: `scratch/results/embedding_sweep_balanced/esmc6b{,_aug}` (0.8812 / 0.8800).
- esmc6b foldx (paper split, cross-split reference): `scratch/foldx_s1102/cv10_esmc6b/`.
- Gap to close for a clean foldx comparison: run the esmc6b *balanced* foldx chain
  (`foldx_esmc6b_balanced_chain.sh` → `cv10_esmc6b_balanced/{baseline,foldx,foldx_mlp}`).
