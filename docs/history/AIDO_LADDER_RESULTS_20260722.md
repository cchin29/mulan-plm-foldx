# AIDO.Protein-16B honest-ladder — final results (2026-07-22)

Full-SKEMPI honest ladder, **5 rungs × 3 arms × 3 folds** (CATH = 1 fold), 300 ep / patience 30,
MAXPAR=3, on an RTX 3090 GPU box. Metric = **mean `test_pcc`** over folds.
Ladder finished; all foldx fold counts met (bycomplex 3, clustered 3, CATH 1,
mp-bycomplex 3, mp-clustered 3). AIDO scored here against the three already-run ladder PLMs.

## Full matrix (mean test_pcc; **bold** = best in row-group per rung)

### full_skempi_bycomplex — single-point, by-complex split
| PLM | base | foldx_scalar | foldx-MLP |
|---|---|---|---|
| AIDO-16B | 0.539 | 0.553 | 0.558 |
| ankh3_large | 0.543 | 0.550 | 0.527 |
| ankh3_xl | 0.545 | 0.562 | **0.565** |
| **esmc6b** | **0.578** | **0.576** | 0.573 |

### full_skempi — single-point, clustered (id60) split
| PLM | base | foldx_scalar | foldx-MLP |
|---|---|---|---|
| AIDO-16B | 0.156 | 0.250 | 0.219 |
| ankh3_large | 0.155 | 0.173 | 0.200 |
| ankh3_xl | 0.246 | 0.256 | 0.215 |
| **esmc6b** | **0.341** | **0.372** | **0.323** |

### full_skempi_cath — single-point, CATH split (n=1 fold, noisy)
| PLM | base | foldx_scalar | foldx-MLP |
|---|---|---|---|
| AIDO-16B | 0.487 | 0.549 | 0.510 |
| ankh3_large | 0.267 | 0.552 | 0.464 |
| **ankh3_xl** | **0.616** | **0.636** | 0.499 |
| esmc6b | 0.520 | 0.600 | **0.516** |

### full_skempi_mp_bycomplex — multipoint, by-complex split
| PLM | base | foldx_scalar | foldx-MLP |
|---|---|---|---|
| AIDO-16B | 0.540 | 0.604 | 0.599 |
| ankh3_large | 0.573 | 0.609 | 0.619 |
| ankh3_xl | 0.605 | 0.652 | 0.670 |
| **esmc6b** | **0.612** | **0.664** | **0.671** |

### full_skempi_mp_clustered — multipoint, clustered split
| PLM | base | foldx_scalar | foldx-MLP |
|---|---|---|---|
| AIDO-16B | 0.159 | 0.374 | 0.269 |
| ankh3_large | 0.177 | 0.420 | 0.492 |
| ankh3_xl | 0.262 | 0.376 | 0.546 |
| **esmc6b** | **0.317** | **0.467** | **0.571** |

## Standings

**esmc6b is the decisive ladder winner** — best (any arm) on 4 of 5 rungs; the only rung it loses
is CATH (single-fold, noisy) to ankh3_xl. Best cell per rung: bycomplex esmc6b-base 0.578 ·
clustered esmc6b-scalar 0.372 · CATH ankh3_xl-scalar 0.636 · mp-bycomplex esmc6b-foldx 0.671 ·
mp-clustered esmc6b-foldx 0.571.

**AIDO-16B — the largest model — is the weakest of the four.**
- **Base arm:** last or tied-last on 4/5 rungs (bycomplex, clustered, mp-bycomplex, mp-clustered).
  Only on CATH-base does it beat ankh3_large (0.487 vs 0.267) — exactly the known base standing.
- **Does FoldX change AIDO's standing?** Partially, and only on the *single-point* rungs:
  - **Single-point (bycomplex / clustered / CATH):** FoldX (esp. scalar) lifts AIDO past
    ankh3_large — bycomplex-foldx 0.558 > 0.527, clustered-foldx 0.219 > both ankh3, CATH-foldx
    0.510 > both ankh3. AIDO rises to ~2nd behind esmc6b here.
  - **Multipoint (mp-bycomplex / mp-clustered):** FoldX does **not** rescue AIDO — it stays clearly
    last in every arm. Worse, AIDO's **foldx-MLP arm underperforms its own foldx_scalar** on the
    clustered/mp-clustered rungs (clustered 0.219 < 0.250; mp-clustered 0.269 < 0.374), the
    **opposite** of esmc6b/ankh3_xl, for whom foldx-MLP is the *strongest* arm on mp-clustered
    (0.571 / 0.546). On mp-clustered-foldx the gap is stark: AIDO 0.269 vs esmc6b 0.571.

**Bottom line:** raw model scale (16B) does not buy honest-ladder performance — AIDO-16B trails
the smaller ankh3/esmc6b backbones. FoldX partially compensates on easy single-point splits (pushing
AIDO ahead of ankh3_large) but fails on the hard multipoint-clustered generalization rung, where
AIDO's FoldX-MLP head actively regresses relative to its scalar arm. Consistent with the
attention→interface finding (both 16B models weak; esmc6b > aido). **esmc6b remains the backbone of
choice across the ladder.**

_Data: `scratch/results/full_skempi{,_bycomplex,_cath,_mp_bycomplex,_mp_clustered}/aido_{base,foldx_scalar,foldx}/fold_*/training_run/all_results.json` (pulled text-only from the GPU box; checkpoints left on the box)._
