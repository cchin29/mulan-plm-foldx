# SKEMPI fuller sweep — P2-interface (A) + monomer contrast (B)

Single-mutation residues scored: **4510** of 5112 (interface 3663 / surface 847). Skipped: {'residue_not_found': 602}.

## A — which contact cutoff identifies interface residues? (cross-chain)

| feature | AUROC (interface vs SUR) | 95% CI |
|---|---|---|
| cross-chain Cα-8Å | 0.690 | [0.678, 0.701] |
| cross-chain Cβ-5Å | 0.600 | [0.591, 0.607] |

## B — monomer contrast: how much does the monomer-only tool miss?

| feature | AUROC (interface vs SUR) | 95% CI |
|---|---|---|
| monomer Cα-8Å | 0.458 | [0.438, 0.479] |
| monomer Cβ-5Å | 0.490 | [0.470, 0.511] |

## Per-class mean contacts (Levy gradient)

| class | n | mean cross-chain Cα-8 | mean monomer Cα-8 |
|---|---|---|---|
| COR | 2033 | 2.1 | 9.0 |
| SUP | 631 | 1.2 | 12.0 |
| INT | 320 | 0.3 | 11.4 |
| RIM | 999 | 1.2 | 8.4 |
| SUR | 527 | 0.1 | 8.9 |

## Verdict

- **A (cutoff):** cross-chain **Cα-8Å** has the higher interface-AUROC (Cα-8Å 0.690 vs Cβ-5Å 0.600). This is the cutoff the future complex-context extension should use for interface detection.
- **B (monomer miss):** monomer intra-chain contacts predict interface at AUROC ≈ **0.490** (vs cross-chain 0.690) — the monomer view is far weaker, quantifying exactly what a monomer-only tool cannot see. Confirms the documented monomer-only limitation with a number, and motivates the complex extension.
