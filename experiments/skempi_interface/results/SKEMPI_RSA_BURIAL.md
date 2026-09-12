# SKEMPI interface — RSA burial (monomer vs complex)

Residues scored: **3669** of 5112 (interface 2989 / non-interface 680). Skipped: {'residue_no_dssp': 464, 'complex_dssp_error': 979}.

## AUROC (interface vs non-interface)

| feature | AUROC | 95% CI |
|---|---|---|
| ΔrASA (monomer−complex, complex-informed) | 0.750 | [0.736, 0.765] |
| rASA monomer only (monomer view) | 0.549 | [0.525, 0.573] |

## Per-class mean rASA

| class | n | rASA monomer | rASA complex | ΔrASA |
|---|---|---|---|---|
| COR | 1667 | 0.44 | 0.16 | 0.28 |
| SUP | 498 | 0.24 | 0.13 | 0.11 |
| INT | 258 | 0.22 | 0.17 | 0.05 |
| RIM | 824 | 0.48 | 0.31 | 0.16 |
| SUR | 422 | 0.45 | 0.43 | 0.02 |

## Verdict

- **ΔrASA (needs the complex)** identifies interface residues at AUROC **0.750** — burial-on-binding is Levy's interface definition, recovered.
- **Monomer rASA alone** manages only AUROC **0.549** — the RSA-side confirmation that the monomer-only tool cannot flag interface residues from burial (they're surface-exposed until the partner arrives), matching the contact-based ~0.49 in `SKEMPI_INTERFACE.md`.
