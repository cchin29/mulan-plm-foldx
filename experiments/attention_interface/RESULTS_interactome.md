# Tier-0: PDB-Interactome attention->interface AUROC (reproduction)

Released **Ankh-large** attention (Zenodo 18175031 `mulan_attentions_pdb.h5`) scored against the released per-residue interface labels (`contacts_data_pdb.pkl.gz`). Goal: recover the paper's published AUROCs and pin down the aggregation protocol (pooled-over-residues vs macro-averaged-over-chains).

- chains scored (h5 ∩ pickle): **224,888**
- attention: per-chain MinMax-scaled, mean over 3 heads × 64 filters (paper Methods)
- per-class AUROC computed over chains with ≥1 positive residue of that class

| class | paper | pooled | Δ | macro (per-chain) | Δ | chains | residues | pos% |
|---|---|---|---|---|---|---|---|---|
| all interactions | 0.642 | 0.642 | +0.000 | 0.642 | -0.000 | 224,888 | 49,712,258 | 20.6% |
| protein-protein | 0.612 | 0.607 | -0.005 | 0.609 | -0.003 | 167,500 | 31,939,214 | 20.2% |
| DNA/RNA | 0.763 | 0.782 | +0.019 | 0.797 | +0.034 | 7,957 | 1,940,736 | 5.9% |
| ligand | 0.660 | 0.671 | +0.011 | 0.675 | +0.015 | 137,829 | 35,951,086 | 11.7% |

Whichever column (pooled / macro) matches the paper column fixes the protocol carried into Tier 1 (the PLM-backbone head-to-head).
