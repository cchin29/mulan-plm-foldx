# Data provenance — embedding sweep S1102

## `S1102_filtered.tsv` (1100 mutations, 110 complexes)

Columns: `wt_chainA_label  wt_chainB_label  mutation(s)  ddG`. Mutation format
`<wt_aa><chain A|B><pos><mut_aa>`, position 1-indexed into the WT chain string.

Derived from the published **S1102** benchmark (`examples/S1102.tsv`, 1102 mutations, 111
complexes) minus complex **2I9B** (2 mutations: `RB129A`, `KB131A`).

**Why 1100, not the paper's 1102:** 2I9B is absent from the local full SKEMPI 2.0 CSV
(`scratch/skempi_v2.csv`, 7085 muts / 345 complexes) that `experiments/build_wt_fasta.py` uses to
rebuild WT sequences with SKEMPI's cleaned numbering, so no WT chains could be produced for it.
Raw RCSB `2I9B` (ATF·uPAR complex) does **not** match the benchmark's numbering (no chain has
R@129/K@131 in author numbering), and both the SKEMPI2 site and the MuLAN GitLab were unreachable,
so an authoritative sequence could not be recovered. The paper kept 2I9B (S1102 = 1102). Impact:
0.18% of data — negligible for PCC. Decision on record: proceed with 1100.

**Handling verified (2026-07-10):** the drop is a clean pre-pipeline excision, not a load-time skip —
`MulanDataset._fill_metadata` does an unguarded `self.sequences[label]` (identical in the GitLab upstream
and my fork), so a row whose complex is missing from the fasta would `KeyError`. `S1102_filtered.tsv` is
exactly `examples/S1102.tsv` minus the two 2I9B rows, order-preserving; all 1100 rows' WT chains are present
in the 220-chain fasta (110×2). Both `splits/paper_seed42/` and `scratch/cv10_splits_reconciled/` train on
1100 with 2I9B absent (they differ only in fold membership — the immaterial #3 reindexing).

**To reach exact 1102 parity:** obtain 2I9B's two WT chains in the benchmark's (SKEMPI-cleaned)
numbering — verify R@129 and K@131 on the mutated partner (chain "B") — append them to
`wt_sequences.fasta`, add the 2 rows back to `S1102_filtered.tsv`, regenerate splits
(`gen_splits.py`), and regenerate each model's 2I9B embeddings. **The GitLab upstream (now cloned at
`../MuLAN_GITLAB`) does NOT help** — it ships only `examples/S1102.tsv` (labels, no sequences) and a
2-sequence demo `example.fasta`; no benchmark WT fasta or SKEMPI data. Recovery still needs a complete
SKEMPI 2.0 copy that contains 2I9B.

## `wt_sequences.fasta` (220 chains)

Two WT chains per complex (`<PDB>_A`, `<PDB>_B`), built by `experiments/build_wt_fasta.py` from
SKEMPI 2.0 cleaned PDBs (CA residues, indexed by residue number so string position == mutation
position; gaps → `X`, non-canonical → `X`). Validated against a known 1A22 chain-A reference.
