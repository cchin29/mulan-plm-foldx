# CATH-superfamily data-leakage exposure

Assigns CATH superfamilies to SKEMPI-derived complexes and computes, per dataset,
how much of the data sits in CATH interface **families** that recur within the split —
the "what each split buys" leakage signal.

## Run
```
python3 experiments/cath_leakage/cath_leakage.py
```
Needs network + EBI/CATH egress (a cloud sandbox that blocks EBI cannot do this; run
on the Mac). Stdlib only. Downloads two reference files on first run (both **gitignored**,
re-downloadable, ~47 MB total):
- `pdb_chain_cath_uniprot.csv.gz` — SIFTS (pdb,chain) → CATH **domain id** (e.g. `101mA00`)
- `cath-domain-list.txt` — CATH domain id → 4-level `C.A.T.H` superfamily

If EBI SSL is intercepted (self-signed cert in the proxy chain), fetch the SIFTS file
manually with curl first, then re-run (the script uses the cached copy):
```
curl -fsSL https://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/pdb_chain_cath_uniprot.csv.gz \
  -o experiments/cath_leakage/pdb_chain_cath_uniprot.csv.gz
```

## Method
- **Complex key** = `PDB_chainsA_chainsB` (benchmarks: join col0/col1 chain letters; SKEMPI: `#Pdb`).
- **Partner superfamilies** = union of 4-level CATH over each side's chains.
- **CATH family** = connected components where two complexes share superfamilies on **both**
  partner sides in either orientation (A∩A′ & B∩B′) or (A∩B′ & B∩A′).
- **Exposure** = fraction of a dataset's mutations in families that contain ≥2 distinct
  complexes *from that dataset*.
- **Unmapped policy:** a complex missing CATH on ≥1 partner side is its own singleton family
  (cannot be "exposed") → exposure % is a conservative lower bound.

## Output (`cath_families.json`)
`meta` (coverage, family count, policy), `per_complex` (partnerA_sf, partnerB_sf, family_id),
`per_dataset` (n_mut, n_cplx, cath_family_exposure_pct, chain_coverage_pct, + antibody breakdown).

## Headline (2026-07-19, CATH release 28.26 / UniProt 2026.03; 70.4% global chain coverage)
| dataset | n_mut | n_cplx | CATH-fam exposure % | chain cov % |
|---|--:|--:|--:|--:|
| S1102 | 1102 | 111 | 22.2 | 52.3 |
| S1131 | 1127 | 110 | 25.8 | 51.8 |
| S2003 | 1124 | 174 | 22.9 | 62.1 |
| S4169 | 2497 | 211 | 38.6 | 65.2 |
| SKEMPI-single | 5112 | 324 | 64.5 | 78.6 |
| SKEMPI-multi | 1973 | 155 | 69.2 | 76.5 |

Curated benchmarks carry ~22–26% fold-family exposure; full SKEMPI is ~65–69% — a random/clustered
split on full SKEMPI leaks fold-family signal that a CATH-superfamily hold-out neutralizes. The Ig
fold (2.60.40.10) drives the growth (6–9% of benchmarks → 32–33% of full SKEMPI).
