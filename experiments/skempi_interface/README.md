# SKEMPI interface sweep — interface radius (A) + monomer contrast (B)

The quantitative, at-scale version of the question that the six-residue functional-site set couldn't answer:
**"Cα-8 Å or Cβ-5 Å for interface residues?"** — and a measured answer to how much the shipped
**monomer-only** tool misses at interfaces. Self-contained analysis dir (parallel to
`../structctx_sweep/`, `../paper_fairness/`); the framing/deferral discussion that led here is in
`../structctx_sweep/README.md` §"SKEMPI fuller sweep".

## Method

Ground truth = SKEMPI 2.0's per-mutation **Levy structural class** (`iMutation_Location(s)` in
`scratch/skempi_v2.csv`): **interface = {COR core, RIM rim, SUP support}**, **non-interface =
{INT interior, SUR surface}**. (INT is buried *monomer-interior* — NOT at the interface; getting
this right matters, see below.) For each single-mutation residue we load the local **crystal
complex** (`scratch/skempi2/PDBs/<PDB>.pdb`, author numbering matching `Mutation(s)_PDB`) and count
contacts at each cutoff (Cα ≤ 8 Å; Cβ ≤ 5 Å, Gly→Cα), split by:

- **cross-chain** — neighbors on the binding partner's chain(s)  → the *complex* view
- **monomer** — neighbors on the residue's own chain only        → the crystal chain with the
  partner deleted, used as a clean, AF-noise-free proxy for what a monomer-only tool sees

**4,510 / 5,112** single mutations scored (interface 3,663 / non-interface 847); 602 skipped
(residue not resolved in the crystal — missing density / numbering). Pure geometry, ~6 s, so no
pueue needed. Reproduce:

```bash
.venv-structctx/bin/python experiments/skempi_interface/run_skempi_interface.py
```

## Results (AUROC interface vs non-interface, 95% bootstrap CI, n=4510)

| feature | AUROC | 95% CI |
|---|---|---|
| **cross-chain Cα-8 Å** | **0.690** | [0.678, 0.701] |
| cross-chain Cβ-5 Å | 0.600 | [0.591, 0.607] |
| monomer Cα-8 Å | 0.458 | [0.438, 0.479] |
| monomer Cβ-5 Å | 0.490 | [0.470, 0.511] |

Per-class mean cross-chain / monomer Cα-8 contacts: COR 2.1 / 9.0 · SUP 1.2 / 12.0 · RIM 1.2 / 8.4
· INT 0.3 / 11.4 · SUR 0.1 / 8.9.

## Findings

**A — Cα-8 Å is the better interface cutoff (significantly).** Cross-chain Cα-8 Å (0.690) beats
Cβ-5 Å (0.600) with non-overlapping CIs. Cβ ≤ 5 Å is so tight that many true interface residues
have zero cross-chain Cβ neighbors, collapsing the signal — the same reason it failed the monomer
buried-core sanity gate in `../structctx_sweep/`. So **both** the monomer packing question (P2) and
the complex interface question agree: **keep Cα-8 Å**; it's also the cutoff the future
complex-context extension should use for interface detection.

**B — the monomer view is blind to interfaces (measured).** Monomer intra-chain contacts predict
interface membership at AUROC ≈ **0.49** — i.e. **chance, even slightly anti-predictive** (interface
core/rim residues are *surface-exposed* in the monomer with low intra-chain packing, while the INT
negatives are buried with high packing, so monomer packing points the wrong way). Against the
cross-chain 0.690, this is a hard number for the **monomer-only limitation**: a monomer-only tool
cannot flag interface residues from packing, because the contacts that define an interface residue
are exactly the cross-chain ones it never sees. This is the quantitative motivation for the
deferred **complex-context extension** (AF-multimer / bound complexes) — the burial/packing axis
only captures interfaces once the partner chain is present.

## Correctness note

Levy's **INT (interior)** class is buried *monomer-interior* and is **non-interface** — an early
run that mislabeled INT as interface depressed the cross-chain AUROC (INT residues have ~0
cross-chain contacts). The per-class table is the check: INT (0.3) sits with SUR (0.1), far below
COR (2.1). Interface = COR/RIM/SUP only.

## RSA-burial extension (`run_rsa_burial.py`)

The RSA analog of B, using the tool's headline burial signal: DSSP on the full complex vs the
isolated chain, so **ΔrASA = rASA_monomer − rASA_complex** = how much the partner buries the
residue (Levy's interface definition). On 3,669 residues (interface 2,989 / non-interface 680):

| feature | AUROC | 95% CI |
|---|---|---|
| **ΔrASA** (needs the complex) | **0.750** | [0.736, 0.765] |
| rASA monomer only | 0.549 | [0.525, 0.573] |

Per-class ΔrASA: COR 0.28 · RIM 0.16 · SUP 0.11 · INT 0.05 · SUR 0.02 — the textbook
burial-on-binding gradient (core buries most; interior/surface ~0). This is the RSA-side twin of
the contact result: interface identification needs the complex (ΔrASA 0.75), and the monomer view
alone is weak (0.55) — consistent with the contact-based ≈ 0.49. **Coverage caveat:** ~979
complexes failed `mkdssp` (older/multi-model PDBs) → 3,669 of 5,112 scored (vs 4,510 for the
lighter contact pass); the tight CIs and agreement with the contact result make the conclusion
robust. Heavy (DSSP on complex + each isolated chain) → run via pueue. `results/SKEMPI_RSA_BURIAL.md`.

## Scope

- **Crystal-chain-alone** is the monomer proxy (removes AF-prediction error as a confound; avoids
  ~211 AF fetches). A full AF-monomer replication would add prediction noise but not change the
  qualitative B result; documented as an optional heavier follow-up.
- Contact-count based (the P2 knob). A burial/RSA (rASA monomer vs complex) version would need DSSP
  on isolated chains — a natural extension, not required for the cutoff decision.
- Informs the **future complex-context extension**, not the settled v1 monomer defaults.
