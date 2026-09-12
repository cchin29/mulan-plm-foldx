# PLAN — Attention → interaction-site prediction across the PDB (attention→interface question)

**Status:** Tier 0 DONE (pipeline validated). Tier 1 DONE (9-backbone S1102 head-to-head).
Tier 2 DONE (bootstrap-CI table + figure, fold-robust). Tier 3 DONE (full-SKEMPI, 7
backbones, leakage-controlled by-complex heads — Ankh clearly #1, ALL successors
significantly below). Remaining: esmc6b (GPU-box, snapshot shipped), esm3/saprot13b
full-SKEMPI caches, INTBuilder labels, DSSP stratification, Interactome-sample zoo.
**Owner:** cchin29 · **Branch:** a1-interface-xattn · **Created:** 2026-07-22

---

## 0. The question

> Attention weights → interaction-site prediction across the full PDB; do the
> tested PLMs improve it?

Concretely: take MuLAN attention weights → per-residue score → **AUROC vs
residue-level interface labels** across the PDB, and ask whether the newer PLM
backbones beat the paper's **Ankh-large** baseline. This is the one
piece not yet built — the four existing artifacts do **not** address it (their
"interface" = leakage clustering, not attention-based site prediction).

---

## 1. What the paper actually reports (verified against the paper)

Two distinct evaluations, with **different label tools**:

| eval | dataset | label tool | headline AUROC | figure |
|---|---|---|---|---|
| **S1102** | 1102-mut SKEMPI subset | **INTBuilder** (Dequeker 2017) | **0.757** avg (0.809 unmut / 0.723 mut) | Fig S1 |
| **PDB Interactome** | 224,888 protein chains / 123,613 entries | **Cα ≤ 8 Å** to partner reference atom | 0.642 all · 0.612 PPI · 0.763 DNA/RNA · 0.660 ligand | Fig 2 / S2 |

- Interactome contact-graph smoothing (Cα–Cα ≤ 7 Å neighbor averaging) → modest
  lift (0.645 all). DSSP 3-state stratification over 177,545 chains → Fig S2.
- **"ProteinNet 0.62" does not exist in this manuscript** (zero repo-wide hits;
  nearest is the PPI 0.612). Dropped from the framing unless an external source
  surfaces.
- **Attention-extraction recipe (paper Methods L316), verified identical to our
  `scripts/extract_attentions.py`:** concatenate LightAtt head outputs → per
  residue a vector of dim N×C (N=3 heads, C=64 filters) → **mean → MinMax** →
  one scalar per residue.

### The "online residue-level labels" = Zenodo record 18175031
Downloaded to `scratch/attention_interface/zenodo/` (gitignored):
- `contacts_data_pdb.pkl.gz` (490 MB) — dict of dicts keyed by chain id
  (`8h7f_A1`): `sequences`, `coords` (Cα [L,3]), `labels_ppi`, `labels_dna_rna`,
  `labels_ligand`, `labels_all` (per-residue **bool** [L]). 232,580 chains.
- `mulan_attentions_pdb.h5` (268 MB) — the paper's **released Ankh-large**
  per-residue attention, keyed by the same chain ids, per-chain MinMax [0,1].
- `mulan_pdb_by_taxonomy.csv` (139 KB) — chain→taxonomy + per-taxon `roc_auc`.
- h5 ∩ pickle = **224,888 chains** (exactly the paper's headline count).

---

## 2. Is any of this already implemented? — **No.**

- **Neither Lombardi release** (`../MuLAN_github_GianLMB`, `../MuLAN_GITLAB`)
  ships the interface-eval pipeline. They provide only the extraction CLI
  (`mulan-att`) + a README note that attention "relates to interface regions".
  The contacts-construction / AUROC / DSSP-stratification code that produced
  Fig 2/S1/S2 was **never released as runnable code** — only the Zenodo outputs.
- Our repo has three **ingredients, none the deliverable**: `extract_attentions.py`
  (faithful recipe, but wired only to shipped `mulan-esm`/`mulan-ankh`),
  `skempi_interface/run_skempi_interface.py` (`auroc`/`auroc_ci`, but on geometric
  contact counts), `interface_xattn/build_masks.py` (pairwise masks for the A1
  ΔΔG head — different task).
- **Reproduction nuance:** GitLab `extract_attentions.py` pulls attention via
  `model([emb]*4, output_attentions=True).attention[0]` (full-model forward, the
  likely path that generated the Zenodo scores); GianLMB/ours use
  `model.encoder(embedding).attention` (encoder-only). Functionally the same
  encoder attention for a single chain — but Tier 1 will mirror the GitLab call
  to be byte-faithful.
- **Checkpoint reality:** upstream ships trained heads for **ESM2-3B + Ankh-large
  only**. All other PLM heads are our own (`scratch/results/embedding_sweep_balanced/`),
  so the "do other PLMs improve it" comparison is inherently our novel extension.

---

## 3. Tiers

### Tier 0 — Reproduce the paper exactly ✅ DONE
Score the released Ankh h5 against the pickle labels; confirm we recover the
published AUROCs. **Result** (`score_interactome.py` → `RESULTS_interactome.md`):

| class | paper | pooled | Δ | macro |
|---|---|---|---|---|
| **all** | 0.642 | **0.642** | **+0.000** | 0.642 |
| PPI | 0.612 | 0.607 | −0.005 | 0.609 |
| DNA/RNA | 0.763 | 0.782 | +0.019 | 0.797 |
| ligand | 0.660 | 0.671 | +0.011 | 0.675 |

**Overall 0.642 reproduces to 3 decimals; pooled ≡ macro.** Pipeline (attention
orientation, label alignment, AUROC, pooled-over-residues protocol) is validated.
Per-class deltas reflect the unspecified per-class negative universe (DNA/RNA
smallest & most sensitive; we land higher, not lower → protocol nuance, not bug).

### Tier 1 — The head-to-head that answers it ✅ DONE (S1102)
`score_s1102.py` extracts per-residue LightAtt attention (mirroring the paper recipe)
from each fold-0 head and scores AUROC vs the interface_masks labels over the 220 WT
chains. Live-embed for pure-AA PLMs (ankh, esm2, prostt5, ankh3_large/xl); cached WT
embeddings (`--emb-dir`) for structure/SDK PLMs (saprot, saprot13b, esm3, esmc600m).
**Ankh validates at macro 0.788 vs paper 0.757** (+0.031; our heavy-8Å labels + fold-0
head) — pipeline trusted. esmc6b deferred (head not kept locally — GPU-only).

### Tier 2 — Ranked head-to-head + stats ✅ DONE (S1102)
`plot_s1102_interface.py` → `RESULTS_s1102.md` + `s1102_interface.{png,svg}`: macro
AUROC [10k-bootstrap 95% CI] per backbone, paired-bootstrap Δ-vs-Ankh (CI-excludes-0 ⇒
significant). **Result: no — the newer PLMs do not beat Ankh-large.**
SaProt (0.799) and ESM-C 600M (0.796) numerically edge Ankh (0.788) but Δ-vs-Ankh is
**ns** (paired CI includes 0); every pure-sequence successor (SaProt-1.3B, Ankh3-xl,
ESM2-3B 0.661, Ankh3-large, ProstT5 0.638) is **significantly below** Ankh. ⚠️ ESM3 =
0.424 (below chance) despite a healthy ddG head (CV PCC ~0.82) — flagged as a likely
embedding-cache provenance mismatch, excluded from the headline.

### Tier 2b — Hardening + extensions (remaining, optional)
The S1102 head-to-head answers it. Optional strengthening, in priority order:
1. ~~**esm3 provenance**~~ ✅ RESOLVED — head trained on the exact `scratch/embeddings_esm3`
   SDK cache, so 0.424 is real. Cause: ESM3 massive-activation outliers (emb std ~325 vs
   ~0.04); attention partially tracks outlier magnitude (Spearman +0.22), not interfaces,
   while the ddG head copes (PCC ~0.82). Genuine dissociation; kept as a flagged scale outlier.
2. **Fold ensemble** — fold-5 spot-check ✅ done (ranking stable: top cluster SaProt/ESM-C/
   Ankh ~0.78–0.80 in both folds, all successors below, no PLM beats Ankh either fold). A
   full 10-fold-averaged attention would further tighten CIs (10× cost) — optional.
3. **esmc6b** — regenerate/pull its head (GPU-only) to include the largest ESM-C.
4. **INTBuilder labels** — reproduce Fig S1's exact label tool to land Ankh *at* 0.757
   (vs our 0.788 heavy-8Å proxy) and remove the label-definition caveat.
5. **DSSP + class stratification** (Fig S2) — needs the multi-partner Interactome labels,
   not the single-partner S1102 masks → belongs with the Interactome-sample run.
6. **Interactome sample** — representative N of the 224,888 chains per backbone (needs
   per-backbone PLM embeddings for those chains; GPU / Linux box). The cross-PLM ranking
   on the paper's own Cα-8Å labels; compare to Tier-0's Ankh 0.642.

### Tier 3 — Full-SKEMPI scale-up ✅ DONE (7 backbones)
`plot_skempi_full_interface.py` → `RESULTS_skempi_full.md` + `skempi_full_interface.{png,svg}`.
**Result — corroborates & sharpens the S1102 finding:** with leakage-controlled by-complex
heads on held-out complexes, **Ankh is unambiguously #1 (0.671)** and **every** newer PLM is
*significantly* below (paired CI excludes 0): SaProt 0.628 (−0.042), ESM-C 600M 0.609 (−0.062),
Ankh3-xl 0.561, ProstT5 0.516, ESM2-3B 0.500 (=chance), Ankh3-large 0.460 (<chance). The S1102
"3-way tie at the top" collapses to "Ankh clearly best" once leakage is controlled — the newer
PLMs' apparent parity was partly a same-distribution artifact. (esmc6b pending GPU box; esm3/
saprot13b pending full-SKEMPI WT caches.)

<details><summary>scaffold detail</summary>
The tractable, self-contained scale target (vs the GPU-gated Interactome sample): re-run
the head-to-head on the broader full-SKEMPI set with the **leakage-controlled by-complex
heads** (evaluated on held-out complexes → honest generalization). All pieces are local —
no new embeddings, no SDK, no GPU.
- **Masks built:** `build_masks_skempi_full.py` → 102 masks in
  `scratch/interface_masks_skempi_full/` (heavy-atom 8Å; multi-chain groups e.g. `1AHW.AB.C_AB`
  handled by concatenation-alignment, all at 1.00 AA identity; 6 align errors, 206 no-PDB skips).
- **Scorer generalized:** `score_s1102.py` now takes `--mask-dir/--wt-fasta/--ckpt/--suffix`
  (S1102 defaults unchanged). Heads: `retrain_bycomplex/{tag}_base/fold_0/`.
- **Driver:** `run_skempi_full.sh` (10 backbones; live-embed pure-AA + cached saprot/esmc600m;
  esm3/saprot13b need full-SKEMPI WT caches verified first). Launch: `bash …/run_skempi_full.sh`
  (MPS-bound — queue it), then aggregate the `s1102_skempi_full_*.csv` like `plot_s1102_interface.py`.
- **Smoke-test preview:** Ankh by-complex head → macro AUROC **0.671** (vs S1102 0.788) —
  lower as expected for the harder, leakage-controlled setting; a genuine generalization number.
</details>

---

## 4. Open decisions / flags

- **[label source]** Proceeding on the assumption that "online labels" = Zenodo
  18175031 (Cα-8Å Interactome). This reading is **unconfirmed**; the S1102/Fig-S1 target uses INTBuilder instead. Geometric Cα-8Å recompute
  is the fallback.
- **[per-class protocol]** Exact per-class negative universe is unspecified in
  the paper; "all" matches exactly so the core is settled. Revisit only if a
  per-class number needs to be exact.
- **[Interactome scale]** Full 224k per non-Ankh PLM is heavy (per-backbone
  embeddings). Default: representative sample first, not all chains.
- **[extraction path]** Mirror GitLab `model([emb]*4, output_attentions=True)`
  for byte-faithfulness to the released scores.

## 5. Provenance / paths
- Labels + attention: `scratch/attention_interface/zenodo/` (gitignored; from
  Zenodo 18175031, CC-BY-4.0, Lombardi 2025-12-18).
- Tier-0 scorer: `experiments/attention_interface/score_interactome.py`.
- Trained heads: `scratch/results/embedding_sweep_balanced/<plm>/…/model.ckpt`.
- Extraction recipe: `scripts/extract_attentions.py` (matches paper Methods L316).
- AUROC/CI reference: `experiments/skempi_interface/run_skempi_interface.py`.
