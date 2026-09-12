# `foldenv` — extraction plan (standalone repo for `mulan/structural_context`)

**Goal:** publish `mulan/structural_context` as a standalone GitHub project named **`foldenv`**
("the environment of a residue within a fold"), decoupled from the `mulan` package.

## Decisions (locked 2026-07-24)

| # | Decision | Choice |
|---|---|---|
| Repo relationship | How mulan relates to the new repo | **Copy out now, leave mulan untouched this pass; delete/rewire mulan later** |
| Git history | Preserve the 25-commit trail? | **No — fresh single initial commit** (history stays in mulan) |
| Name | Package/import + distribution | **`foldenv`** (import `foldenv`, PyPI `foldenv` — verified available). *Renamed 2026-07-25 from `resenv`:* that string is owned by the MIT Media Lab "Responsive Environments" group (resenv.media.mit.edu, GitHub org `ResEnv`) and an arXiv earthquake-sim paper, so it was unwinnable in search. `foldenv` has an empty SERP, 0 GitHub repos, and fits the field's prefix idiom (Foldseek/Foldcomp/FoldX/Foldy). Note `fold-` reads as "structure tool", *not* "AlphaFold" — the AF provenance is carried by the README/keywords, deliberately not the name. |
| License | Repo license | Undecided at the time of writing; resolved to **CC BY-NC-SA 4.0**, inherited from MuLAN, because two helper modules adapt from it |
| Push | A separate step | The extraction ends with a tree that is ready to push; the push itself is a separate decision. |

**Discoverability strategy:** the name is short/citable; findability comes from *metadata*
(PyPI keywords, GitHub topics, README H1) loaded with the real feature search terms:
`relative solvent accessibility`, `RSA`, `secondary structure`, `residue contacts`,
`AlphaFold`, `pLDDT`, `residue microenvironment`, `per-residue PLM embedding`.

**Build location:** sibling dir `../foldenv` (i.e. `MuLAN/foldenv/`), outside the mulan tree.

---

## Coupling map (why this is low-risk)

`structural_context` is a **leaf** — nothing else in `mulan/` imports it. Its entire coupling
to the parent package is **5 import sites** across 2 files:

| File | Import | Vendor target |
|---|---|---|
| `contacts.py` | `from mulan.constants import three2one` | `foldenv/constants.py` |
| `context.py` | `from mulan.constants import three2one` | `foldenv/constants.py` |
| `validation.py` | `from mulan.constants import three2one` | `foldenv/constants.py` |
| `embedding.py` | `from mulan import constants as C` | `foldenv/constants.py` |
| `embedding.py` (lazy) | `from mulan.utils import load_pretrained_plm, get_device, embed_sequence` | `foldenv/plm.py` |

All *internal* imports are already relative (`from . import ...`) → they survive the move
untouched. Only the 5 sites above change.

**Vendored surface (~110 lines):** `constants.py` = amino-acid tables (`three2one`, `AAs`,
etc.) + the `PLM_ENCODERS` HF-id registry. `plm.py` = `get_device`, `load_pretrained_plm`,
`embed_sequence`. AA tables + model-id registry are non-copyrightable facts; `embed_sequence` /
`load_pretrained_plm` are MuLAN-derived (see License note).

---

## Work breakdown

### Phase 1 — Decouple + rename (in the `../foldenv` copy)
1. Copy `mulan/structural_context/` → `../foldenv/foldenv/` (drop `__pycache__`).
2. Add `foldenv/constants.py` (AA tables + used `PLM_ENCODERS` entries) and `foldenv/plm.py`
   (`get_device`, `load_pretrained_plm`, `embed_sequence`).
3. Rewrite the 5 import sites → `from .constants import ...` / `from .plm import ...`.
4. Update docstring/README references from `mulan.structural_context` → `foldenv`.
5. Sanity: `grep -rn "mulan" foldenv/` returns nothing (except intentional prose/history).

### Phase 2 — Packaging
6. `pyproject.toml`: name `foldenv`, v0.1.0, `requires-python>=3.9`, deps
   (`numpy`, `requests`, `pyyaml`, `biopython`, `mini3di`, `torch`, `transformers`),
   optional-deps groups (`[esmc]` for transformers≥4.57, `[dev]` pytest), keywords + classifiers,
   license = placeholder. External binary `mkdssp` noted in README (not pip-installable).
7. `tests/` → top-level; confirm collection. Keep the live-fetch self-skip + `RUN_HEAVY_EMB` opt-in.
8. `.gitignore`: `__pycache__/`, `*.pyc`, `.venv*/`, `.struct_context_cache/`, `build/`, `dist/`, `*.egg-info/`.
9. `.github/workflows/ci.yml`: pytest on the fast subset; install `mkdssp` (apt/conda) or skip DSSP tests when absent.
10. `LICENSE` placeholder + `NOTICE` recording provenance of the vendored MuLAN-derived helpers.

### Phase 3 — Docs & discoverability
11. Promote `README.md` to a standalone project README: H1 + first paragraph carry the feature
    search terms; quickstart, install (incl. `mkdssp`), API, config, model table.
12. Fix mulan-internal cross-links (e.g. `../EMBEDDING_PREPROCESSING.md`) — inline the needed bit or drop.
13. Curate carried docs (see inventory) — drop internal dev logs, keep user-facing.
14. Draft GitHub repo description + topics: `alphafold protein-structure solvent-accessibility
    dssp secondary-structure contact-map plddt protein-language-model residue-microenvironment bioinformatics`.

### Phase 4 — Verify (standalone)
15. Fresh venv, `pip install -e .`, run pytest (fast subset green).
16. Run the README quickstart live (`get_structural_context`, `tool.invoke`) to prove the
    decoupled package works with **zero `mulan` on the path**.
17. Optional `RUN_HEAVY_EMB=1` single-embedding smoke.

### Phase 5 — Create repo (STOP before push)
18. `git init` + one initial commit. `gh repo create foldenv --private`.
19. **Stop at "ready to push."** The push is a separate decision.

### Phase 6 — mulan reconciliation (DEFERRED, separate pass)
20. Delete `mulan/structural_context/`; `pip install foldenv`; rewire the 6 consumers'
    imports `mulan.structural_context` → `foldenv`:
    `experiments/structctx_sweep/{run_sweep,run_p4_rsa_table,run_p3_mutation_sensitivity}.py`,
    `experiments/embedding_sweep/gen_esmc600m_emb.py`.
    (Two further consumers named in the original plan were withdrawn before publication.)

---

## File inventory — carry vs. drop

| Item | Action |
|---|---|
| `*.py` source (11 modules) | **Carry** (decoupled) |
| `tests/` (13 tests + TESTS.md) | **Carry** → top-level `tests/` |
| `config.py` + `decisions.yaml` + `tool_spec.json` | **Carry** |
| `README.md` | **Carry + rewrite** as project README |
| `SETUP_NOTES.md` | **Carry** (env/mkdssp/porting notes) |
| `M7_TIER2.md`, `PHASE2_REPORT.md` | **Carry as `docs/`** (optional; user-facing analysis) |
| `export_structural_context_dev1.md`, `dev2.md` | **Drop** (internal dev logs) |
| `__pycache__/`, `*.pyc` | **Drop** |
| `constants.py`, `plm.py` | **New** (vendored coupling) |
| `pyproject.toml`, `.gitignore`, `LICENSE`, `NOTICE`, `.github/` | **New** |

---

## Open items still to settle (non-blocking for Phases 1–4)
- **License (Q4):** placeholder now. If a *permissive* license (MIT/Apache-2.0) is later chosen,
  clean-room `embed_sequence` / `load_pretrained_plm` (currently MuLAN CC BY-NC-SA-derived).
  AA tables + PLM registry are facts → no obligation. Simplest alternative: inherit CC BY-NC-SA 4.0.
- **Repo visibility:** create **private** first (nothing is public until it has been checked); flip to public on release.
- **CI mkdssp:** install in CI vs. skip DSSP-dependent tests when the binary is absent.
- **Docs depth:** how much of `M7_TIER2` / `PHASE2_REPORT` narrative to publish vs. trim.

## Rough effort
Phases 1–4 ≈ a focused half-day (decouple + package + verify). Phase 5 minutes. Phase 6 deferred.
