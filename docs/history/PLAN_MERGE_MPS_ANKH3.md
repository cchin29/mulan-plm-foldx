# Plan — merge Mac MPS-support code + ankh3 run data into Linux work area

**Source:** `~/mulan_snapshot_20260703_082412.tar.gz` (+ `_UPDATE.md`), FROM the Mac
(branch `mps-support`, HEAD `7149b47`). Reverse of the usual Linux→Mac flow.
**Target:** Linux work area, branch `consolidate-results`.
**Constraint:** must not disturb runs in flight (the re-launch of A8→C2 AIDO +aug).
**Written:** 2026-07-03. AIDO work is already shared — do **not** re-import it.

---

## Key safety fact (why this merge is safe)

The main venv's `mulan` is a **non-editable install frozen at Jun 25** —
`~/.venvs/mulan/lib/python3.12/site-packages/mulan/` has **no ankh3 and no MPS code**,
and the `mulan-train` shim imports **that installed copy**, not the working tree.
(ankh3 gen scripts only work because they run as `python scratch/…` from the repo root,
where cwd shadows site-packages.)

Consequences:
- Editing the working tree + `git commit` does **NOT** affect any `mulan-train` run,
  running or freshly launched. Only a `pip install` changes what `mulan-train` sees.
- The AIDO C2 path needs **none** of these changes (MPS guards are no-ops on Linux;
  `get_device()`→CPU is identical; the ankh3 branch never fires for AIDO).

So: **steps 1–3 + 6 below are safe anytime, even mid-run.** Only the reinstall (step 4)
is run-affecting and must happen in an idle window. As of writing the box is **idle**
(no training/gen/watchers, no tmux/screen) — the ideal window.

---

## Change inventory

### A. Code — small, portable (MPS guards are no-ops on Linux)

| File | Change |
|---|---|
| `mulan/__init__.py` | `os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK","1")` **before** torch import; export `get_device` |
| `mulan/utils.py` | new `get_device()` (MPS>CUDA>CPU); `embed_sequence` ankh3 fix: prepend `os.environ.get("ANKH3_PREFIX","[NLU]")`, then strip 2 leading tokens (`embedding[:, 2:, :]`); `device = get_device()` |
| `mulan/modules.py` | `from mulan.utils import TorchDevice, get_device`; `device = get_device()` |
| `scripts/__init__.py` | **NEW** (1-line docstring) — makes `mulan.scripts` importable for console-script resolution |
| `scripts/train.py` | MPS-guard preamble; `dataloader_pin_memory=False` |
| `scripts/generate_embeddings.py` | MPS-guard preamble; `device = mulan.get_device()` |
| `scripts/predict.py` | MPS-guard preamble; `device = mulan.get_device()` |
| `scripts/compute_landscape.py` | MPS-guard preamble; `device = mulan.get_device()`; `zs_score ….to(device)` |
| `scripts/extract_attentions.py` | MPS-guard preamble; `device = mulan.get_device()` |

**No action** — already identical on Linux: `constants.py` (ankh3 entries present),
`config.py`, `data.py`, `train_utils.py`. Merge hunk-by-hunk; never overwrite whole files.

### B. Run data — 11.5 MB, no heavy embeddings (all embedding dirs empty in tarball)

Copy into `scratch/results/` (strip `._*` macOS AppleDouble junk). Linux has none of these:
- `cv10_ankh3_large_nlu`
- `cv10_ankh3_large_s2s`
- `cv10_ankh3_large_aug_nlu`
- `cv10_ankh3_xl_nlu`
- `cv10_ankh3_xl_aug_nlu`

### C. Docs / plots — hunk-merge, both sides edited (do NOT overwrite)

- `RESULTS.md` — splice Mac's §13 (ankh3-large 0.823, ankh3-xl 0.831, ◆=MPS-measured) +
  platform columns. The Mac copy already has an AIDO §14 — watch for duplication.
- `TODO.md` — splice Mac's "🍎 Mac/MPS parallel track" overlay onto current Linux status.
- `MPS_COMPATIBILITY.md` — **NEW**, copy wholesale.
- `scripts_plots/` (`plot_ddg_scaling.py`, `ddg_scaling_data.csv`, `README.md`,
  `ddg_scaling.png/.svg`) — Mac authoritative, take wholesale; optionally regen with the
  plotting venv (`~/.venvs/mulan-esmc/bin/python scripts_plots/plot_ddg_scaling.py`).

---

## Two landmines to handle explicitly

1. **Stale ankh3 caches.** Linux `scratch/embeddings_ankh3_large` (1.9 G) +
   `embeddings_ankh3_xl` (3.1 G) were generated **before** the prefix fix and are **wrong**
   (raw seq, no `[NLU]`, off-by-one leading `<unk>`). Once utils.py lands they are
   code-inconsistent. **Do NOT delete** — the Mac already has correct ankh3 results, Linux
   C5/C6 are deferred, and regen is ~5 GB + hours. Instead drop a
   `STALE_pre_prefix_fix.txt` marker in each dir. Delete+regen only if reproducing ankh3 on
   Linux (`ANKH3_PREFIX="[NLU]" python scratch/gen_plm_emb.py …`).

2. **The reinstall = the only run-affecting step.** To make the new code live for
   `mulan-train` (non-editable install), `pip install -e .` (switch to **editable** so tree
   and install stay in sync). Do it in an idle window, before re-launching C2. C2 does not
   need it.

---

## Sequence (non-disturbing)

1. Merge code into working tree; copy the 5 run-data dirs (strip `._*`); hunk-merge the
   docs; take `scripts_plots/` wholesale.
2. `git commit` on `consolidate-results`.
3. Add `STALE_pre_prefix_fix.txt` markers to the two ankh3 caches.
4. **(idle-only)** `pip install -e .` → sanity:
   `mulan-train --help`; `python -c "import mulan; print(mulan.get_device())"` (expect `cpu`).
5. Re-launch A8→C2 in the dedicated systemd scope from the consistent tree.

Steps 1–3 safe anytime (even mid-run). Step 4 gated on idle. If keeping the reinstall out
until C2 finishes: do 1–3 now, leave 4–5 for later.

---

## Provenance / reference paths

- Unpacked snapshot: a scratch directory, not committed
- Mac branch commits new to Linux (newest→base): `7149b47` plot y-axis · `054588b` AIDO
  merge · `f4bbee8` ankh3 results+lineage · `87939a3` plot platform/MPS · `42b2eaa`
  mulan.scripts fix · `51088e8` **ankh3 prefix fix** · `14034a0` MPS_COMPATIBILITY.md ·
  `54af1ca` merge Linux snapshot · `67f39f6` MPS device support.
