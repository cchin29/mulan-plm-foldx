# MuLAN — Usage Guide

This document is a practical, end-to-end guide to running **MuLAN** (Mutational
effects with Light Attention Networks): installing it, predicting ΔΔG for
mutations, extracting interface attention, building mutational landscapes,
generating PLM embeddings, and **re-training / reproducing the SKEMPI models from
the paper**.

For background and the model description, see [README.md](../README.md) and the
[bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2024.08.24.609515).

---

## 1. Installation

MuLAN needs **PyTorch ≥ 2.0** installed first. Install the right build for the
hardware from the [PyTorch site](https://pytorch.org/get-started/locally/). For
example:

```bash
# CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# or CPU / Apple Silicon (MPS)
pip install torch
```

Then install MuLAN and its dependencies:

```bash
git clone https://github.com/cchin29/mulan-plm-foldx
cd mulan-plm-foldx
pip install .             # or `pip install -e .` to develop against it
```

> A dedicated conda/venv environment is strongly recommended.

### The `MULAN` environment variable (required)

The pretrained checkpoints are **not** redistributed in this fork — download them from
upstream (<https://github.com/GianLMB/mulan>, `models/pretrained/`) into a local
`models/pretrained/` directory. Then point `MULAN` at the repo root:

```bash
export MULAN="/absolute/path/to/mulan"     # the cloned repo directory
```

If the `.ckpt` files live elsewhere, instead set the directory directly:

```bash
export MULAN_MODELS_PATH="/path/to/checkpoints"
```

> Note: the variable is `MULAN_MODELS_PATH` (see
> [mulan/constants.py](../mulan/constants.py)). **Upstream's README says `MULAN_MODELS_DIR`**,
> which its own `constants.py` has never read — so that name silently does nothing, on this fork
> and on upstream alike. If upstream's instructions were followed and the checkpoints were not
> found, this is why.

### Device selection

MuLAN auto-selects the compute device (see
[`get_device`](../mulan/utils.py)) in this order: **MPS** (Apple Silicon) →
**CUDA** → **CPU**. No flag is needed.

### Verify the install

```bash
python -c "import mulan; print(mulan.get_available_models()); print(mulan.get_available_plms())"
```

Expected models:
`['mulan-esm', 'mulan-esm-multiple', 'imulan-esm', 'mulan-ankh', 'imulan-ankh', 'mulan-ankh-multiple']`

---

## 2. Models and PLMs

### MuLAN checkpoints (`models/pretrained/`, obtained from upstream — see above)

| Name | PLM backbone | Notes |
|------|--------------|-------|
| `mulan-esm` | ESM-2 (3B) | single-point mutations |
| `mulan-esm-multiple` | ESM-2 (3B) | trained incl. multiple-point mutations |
| `imulan-esm` | ESM-2 (3B) | **i**mproved: also takes a zero-shot score input |
| `mulan-ankh` | Ankh-large | **default** |
| `mulan-ankh-multiple` | Ankh-large | multiple-point mutations |
| `imulan-ankh` | Ankh-large | requires zero-shot scores |

The `imulan-*` variants **require** a zero-shot scores file (`-s/--scores-file`).

### Protein language models ([mulan/constants.py](../mulan/constants.py))

The PLM name is derived from the model name (`mulan-ankh` → `ankh`). PLM weights
are downloaded from the HuggingFace Hub on first use and cached in
`~/.cache/huggingface/hub`.

Call `mulan.get_available_plms()` for the live list — **17 backbones** as of writing, not the
seven this section used to name. The registry is [mulan/plm/registry.py](../mulan/plm/registry.py)
and [docs/PLM_BACKEND.md](PLM_BACKEND.md) explains it. Two caveats it encodes: `esmc_600m`,
`esmc_6b` and `esm3_sm_open_v1` are generated in **a separate environment** built from
`requirements-esmc.txt`, never as an extra of this package (their SDK pins `transformers<4.48.2`,
below what training uses); and `aido`, `esmc_6b` and `mint`
embeddings are generated out of band and read from cache rather than via `plm-embed`.

> ⚠️ The default backbones (`ankh-large`, `esm2_t36_3B`) are multi-GB downloads
> and memory-heavy. For quick local tests, use a smaller PLM with `plm-embed`
> (e.g. `esm_35M`), or expect the first run to be slow.

---

## 3. Command-line tools

All commands accept `--help`. Mutation syntax used throughout:

```
<wt_aa><chain><position><mut_aa>          e.g.  SA51A  =  Ser→Ala at position 51 of chain A
```

- `chain` is `A` or `B` (which of the two interacting partners).
- `position` is **1-based** along that chain's sequence.
- Multiple-point mutations are joined with `:`  → `AA1G:CB4A`.
- In input/score files, multiple *independent* mutation entries are joined with `,`.

> With `pip install -e .` done and the console scripts working, the equivalent call is
> `mulan-predict`, `mulan-att`, `mulan-landscape`, `mulan-train`, `plm-embed`
> directly. Otherwise call the script by path, e.g.
> `python scripts/predict.py ...`. Both forms are shown below.

### 3.1 `mulan-predict` — ΔΔG for specific mutations

Input file: whitespace-separated, one complex per line —
`name  seqA  seqB  mut1,mut2,...`

```
C1 MVKQIES...LV MKMSRL...NA SA51A,FA165A:RB11A
```

Run:

```bash
mulan-predict examples/sample_mut.txt -o output/predictions.tsv
# or
python scripts/predict.py examples/sample_mut.txt -o output/predictions.tsv
```

> `predict` takes the **table** above, not a FASTA. `examples/example.fasta` is the input for
> `mulan-att` (§3.2) and `plm-embed` (§3.4); handing it to `predict` is a parse error.

Useful flags:

| Flag | Meaning |
|------|---------|
| `--model-name` | one of the models in §2 (default `mulan-ankh`) |
| `-s, --scores-file` | zero-shot scores (required for `imulan-*`) |
| `-o, --output-file` | output TSV (default `output.txt`) |
| `--store-embeddings` | also dump per-sequence `.pt` embeddings next to the output |

Zero-shot scores file format (for `imulan-*`): `complex_name  mutation  score` per line, with
a multi-point mutation group colon-joined exactly as in the input table (`SA51A:RB11A`). **No
sample ships** — [`examples/sample_mut.txt`](../examples/sample_mut.txt) is a *predict input
table*, not a scores file.

Output is a TSV of `complex  mutation  score` (predicted ΔΔG).

### 3.2 `mulan-att` — interface attention weights

Extracts per-residue normalized attention (a proxy for interface involvement)
for every sequence in a FASTA file, into an HDF5 file.

```bash
mulan-att examples/example.fasta -o output/attentions.h5
python scripts/extract_attentions.py examples/example.fasta -o output/attentions.h5
```

Flags: `--model-name`, `-e/--embeddings-dir` (also save embeddings),
`-m/--no-minmax` (disable min-max scaling; instead rescale to mean 1).
Read results with `h5py`: each dataset key is a FASTA id, value is a 1-D array
over residues.

### 3.3 `mulan-landscape` — full single-mutation landscape

Scores **every** single-point substitution of the first sequence against a
partner. Sequences are passed inline, separated by `:` (scored sequence first).

```bash
mulan-landscape "MVKQIESKTAFQEALDAA:MKMSRLCLSVALLVLL" -o output/landscape
python scripts/compute_landscape.py "SEQ_TO_SCORE:PARTNER_SEQ" -o output/landscape
```

Flags: `-m/--model-name`, `-s/--scores-file` (matrix of zero-shot scores for
`imulan-*`: rows = positions, cols = amino acids in alphabetical order),
`-e/--embeddings-dir`, `--no-ranksort` (keep raw scores instead of rank-sorted).
Output: `landscape.csv` (positions × 20 amino acids).

Visualize a region against UniProt features with
[scripts/uniprot_visualize.py](../scripts/uniprot_visualize.py).

### 3.4 `plm-embed` — raw PLM embeddings

```bash
plm-embed examples/example.fasta ankh -o output/embeddings
python scripts/generate_embeddings.py examples/example.fasta esm_35M -o output/embeddings
```

Stores one `<fasta_id>.pt` tensor per sequence. Any PLM key from §2 is accepted.
These `.pt` files are exactly what training consumes as a precomputed embeddings
cache (§4).

---

## 4. Training & reproducing the paper

MuLAN training is a thin wrapper over the HuggingFace `Trainer`
([scripts/train.py](../scripts/train.py),
[mulan/train_utils.py](../mulan/train_utils.py)). The light-attention head is
tiny; the expensive part is computing PLM embeddings, which are **cached to
disk** and reused across epochs and folds.

### 4.1 Data format

Two files are needed (whitespace-separated, **no header**):

**(a) Mutated-complex table** — parsed by
[`MulanDataset.from_table`](../mulan/data.py):

```
seqA_label  seqB_label  mutations  [score]  [zs_score]
```

Column rules:
- `seqA_label`, `seqB_label` — FASTA ids that must exist in the FASTA file (b).
- `mutations` — comma-separated mutation string(s), e.g. `SA51A` or `AA1G:CB4A`.
- 4th column → `score` (the ΔΔG label) for normal models.
- 5th column → zero-shot score, used by `imulan-*` models. (With exactly 4
  columns, pass `add_zs_scores` only if that 4th column is the zero-shot score.)

The bundled [examples/S1102.tsv](../examples/S1102.tsv) shows this layout:

```
1A22_A  1A22_B  SA51A  0.3480601775656975
1A22_A  1A22_B  FA165A 0.4104498198934685
```

**(b) Wild-type FASTA** — every label used in (a) must appear here:

```
>1A22_A
FPTIPLSRLFDNAMLRAHRLHQ...
>1A22_B
PKFTKCRSPERETFSCHWTLGP...
```

> Note: the FASTA in [examples/example.fasta](../examples/example.fasta) uses
> UniProt accessions and is for the `predict` CLI; for training the input is a
> FASTA whose ids match the training table (e.g. `1A22_A`/`1A22_B`).

### 4.2 Splitting data (train / val / test, or CV folds)

Use [`mulan.data.split_data`](../mulan/data.py) to produce reproducible splits or
cross-validation folds (`random_state=42` by default):

```python
from mulan.data import split_data

# single split: 70/15/15 train/val/test
split_data("data/skempi.tsv", output_dir="data/split",
           add_validation_set=True, validation_size=0.15, test_size=0.15)

# k-fold cross-validation (as in the paper)
split_data("data/skempi.tsv", output_dir="data/cv", num_folds=10)
```

### 4.3 Training command

The trainer is configured by three dataclasses (`DatasetArguments`,
`ModelArguments`, `CustomisableTrainingArguments` in
[mulan/train_utils.py](../mulan/train_utils.py)). HuggingFace's argument parser
exposes each field as `--field_name`:

```bash
export MULAN="$PWD"

mulan-train \
  --train_data        data/split/train.tsv \
  --eval_data         data/split/val.tsv \
  --test_data         data/split/test.tsv \
  --train_fasta_file  data/sequences.fasta \
  --test_fasta_file   data/sequences.fasta \
  --embeddings_dir    data/embeddings_ankh \
  --plm_model_name    ankh \
  --model_name_or_config_path models/config/lightatt_default_config.json \
  --output_dir        runs/mulan-ankh-reproduce \
  --num_epochs        30 \
  --batch_size        8 \
  --learning_rate     5e-4 \
  --early_stopping_patience 10 \
  --save_model
# or: python scripts/train.py --train_data ...
```

Key arguments:

| Argument | Default | Purpose |
|----------|---------|---------|
| `--train_data` / `--eval_data` / `--test_data` | — | split tables (§4.1) |
| `--train_fasta_file` / `--test_fasta_file` | — | WT sequences for those tables |
| `--embeddings_dir` | — | precomputed embeddings cache; **generated here if missing** (needs `--plm_model_name`) |
| `--plm_model_name` | `None` | PLM key (`ankh`, `esm`, …) used to build missing embeddings |
| `--model_name_or_config_path` | — | a model name to **fine-tune**, or a JSON config to train **from scratch** |
| `--num_epochs` | 30 | — |
| `--batch_size` | 8 | — |
| `--learning_rate` | 5e-4 | AdamW; `ReduceLROnPlateau` (factor 0.5, patience 5), weight decay 0.01 |
| `--early_stopping_patience` | `None` | epochs without val improvement before stopping |
| `--save_model` | off | write `model.ckpt` to `--output_dir` |
| `--report_to` | `none` | e.g. `tensorboard`, `wandb` |

Training internals (fixed): global seed **42**, MSE loss, metrics **MAE / RMSE /
PCC / SCC** computed each epoch, best model (by eval loss) restored at the end
when eval data + `--save_model` are given.

**Outputs** in `--output_dir`: `model.ckpt` (state dict + config, loadable with
`LightAttModel.from_pretrained`), `all_results.json` (metrics), and
`test_predictions.tsv` if `--test_data` was provided.

### 4.4 Model configs

- [models/config/lightatt_default_config.json](../models/config/lightatt_default_config.json)
  — standard MuLAN head (`add_scores: false`).
- [models/config/lightatt_scores_config.json](../models/config/lightatt_scores_config.json)
  — `imulan` head (`add_scores: true`), which additionally consumes a zero-shot
  score; use this with a 5-column training table.

### 4.5 Reproducing the SKEMPI models from the paper

1. Obtain the **SKEMPI v2** dataset and convert it to the table + FASTA format
   in §4.1 (one row per mutation with its experimental ΔΔG; for `imulan`, add a
   zero-shot column from the method of choice, e.g. an ESM log-likelihood
   ratio). The shipped `mulan_ankh.ckpt` etc. were trained this way on SKEMPI.
2. Build cross-validation folds with `split_data(..., num_folds=10)` (§4.2) for
   the reported CV numbers, or a single 70/15/15 split for a single run.
3. Precompute embeddings once with `plm-embed` (§3.4) into the directory
   pass as `--embeddings_dir`, or let the first training run generate them.
4. Train per §4.3:
   - `mulan-ankh` → `lightatt_default_config.json` + `--plm_model_name ankh`
   - `mulan-esm`  → `lightatt_default_config.json` + `--plm_model_name esm`
   - `imulan-*`   → `lightatt_scores_config.json` + a 5-column table
   - `*-multiple` → include multiple-point mutations (`:`-joined) in the table
5. Aggregate `all_results.json` / `test_predictions.tsv` across folds for the
   PCC/SCC/RMSE reported in the paper.

---

## 5. Python API (quick reference)

```python
import mulan

mulan.get_available_models()      # MuLAN checkpoints
mulan.get_available_plms()        # PLM backbones
mulan.get_device()                # mps / cuda / cpu

model = mulan.load_pretrained("mulan-ankh")            # LightAttModel
plm, tok = mulan.load_pretrained_plm("ankh")           # backbone + tokenizer

from mulan.data import MulanDataset, MulanDataCollator, split_data
from mulan.config import MulanConfig
```

---

## 6. Troubleshooting / known issues

Verified against this branch. Two long-standing entries were removed because they were fixed:
`scripts/__init__.py` now exists and `pyproject.toml` packages it as `mulan.scripts`, so the
console scripts install correctly; and [scripts/train.py](../scripts/train.py) uses
`hf_logging.get_logger`, not the standard-library `logging`.

1. **`ValueError: ... expected 4 whitespace-separated fields` from `mulan-predict`.**
   Its input is a table (`name  seqA  seqB  mut1,mut2,...`), not a FASTA. Use
   [`examples/sample_mut.txt`](../examples/sample_mut.txt); `examples/example.fasta` is for
   `mulan-att` and `plm-embed`. The error names the file, the line and the field count, and
   says so explicitly when the line looks like a FASTA record.

2. **`--store-embeddings` used to crash with the default `-o`.** Embeddings go to
   `os.path.dirname(--output-file)`, which is `""` for the default `output.txt`, and
   `os.makedirs("")` raises `FileNotFoundError`. Fixed — it falls back to `.` — but on an older checkout, pass an `-o` containing a directory, e.g. `-o output/predictions.tsv`.

3. **`RuntimeWarning: transformers X is outside the validated range 4.27-4.45.`** The T5/Ankh
   and ProstT5 prefix and token-strip behaviour is what produced every cached embedding, so a
   version outside that range may silently change what an embedding *means*. The ESM-C/ESM3 SDK
   pins `transformers<4.48.2` and therefore **must go in its own environment** — see
   [EMBEDDING_SETUP.md](EMBEDDING_SETUP.md#0-environments-venvs). Installing it alongside the base
   environment invalidates the provenance of every cached Ankh/ProstT5 embedding without any error.

4. **ESM3 and ESM-C 6B do not run on Apple Silicon.** Both are registered `mps_ok=False`
   ([mulan/plm/registry.py](../mulan/plm/registry.py)); ESM3's structure track hard-raises under
   MPS autocast. They fall back to CPU. AIDO, ESM-C 6B and MINT embeddings are generated out of
   band and read from cache — `plm-embed` is not the path for those.

5. **`ImportError: cannot import name 'structural_context'`.** That module was extracted to
   [foldenv](https://github.com/cchin29/foldenv). Install with `pip install ".[struct]"` and use
   `from foldenv import context`.

6. **First run is slow / large download.** PLM weights are fetched from HuggingFace into
   `~/.cache/huggingface/hub`. Pre-download, or use a smaller PLM (`esm_35M`) for testing.

7. **`Invalid model name` / checkpoints not found.** Make sure `MULAN` (or `MULAN_MODELS_PATH` —
   not `MULAN_MODELS_DIR`, see §1) is exported in the current shell.

---

## 7. Citation

```bibtex
@article{Lombardi2024.08.24.609515,
  author    = {Lombardi, Gianluca and Carbone, Alessandra},
  title     = {MuLAN: Mutation-driven Light Attention Networks for investigating protein-protein interactions from sequences},
  year      = {2024},
  doi       = {10.1101/2024.08.24.609515},
  publisher = {Cold Spring Harbor Laboratory},
  journal   = {bioRxiv}
}
```
