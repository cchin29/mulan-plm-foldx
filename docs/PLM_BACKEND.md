# The PLM backend

MuLAN works with many protein language models through a two-stage separation: a **front end**
that turns a sequence into a per-residue embedding, and the **MuLAN model**, which never sees a
sequence — only cached embeddings — and is therefore backbone-agnostic by construction.

This document describes the front end. For *which generator to run for which cache*, see
[`EMBEDDING_SETUP.md`](EMBEDDING_SETUP.md).

## One registry, one key

Every backbone is one row in `mulan/plm/registry.py`. The **tag** is the only key anything
needs — configs, the CLI, the tag tables in `experiments/`:

```python
from mulan import plm

plm.list_plms()                    # every registered tag
spec = plm.get_spec("ankh3_large") # dim, backend, prefix, trimming rule, required extra
backend = plm.load_backend("ankh") # loaded, device resolved for this backbone
emb = backend.embed_checked(seq)   # [1, L, dim], shape- and finiteness-checked
```

or from the command line:

```bash
plm-embed --list
plm-embed --model prostt5 --table muts.tsv --wt-fasta wt.fasta -o emb/prostt5
```

A `PlmSpec` carries everything that used to be scattered: the model id, the embedding width,
which backend loads it, the input convention (prefix, space-joining), how the output is trimmed
back to one vector per residue, which pip extra it needs, and whether it runs on Apple Silicon.

**To add a backbone, add a row. Do not add an `if`.**

## Why it is a registry

Before, choosing a PLM meant three places agreeing at once:

| Where | Branched on | Decided |
|---|---|---|
| `constants.PLM_ENCODERS` | — | name → HuggingFace id, and nothing else |
| `utils.load_pretrained_plm` | substrings of the **model id** | model class, tokenizer class |
| `utils.embed_sequence` | substrings of the **tokenizer's** `name_or_path` | prefix, how many leading tokens to strip |

Three surfaces, three chances to disagree — and the failure mode is silent. A prefix applied
without a matching strip shifts every residue by one position, producing a correctly-shaped
tensor full of wrong numbers.

It also left real gaps. Several backbones used in this work were absent from the registry
entirely: ESM-C 600M, ESM3, MINT and SaProt-1.3B were referenced by the tag tables under names
`PLM_ENCODERS` did not contain, and worked only because their caches happened to be complete. A
single missing file surfaced as `Invalid model_name` rather than as a missing embedding. The two
tag tables also disagreed with each other — the same backbone was `saprot13b` in one and
`saprot_1.3b` in the other. Both now resolve, as aliases, to one canonical tag.

## Input conventions and trimming

The part most easily got wrong. Backbones disagree about what surrounds the residues, and the
tokenizer's `special_tokens_mask` does not flag all of it:

| Backbone | Input | Extra positions to remove |
|---|---|---|
| ESM-2, Ankh v1 | raw sequence | none |
| ProtBERT, ProtT5 | space-separated residues | none |
| ProstT5 | `<AA2fold> ` + space-separated residues | 1 leading (the prefix is an ordinary vocabulary token) |
| Ankh3 | `[NLU]` + raw sequence | anchored on the residue count |
| AIDO | raw sequence | 1 trailing (`[SEP]`, and no `[CLS]`) |
| ESM-C, ESM3 (SDK) | raw sequence | BOS and EOS |

Ankh3 is the one that must be anchored rather than counted: depending on the tokenizer build, a
spurious leading `<unk>` may accompany the prefix. Stripping `len(tokens) - len(residues)`
leading positions is correct for both builds; a hardcoded strip silently deletes the first
residue on one of them.

`embed_checked()` asserts the result has one position per residue and the registered width, so a
mismatched rule fails at generation time instead of poisoning a cache.

## Backends

| Backend | Backbones | Notes |
|---|---|---|
| `hf_t5` | Ankh v1/v3, ProstT5, ProtT5 | `T5EncoderModel`; Ankh uses `AutoTokenizer` with a slow-tokenizer fallback (the fast build needs protobuf) |
| `hf_auto` | ESM-2, ProtBERT | plain `AutoModel` |
| `hf_esmc` | ESM-C via transformers | needs `AutoModelForMaskedLM(output_hidden_states=True)` — ESM-C ships only an MLM head, so there is no `last_hidden_state` |
| `esm_sdk` | ESM-C 600M, ESM3 | EvolutionaryScale SDK; own generation environment via `requirements-esmc.txt` |
| `esmc_6b_raw` | ESM-C 6B | hand-built from the published safetensors; no released transformers registers it and the SDK registry omits it |
| `aido` | AIDO.Protein-16B | `trust_remote_code`, bfloat16 |
| `saprot` | SaProt 650M / 1.3B | structure-aware vocabulary — not embeddable from a bare sequence; see `experiments/gen_saprot_emb.py` |
| `mint` | MINT | partner-context embeddings, keyed per (complex, mutation); see `experiments/mint/gen_mint_emb.py` |

The last two raise a specific, actionable error naming the generator to use, rather than failing
generically.

## Device policy

`plm.load_backend` resolves the device once, honouring each backbone's constraints. ESM3's
forward pass always runs its structure geometry track, whose fp32 autocast guard raises on MPS;
ESM-C 6B and AIDO were never validated there. Those specs carry `mps_ok=False` and demote to CPU
rather than failing deep inside a forward pass. `MULAN_FORCE_CPU=1` overrides globally.

## The model side absorbs the width

Nothing downstream needs to know the embedding width: the first convolution is lazily shaped on
first use, so a new backbone needs no model-side change and no config edit. This is why one
training recipe spans backbones from 480 to 2560 dimensions — and why per-step training cost
tracks the *width*, not the parameter count. ESM-C 6B trains faster than ESM-2 3B: both are
2560-wide, and the PLM does not run in the training loop at all.

## Compatibility

`constants.PLM_ENCODERS`, `utils.load_pretrained_plm` and `utils.embed_sequence` all still work
with their original signatures and now route through the registry; `PLM_ENCODERS` is a derived
view of it.

This is verified, not asserted. `tests/test_plm_registry.py` compares every registered backbone
against the original inline branches, copied verbatim into the test as the reference. On real
weights, ProstT5 and Ankh3-large — the two hardest paths, both carrying a prefix *and* a trimming
rule — produce **bit-for-bit identical** tensors before and after the refactor, across short,
single-residue, long, and non-canonical-residue sequences.

> One caveat that predates this work and is worth knowing when comparing any embedding against a
> cached one: **MPS and CPU do not produce identical floats.** Regenerating an MPS-built cache
> entry on CPU gives a max absolute difference around 2e-6 — with the old code and the new code
> alike. Compare within a platform, or compare with a tolerance.
