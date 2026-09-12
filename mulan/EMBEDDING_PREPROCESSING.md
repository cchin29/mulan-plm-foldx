# Embedding preprocessing across PLM models

How each protein language model's **input is preprocessed** before the forward pass, and how
its **per-residue output is extracted**. This covers both layers: the shared
`mulan.utils.{embed_sequence, load_pretrained_plm}` used across the project, and the
[foldenv](https://github.com/cchin29/foldenv) embedding registry (`embedding.EMBEDDING_MODELS`) that selects among
them and adds a dedicated structure-aware path for SaProt. Every non-obvious choice below has
bitten someone; the gotchas and env conflicts are at the end.

## Summary table

| registry name | model (HF id) | dim | prefix token | residue formatting | non-residue tokens stripped | structure input | loaded as → hidden state | verified¹ |
|---|---|---|---|---|---|---|---|---|
| `ankh` *(default)* | Ankh-large (`ElnaggarLab/ankh-large`) | 1536 | — (1:1 raw) | raw chars | special-mask only | — | `T5EncoderModel` → `last_hidden_state` | ✅ 70× |
| `ankh3_large` | Ankh3-large (`ElnaggarLab/ankh3-large`) | 1536 | **`[NLU]`** (or `[S2S]`) | raw chars, prefixed | special-mask **+ N leading (anchored)** | — | `T5EncoderModel` | ✅ 63× |
| `ankh3_xl` | Ankh3-XL (`ElnaggarLab/ankh3-xl`) | 2560 | **`[NLU]`** | raw chars, prefixed | special-mask + N leading | — | `T5EncoderModel` | ✅ 80× |
| `prostt5_aa` | ProstT5 AA-mode (`Rostlab/ProstT5`) | 1024 | **`<AA2fold>`** | **space-joined** | special-mask **+ 1 leading** | — | `T5EncoderModel` | ✅ 63× |
| `saprot` | SaProt-650M (`westlake-repl/SaProt_650M_AF2`) | 1280 | — | **SA tokens** (AA+3Di, 2 chars/res) | special-mask only | **3Di (mini3di)** | `EsmForMaskedLM` → `hidden_states[-1]` | ✅ 66× |
| `saprot_1.3b` | SaProt-1.3B (`westlake-repl/SaProt_1.3B_AFDB_OMG_NCBI`) | 1280 | — | **SA tokens** | special-mask only | **3Di (mini3di)** | `EsmForMaskedLM` → `hidden_states[-1]` | ✅ 87× |
| `esm2_3b` | ESM2-3B (`facebook/esm2_t36_3B_UR50D`) | 2560 | — | raw chars | special-mask only | — | `AutoModel` (EsmModel) → `last_hidden_state` | ✅ 14× |
| `esm2_650m` | ESM2-650M (`facebook/esm2_t33_650M_UR50D`) | 1280 | — | raw chars | special-mask only | — | `AutoModel` | wired² |
| `esmc_6b` | ESM C 6B (`EvolutionaryScale/esmc-6b-2024-12`) | 2560 | — | raw chars | special-mask only | — | `AutoModelForMaskedLM` → `hidden_states[-1]` | env³ |

¹ *Verified by forward pass on TEM-1 (P62593) + a row↔residue alignment check: an
L150A point mutation must perturb the embedding most at row 150. The number is the ratio of the
mutated row's change to the median row change (higher = sharper localization; all argmax = 150).*
² *`esm2_650m` wired but not cached locally (not run).*  ³ *ESM C needs `transformers≥4.57` — a
separate env (`.venv-esmc`); never on MPS. See conflicts below.*

## Common preprocessing (all models)

In `embed_sequence`, before any model-specific step:
- `sequence.upper()`
- `re.sub("[UZOB]", "X", …)` — non-canonical/ambiguous residues (Sec/Pyl, B, Z, O) → `X`.
- Per-residue count `n_residues` is captured here (before prefixing/space-joining) and used to
  anchor the ankh3 strip (see gotcha #1).

Output extraction: run the model, take `last_hidden_state` if present, else `hidden_states[-1]`
(masked-LM-head models like ESM C have no `last_hidden_state`). Then drop tokens flagged by
`special_tokens_mask` (BOS/EOS/pad), and finally strip any **prefix** tokens the mask misses.

## Per-model details

### Ankh-large (`ankh`) — the tool default
No prefix, no spacing — the tokenizer is 1:1 with residues. Only special tokens are stripped.
Chosen as the default: similar size to ProstT5 but stronger on ΔΔG (S1102 CV10).

### Ankh3 (`ankh3_large`, `ankh3_xl`)
**Prefix-conditioned.** Prepend a task prefix — `[NLU]` (default, best for embedding extraction
per the model card) or `[S2S]` — selectable via the `ANKH3_PREFIX` env var. Residues are raw
(not spaced). The prefix token is **not** flagged special, so it must be stripped after the
forward pass. **The number of leading tokens to strip is tokenizer-version dependent** (see
gotcha #1) — we strip `n_nonspecial − n_residues` rather than a hardcoded count.

### ProstT5 AA-mode (`prostt5_aa`)
Bilingual AA↔3Di model; we use **AA mode**: prepend `<AA2fold>` and **space-join** residues
(`"<AA2fold> M K T A Y …"`). The `<AA2fold>` prefix isn't special-masked → strip **1** leading
token. (3Di/fold mode uses `<fold2AA>` and lowercase 3Di — not used here; that's the separate
structure-channel experiment.)

### SaProt (`saprot`, `saprot_1.3b`) — structure-aware, NOT via `embed_sequence`
Dedicated path in `structural_context/embedding.py`:
1. **3Di from the AlphaFold backbone** via **mini3di** (`structure_to_3di`): encode N/CA/C/CB
   coords → one lowercase 3Di letter per residue; string length forced to `len(seq)` with a
   valid placeholder (`d`) for un-encodable residues. Glycine has no CB → passed as NaN (mini3di
   handles it).
2. **SA (structure-aware) sequence** (`sa_sequence`): interleave `AA(upper) + 3Di(lower)` into
   one **2-character token per residue** (e.g. `Md Kp …`). A non-canonical AA has no valid SA
   token → map its AA half to the **`#` mask** (`#d`), the idiomatic "unknown residue" — critical
   because two adjacent OOV pairs would collapse into one token and break per-residue alignment.
3. Forward through `EsmForMaskedLM` (ESM-2 arch + SA vocab), take `hidden_states[-1]`, strip
   special tokens. 1.3B is *deeper* (66 layers) not wider — same 1280-d as 650M.

**Why WT 3Di (mini3di) is sufficient — the FoldX mutant-3Di no-op.** The 3Di structure half is
computed once from the **WT/AlphaFold backbone** (mini3di, a pure-Python Foldseek-3Di encoder) and
reused for every variant of a chain; the point mutation is carried entirely by the **AA half** of
the SA token. The obvious objection is "shouldn't the mutant have its own 3Di?" — so the project
tested exactly that with FoldX (`RepairPDB` → `BuildModel(mutant)` → mini3di), and it was a
**confirmed no-op**: a probe of **39 mutations** — all 36 of 1A22's S1102 mutations *plus* the two
most backbone-perturbing substitutions in the set (1ACB Leu38→Pro and →Gly) — changed the 3Di at
**0 / 39 positions**, including →Pro/→Gly and even *at the mutated residue itself* (e.g. L38P: pos-38
3Di `p`→`p`). Reason: FoldX `BuildModel` is a **side-chain repacker** with sub-Ångström backbone
motion, while Foldseek-3Di is a coarse **20-state backbone** descriptor — sub-Å moves never cross a
3Di state boundary, so **FoldX-mutant 3Di ≡ WT 3Di** (a property of FoldX + 3Di, dataset-independent;
CV10 would be byte-identical to Stage-1's 0.842). So WT 3Di *is* the mutant 3Di here; capturing real
mutation-induced 3Di change would need a **backbone-relaxing** method (energy minimization / MD, or
AlphaFold-on-mutant), not FoldX. (RESULTS.md §15; the SaProt+FoldX route is deferred.)

The 3Di half is **backbone-only**, so a point mutation that doesn't move the backbone leaves the
3Di unchanged; the SA token still flips because its AA half changes (confirmed: L150A → 66–87×
localization at row 150).

### ESM2 (`esm2_3b`, `esm2_650m`)
Plain sequence — no prefix, no spacing. Tokenizer adds BOS/EOS (special-masked). Loaded via
`AutoModel` (→ `EsmModel`), read `last_hidden_state`. Note ESM2 gives a **more diffuse** mutation
response than the T5 encoders (esm2_3b ratio ~14× vs Ankh ~70×) — still correctly argmax-aligned,
just softer; the length match (L rows) is the real alignment guarantee. (A benign warning about
`pooler` weights being newly initialized is expected — the pooler is unused for per-residue states.)

### ESM C 6B (`esmc_6b`)
Masked-LM head only (`AutoModelForMaskedLM`, `output_hidden_states=True`) — per-residue states
come from `hidden_states[-1]`, not `last_hidden_state`. Flagged `mps_ok=False` (too large /
MPS-incompatible) → routed to CPU/CUDA. **Needs `transformers≥4.57`** (native `esmc` model_type).

## Gotchas & lessons learned

1. **Ankh3 leading-token strip is protobuf/tokenizer-version dependent (the off-by-one).** The
   ankh3 tokenizer's leading tokens differ by tokenizer build:
   - **fast** tokenizer (`T5TokenizerFast`, needs **protobuf**): emits `[NLU]` **+ a spurious
     `<unk>`** + residues → **2** leading non-residue tokens.
   - **slow** tokenizer (no protobuf → `use_fast=False` fallback): emits `[NLU]` + residues →
     **1** leading token.
   The old code hardcoded stripping **2**, which **dropped the first residue** (L−1, misaligned)
   in any protobuf-less env. Fixed by anchoring on the residue count (`n_nonspecial − n_residues`)
   → correct in both. **The recorded Ankh3 sweep results are valid** (they ran in `.venv` with
   protobuf → fast tokenizer → strip-2 was correct; all cached ankh3 tensors are length L).
   Lesson: never hardcode a prefix-strip count; anchor on the known residue count. The tool's
   length guard (`per_res rows == len(seq)`) is what surfaced this — keep such guards.

2. **Prefix tokens are not in `special_tokens_mask`.** `<AA2fold>` (ProstT5) and `[NLU]`/`[S2S]`
   (Ankh3) look like normal tokens to the mask, so they survive the special-token filter and must
   be stripped explicitly. Only BOS/EOS/pad are masked.

3. **SaProt non-canonical residues must be `#`-masked, not `X`.** SaProt's SA vocab only has the
   20 canonical AAs in the AA half; an `X` has no valid 2-char token and adjacent OOV pairs merge,
   desyncing the per-residue length. Map non-canonical → `#`.

4. **mini3di needs N/CA/C (CB optional).** Residues missing backbone atoms can't be encoded →
   filled with a valid placeholder 3Di letter (`d`) to preserve length parity with the AA string.

5. **ESM2 mutation signal is diffuse** — don't treat "argmax exactly at the mutated residue with
   huge dominance" as the only validity check for every model; the **length == n_residues** check
   is the robust alignment guarantee, the argmax/ratio is a secondary sanity signal.

6. **Non-canonical AAs are globally coerced to `X`** (`[UZOB]→X`) before anything else — keep this
   in mind when comparing to external pipelines that keep U/O.

## venv / library conflicts

- **`transformers` version split.** The repo pins `transformers<4.45,>=4.27` (works for Ankh,
  Ankh3, ProstT5, ESM2, SaProt). **ESM C 6B needs `transformers≥4.57`** — the two cannot coexist,
  so ESM C runs in a **separate venv** (`.venv-esmc`, currently transformers 5.12.1). Don't bump
  the main pin.
- **protobuf → fast vs slow tokenizer (ankh3).** With protobuf installed, ankh3 loads the fast
  tokenizer (emits the `<unk>`); without it, the slow tokenizer (no `<unk>`). Post-fix both are
  aligned, but they are **numerically different** embeddings (the `<unk>` changes the model's input
  context). To reproduce the sweep's ankh3 embeddings exactly, install protobuf (fast tokenizer).
  `.venv-structctx` currently has **no** protobuf → slow ankh3 tokenizer.
- **sentencepiece** is required for the T5/Ankh tokenizers; **mini3di** for the SaProt 3Di channel.
- **Python 3.11** for the whole embedding stack (torch/transformers wheels; 3.14 has none yet).
- **MPS:** `PYTORCH_ENABLE_MPS_FALLBACK=1` is set in `mulan/__init__`. ESM C is forced off MPS
  (`mps_ok=False` → CPU/CUDA). ESM2-3B and the others run on MPS/CPU/CUDA fine.
- **Model classes differ by architecture** (see the table): T5 encoders → `T5EncoderModel`; ESM2 →
  `AutoModel`/`EsmModel` with `last_hidden_state`; SaProt & ESM C → `*ForMaskedLM` with
  `output_hidden_states` (no `last_hidden_state`). Loader logic lives in `mulan.utils.load_pretrained_plm`.

_Companion docs: `structural_context/README.md` (registry + device routing),
`structural_context/SETUP_NOTES.md` (cross-machine setup), `../docs/PLM_BACKEND.md` /
`../docs/history/PLM_COMPARISON_S1102.md` (repo-level PLM notes)._
