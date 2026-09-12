"""The protein-language-model registry — one row per backbone, one key per backbone.

Before this module, choosing a PLM meant agreeing with three independent places at once:

  * ``mulan.constants.PLM_ENCODERS`` — a name -> HuggingFace-id map, carrying nothing else;
  * ``mulan.utils.load_pretrained_plm`` — branching on substrings of the *model id* to pick a
    model class and tokenizer class;
  * ``mulan.utils.embed_sequence`` — branching **again**, on substrings of the *tokenizer's*
    ``name_or_path``, to decide the input prefix and how many leading tokens to strip.

Three surfaces meant three chances to disagree, and several backbones used in this work were
absent from the first one entirely (they were generated out of band and read from cache, which
worked only for as long as the cache stayed complete).

Here, a backbone is one :class:`PlmSpec`. The **tag** is the only key a caller ever needs; the
spec carries the model id, the embedding width, which backend loads it, how sequences are
preprocessed, how the output is trimmed, and which optional dependency it requires.

Preserving cached embeddings
----------------------------
The preprocessing and trimming fields below reproduce the *exact* behaviour of the previous
inline branches. This is not optional politeness: tens of gigabytes of cached embeddings exist,
and a run that regenerates one of them must produce a bit-identical tensor. ``tests/`` asserts
the equivalence branch by branch. If you add a backbone, add a row here — do not add an ``if``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# --- how a sequence is trimmed back to one vector per residue -------------------------------
#
# Every backend first drops whatever the tokenizer flags via `special_tokens_mask`. What differs
# is the tokens that mask does NOT flag:
STRIP_NONE = "none"
#   ProstT5's "<AA2fold>" translation prefix is a normal vocabulary token, so it survives the
#   mask and occupies exactly one leading position.
STRIP_LEADING_1 = "leading_1"
#   Ankh3's "[NLU]"/"[S2S]" prefix likewise survives — and depending on the tokenizer build, a
#   spurious leading <unk> may accompany it. Anchoring on the known residue count strips however
#   many there actually are, so the same code is correct for both tokenizer generations. A fixed
#   count silently deletes the first residue on one of them.
STRIP_TO_RESIDUE_COUNT = "to_residue_count"
#   AIDO emits a trailing [SEP] and no [CLS].
STRIP_TRAILING_1 = "trailing_1"
#   The ESM SDK returns BOS/EOS around the residues.
STRIP_BOS_EOS = "bos_eos"


@dataclass(frozen=True)
class PlmSpec:
    """Everything needed to load a backbone and turn a sequence into ``[L, dim]``."""

    tag: str
    """Canonical short name. The single key used by configs, CLIs and the tag registries."""

    backend: str
    """Which backend loads it: ``hf_t5``, ``hf_auto``, ``hf_esmc``, ``aido``, ``esm_sdk``,
    ``esmc_6b_raw``, ``saprot``, ``mint``."""

    model_id: Optional[str]
    """HuggingFace hub id, SDK model name, or ``None`` for locally-supplied weights."""

    dim: Optional[int]
    """Per-residue embedding width. ``None`` where it is not fixed."""

    prefix: str = ""
    """Literal prepended to the sequence *before* any space-joining."""

    prefix_env: Optional[str] = None
    """Environment variable overriding :attr:`prefix` (kept for Ankh3, whose prefix selects
    between the model card's ``[NLU]`` and ``[S2S]`` conditioning modes)."""

    space_join: bool = False
    """Whether residues are separated by spaces, as the ProtTrans tokenizers expect."""

    strip: str = STRIP_NONE
    """Which non-residue positions to remove after the special-token mask. See the constants."""

    extra: Optional[str] = None
    """What this backbone needs beyond the base install; ``None`` means nothing.

    Usually a pip extra (``"mint"``, ``"struct3di"``). The ESM-C / ESM3 rows instead name
    ``"requirements-esmc.txt"``, because they are **not** installable as an extra of this
    package: their SDK pins ``transformers`` below the version training uses, so generation
    runs in a separate environment. See ``docs/EMBEDDING_SETUP.md`` §0.
    """

    mps_ok: bool = True
    """Whether it runs on Apple Silicon. ``False`` backbones fall back to CPU/CUDA."""

    aliases: Tuple[str, ...] = ()
    """Historical names that resolve to this tag."""

    note: str = ""
    """Anything a caller would otherwise have to discover by reading code."""

    def resolved_prefix(self) -> str:
        """The prefix actually used, after applying :attr:`prefix_env`."""
        if self.prefix_env is not None:
            return os.environ.get(self.prefix_env, self.prefix)
        return self.prefix

    def preprocess(self, sequence: str) -> str:
        """Apply this backbone's input convention to an already-normalized sequence."""
        body = " ".join(sequence) if self.space_join else sequence
        return self.resolved_prefix() + body


def _spec(tag: str, **kw) -> PlmSpec:
    return PlmSpec(tag=tag, **kw)


# --------------------------------------------------------------------------------------------
# The registry.
#
# Ordering is by family, then by size. `dim` is the measured width of the cached tensors, not a
# number copied from a model card.
# --------------------------------------------------------------------------------------------
_SPECS: Tuple[PlmSpec, ...] = (
    # --- ESM-2 (HuggingFace, plain encoder) --------------------------------------------------
    _spec("esm", backend="hf_auto", model_id="facebook/esm2_t36_3B_UR50D", dim=2560,
          aliases=("esm2", "esm2_3b"),
          note="ESM-2 3B. The widest of the routinely-used backbones, and therefore the "
               "slowest to train on: per-step cost tracks embedding width, not parameter count."),
    _spec("esm_650M", backend="hf_auto", model_id="facebook/esm2_t33_650M_UR50D", dim=1280),
    _spec("esm_35M", backend="hf_auto", model_id="facebook/esm2_t12_35M_UR50D", dim=480),

    # --- Ankh v1 (T5 encoder) ----------------------------------------------------------------
    _spec("ankh", backend="hf_t5", model_id="ElnaggarLab/ankh-large", dim=1536,
          aliases=("ankh_large",),
          note="Ankh v1 Large — the upstream paper's reference backbone. Tokenization is 1:1 "
               "with raw residues, unlike Ankh3."),
    _spec("ankh_base", backend="hf_t5", model_id="ElnaggarLab/ankh-base", dim=768),

    # --- Ankh v3 (T5 encoder, prefix-conditioned) --------------------------------------------
    _spec("ankh3_large", backend="hf_t5", model_id="ElnaggarLab/ankh3-large", dim=1536,
          prefix="[NLU]", prefix_env="ANKH3_PREFIX", strip=STRIP_TO_RESIDUE_COUNT,
          note="Prefix-conditioned: '[NLU]' for encoder embedding extraction, '[S2S]' the "
               "alternative the model card suggests may be stronger. Existing caches were built "
               "with [NLU] (hence the *_nlu directory names) — do not change the default."),
    _spec("ankh3_xl", backend="hf_t5", model_id="ElnaggarLab/ankh3-xl", dim=2560,
          prefix="[NLU]", prefix_env="ANKH3_PREFIX", strip=STRIP_TO_RESIDUE_COUNT),

    # --- ProtTrans (space-separated residues) ------------------------------------------------
    _spec("prostt5", backend="hf_t5", model_id="Rostlab/ProstT5", dim=1024,
          prefix="<AA2fold> ", space_join=True, strip=STRIP_LEADING_1,
          note="Bilingual amino-acid/3Di model. '<AA2fold>' selects amino-acid mode; the "
               "'<fold2AA>' direction is used separately to embed 3Di strings."),
    _spec("prott5_xl_half", backend="hf_t5",
          model_id="Rostlab/prot_t5_xl_half_uniref50-enc", dim=1024, space_join=True),
    _spec("protbert", backend="hf_auto", model_id="Rostlab/prot_bert", dim=1024,
          space_join=True),

    # --- SaProt (structure-aware vocabulary) -------------------------------------------------
    _spec("saprot", backend="saprot", model_id="westlake-repl/SaProt_650M_AF2", dim=1280,
          note="ESM-2 650M architecture over a structure-aware (residue + 3Di) vocabulary, so "
               "it needs 3Di tokens alongside the sequence. Mid-pack for training speed despite "
               "its structural input — the 1280-wide embedding is what costs, not the backbone."),
    _spec("saprot_1.3b", backend="saprot",
          model_id="westlake-repl/SaProt_1.3B_AFDB_OMG_NCBI", dim=1280,
          aliases=("saprot13b",),
          note="Deeper (66 layers) but the SAME width as SaProt 650M — not wider."),

    # --- ESM Cambrian ------------------------------------------------------------------------
    _spec("esmc_600m", backend="esm_sdk", model_id="esmc_600m", dim=1152,
          strip=STRIP_BOS_EOS, extra="requirements-esmc.txt", aliases=("esmc600m",),
          note="Loaded through the EvolutionaryScale `esm` SDK, not transformers. Runs on MPS."),
    _spec("esmc_6b", backend="esmc_6b_raw",
          model_id="EvolutionaryScale/esmc-6b-2024-12", dim=2560,
          strip=STRIP_BOS_EOS, extra="requirements-esmc.txt", mps_ok=False, aliases=("esmc6b",),
          note="No released transformers version registers the 6B, and the SDK registry omits "
               "it — but the published safetensors are in native esm-package format, so they "
               "load into a hand-built ESMC(d_model=2560, n_heads=40, n_layers=80). CPU or CUDA "
               "only. Trains FASTER than ESM-2 3B despite being larger: 2560 is the same width."),

    # --- ESM3 --------------------------------------------------------------------------------
    _spec("esm3_sm_open_v1", backend="esm_sdk", model_id="esm3_sm_open_v1", dim=1536,
          strip=STRIP_BOS_EOS, extra="requirements-esmc.txt", mps_ok=False, aliases=("esm3",),
          note="Multimodal: forward() always runs the structure geometry track, whose fp32 "
               "autocast guard hard-raises on MPS. CPU only, unlike the pure-sequence ESM-C."),

    # --- AIDO --------------------------------------------------------------------------------
    _spec("aido", backend="aido", model_id="genbio-ai/AIDO.Protein-16B", dim=2304,
          mps_ok=False, strip=STRIP_TRAILING_1,
          note="Needs trust_remote_code and bfloat16; emits a trailing [SEP] and no [CLS]. "
               "Embeddings are generated offline (CPU/GPU box) and read from cache."),

    # --- MINT --------------------------------------------------------------------------------
    _spec("mint", backend="mint", model_id=None, dim=1280,
          extra="mint", aliases=("mint_mono",),
          note="Multi-chain: embeddings are partner-context and keyed per (complex, mutation), "
               "not per sequence label, so the cache layout differs (wt/<s1>__<s2>.pt and "
               "mut/<s1>__<s2>__<muts>.pt bundles). Requires the MINT source and checkpoint out "
               "of band. Enable with --mint_pair."),
)

PLM_REGISTRY: Dict[str, PlmSpec] = {s.tag: s for s in _SPECS}

_ALIASES: Dict[str, str] = {a: s.tag for s in _SPECS for a in s.aliases}


def resolve_tag(name: str) -> str:
    """Map a tag or a historical alias to the canonical tag.

    Aliases exist because the tag registries in ``experiments/`` accumulated inconsistent
    spellings — ``saprot13b`` in one file and ``saprot_1.3b`` in another for the same backbone.
    Resolving here means both keep working while there is exactly one canonical name.
    """
    if name in PLM_REGISTRY:
        return name
    if name in _ALIASES:
        return _ALIASES[name]
    raise KeyError(
        f"Unknown PLM {name!r}. Available: {', '.join(sorted(PLM_REGISTRY))}"
        + (f" (aliases: {', '.join(sorted(_ALIASES))})" if _ALIASES else "")
    )


def get_spec(name: str) -> PlmSpec:
    """Return the :class:`PlmSpec` for a tag or alias."""
    return PLM_REGISTRY[resolve_tag(name)]


def list_plms(include_aliases: bool = False) -> List[str]:
    """All canonical tags, sorted."""
    tags = sorted(PLM_REGISTRY)
    if include_aliases:
        tags += sorted(_ALIASES)
    return tags


def embedding_dim(name: str) -> Optional[int]:
    """Per-residue embedding width for a tag, or ``None`` if not fixed."""
    return get_spec(name).dim


def hub_ids() -> Dict[str, str]:
    """``{tag: model_id}`` for every backbone with an external id.

    This is the backwards-compatible view consumed by ``mulan.constants.PLM_ENCODERS``.
    """
    return {s.tag: s.model_id for s in _SPECS if s.model_id is not None}


def spec_for_model_path(name_or_path: str) -> Optional[PlmSpec]:
    """Recover a spec from a loaded tokenizer's ``name_or_path``.

    Needed only by the legacy ``embed_sequence(model, tokenizer, seq)`` entry point, which is
    handed an already-loaded pair and no tag. Matching is on the model id's last path segment so
    that a local HuggingFace snapshot directory resolves as readily as a hub id.

    Returns ``None`` for an unrecognised backbone; the caller then falls back to the historical
    substring rules, so a model that was never registered behaves exactly as it did before.
    """
    haystack = str(name_or_path).lower()
    for spec in _SPECS:
        if not spec.model_id:
            continue
        if spec.model_id == name_or_path:
            return spec
        if spec.model_id.split("/")[-1].lower() in haystack:
            return spec
    return None
