"""HuggingFace-hosted backbones: the T5/Ankh encoders, the plain encoders, and ESM-C via
``transformers``."""

from __future__ import annotations

import torch

from .base import Backend


class HFBackend(Backend):
    """Wraps a ``transformers`` model + tokenizer pair.

    Accepts an already-loaded pair so the legacy ``load_pretrained_plm`` / ``embed_sequence``
    entry points can route through the same trimming logic without loading anything twice.
    """

    def __init__(self, spec, device, model, tokenizer):
        super().__init__(spec, device)
        self.model = model
        self.tokenizer = tokenizer

    def _forward(self, text: str):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=True,
            return_special_tokens_mask=True,
        ).to(self.model.device)
        outputs = self.model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        # Encoders expose `last_hidden_state`. Masked-LM-head models (ESM-C ships only an MLM
        # head) carry per-residue states in `hidden_states` instead.
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            hidden = outputs.hidden_states[-1]
        return hidden, inputs["special_tokens_mask"]


#: The T5/Ankh and ProstT5 tokenizer paths are validated against these. Newer transformers may
#: work, but the prefix/strip behaviour is what produced every cached embedding, so a mismatch is
#: worth a warning rather than silence. This replaces an upper pin that made the `esmc` extra
#: uninstallable — see pyproject.toml.
VALIDATED_TRANSFORMERS = (4, 27, 4, 45)


def check_transformers(warn=True):
    """Warn if transformers is outside the range the tokenizer handling was validated on."""
    try:
        import transformers
        parts = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    except Exception:
        return None
    lo_major, lo_minor, hi_major, hi_minor = VALIDATED_TRANSFORMERS
    ok = (lo_major, lo_minor) <= parts < (hi_major, hi_minor)
    if not ok and warn:
        import warnings
        warnings.warn(
            f"transformers {transformers.__version__} is outside the validated range "
            f"{lo_major}.{lo_minor}-{hi_major}.{hi_minor}. The T5/Ankh and ProstT5 prefix and "
            f"token-strip behaviour is what produced the cached embeddings; verify a regenerated "
            f"embedding matches before trusting it.", RuntimeWarning, stacklevel=2)
    return ok


def load_hf(spec, device):
    """Load a HuggingFace backbone according to ``spec.backend``.

    Returns ``(model, tokenizer)``; the caller wraps them in :class:`HFBackend`. The three
    branches here are the same three that used to be selected by substring-matching the model
    id — now selected by an explicit field.

    Calls :func:`check_transformers` first. It was exported and documented as a runtime guard for
    a release without being called anywhere, so the warning `docs/USAGE.md` tells a user to expect
    could not fire — on an install that resolves a transformers well outside the validated range,
    which a bare ``pip install .`` now does.
    """
    check_transformers()
    model_id = spec.model_id

    if spec.backend == "hf_t5":
        from transformers import T5EncoderModel

        model = T5EncoderModel.from_pretrained(model_id)
        if "ankh" in model_id.lower():
            from transformers import AutoTokenizer

            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id)
            except ImportError:
                # Some Ankh checkpoints ship only a slow SentencePiece tokenizer; building the
                # fast one needs protobuf. The slow tokenizer yields the same per-residue ids.
                tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        else:
            from transformers import T5Tokenizer

            tokenizer = T5Tokenizer.from_pretrained(model_id, do_lower_case=False)

    elif spec.backend == "hf_esmc":
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        model = AutoModelForMaskedLM.from_pretrained(model_id, output_hidden_states=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

    else:  # hf_auto, saprot
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)

    return model.to(device).eval(), tokenizer
