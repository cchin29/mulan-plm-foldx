"""The backend interface: load a backbone, turn a sequence into ``[1, L, dim]``."""

from __future__ import annotations

import re
from typing import Optional

import torch

from ..registry import (
    PlmSpec,
    STRIP_BOS_EOS,
    STRIP_LEADING_1,
    STRIP_TO_RESIDUE_COUNT,
    STRIP_TRAILING_1,
)


def normalize_sequence(sequence: str) -> str:
    """Uppercase, and map the non-canonical codes to ``X``.

    ``U`` (selenocysteine), ``Z``/``B`` (ambiguous Glx/Asx) and ``O`` (pyrrolysine) are not in
    every backbone's vocabulary. Substituting them is the upstream convention and every cached
    embedding in this project was produced through it, so it is load-bearing for reproducibility
    rather than a detail.
    """
    return re.sub(r"[UZOB]", "X", sequence.upper())


class Backend:
    """A loaded backbone plus the knowledge of how to embed one sequence with it.

    Subclasses implement :meth:`_forward`, returning the raw per-position hidden states together
    with whatever mask the tokenizer produced. Trimming to one vector per residue is shared, so
    the rules live in exactly one place.
    """

    def __init__(self, spec: PlmSpec, device: torch.device):
        self.spec = spec
        self.device = device

    # -- subclass contract -------------------------------------------------------------------
    def _forward(self, text: str):
        """Run the model on preprocessed ``text``.

        Returns ``(hidden, special_tokens_mask)`` where ``hidden`` is ``[1, T, dim]`` and the
        mask is ``[1, T]`` (or ``None`` when the backend already returns residues only).
        """
        raise NotImplementedError

    # -- shared ------------------------------------------------------------------------------
    @torch.inference_mode()
    def embed(self, sequence: str) -> torch.Tensor:
        """Embed one sequence, returning ``[1, L, dim]`` with one position per residue."""
        sequence = normalize_sequence(sequence)
        n_residues = len(sequence)  # captured BEFORE prefixing/space-joining
        hidden, special_mask = self._forward(self.spec.preprocess(sequence))

        if special_mask is not None:
            hidden = hidden[~special_mask.bool()].unsqueeze(0)

        strip = self.spec.strip
        if strip == STRIP_LEADING_1:
            # A translation prefix that the tokenizer does not flag as special.
            hidden = hidden[:, 1:, :]
        elif strip == STRIP_TO_RESIDUE_COUNT:
            # Strip however many leading non-residue positions there actually are. The count is
            # tokenizer-build dependent (some emit a spurious <unk> beside the prefix), so
            # anchoring on the residue count is correct for both; a fixed strip is not.
            hidden = hidden[:, hidden.shape[1] - n_residues:, :]
        elif strip == STRIP_TRAILING_1:
            hidden = hidden[:, :-1, :]
        elif strip == STRIP_BOS_EOS:
            hidden = hidden[:, 1:-1, :]

        return hidden

    def embed_checked(self, sequence: str) -> torch.Tensor:
        """:meth:`embed`, plus the two assertions every generator in this project made by hand.

        A shape mismatch here means the trimming rule is wrong for this backbone — which is a
        silent, cache-poisoning failure if it is not caught at generation time.
        """
        emb = self.embed(sequence)
        n = len(normalize_sequence(sequence))
        if emb.shape[1] != n:
            raise RuntimeError(
                f"{self.spec.tag}: got {emb.shape[1]} positions for a {n}-residue sequence. "
                f"The '{self.spec.strip}' trimming rule does not match this tokenizer."
            )
        if self.spec.dim is not None and emb.shape[2] != self.spec.dim:
            raise RuntimeError(
                f"{self.spec.tag}: embedding width {emb.shape[2]} != registered dim "
                f"{self.spec.dim}."
            )
        if not torch.isfinite(emb).all():
            raise RuntimeError(f"{self.spec.tag}: non-finite values in embedding.")
        return emb


class UnavailableBackend(Backend):
    """Placeholder for a backbone whose embeddings are produced out of band.

    Raising a specific, actionable error beats the previous behaviour, where an unregistered
    backbone failed with a generic 'Invalid model_name' — or worse, appeared to work because the
    cache happened to be complete, and failed only when a single file was missing.
    """

    def __init__(self, spec: PlmSpec, device: torch.device, how: str):
        super().__init__(spec, device)
        self.how = how

    def _forward(self, text: str):
        raise RuntimeError(
            f"{self.spec.tag!r} embeddings are not generated in-process. {self.how}"
        )
