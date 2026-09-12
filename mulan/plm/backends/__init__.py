"""Backend dispatch: tag -> loaded backbone."""

from __future__ import annotations

from typing import Optional

import torch

from ..registry import PlmSpec, get_spec
from .base import Backend, UnavailableBackend, normalize_sequence

__all__ = ["Backend", "UnavailableBackend", "normalize_sequence", "load_backend"]


def resolve_device(spec: PlmSpec, device: Optional[torch.device] = None) -> torch.device:
    """Pick a device for ``spec``, honouring the backbone's own constraints.

    Some backbones cannot run on Apple Silicon at all — ESM3's structure track raises there, and
    the ESM-C 6B / AIDO paths were never validated on it. Rather than let a Mac run fail deep
    inside a forward pass, demote to CPU here.
    """
    from ...utils import get_device

    if device is None:
        device = get_device()
    device = torch.device(device)
    if device.type == "mps" and not spec.mps_ok:
        return torch.device("cpu")
    return device


def load_backend(name: str, device=None) -> Backend:
    """Load the backbone registered under ``name`` (tag or alias)."""
    spec = get_spec(name)
    device = resolve_device(spec, device)

    if spec.backend in ("hf_t5", "hf_auto", "hf_esmc"):
        from .hf import HFBackend, load_hf

        model, tokenizer = load_hf(spec, device)
        return HFBackend(spec, device, model, tokenizer)

    if spec.backend == "esm_sdk":
        from .esm_sdk import ESM3Backend, ESMCBackend

        return (ESM3Backend if spec.tag.startswith("esm3") else ESMCBackend)(spec, device)

    if spec.backend == "esmc_6b_raw":
        from .esm_sdk import ESMC6BBackend

        return ESMC6BBackend(spec, device)

    if spec.backend == "aido":
        from .aido import AIDOBackend

        return AIDOBackend(spec, device)

    if spec.backend == "saprot":
        # SaProt consumes a structure-aware vocabulary: each position is a residue token paired
        # with a 3Di token, so embedding a bare sequence is not defined. Generation needs the
        # 3Di strings alongside, which is what the dedicated generator assembles.
        return UnavailableBackend(
            spec, device,
            "SaProt needs 3Di tokens alongside the sequence. Generate with "
            "`experiments/gen_saprot_emb.py` (which needs `experiments/gen_3di.py` output "
            "first), then train against the resulting cache.",
        )

    if spec.backend == "mint":
        return UnavailableBackend(
            spec, device,
            "MINT embeddings are partner-context: they are keyed per (complex, mutation) rather "
            "than per sequence, so there is no per-sequence embedding to compute. Generate with "
            "`experiments/mint/gen_mint_emb.py` and train with --mint_pair.",
        )

    raise ValueError(f"Unknown backend {spec.backend!r} for PLM {spec.tag!r}.")
