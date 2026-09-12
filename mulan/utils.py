"""Util functions to process data and models."""

from typing import Dict, Iterator, List, Optional, Tuple, Union

import os
import re
import torch
import numpy as np
from scipy.stats import rankdata
from transformers import PreTrainedTokenizerBase, PreTrainedModel

import mulan.constants as C
from mulan.constants import AAs, aa2idx, idx2aa, one2three, three2one


TorchDevice = Union[str, torch.device]


def get_device() -> torch.device:
    """Return the best available device, preferring Apple Silicon (MPS), then CUDA, then CPU.

    Centralizes device selection so a Mac run lands on MPS instead of silently falling back to
    CPU. Note that argument defaults are evaluated at definition time, so callers that want this
    must default the ``device`` argument to ``None`` and resolve it in-body via this helper.
    """
    if os.environ.get("MULAN_FORCE_CPU") == "1":
        return torch.device("cpu")  # opt-in override (e.g. MPS-vs-CPU parity, or freeing MPS)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def mutation_generator(sequence: str) -> Iterator[Tuple[str, str]]:
    """Generate all possible single-point mutations for a given sequence.
    Args:
        sequence (str): The input sequence for which mutations are generated.
    Yields:
        A tuple containing the mutation identifier and the mutated sequence.
    """
    for i, aa in enumerate(sequence):
        for new_aa in C.AAs:
            if new_aa != aa:
                yield (f"{aa}{i + 1}{new_aa}", sequence[:i] + new_aa + sequence[i + 1 :])


def parse_mutations(mutations: Tuple[str], seq1: str, seq2: str) -> Tuple[str, str]:
    """
    Parses a list of mutations and applies them to two sequences.
    Args:
        mutations (Tuple[str]): A tuple of strings representing the mutations.
            Each mutation should be in the format '<wt_aa><chain:A,B><position><mut_aa>'.
            Example: 'AB23G'.
        seq1 (str): The first sequence.
        seq2 (str): The second sequence.
    Returns:
        Tuple[str, str]: A tuple representing the modified sequences.
    """
    seq1, seq2 = list(seq1), list(seq2)
    for single_mut in mutations:
        chain = single_mut[1]
        if chain == "A":
            seq1[int(single_mut[2:-1]) - 1] = single_mut[-1]
        elif chain == "B":
            seq2[int(single_mut[2:-1]) - 1] = single_mut[-1]
        else:
            # Reject any non-A/B chain code loudly. The old `else` applied it to seq2,
            # but the embedding-id labels in data.py filter on chain=='A'/'B', so a
            # non-A/B code was mutated in the sequence yet excluded from BOTH labels —
            # silently aliasing the mutant onto the wild-type embedding.
            raise ValueError(
                f"Unexpected chain code {chain!r} in mutation {single_mut!r}; expected "
                "'A' or 'B'."
            )
    return "".join(seq1), "".join(seq2)


def alphabetic_tokens_permutation(tokenizer: PreTrainedTokenizerBase) -> List[int]:
    """Get indices of standard amino acids in alphabetic order for input tokenizer."""
    vocab = tokenizer.get_vocab()
    aas_idx = [vocab[tok] for tok in C.AAs]
    return aas_idx


def parse_fasta(fasta_file: str) -> Dict[str, str]:
    """Parse a fasta file and return a dictionary."""
    with open(fasta_file) as f:
        lines = f.readlines()
    fasta_dict = {}
    for line in lines:
        if line.startswith("#") or not line.strip():
            continue
        if line.startswith(">"):
            key = line.strip().split()[0][1:]
            key = key.split("|")[0]
            fasta_dict[key] = ""
        else:
            fasta_dict[key] += line.strip().upper()
    return fasta_dict


def dict_to_fasta(fasta_dict: Dict[str, str], fasta_file: str):
    """Write a dictionary to a fasta file."""
    with open(fasta_file, "w") as f:
        for key, value in fasta_dict.items():
            f.write(f">{key}\n")
            f.write(f"{value}\n")


def get_available_plms() -> List[str]:
    """Return the names of the available pretrained language models.

    Sourced from `mulan.plm.registry`, so it includes backbones that have no HuggingFace id —
    MINT ships its weights out of band and would be invisible in the hub-id view.
    """
    from mulan.plm import list_plms

    return list_plms()


def get_available_models() -> List[str]:
    """Return the names of the available Mulan models."""
    return list(C.MODELS.keys())


def load_pretrained_plm(model_name: str, device: Optional[TorchDevice] = None):
    """Load a PLM by tag, returning ``(model, tokenizer)``.

    Retained with its original signature — `MulanDataset` and several generators call it — but
    the model-class and tokenizer-class selection now comes from `mulan.plm.registry` instead of
    substring-matching the model id. Behaviour for every previously-supported tag is unchanged.

    Only backbones with a `transformers` loader return a tokenizer. The SDK-loaded ones
    (ESM-C, ESM3), AIDO, SaProt and MINT have no equivalent pair; use
    :func:`mulan.plm.load_backend` for those, which returns an object exposing ``.embed()``.
    """
    from mulan.plm import get_spec
    from mulan.plm.backends import resolve_device
    from mulan.plm.backends.hf import load_hf

    try:
        spec = get_spec(model_name)
    except KeyError as exc:
        raise ValueError(
            f"Invalid model_name: {model_name}. Must be one of {get_available_plms()}"
        ) from exc

    if spec.backend not in ("hf_t5", "hf_auto", "hf_esmc", "saprot"):
        raise ValueError(
            f"{spec.tag!r} does not load as a (model, tokenizer) transformers pair "
            f"(backend={spec.backend!r}). Use mulan.plm.load_backend({spec.tag!r}) instead."
        )
    return load_hf(spec, resolve_device(spec, device))


def load_pretrained(pretrained_model_name: str, device: TorchDevice = None, **kwargs):
    """Load a pretrained model from disk."""
    model_path = C.MODELS.get(pretrained_model_name)
    if model_path is None:
        raise ValueError(f"Invalid model_name: {pretrained_model_name}")
    from mulan.modules import LightAttModel

    return LightAttModel.from_pretrained(model_path, device=device, **kwargs)


@torch.inference_mode()
def embed_sequence(
    plm_model: PreTrainedModel, plm_tokenizer: PreTrainedTokenizerBase, sequence: str
):
    """Embed a sequence with an already-loaded (model, tokenizer) pair -> ``[1, L, dim]``.

    The input convention (prefix, space-joining) and the trimming rule now come from the
    registry, recovered from the tokenizer's ``name_or_path``. A backbone that is not registered
    falls back to the historical substring rules below, so nothing that worked before stops
    working — but a registered backbone is described in exactly one place.
    """
    from mulan.plm import spec_for_model_path
    from mulan.plm.backends.hf import HFBackend
    from mulan.plm.registry import (
        PlmSpec, STRIP_LEADING_1, STRIP_NONE, STRIP_TO_RESIDUE_COUNT,
    )

    name_or_path = str(plm_tokenizer.name_or_path)
    spec = spec_for_model_path(name_or_path)
    if spec is None:
        # Unregistered backbone: reproduce the original inline branches verbatim.
        if "ProstT5" in name_or_path:
            spec = PlmSpec(tag="<unregistered>", backend="hf_auto", model_id=None, dim=None,
                           prefix="<AA2fold> ", space_join=True, strip=STRIP_LEADING_1)
        elif "ankh3" in name_or_path.lower():
            spec = PlmSpec(tag="<unregistered>", backend="hf_auto", model_id=None, dim=None,
                           prefix="[NLU]", prefix_env="ANKH3_PREFIX",
                           strip=STRIP_TO_RESIDUE_COUNT)
        elif "Rostlab/prot" in name_or_path:
            spec = PlmSpec(tag="<unregistered>", backend="hf_auto", model_id=None, dim=None,
                           space_join=True, strip=STRIP_NONE)
        else:
            spec = PlmSpec(tag="<unregistered>", backend="hf_auto", model_id=None, dim=None)

    return HFBackend(spec, plm_model.device, plm_model, plm_tokenizer).embed(sequence)


def save_embedding(embedding: torch.Tensor, output_dir: str, name: str):
    """Save an embedding to disk."""
    embedding = embedding.squeeze(0).cpu()
    torch.save(embedding, os.path.join(output_dir, name + ".pt"))


def ranksort(array: np.ndarray) -> np.ndarray:
    """Ranksort an array."""
    return (rankdata(array) / array.size).reshape(array.shape)


def minmax_scale(array: np.ndarray) -> np.ndarray:
    """Min-max scale an array."""
    return (array - array.min()) / (array.max() - array.min())
