"""Protein-language-model front end.

One registry, one key, one place that knows how each backbone wants its input::

    from mulan import plm

    plm.list_plms()                       # every registered tag
    spec = plm.get_spec("ankh3_large")    # dim, backend, prefix, trimming rule, extra
    backend = plm.load_backend("ankh")    # loaded model, device resolved for it
    emb = backend.embed_checked(seq)      # [1, L, dim], validated

The legacy ``mulan.utils.load_pretrained_plm`` / ``embed_sequence`` entry points still work and
now route through here, so existing scripts and caches are unaffected.
"""

from .backends import Backend, UnavailableBackend, load_backend, normalize_sequence
from .backends.hf import check_transformers
from .ids import enumerate_sequence_ids, missing_ids, mutant_labels, read_mutation_table
from .registry import (
    PLM_REGISTRY,
    PlmSpec,
    embedding_dim,
    get_spec,
    hub_ids,
    list_plms,
    resolve_tag,
    spec_for_model_path,
)

__all__ = [
    "Backend",
    "check_transformers",
    "UnavailableBackend",
    "PLM_REGISTRY",
    "PlmSpec",
    "embedding_dim",
    "enumerate_sequence_ids",
    "get_spec",
    "hub_ids",
    "list_plms",
    "load_backend",
    "missing_ids",
    "mutant_labels",
    "normalize_sequence",
    "read_mutation_table",
    "resolve_tag",
    "spec_for_model_path",
]
