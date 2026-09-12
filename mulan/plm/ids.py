"""Sequence-id enumeration — the naming contract between generators and the trainer.

An embedding cache is a flat directory of ``<id>.pt``, and the trainer finds a tensor purely by
constructing the same id string the generator used. That contract was previously re-implemented
in **eight** places: ``MulanDataset._fill_metadata`` (the real one) plus a copy in every
standalone generator, each carrying the comment ``== mulan.data.MulanDataset._fill_metadata``.
Copies that must agree exactly, don't — and a disagreement here is silent: the trainer simply
regenerates the "missing" embedding under its own name, or fails on a lookup, long after the
expensive generation run finished.

One implementation, imported everywhere.

The id scheme
-------------
Wild-type chains are named by their FASTA label. A mutant chain appends the mutations that fall
on *that* chain::

    1AHW_A                      wild-type chain A
    1AHW_A_YA33A-SA35G          the same chain carrying two chain-A mutations
    1AHW_B_                     chain B of a complex whose mutations are all on chain A

Note the trailing underscore in the last case: a chain with no mutations of its own still gets a
distinct label, because its *partner* changed. This is deliberate — the mutant complex's chain B
embedding is identical in content to the wild type's, but is stored separately so a mutant
complex always resolves four ids.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Sequence, Tuple


def mutant_labels(
    seq1_label: str, seq2_label: str, mutations: Sequence[str]
) -> Tuple[str, str]:
    """Return the ``(chain-A, chain-B)`` mutant labels for one row.

    Must stay identical to ``MulanDataset._fill_metadata``; ``tests/`` asserts it.
    """
    a = "-".join(m for m in mutations if m[1] == "A")
    b = "-".join(m for m in mutations if m[1] == "B")
    return f"{seq1_label}_{a}", f"{seq2_label}_{b}"


def read_mutation_table(path: str) -> List[Tuple[str, str, Tuple[str, ...]]]:
    """Read the first three columns of a mutation table: ``(seq1, seq2, mutations)``.

    Whitespace-separated and headerless, matching what the trainer consumes. Extra columns
    (labels, zero-shot scores) are ignored here.
    """
    rows = []
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            rows.append((fields[0], fields[1], tuple(fields[2].split(","))))
    return rows


def enumerate_sequence_ids(table_path: str, wt_fasta_path: str) -> Dict[str, str]:
    """Map every id referenced by a mutation table to its sequence.

    This is exactly the set of ``.pt`` files a complete embedding cache holds for that table —
    wild-type chains and mutant chains alike.

    Note that mutations on a chain other than ``A``/``B`` raise here, via
    :func:`mulan.utils.parse_mutations`. The standalone generators this replaces silently
    applied such mutations to chain B while the label logic excluded them, which aliased the
    mutant onto the wild-type embedding.
    """
    from .. import utils  # deferred: mulan.utils -> mulan.constants -> this package

    wt = utils.parse_fasta(wt_fasta_path)
    sequences: Dict[str, str] = dict(wt)
    for seq1_label, seq2_label, mutations in read_mutation_table(table_path):
        mut_seq1, mut_seq2 = utils.parse_mutations(mutations, wt[seq1_label], wt[seq2_label])
        label1, label2 = mutant_labels(seq1_label, seq2_label, mutations)
        sequences[label1] = mut_seq1
        sequences[label2] = mut_seq2
    return sequences


def missing_ids(sequences: Iterable[str], cache_dir: str) -> List[str]:
    """Which of ``sequences`` have no ``<id>.pt`` in ``cache_dir`` yet."""
    have = (
        {os.path.splitext(f)[0] for f in os.listdir(cache_dir)}
        if os.path.isdir(cache_dir)
        else set()
    )
    return [i for i in sequences if i not in have]
