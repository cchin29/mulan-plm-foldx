"""The evaluation protocols, as data.

Which split a number came from is the single most important thing about it, and it was
previously encoded only in the *name of a directory* referenced by one of fourteen near-identical
shell configs. Two of those configs were byte-identical in length and differed in four path
strings; one protocol had no builder script at all, only a Python one-liner inside a comment.

Here each protocol is a row. The point is not to hide the builders — they are real code doing
real clustering — but to make the *properties* of a protocol answerable without reading a shell
script: what unit is held out, how many folds, what leakage it controls, and crucially whether
it is comparable to a published number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Protocol:
    name: str
    held_out_unit: str
    folds: int
    seed: int
    leakage_controlled: bool
    """Whether the protocol prevents the test complex's own mutations from being trained on."""
    homology_controlled: bool
    """Whether it additionally prevents *homologs* of the test complex from being trained on.
    A by-complex hold-out does not: the published frontier measures ~88.7% of such a test set as
    'easy' by this criterion, which is why by-complex numbers are NOT comparable to CATH ones."""
    frontier_comparable: bool
    """Whether a number from this protocol can be placed directly in a published table."""
    builder: str
    description: str

    @property
    def is_legacy(self) -> bool:
        return not self.leakage_controlled


PROTOCOLS: Tuple[Protocol, ...] = (
    Protocol(
        name="legacy_cv10",
        held_out_unit="individual mutations",
        folds=10, seed=42,
        leakage_controlled=False, homology_controlled=False, frontier_comparable=False,
        builder="mulan.data.split_data (via experiments/embedding_sweep/gen_splits.py)",
        description=(
            "Random k-fold over mutations, the upstream paper's protocol. The same complex "
            "appears in train and test, so a model can score well by memorising a complex "
            "rather than learning the effect of a mutation. Retained ONLY to reproduce the "
            "published number; no claim in this work rests on it."
        ),
    ),
    Protocol(
        name="bycomplex",
        held_out_unit="whole complexes",
        folds=3, seed=42,
        leakage_controlled=True, homology_controlled=False, frontier_comparable=False,
        builder="experiments/retrain_split/build_splits_bycomplex.py",
        description=(
            "Whole complexes are held out, with the validation set carved as whole complexes "
            "too. Controls memorising a complex's energetics from its other mutations — but "
            "close homologs of the test complex remain in training, so this is easier than it "
            "sounds and is not comparable to a CATH-split number."
        ),
    ),
    Protocol(
        name="clustered_id60",
        held_out_unit="sequence families (<=60% identity)",
        folds=3, seed=42,
        leakage_controlled=True, homology_controlled=True, frontier_comparable=True,
        builder="experiments/retrain_split/cluster_split.py",
        description=(
            "Interface chains clustered at 60% identity / 0.8 coverage, then complexes linked "
            "into families by shared cluster membership. Whole families are quarantined per "
            "fold. Matches the construction one frontier comparator uses — though that "
            "comparator reports pooled correlations only, so no per-structure comparison is "
            "possible against it."
        ),
    ),
    Protocol(
        name="cath",
        held_out_unit="CATH superfamilies",
        folds=1, seed=2024,
        leakage_controlled=True, homology_controlled=True, frontier_comparable=True,
        builder="experiments/full_skempi_seqonly/build_split_cath.py",
        description=(
            "Not our split: the partition is taken verbatim from the column USP-ddG ship with "
            "their SKEMPI table, joined onto our rows by complex. That makes a row from this "
            "repository placeable directly in their Table 1. It is a single fixed partition, "
            "not cross-validation — so it has no fold spread to measure variance against, and "
            "small per-arm differences on it should not be over-read."
        ),
    ),
)

_BY_NAME: Dict[str, Protocol] = {p.name: p for p in PROTOCOLS}


def get_protocol(name: str) -> Protocol:
    if name not in _BY_NAME:
        raise KeyError(f"Unknown protocol {name!r}. Available: {', '.join(_BY_NAME)}")
    return _BY_NAME[name]


def list_protocols(include_legacy: bool = False) -> List[str]:
    """Leakage-controlled protocols by default.

    ``legacy_cv10`` is opt-in rather than merely documented-as-legacy: the whole point of this
    work is that the default protocol should be an honest one.
    """
    return [p.name for p in PROTOCOLS if include_legacy or not p.is_legacy]


def frontier_comparable(name: str) -> bool:
    return get_protocol(name).frontier_comparable


def comparability_warning(name: str) -> Optional[str]:
    """A sentence to print beside a number from this protocol, or ``None`` if it is safe."""
    protocol = get_protocol(name)
    if not protocol.leakage_controlled:
        return (f"{name}: LEAKY — the same complex appears in train and test. "
                f"Not comparable to any published leakage-controlled number.")
    if not protocol.homology_controlled:
        return (f"{name}: leakage-controlled but NOT homology-controlled — homologs of the "
                f"test complex remain in training. Not comparable to a CATH-split number.")
    if protocol.folds == 1:
        return (f"{name}: a single fixed partition, so there is no fold spread to judge "
                f"variance against. Do not over-read small per-arm differences.")
    return None
